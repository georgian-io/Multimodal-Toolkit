import logging
import os
import sys
from typing import Callable, Dict

import numpy as np
from pprint import pformat
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    EvalPrediction,
    set_seed
)

from multimodal_exp_args import MultimodalDataTrainingArguments, ModelArguments, OurTrainingArguments
from evaluation import calc_classification_metrics, calc_regression_metrics
from multimodal_transformers.data.load_data import load_data_from_folder
from multimodal_transformers.model.tabular_config import TabularConfig
from multimodal_transformers.model.tabular_modeling_auto import AutoModelWithTabular
from util import create_dir_if_not_exists, get_args_info_as_str

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, MultimodalDataTrainingArguments,
                               OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    create_dir_if_not_exists(training_args.output_dir)
    stream_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename=os.path.join(training_args.output_dir, 'train_log.txt'),
                                       mode='w+')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.DEBUG if training_args.local_rank in [-1, 0] else logging.WARN,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[stream_handler, file_handler]
    )

    logger.info(f"======== Model Args ========\n{get_args_info_as_str(model_args)}\n")
    logger.info(f"======== Data Args ========\n{get_args_info_as_str(data_args)}\n")
    logger.info(f"======== Training Args ========\n{get_args_info_as_str(training_args)}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    train_dataset, val_dataset, test_dataset = load_data_from_folder(
        data_args.data_path,
        data_args.column_info['text_cols'],
        tokenizer,
        label_col=data_args.column_info['label_col'],
        label_list=data_args.column_info['label_list'],
        categorical_cols=data_args.column_info['cat_cols'],
        numerical_cols=data_args.column_info['num_cols'],
        categorical_encode_type=data_args.categorical_encode_type,
        numerical_transformer_method=data_args.numerical_transformer_method,
        sep_text_token_str=tokenizer.sep_token,
        max_token_length=training_args.max_token_length,
        debug=training_args.debug,
    )

    set_seed(training_args.seed)
    task = data_args.task
    if task == 'regression':
        num_labels = 1
    else:
        num_labels = len(np.unique(train_dataset.labels))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tabular_config = TabularConfig(num_labels=num_labels,
                                   cat_feat_dim=train_dataset.cat_feats.shape[1],
                                   numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                                   **vars(data_args))
    config.tabular_config = tabular_config
    logger.info(tabular_config)

    model = AutoModelWithTabular.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )
    logger.info(model)

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if task_name == "classification":
                preds_labels = np.argmax(p.predictions, axis=1)
                pred_scores = softmax(p.predictions, axis=1)[:, 1]
                return calc_classification_metrics(pred_scores, preds_labels,
                                                   p.label_ids)
            elif task_name == "regression":
                preds = np.squeeze(p.predictions)
                return calc_regression_metrics(preds, p.label_ids)
            else:
                return {}
        return compute_metrics_fn

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=build_compute_metrics_fn(task),
    )
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_result = trainer.evaluate(eval_dataset=val_dataset)
        logger.info(pformat(eval_result, indent=4))

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_metric_results_{task}.txt"
        )
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(task))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")

        predictions = trainer.predict(test_dataset=test_dataset).predictions
        output_test_file = os.path.join(
            training_args.output_dir, f"test_results_{task}.txt"
        )
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        logger.info(pformat(eval_result, indent=4))
        if trainer.is_world_master():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results {} *****".format(task))
                writer.write("index\tprediction\n")
                if task == "classification":
                    predictions = np.argmax(predictions, axis=1)
                for index, item in enumerate(predictions):
                    if task == "regression":
                        writer.write("%d\t%3.3f\t%d\n" % (index, item, test_dataset.labels[index]))
                    else:
                        item = test_dataset.get_labels()[item]
                        writer.write("%d\t%s\n" % (index, item))
            output_test_file = os.path.join(
                training_args.output_dir, f"test_metric_results_{task}.txt"
            )
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results {} *****".format(task))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)

    return eval_results


if __name__ == '__main__':
    main()
