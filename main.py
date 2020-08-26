import logging
import os
import sys
from typing import Callable, Dict, Optional

import numpy as np
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    EvalPrediction,
    glue_compute_metrics,
    set_seed
)


from args import MultimodalDataTrainingArguments, ModelArguments, OurTrainingArguments
from eval import calc_classification_metrics, calc_regression_metrics
from load_data import load_data_from_folder
from model.multimodal_config import TabularConfig
from model.multimodal_modeling_auto import AutoModelWithTabular

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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    train_dataset, val_dataset, test_dataset = load_data_from_folder(
        data_args.data_path,
        data_args.column_info['text_cols'],
        tokenizer,
        label_col=data_args.column_info['label_col'],
        categorical_cols=data_args.column_info['cat_cols'],
        numerical_cols=data_args.column_info['num_cols'],
        sep_text_token_str=tokenizer.sep_token,
        max_token_length=training_args.max_token_length
    )

    set_seed(training_args.seed)
    task = data_args.task
    if task == 'regression':
        num_labels = 1
    else:
        num_labels = len(np.unique(train_dataset.labels))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    tabular_config = TabularConfig(num_labels=num_labels,
                                   cat_feat_dim=train_dataset.cat_feats.shape[1],
                                   numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                                   **vars(data_args))
    config.tabular_config = tabular_config

    model = AutoModelWithTabular.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        config=config
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

    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )

    print('here')


if __name__ == '__main__':
    main()