import os
import sys

sys.path.append("./")
from typing import Callable, Dict

import numpy as np
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    EvalPrediction,
    set_seed,
)

from multimodal_exp_args import (
    MultimodalDataTrainingArguments,
    ModelArguments,
    OurTrainingArguments,
)
from evaluation import calc_classification_metrics, calc_regression_metrics
from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular

import pytest

os.environ["COMET_MODE"] = "DISABLED"

DEBUG_DATASET_SIZE = 50

CONFIGS = [
    "./tests/test_airbnb.json",
    "./tests/test_clothing.json",
    "./tests/test_petfinder.json",
]

MODELS = [
    "albert-base-v2",
    "bert-base-multilingual-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "xlm-mlm-100-1280",
    "xlm-roberta-base",
    "xlnet-base-cased",
]


def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        # p.predictions is now a list of objects
        # The first entry is the actual predictions
        predictions = p.predictions[0]
        if task_name == "classification":
            preds_labels = np.argmax(predictions, axis=1)
            if predictions.shape[-1] == 2:
                pred_scores = softmax(predictions, axis=1)[:, 1]
            else:
                pred_scores = softmax(predictions, axis=1)
            return calc_classification_metrics(pred_scores, preds_labels, p.label_ids)
        elif task_name == "regression":
            preds = np.squeeze(predictions)
            return calc_regression_metrics(preds, p.label_ids)
        else:
            return {}

    return compute_metrics_fn


@pytest.mark.parametrize("json_file", CONFIGS)
@pytest.mark.parametrize("model_string", MODELS)
def test_model(json_file: str, model_string: str):
    # Parse our input json files
    parser = HfArgumentParser(
        (ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath(json_file)
    )

    # Set model string
    # We don't use the value from the config here since we test multiple models
    training_args.experiment_name = model_string
    model_args.model_name_or_path = model_string
    model_args.tokenizer_name = model_string

    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Load and preprocess datasets
    # We force debug=True so we load only DEBUG_DATASET_SIZE entries
    train_dataset, val_dataset, test_dataset = load_data_from_folder(
        data_args.data_path,
        data_args.column_info["text_cols"],
        tokenizer,
        label_col=data_args.column_info["label_col"],
        label_list=data_args.column_info["label_list"],
        categorical_cols=data_args.column_info["cat_cols"],
        numerical_cols=data_args.column_info["num_cols"],
        categorical_encode_type=data_args.categorical_encode_type,
        numerical_transformer_method=data_args.numerical_transformer_method,
        sep_text_token_str=tokenizer.sep_token
        if not data_args.column_info["text_col_sep_token"]
        else data_args.column_info["text_col_sep_token"],
        max_token_length=training_args.max_token_length,
        debug=True,
        debug_dataset_size=DEBUG_DATASET_SIZE,
    )

    set_seed(training_args.seed)
    task = data_args.task

    # Regression tasks have only one "label"
    if task == "regression":
        num_labels = 1
    else:
        num_labels = (
            len(np.unique(train_dataset.labels))
            if data_args.num_classes == -1
            else data_args.num_classes
        )

    # Setup configs
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tabular_config = TabularConfig(
        num_labels=num_labels,
        cat_feat_dim=train_dataset.cat_feats.shape[1]
        if train_dataset.cat_feats is not None
        else 0,
        numerical_feat_dim=train_dataset.numerical_feats.shape[1]
        if train_dataset.numerical_feats is not None
        else 0,
        **vars(data_args)
    )
    config.tabular_config = tabular_config

    # Make model
    model = AutoModelWithTabular.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model_path = (
        model_args.model_name_or_path
        if os.path.isdir(model_args.model_name_or_path)
        else None
    )

    # Make trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=build_compute_metrics_fn(task),
    )

    # Train
    trainer.train(resume_from_checkpoint=model_path)

    # Get predictions
    test_results = trainer.predict(test_dataset=test_dataset)
    assert test_results.predictions[0].shape == (DEBUG_DATASET_SIZE, num_labels)
