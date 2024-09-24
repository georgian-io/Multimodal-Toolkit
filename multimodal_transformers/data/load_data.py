import logging
from functools import partial
from os import makedirs
from os.path import exists, join
from typing import List, Optional, Tuple

import joblib
import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import KFold, train_test_split

from .data_utils import (
    CategoricalFeatures,
    NumericalFeatures,
    agg_text_columns_func,
    convert_to_func,
    get_matching_cols,
)
from .tabular_torch_dataset import TorchTabularTextDataset

logger = logging.getLogger(__name__)


def load_data_into_folds(
    data_csv_path: str,
    num_splits: int,
    validation_ratio: float,
    text_cols: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    label_col: str,
    label_list: Optional[List[str]] = [],
    categorical_cols: Optional[List[str]] = [],
    numerical_cols: Optional[List[str]] = [],
    sep_text_token_str: str = " ",
    categorical_encode_type: str = "ohe",
    categorical_handle_na: bool = False,
    categorical_na_value: str = "-9999999",
    ohe_handle_unknown: str = "error",
    numerical_transformer_method: str = "quantile_normal",
    numerical_handle_na: bool = False,
    numerical_how_handle_na: str = "value",
    numerical_na_value: float = 0.0,
    empty_text_values: Optional[List[str]] = None,
    replace_empty_text: Optional[str] = None,
    max_token_length: Optional[int] = None,
    debug: bool = False,
    debug_dataset_size: int = 100,
    output_dir: Optional[str] = None,
) -> Tuple[
    List[TorchTabularTextDataset],
    List[Optional[TorchTabularTextDataset]],
    List[TorchTabularTextDataset],
]:
    """
    Load tabular and text data from a specified folder into folds.

    This function loads training, testing, and optionally validation data from a CSV file into the specified number of cross-validation folds. The function performs categorical and numerical preprocessing as specified.

    Returns a tuple containing three lists representing the splits of training, validation, and testing sets. The length of the lists is equal to the number of folds specified by `num_splits`.

    :param data_csv_path:
        The path to the CSV containing the data.

    :param num_splits:
        The number of cross-validation folds to split the data into.

    :param validation_ratio:
        A float between 0 and 1 representing the percentage of the data to hold as a consistent validation set.

    :param text_cols:
        The column names in the dataset that contain text data.

    :param tokenizer:
        HuggingFace tokenizer used for tokenizing text columns specified in `text_cols`.

    :param label_col:
        Column name containing the target labels. For classification, the values should be integers representing class indices. For regression, they can be any numeric values.

    :param label_list:
        List of class names for classification tasks, corresponding to the values in `label_col`.

    :param categorical_cols:
        List of column names containing categorical features. These features can either be preprocessed numerically or processed according to the `categorical_encode_type`.

    :param numerical_cols:
        List of column names containing numerical features, which should contain only numeric values.

    :param sep_text_token_str:
        String used to separate different text columns for a single data sample. For example, for BERT models, this could be the `[SEP]` token.

    :param categorical_encode_type:
        Method for encoding categorical features. Options are:
        - `'ohe'`: One-hot encoding
        - `'binary'`: Binary encoding
        - `None`: No encoding

    :param categorical_handle_na:
        Whether to handle missing values in categorical features.

    :param categorical_na_value:
        Value used to replace missing categorical values, if `categorical_handle_na` is set to `True`.

    :param ohe_handle_unknown:
        Strategy for handling unknown categories in one-hot encoding. Options are:
        - `'error'`: Raise an error for unknown categories
        - `'ignore'`: Ignore unknown categories during encoding

    :param numerical_transformer_method:
        Method for normalizing numerical features. Options are:
        - `'yeo_johnson'`
        - `'box_cox'`
        - `'quantile_normal'`
        - `None`: No transformation

    :param numerical_handle_na:
        Whether to handle missing values in numerical features.

    :param numerical_how_handle_na:
        Method for handling missing numerical values. Options are:
        - `'value'`: Replace with a specific value.
        - `'mean'`: Replace with the mean of the column.

    :param numerical_na_value:
        Value used to replace missing numerical values, if `numerical_handle_na` is set to `True`.

    :param empty_text_values:
        List of text values that should be treated as missing.

    :param replace_empty_text:
        Value to replace empty text values (specified by `empty_text_values`). If `None`, empty text values are ignored.

    :param max_token_length:
        Maximum token length to pad or truncate the input text to.

    :param debug:
        Whether to load a smaller debug version of the dataset.

    :param debug_dataset_size:
        The size of the dataset to load when `debug` is set to `True`.

    :param output_dir:
        Directory to save the processed dataset files.
    """

    assert 0 <= validation_ratio <= 1, "validation ratio needs to be between 0 and 1"
    all_data_df = pd.read_csv(data_csv_path)
    folds_df, val_df = train_test_split(
        all_data_df,
        test_size=validation_ratio,
        shuffle=True,
        train_size=1 - validation_ratio,
        random_state=5,
    )
    kfold = KFold(num_splits, shuffle=True, random_state=5)

    train_splits, val_splits, test_splits = [], [], []

    for train_index, test_index in kfold.split(folds_df):
        train_df = folds_df.copy().iloc[train_index]
        test_df = folds_df.copy().iloc[test_index]

        train, val, test = load_train_val_test_helper(
            train_df=train_df,
            val_df=val_df.copy(),
            test_df=test_df,
            text_cols=text_cols,
            tokenizer=tokenizer,
            label_col=label_col,
            label_list=label_list,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            sep_text_token_str=sep_text_token_str,
            categorical_encode_type=categorical_encode_type,
            categorical_handle_na=categorical_handle_na,
            categorical_na_value=categorical_na_value,
            ohe_handle_unknown=ohe_handle_unknown,
            numerical_transformer_method=numerical_transformer_method,
            numerical_handle_na=numerical_handle_na,
            numerical_how_handle_na=numerical_how_handle_na,
            numerical_na_value=numerical_na_value,
            empty_text_values=empty_text_values,
            replace_empty_text=replace_empty_text,
            max_token_length=max_token_length,
            debug=debug,
            debug_dataset_size=debug_dataset_size,
            output_dir=output_dir,
        )
        train_splits.append(train)
        val_splits.append(val)
        test_splits.append(test)

    return train_splits, val_splits, test_splits


def load_data_from_folder(
    folder_path: str,
    text_cols: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    label_col: str,
    label_list: Optional[List[str]] = [],
    categorical_cols: Optional[List[str]] = [],
    numerical_cols: Optional[List[str]] = [],
    sep_text_token_str: str = " ",
    categorical_encode_type: str = "ohe",
    categorical_handle_na: bool = False,
    categorical_na_value: str = "-9999999",
    ohe_handle_unknown: str = "error",
    numerical_transformer_method: str = "quantile_normal",
    numerical_handle_na: bool = False,
    numerical_how_handle_na: str = "value",
    numerical_na_value: float = 0.0,
    empty_text_values: Optional[List[str]] = None,
    replace_empty_text: Optional[str] = None,
    max_token_length: Optional[int] = None,
    debug: bool = False,
    debug_dataset_size: int = 100,
    output_dir: Optional[str] = None,
) -> Tuple[
    TorchTabularTextDataset, Optional[TorchTabularTextDataset], TorchTabularTextDataset
]:
    """
    Load tabular and text data from a specified folder.

    This function loads training, testing, and optionally validation data from
    a folder into the `TorchTextDataset` class, performing preprocessing on
    categorical and numerical data as specified. The folder should contain
    `train.csv`, `test.csv`, and optionally `val.csv` files.

    Returns a tuple containing the training, validation, and testing datasets. The validation dataset is `None` if no `val.csv` is found in `folder_path`.

    :param folder_path:
        Path to the folder containing the `train.csv`, `test.csv`, and optionally `val.csv` files.

    :param text_cols:
        List of column names in the dataset that contain text data.

    :param tokenizer:
        HuggingFace tokenizer used for tokenizing text columns specified in `text_cols`.

    :param label_col:
        Column name containing the target labels. For classification, the values should be integers representing class indices. For regression, they can be any numeric values.

    :param label_list:
        List of class names for classification tasks, corresponding to the values in `label_col`.

    :param categorical_cols:
        List of column names containing categorical features. These features can either be preprocessed numerically or processed according to the `categorical_encode_type`.

    :param numerical_cols:
        List of column names containing numerical features, which should contain only numeric values.

    :param sep_text_token_str:
        String used to separate different text columns for a single data sample. For example, for BERT models, this could be the `[SEP]` token.

    :param categorical_encode_type:
        Method for encoding categorical features. Options are:
        - `'ohe'`: One-hot encoding
        - `'binary'`: Binary encoding
        - `None`: No encoding

    :param categorical_handle_na:
        Whether to handle missing values in categorical features.

    :param categorical_na_value:
        Value used to replace missing categorical values, if `categorical_handle_na` is set to `True`.

    :param ohe_handle_unknown:
        Strategy for handling unknown categories in one-hot encoding. Options are:
        - `'error'`: Raise an error for unknown categories
        - `'ignore'`: Ignore unknown categories during encoding

    :param numerical_transformer_method:
        Method for normalizing numerical features. Options are:
        - `'yeo_johnson'`
        - `'box_cox'`
        - `'quantile_normal'`
        - `None`: No transformation

    :param numerical_handle_na:
        Whether to handle missing values in numerical features.

    :param numerical_how_handle_na:
        Method for handling missing numerical values. Options are:
        - `'value'`: Replace with a specific value.
        - `'mean'`: Replace with the mean of the column.

    :param numerical_na_value:
        Value used to replace missing numerical values, if `numerical_handle_na` is set to `True`.

    :param empty_text_values:
        List of text values that should be treated as missing.

    :param replace_empty_text:
        Value to replace empty text values (specified by `empty_text_values`). If `None`, empty text values are ignored.

    :param max_token_length:
        Maximum token length to pad or truncate the input text to.

    :param debug:
        Whether to load a smaller debug version of the dataset.

    :param debug_dataset_size:
        The size of the dataset to load when `debug` is set to `True`.

    :param output_dir:
        Directory to save the processed dataset files.

    """
    train_df = pd.read_csv(join(folder_path, "train.csv"), index_col=0)
    test_df = pd.read_csv(join(folder_path, "test.csv"), index_col=0)
    if exists(join(folder_path, "val.csv")):
        val_df = pd.read_csv(join(folder_path, "val.csv"), index_col=0)
    else:
        val_df = None

    return load_train_val_test_helper(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        text_cols=text_cols,
        tokenizer=tokenizer,
        label_col=label_col,
        label_list=label_list,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        sep_text_token_str=sep_text_token_str,
        categorical_encode_type=categorical_encode_type,
        categorical_handle_na=categorical_handle_na,
        categorical_na_value=categorical_na_value,
        ohe_handle_unknown=ohe_handle_unknown,
        numerical_transformer_method=numerical_transformer_method,
        numerical_handle_na=numerical_handle_na,
        numerical_how_handle_na=numerical_how_handle_na,
        numerical_na_value=numerical_na_value,
        empty_text_values=empty_text_values,
        replace_empty_text=replace_empty_text,
        max_token_length=max_token_length,
        debug=debug,
        debug_dataset_size=debug_dataset_size,
        output_dir=output_dir,
    )


def load_train_val_test_helper(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    label_col: str,
    label_list: Optional[List[str]] = [],
    categorical_cols: Optional[List[str]] = [],
    numerical_cols: Optional[List[str]] = [],
    sep_text_token_str: str = " ",
    categorical_encode_type: str = "ohe",
    categorical_handle_na: bool = False,
    categorical_na_value: str = "-9999999",
    ohe_handle_unknown: str = "error",
    numerical_transformer_method: str = "quantile_normal",
    numerical_handle_na: bool = False,
    numerical_how_handle_na: str = "value",
    numerical_na_value: float = 0.0,
    empty_text_values: Optional[List[str]] = None,
    replace_empty_text: Optional[str] = None,
    max_token_length: Optional[int] = None,
    debug: bool = False,
    debug_dataset_size: int = 100,
    output_dir: Optional[str] = None,
) -> Tuple[
    TorchTabularTextDataset, Optional[TorchTabularTextDataset], TorchTabularTextDataset
]:
    if categorical_encode_type == "ohe" or categorical_encode_type == "binary":
        # Combine all DFs so that we don't run into encoding errors with the
        # test or validation sets
        dfs = [df for df in [train_df, val_df, test_df] if df is not None]
        data_df = pd.concat(dfs, axis=0).reset_index(drop=False)

        # Build feature encoder
        categorical_transformer = CategoricalFeatures(
            categorical_cols,
            categorical_encode_type,
            handle_na=categorical_handle_na,
            na_value=categorical_na_value,
            ohe_handle_unknown=ohe_handle_unknown,
        )
        categorical_transformer.fit(data_df)
    else:
        categorical_transformer = None

    if numerical_transformer_method != "none":
        numerical_transformer = NumericalFeatures(
            numerical_cols=numerical_cols,
            numerical_transformer_method=numerical_transformer_method,
            handle_na=numerical_handle_na,
            how_handle_na=numerical_how_handle_na,
            na_value=numerical_na_value,
        )
        numerical_transformer.fit(data_df)
    else:
        numerical_transformer = None

    train_dataset = load_data(
        data_df=train_df,
        text_cols=text_cols,
        tokenizer=tokenizer,
        label_col=label_col,
        label_list=label_list,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        sep_text_token_str=sep_text_token_str,
        categorical_transformer=categorical_transformer,
        numerical_transformer=numerical_transformer,
        empty_text_values=empty_text_values,
        replace_empty_text=replace_empty_text,
        max_token_length=max_token_length,
        debug=debug,
        debug_dataset_size=debug_dataset_size,
    )
    test_dataset = load_data(
        data_df=test_df,
        text_cols=text_cols,
        tokenizer=tokenizer,
        label_col=label_col,
        label_list=label_list,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        sep_text_token_str=sep_text_token_str,
        categorical_transformer=categorical_transformer,
        numerical_transformer=numerical_transformer,
        empty_text_values=empty_text_values,
        replace_empty_text=replace_empty_text,
        max_token_length=max_token_length,
        debug=debug,
        debug_dataset_size=debug_dataset_size,
    )

    if val_df is not None:
        val_dataset = load_data(
            data_df=val_df,
            text_cols=text_cols,
            tokenizer=tokenizer,
            label_col=label_col,
            label_list=label_list,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            sep_text_token_str=sep_text_token_str,
            categorical_transformer=categorical_transformer,
            numerical_transformer=numerical_transformer,
            empty_text_values=empty_text_values,
            replace_empty_text=replace_empty_text,
            max_token_length=max_token_length,
            debug=debug,
            debug_dataset_size=debug_dataset_size,
        )
    else:
        val_dataset = None

    # Save transformers and datasets if an output dir is specified
    if output_dir:
        makedirs(output_dir, exist_ok=True)
        if numerical_transformer:
            joblib.dump(
                numerical_transformer,
                join(output_dir, "numerical_transformer.pkl"),
            )
        if categorical_transformer:
            joblib.dump(
                categorical_transformer, join(output_dir, "categorical_transformer.pkl")
            )
        torch.save(train_dataset, join(output_dir, "train_data.pt"))
        torch.save(test_dataset, join(output_dir, "test_data.pt"))
        if val_dataset:
            torch.save(val_dataset, join(output_dir, "val_data.pt"))

    return train_dataset, val_dataset, test_dataset


def build_categorical_features(
    data_df: pd.DataFrame,
    categorical_cols: List[str],
    categorical_transformer: CategoricalFeatures,
) -> Optional[pd.DataFrame]:
    if len(categorical_cols) > 0:
        # Find columns in the dataset that are in categorical_cols
        categorical_cols_func = convert_to_func(categorical_cols)
        categorical_cols = get_matching_cols(data_df, categorical_cols_func)
        if categorical_transformer is not None:
            return categorical_transformer.transform(data_df[categorical_cols])
        else:
            return data_df[categorical_cols]
    else:
        return None


def build_numerical_features(
    data_df: pd.DataFrame,
    numerical_cols: List[str],
    numerical_transformer: NumericalFeatures,
) -> Optional[np.ndarray]:
    if len(numerical_cols) > 0:
        # Find columns in the dataset that are in numerical_cols
        numerical_cols_func = convert_to_func(numerical_cols)
        numerical_cols = get_matching_cols(data_df, numerical_cols_func)
        if numerical_transformer is not None:
            return numerical_transformer.transform(data_df[numerical_cols])
        else:
            return data_df[numerical_cols].values
    else:
        return None


def build_text_features(
    data_df: pd.DataFrame,
    text_cols: List[str],
    empty_text_values: List[str],
    replace_empty_text: str,
    sep_text_token_str: str,
) -> List[str]:
    text_cols_func = convert_to_func(text_cols)
    agg_func = partial(agg_text_columns_func, empty_text_values, replace_empty_text)
    text_cols = get_matching_cols(data_df, text_cols_func)
    logger.info(f"Text columns: {text_cols}")
    texts_list = data_df[text_cols].agg(agg_func, axis=1).tolist()
    for i, text in enumerate(texts_list):
        texts_list[i] = f" {sep_text_token_str} ".join(text)
    logger.info(f"Raw text example: {texts_list[0]}")
    return texts_list


def load_data(
    data_df: pd.DataFrame,
    text_cols: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    label_col: Optional[str] = None,
    label_list: Optional[List[str]] = [],
    categorical_cols: Optional[List[str]] = [],
    numerical_cols: Optional[List[str]] = [],
    sep_text_token_str: str = " ",
    categorical_transformer: Optional[CategoricalFeatures] = None,
    numerical_transformer: Optional[NumericalFeatures] = None,
    empty_text_values: Optional[List[str]] = None,
    replace_empty_text: Optional[str] = None,
    max_token_length: Optional[int] = None,
    debug: bool = False,
    debug_dataset_size: int = 100,
) -> TorchTabularTextDataset:
    """
    Load a single dataset from a pandas DataFrame.

    Given a DataFrame, this function loads the data into a :obj:`torch_dataset.TorchTextDataset` object, which can be used in a :obj:`torch.utils.data.DataLoader`.

    :param data_df:
        The DataFrame to convert to a TorchTextDataset.

    :param text_cols:
        The column names in the dataset that contain text data.

    :param tokenizer:
        HuggingFace tokenizer used for tokenizing text columns specified in `text_cols`.

    :param label_col:
        The column name containing the target labels. For classification, the values should be integers representing class indices. For regression, they can be any numeric values.

    :param label_list:
        List of class names for classification tasks, corresponding to the values in `label_col`.

    :param categorical_cols:
        List of column names containing categorical features. These features can either be preprocessed numerically or processed according to the specified transformer.

    :param numerical_cols:
        List of column names containing numerical features, which should contain only numeric values.

    :param sep_text_token_str:
        String used to separate different text columns for a single data sample. For example, for BERT models, this could be the `[SEP]` token.

    :param categorical_transformer:
        Sklearn transformer instance for preprocessing categorical features.

    :param numerical_transformer:
        Sklearn transformer instance for preprocessing numerical features.

    :param empty_text_values:
        List of text values that should be treated as missing.

    :param replace_empty_text:
        Value to replace empty text values (specified by `empty_text_values`). If `None`, empty text values are ignored.

    :param max_token_length:
        Maximum token length to pad or truncate the input text to.

    :param debug:
        Whether to load a smaller debug version of the dataset.

    :param debug_dataset_size:
        The size of the dataset to load when `debug` is set to `True`.

    """
    if debug:
        data_df = data_df[:debug_dataset_size]
    if empty_text_values is None:
        empty_text_values = ["nan", "None"]

    # Build categorical features
    categorical_feats = build_categorical_features(
        data_df=data_df,
        categorical_cols=categorical_cols,
        categorical_transformer=categorical_transformer,
    )
    # Build numerical features
    numerical_feats = build_numerical_features(
        data_df=data_df,
        numerical_cols=numerical_cols,
        numerical_transformer=numerical_transformer,
    )

    # Build text features
    texts_list = build_text_features(
        data_df, text_cols, empty_text_values, replace_empty_text, sep_text_token_str
    )

    # Create tokenized text features
    hf_model_text_input = tokenizer(
        texts_list, padding=True, truncation=True, max_length=max_token_length
    )

    tokenized_text_ex = " ".join(
        tokenizer.convert_ids_to_tokens(hf_model_text_input["input_ids"][0])
    )
    logger.debug(f"Tokenized text example: {tokenized_text_ex}")

    # Setup labels, if any
    if label_col:
        labels = data_df[label_col].values
    else:
        labels = None

    return TorchTabularTextDataset(
        encodings=hf_model_text_input,
        categorical_feats=categorical_feats,
        numerical_feats=numerical_feats,
        labels=labels,
        df=data_df,
        label_list=label_list,
    )
