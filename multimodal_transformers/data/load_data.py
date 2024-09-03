from functools import partial
import logging
from os.path import join, exists

import pandas as pd
import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from .tabular_torch_dataset import TorchTabularTextDataset
from .data_utils import (
    CategoricalFeatures,
    agg_text_columns_func,
    convert_to_func,
    get_matching_cols,
    load_num_feats,
    load_cat_and_num_feats,
    normalize_numerical_feats,
)

logger = logging.getLogger(__name__)


def load_data_into_folds(
    data_csv_path,
    num_splits,
    validation_ratio,
    text_cols,
    tokenizer,
    label_col,
    label_list=None,
    categorical_cols=None,
    numerical_cols=None,
    sep_text_token_str=" ",
    categorical_encode_type="ohe",
    numerical_transformer_method="quantile_normal",
    empty_text_values=None,
    replace_empty_text=None,
    max_token_length=None,
    debug=False,
    debug_dataset_size=100,
    encoder_save_path=None,
):
    """
    Function to load tabular and text data from a specified folder into folds

    Loads train, test and/or validation text and tabular data from specified
    csv path into num_splits of train, val and test for Kfold cross validation.
    Performs categorical and numerical data preprocessing if specified. `data_csv_path` is a path to

    Args:
        data_csv_path (str): The path to the csv containing the data
        num_splits (int): The number of cross validation folds to split the data into.
        validation_ratio (float): A float between 0 and 1 representing the percent of the data to hold as a consistent validation set.
        text_cols (:obj:`list` of :obj:`str`): The column names in the dataset that contain text
            from which we want to load
        tokenizer (:obj:`transformers.tokenization_utils.PreTrainedTokenizer`):
            HuggingFace tokenizer used to tokenize the input texts as specifed by text_cols
        label_col (str): The column name of the label, for classification the column should have
            int values from 0 to n_classes-1 as the label for each class.
            For regression the column can have any numerical value
        label_list (:obj:`list` of :obj:`str`, optional): Used for classification;
            the names of the classes indexed by the values in label_col.
        categorical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that
            contain categorical features. The features can be already prepared numerically, or
            could be preprocessed by the method specified by categorical_encode_type
        numerical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that contain numerical features.
            These columns should contain only numeric values.
        sep_text_token_str (str, optional): The string token that is used to separate between the
            different text columns for a given data example. For Bert for example,
            this could be the [SEP] token.
        categorical_encode_type (str, optional): Given categorical_cols, this specifies
            what method we want to preprocess our categorical features.
            choices: [ 'ohe', 'binary', None]
            see encode_features.CategoricalFeatures for more details
        numerical_transformer_method (str, optional): Given numerical_cols, this specifies
            what method we want to use for normalizing our numerical data.
            choices: ['yeo_johnson', 'box_cox', 'quantile_normal', None]
            see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
            for more details
        empty_text_values (:obj:`list` of :obj:`str`, optional): specifies what texts should be considered as
            missing which would be replaced by replace_empty_text
        replace_empty_text (str, optional): The value of the string that will replace the texts
            that match with those in empty_text_values. If this argument is None then
            the text that match with empty_text_values will be skipped
        max_token_length (int, optional): The token length to pad or truncate to on the
            input text
        debug (bool, optional): Whether or not to load a smaller debug version of the dataset

    Returns:
        :obj:`tuple` of `list` of `tabular_torch_dataset.TorchTextDataset`:
            This tuple contains three lists representing the splits of
            training, validation and testing sets. The length of the lists is
            equal to the number of folds specified by `num_splits`
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
            numerical_transformer_method=numerical_transformer_method,
            empty_text_values=empty_text_values,
            replace_empty_text=replace_empty_text,
            max_token_length=max_token_length,
            debug=debug,
            debug_dataset_size=debug_dataset_size,
            encoder_save_path=encoder_save_path,
        )
        train_splits.append(train)
        val_splits.append(val)
        test_splits.append(test)

    return train_splits, val_splits, test_splits


def load_data_from_folder(
    folder_path,
    text_cols,
    tokenizer,
    label_col,
    label_list=None,
    categorical_cols=None,
    numerical_cols=None,
    sep_text_token_str=" ",
    categorical_encode_type="ohe",
    numerical_transformer_method="quantile_normal",
    empty_text_values=None,
    replace_empty_text=None,
    max_token_length=None,
    debug=False,
    debug_dataset_size=100,
    encoder_save_path=None,
):
    """
    Function to load tabular and text data from a specified folder

    Loads train, test and/or validation text and tabular data from specified
    folder path into TorchTextDataset class and does categorical and numerical
    data preprocessing if specified. Inside the folder, there is expected to be
    a train.csv, and test.csv (and if given val.csv) containing the training, testing,
    and validation sets respectively

    Args:
        folder_path (str): The path to the folder containing `train.csv`, and `test.csv` (and if given `val.csv`)
        text_cols (:obj:`list` of :obj:`str`): The column names in the dataset that contain text
            from which we want to load
        tokenizer (:obj:`transformers.tokenization_utils.PreTrainedTokenizer`):
            HuggingFace tokenizer used to tokenize the input texts as specifed by text_cols
        label_col (str): The column name of the label, for classification the column should have
            int values from 0 to n_classes-1 as the label for each class.
            For regression the column can have any numerical value
        label_list (:obj:`list` of :obj:`str`, optional): Used for classification;
            the names of the classes indexed by the values in label_col.
        categorical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that
            contain categorical features. The features can be already prepared numerically, or
            could be preprocessed by the method specified by categorical_encode_type
        numerical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that contain numerical features.
            These columns should contain only numeric values.
        sep_text_token_str (str, optional): The string token that is used to separate between the
            different text columns for a given data example. For Bert for example,
            this could be the [SEP] token.
        categorical_encode_type (str, optional): Given categorical_cols, this specifies
            what method we want to preprocess our categorical features.
            choices: [ 'ohe', 'binary', None]
            see encode_features.CategoricalFeatures for more details
        numerical_transformer_method (str, optional): Given numerical_cols, this specifies
            what method we want to use for normalizing our numerical data.
            choices: ['yeo_johnson', 'box_cox', 'quantile_normal', None]
            see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
            for more details
        empty_text_values (:obj:`list` of :obj:`str`, optional): specifies what texts should be considered as
            missing which would be replaced by replace_empty_text
        replace_empty_text (str, optional): The value of the string that will replace the texts
            that match with those in empty_text_values. If this argument is None then
            the text that match with empty_text_values will be skipped
        max_token_length (int, optional): The token length to pad or truncate to on the
            input text
        debug (bool, optional): Whether or not to load a smaller debug version of the dataset

    Returns:
        :obj:`tuple` of `tabular_torch_dataset.TorchTextDataset`:
            This tuple contains the
            training, validation and testing sets. The val dataset is :obj:`None` if
            there is no `val.csv` in folder_path
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
        numerical_transformer_method=numerical_transformer_method,
        empty_text_values=empty_text_values,
        replace_empty_text=replace_empty_text,
        max_token_length=max_token_length,
        debug=debug,
        debug_dataset_size=debug_dataset_size,
        encoder_save_path=encoder_save_path,
    )


def load_train_val_test_helper(
    train_df,
    val_df,
    test_df,
    text_cols,
    tokenizer,
    label_col,
    label_list=None,
    categorical_cols=None,
    numerical_cols=None,
    sep_text_token_str=" ",
    categorical_encode_type="ohe",
    numerical_transformer_method="quantile_normal",
    empty_text_values=None,
    replace_empty_text=None,
    max_token_length=None,
    debug=False,
    debug_dataset_size=100,
    encoder_save_path=None,
):
    if categorical_encode_type == "ohe" or categorical_encode_type == "binary":
        dfs = [df for df in [train_df, val_df, test_df] if df is not None]
        data_df = pd.concat(dfs, axis=0).reset_index(drop=False)
        cat_feat_processor = CategoricalFeatures(
            categorical_cols, categorical_encode_type
        )
        data_df = cat_feat_processor.fit_transform(data_df)
        categorical_cols = cat_feat_processor.feat_names

        len_train = len(train_df)
        len_val = len(val_df) if val_df is not None else 0

        train_df = data_df.iloc[:len_train]
        if val_df is not None:
            val_df = data_df.iloc[len_train : len_train + len_val]
            len_train = len_train + len_val
        test_df = data_df.iloc[len_train:]

        categorical_encode_type = None
    else:
        cat_feat_processor = None

    if numerical_transformer_method != "none":
        if numerical_transformer_method == "yeo_johnson":
            numerical_transformer = PowerTransformer(method="yeo-johnson")
        elif numerical_transformer_method == "box_cox":
            numerical_transformer = PowerTransformer(method="box-cox")
        elif numerical_transformer_method == "quantile_normal":
            numerical_transformer = QuantileTransformer(output_distribution="normal")
        else:
            raise ValueError(
                f"preprocessing transformer method "
                f"{numerical_transformer_method} not implemented"
            )
        num_feats = load_num_feats(train_df, convert_to_func(numerical_cols))
        numerical_transformer.fit(num_feats)
    else:
        numerical_transformer = None

    # Save the categorical & numerical transformer if needed
    if encoder_save_path:
        if numerical_transformer:
            joblib.dump(
                numerical_transformer,
                join(encoder_save_path, "numerical_transformer.pkl"),
            )
        if cat_feat_processor:
            joblib.dump(
                cat_feat_processor, join(encoder_save_path, "cat_feat_processor.pkl")
            )

    train_dataset = load_data(
        data_df=train_df,
        text_cols=text_cols,
        tokenizer=tokenizer,
        label_col=label_col,
        label_list=label_list,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        sep_text_token_str=sep_text_token_str,
        categorical_encode_type=categorical_encode_type,
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
        categorical_encode_type=categorical_encode_type,
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
            categorical_encode_type=categorical_encode_type,
            numerical_transformer=numerical_transformer,
            empty_text_values=empty_text_values,
            replace_empty_text=replace_empty_text,
            max_token_length=max_token_length,
            debug=debug,
            debug_dataset_size=debug_dataset_size,
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


def load_data(
    data_df,
    text_cols,
    tokenizer,
    label_col=None,
    label_list=None,
    categorical_cols=None,
    numerical_cols=None,
    sep_text_token_str=" ",
    categorical_encode_type="ohe",
    numerical_transformer=None,
    empty_text_values=None,
    replace_empty_text=None,
    max_token_length=None,
    debug=False,
    debug_dataset_size=100,
):
    """Function to load a single dataset given a pandas DataFrame

    Given a DataFrame, this function loads the data to a :obj:`torch_dataset.TorchTextDataset`
    object which can be used in a :obj:`torch.utils.data.DataLoader`.

    Args:
        data_df (:obj:`pd.DataFrame`): The DataFrame to convert to a TorchTextDataset
        text_cols (:obj:`list` of :obj:`str`): the column names in the dataset that contain text
            from which we want to load
        tokenizer (:obj:`transformers.tokenization_utils.PreTrainedTokenizer`):
            HuggingFace tokenizer used to tokenize the input texts as specifed by text_cols
        label_col (str): The column name of the label, for classification the column should have
            int values from 0 to n_classes-1 as the label for each class.
            For regression the column can have any numerical value
        label_list (:obj:`list` of :obj:`str`, optional): Used for classification;
            the names of the classes indexed by the values in label_col.
        categorical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that
            contain categorical features. The features can be already prepared numerically, or
            could be preprocessed by the method specified by categorical_encode_type
        numerical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that contain numerical features.
            These columns should contain only numeric values.
        sep_text_token_str (str, optional): The string token that is used to separate between the
            different text columns for a given data example. For Bert for example,
            this could be the [SEP] token.
        categorical_encode_type (str, optional): Given categorical_cols, this specifies
            what method we want to preprocess our categorical features.
            choices: [ 'ohe', 'binary', None]
            see encode_features.CategoricalFeatures for more details
        numerical_transformer (:obj:`sklearn.base.TransformerMixin`): The sklearn numeric
            transformer instance to transform our numerical features
        empty_text_values (:obj:`list` of :obj:`str`, optional): Specifies what texts should be considered as
            missing which would be replaced by replace_empty_text
        replace_empty_text (str, optional): The value of the string that will replace the texts
            that match with those in empty_text_values. If this argument is None then
            the text that match with empty_text_values will be skipped
        max_token_length (int, optional): The token length to pad or truncate to on the
            input text
        debug (bool, optional): Whether or not to load a smaller debug version of the dataset

    Returns:
        :obj:`tabular_torch_dataset.TorchTextDataset`: The converted dataset
    """
    if debug:
        data_df = data_df[:debug_dataset_size]
    if empty_text_values is None:
        empty_text_values = ["nan", "None"]

    text_cols_func = convert_to_func(text_cols)
    categorical_cols_func = convert_to_func(categorical_cols)
    numerical_cols_func = convert_to_func(numerical_cols)

    categorical_feats, numerical_feats = load_cat_and_num_feats(
        data_df, categorical_cols_func, numerical_cols_func, categorical_encode_type
    )
    numerical_feats = normalize_numerical_feats(numerical_feats, numerical_transformer)
    agg_func = partial(agg_text_columns_func, empty_text_values, replace_empty_text)
    texts_cols = get_matching_cols(data_df, text_cols_func)
    logger.info(f"Text columns: {texts_cols}")
    texts_list = data_df[texts_cols].agg(agg_func, axis=1).tolist()
    for i, text in enumerate(texts_list):
        texts_list[i] = f" {sep_text_token_str} ".join(text)
    logger.info(f"Raw text example: {texts_list[0]}")
    hf_model_text_input = tokenizer(
        texts_list, padding=True, truncation=True, max_length=max_token_length
    )
    tokenized_text_ex = " ".join(
        tokenizer.convert_ids_to_tokens(hf_model_text_input["input_ids"][0])
    )
    logger.debug(f"Tokenized text example: {tokenized_text_ex}")
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
