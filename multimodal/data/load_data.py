from functools import partial
import logging
from os.path import join, exists

import pandas as pd
import transformers
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from multimodal.data.tabular_torch_dataset import TorchTabularTextDataset

from multimodal.data.data_utils import (
    CategoricalFeatures,
    agg_text_columns_func,
    convert_to_func,
    get_matching_cols,
    load_num_feats,
    load_cat_and_num_feats,
    normalize_numerical_feats,
)

logger = logging.getLogger(__name__)


def load_data_from_folder(folder_path,
                          text_cols,
                          tokenizer,
                          label_col,
                          label_list=None,
                          categorical_cols=None,
                          numerical_cols=None,
                          sep_text_token_str=' ',
                          categorical_encode_type='ohe',
                          numerical_transformer_method='quantile_normal',
                          empty_text_values=None,
                          replace_empty_text=None,
                          max_token_length=None,
                          debug=False,
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
    train_df = pd.read_csv(join(folder_path, 'train.csv'), index_col=0)
    test_df = pd.read_csv(join(folder_path, 'test.csv'), index_col=0)
    if exists(join(folder_path, 'val.csv')):
        val_df = pd.read_csv(join(folder_path, 'val.csv'), index_col=0)
    else:
        val_df = None

    if categorical_encode_type == 'ohe' or categorical_encode_type == 'binary':
        dfs = [df for df in [train_df, val_df, test_df] if df is not None]
        data_df = pd.concat(dfs, axis=0)
        if categorical_encode_type == 'ohe':
            data_df = pd.get_dummies(data_df, columns=categorical_cols,
                                     dummy_na=True)
            categorical_cols = [col for col in data_df.columns for old_col in categorical_cols
                                if col.startswith(old_col) and len(col) > len(old_col)]
        elif categorical_encode_type == 'binary':
            cat_feat_processor = CategoricalFeatures(data_df, categorical_cols, 'binary')
            vals = cat_feat_processor.fit_transform()
            cat_df = pd.DataFrame(vals, columns=cat_feat_processor.feat_names)
            data_df = pd.concat([data_df, cat_df], axis=1)
            categorical_cols = cat_feat_processor.feat_names

        train_df = data_df.loc[train_df.index]
        if val_df is not None:
            val_df = data_df.loc[val_df.index]
        test_df = data_df.loc[test_df.index]

        categorical_encode_type = None

    if numerical_transformer_method != 'none':
        if numerical_transformer_method == 'yeo_johnson':
            numerical_transformer = PowerTransformer(method='yeo-johnson')
        elif numerical_transformer_method == 'box_cox':
            numerical_transformer = PowerTransformer(method='box-cox')
        elif numerical_transformer_method == 'quantile_normal':
            numerical_transformer = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f'preprocessing transformer method '
                             f'{numerical_transformer_method} not implemented')
        num_feats = load_num_feats(train_df, convert_to_func(numerical_cols))
        numerical_transformer.fit(num_feats)
    else:
        numerical_transformer = None

    train_dataset = load_data(train_df,
                              text_cols,
                              tokenizer,
                              label_col,
                              label_list,
                              categorical_cols,
                              numerical_cols,
                              sep_text_token_str,
                              categorical_encode_type,
                              numerical_transformer,
                              empty_text_values,
                              replace_empty_text,
                              max_token_length,
                              debug
                              )
    test_dataset = load_data(test_df,
                             text_cols,
                             tokenizer,
                             label_col,
                             label_list,
                             categorical_cols,
                             numerical_cols,
                             sep_text_token_str,
                             categorical_encode_type,
                             numerical_transformer,
                             empty_text_values,
                             replace_empty_text,
                             max_token_length,
                             debug
                             )

    if val_df is not None:
        val_dataset = load_data(val_df,
                                text_cols,
                                tokenizer,
                                label_col,
                                label_list,
                                categorical_cols,
                                numerical_cols,
                                sep_text_token_str,
                                categorical_encode_type,
                                numerical_transformer,
                                empty_text_values,
                                replace_empty_text,
                                max_token_length,
                                debug
                                )
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


def load_data(data_df,
              text_cols,
              tokenizer,
              label_col,
              label_list=None,
              categorical_cols=None,
              numerical_cols=None,
              sep_text_token_str=' ',
              categorical_encode_type='ohe',
              numerical_transformer=None,
              empty_text_values=None,
              replace_empty_text=None,
              max_token_length=None,
              debug=False,
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
        data_df = data_df[:50]
    if empty_text_values is None:
        empty_text_values = ['nan', 'None']

    text_cols_func = convert_to_func(text_cols)
    categorical_cols_func = convert_to_func(categorical_cols)
    numerical_cols_func = convert_to_func(numerical_cols)

    categorical_feats, numerical_feats = load_cat_and_num_feats(data_df,
                                                                categorical_cols_func,
                                                                numerical_cols_func,
                                                                categorical_encode_type)
    numerical_feats = normalize_numerical_feats(numerical_feats, numerical_transformer)
    agg_func = partial(agg_text_columns_func, empty_text_values, replace_empty_text)
    texts_cols = get_matching_cols(data_df, text_cols_func)
    logger.info(f'Text columns: {texts_cols}')
    texts_list = data_df[texts_cols].agg(agg_func, axis=1).tolist()
    for i, text in enumerate(texts_list):
        texts_list[i] = f' {sep_text_token_str} '.join(text)
    logger.info(f'Raw text example: {texts_list[0]}')
    hf_model_text_input = tokenizer(texts_list, padding=True, truncation=True,
                                    max_length=max_token_length)
    tokenized_text_ex = ' '.join(tokenizer.convert_ids_to_tokens(hf_model_text_input['input_ids'][0]))
    logger.debug(f'Tokenized text example: {tokenized_text_ex}')
    labels = data_df[label_col].values

    return TorchTabularTextDataset(hf_model_text_input, categorical_feats,
                                   numerical_feats, labels, data_df, label_list)
