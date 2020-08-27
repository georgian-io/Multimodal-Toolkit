from functools import partial
from os.path import join, exists
import types

from encode_features import CategoricalFeatures
import pandas as pd
from torch_dataset import TorchTextDataset
from tqdm import tqdm


def load_data_from_folder(folder_path,
                          text_cols,
                          tokenizer,
                          label_col,
                          label_list=None,
                          categorical_cols=None,
                          numerical_cols=None,
                          sep_text_token_str=' ',
                          categorical_encode_type='ohe',
                          empty_text_values=None,
                          replace_empty_text=None,
                          max_token_length=None,
                          debug=False,
                          ):
    train_df = pd.read_csv(join(folder_path, 'train.csv'), index_col=0)
    test_df = pd.read_csv(join(folder_path, 'test.csv'), index_col=0)
    if exists(join(folder_path, 'val.csv')):
        val_df = pd.read_csv(join(folder_path, 'val.csv'), index_col=0)
    else:
        val_df = None

    if categorical_encode_type == 'ohe':
        dfs = [df for df in [train_df, val_df, test_df] if df is not None]
        data_df = pd.concat(dfs, axis=0)
        data_df = pd.get_dummies(data_df, columns=categorical_cols,
                                 dummy_na=True)
        categorical_cols = [col for col in data_df.columns for old_col in categorical_cols
                            if col.startswith(old_col) and len(col) > len(old_col)]
        train_df = data_df.loc[train_df.index]
        if val_df is not None:
            val_df = data_df.loc[val_df.index]
        test_df = data_df.loc[test_df.index]

        categorical_encode_type = None

    train_dataset = load_data(train_df,
                              text_cols,
                              tokenizer,
                              label_col,
                              label_list,
                              categorical_cols,
                              numerical_cols,
                              sep_text_token_str,
                              categorical_encode_type,
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
              empty_text_values=None,
              replace_empty_text=None,
              max_token_length=None,
              debug=False,
              ):
    if debug:
        data_df = data_df[:50]
    if empty_text_values is None:
        empty_text_values = ['nan', 'None']

    def convert_to_func(arg):
        """convert arg to func that returns True if element in arg"""
        if arg is None:
            return lambda df, x: False
        if not isinstance(arg, types.FunctionType):
            assert type(arg) is list or type(arg) is set
            return lambda df, x: x in arg
        else:
            return arg

    text_cols_func = convert_to_func(text_cols)
    categorical_cols_func = convert_to_func(categorical_cols)
    numerical_cols_func = convert_to_func(numerical_cols)

    categorical_feats, numerical_feats = load_cat_and_num_feats(data_df,
                                                                categorical_cols_func,
                                                                numerical_cols_func,
                                                                categorical_encode_type)
    agg_func = partial(agg_text_columns_func, empty_text_values, replace_empty_text)
    texts_cols = get_matching_cols(data_df, text_cols_func)
    print(f'Text columns: {texts_cols}')
    texts_list = data_df[texts_cols].agg(agg_func, axis=1).tolist()
    for i, text in tqdm(enumerate(texts_list), desc='looping texts'):
        texts_list[i] = f' {sep_text_token_str} '.join(text)
    print(f'Raw text example: {texts_list[0]}')
    hf_model_text_input = tokenizer(texts_list, padding=True, truncation=True,
                                    max_length=max_token_length)
    tokenized_text_ex = ' '.join(tokenizer.convert_ids_to_tokens(hf_model_text_input['input_ids'][0]))
    print(f'Tokenized text example: {tokenized_text_ex}')
    labels = data_df[label_col].values

    return TorchTextDataset(data_df, hf_model_text_input, categorical_feats,
                            numerical_feats,  labels, label_list)


def agg_text_columns_func(empty_row_values, replace_text, texts):
    processed_texts = []
    for text in texts.astype('str'):
        if text not in empty_row_values:
            processed_texts.append(text)
        else:
            if replace_text is not None:
                processed_texts.append(replace_text)
    return processed_texts


def load_cat_and_num_feats(df, cat_bool_func, num_bool_func, enocde_type=None):
    cat_feats = load_cat_feats(df, cat_bool_func, enocde_type)
    num_feats = load_num_feats(df, num_bool_func)
    return cat_feats, num_feats


def load_cat_feats(df, cat_bool_func, encode_type=None):
    cat_cols = get_matching_cols(df, cat_bool_func)
    print(f'{len(cat_cols)} categorical columns')
    if len(cat_cols) == 0:
        return None
    cat_feat_processor = CategoricalFeatures(df, cat_cols, encode_type)
    return cat_feat_processor.fit_transform()


def load_num_feats(df, num_bool_func):
    num_cols = get_matching_cols(df, num_bool_func)
    print(f'{len(num_cols)} numerical columns')
    if len(num_cols) == 0:
        return None
    return df[num_cols].values


def get_matching_cols(df, col_match_func):
    return [c for c in df.columns if col_match_func(df, c)]