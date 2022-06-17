import logging
import types

import numpy as np
from sklearn import preprocessing

logger = logging.getLogger(__name__)


class CategoricalFeatures:
    """ Class to help encode categorical features
    From https://github.com/abhishekkrthakur/mlframework/blob/master/src/categorical.py
    """

    def __init__(self, df, categorical_cols, encoding_type, handle_na=False):
        """
        Args:
            df (:obj: `pd.DataFrame`): DataFrame which contains categorical features
            categorical_cols (:obj:`list` of :obj:`str`, optional):
                the column names in the dataset that contain categorical features
            encoding_type (str): method we want to preprocess our categorical features.
                choices: [ 'ohe', 'binary', None]
            handle_na (bool): whether to handle nan by treating them as a separate
                categorical value
        """
        self.df = df
        self.cat_feats = categorical_cols
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df[self.cat_feats].values

    def _label_binarization(self):
        vals =[]
        self.feat_names = []

        def change_name_func(x):
            return x.lower().replace(', ', '_').replace(' ', '_')
        for c in self.cat_feats:
            self.df[c] = self.df[c].astype(str)
            classes_orig = self.df[c].unique()
            val = preprocessing.label_binarize(self.df[c].values, classes=classes_orig)
            vals.append(val)
            if len(classes_orig) == 2:
                classes = [c + '_binary']
            else:
                change_classes_func_vec = np.vectorize(lambda x: c + '_' + change_name_func(x))
                classes = change_classes_func_vec(classes_orig)
            self.feat_names.extend(classes)
        return np.concatenate(vals, axis=1)

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder(sparse=False)
        ohe.fit(self.df[self.cat_feats].values)
        self.feat_names = list(ohe.get_feature_names(self.cat_feats))
        return ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        elif self.enc_type is None or self.enc_type == "none":
            return self.df[self.cat_feats].values
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)

        else:
            raise Exception("Encoding type not understood")


def normalize_numerical_feats(numerical_feats, transformer=None):
    if numerical_feats is None or transformer is None:
        return numerical_feats
    return transformer.transform(numerical_feats)


def convert_to_func(container_arg):
    """convert container_arg to function that returns True if an element is in container_arg"""
    if container_arg is None:
        return lambda df, x: False
    if not isinstance(container_arg, types.FunctionType):
        assert type(container_arg) is list or type(container_arg) is set
        return lambda df, x: x in container_arg
    else:
        return container_arg


def agg_text_columns_func(empty_row_values, replace_text, texts):
    """replace empty texts or remove empty text str from a list of text str"""
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
    """load categorical features from DataFrame and do encoding if specified"""
    cat_cols = get_matching_cols(df, cat_bool_func)
    logger.info(f'{len(cat_cols)} categorical columns')
    if len(cat_cols) == 0:
        return None
    cat_feat_processor = CategoricalFeatures(df, cat_cols, encode_type)
    return cat_feat_processor.fit_transform()


def load_num_feats(df, num_bool_func):
    num_cols = get_matching_cols(df, num_bool_func)
    logger.info(f'{len(num_cols)} numerical columns')
    df = df.copy()
    df[num_cols] = df[num_cols].astype(float)
    df[num_cols] = df[num_cols].fillna(dict(df[num_cols].median()), inplace=False)
    if len(num_cols) == 0:
        return None
    return df[num_cols].values


def get_matching_cols(df, col_match_func):
    return [c for c in df.columns if col_match_func(df, c)]