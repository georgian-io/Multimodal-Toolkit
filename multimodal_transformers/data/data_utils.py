import logging
import types

import numpy as np
import pandas as pd
from typing import List
from sklearn import preprocessing

logger = logging.getLogger(__name__)


class CategoricalFeatures:
    """Class to help encode categorical features
    From https://github.com/abhishekkrthakur/mlframework/blob/master/src/categorical.py
    """

    def __init__(
        self,
        categorical_cols: List[str],
        encoding_type: str,
        handle_na: bool = False,
        na_value: str = "-9999999",
    ):
        """
        Args:
            categorical_cols (:obj:`list` of :obj:`str`, optional):
                the column names in the dataset that contain categorical
                features
            encoding_type (str): method we want to preprocess our categorical
            features.
                choices: [ 'ohe', 'binary', None]
            handle_na (bool): whether to handle nan by treating them as a
                separate categorical value
            na_value (string): what the nan values should be converted to
        """
        self.cat_feats = categorical_cols
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None
        self.handle_na = handle_na
        self.na_value = na_value
        self.feat_names = []

    def _label_encoding(self, dataframe: pd.DataFrame):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(dataframe[c].values)
            self.label_encoders[c] = lbl

    def _label_binarization(self, dataframe: pd.DataFrame):
        for c in self.cat_feats:
            dataframe[c] = dataframe[c].astype(str)
            lb = preprocessing.LabelBinarizer()
            lb.fit(dataframe[c].values)
            self.binary_encoders[c] = lb

            # Create new class names
            for class_name in lb.classes_:
                new_col_name = f"{c}__{change_name_func(class_name)}"
                self.feat_names.append(new_col_name)
                if len(lb.classes_) == 2:
                    break

    def _one_hot(self, dataframe: pd.DataFrame):
        self.ohe = preprocessing.OneHotEncoder(sparse=False)
        self.ohe.fit(dataframe[self.cat_feats].values)
        self.feat_names = list(self.ohe.get_feature_names_out(self.cat_feats))

    def fit(self, dataframe: pd.DataFrame):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = (
                    dataframe.loc[:, c].astype(str).fillna(self.na_value)
                )
        if self.enc_type == "label":
            self._label_encoding(dataframe)
        elif self.enc_type == "binary":
            self._label_binarization(dataframe)
        elif self.enc_type == "ohe":
            self._one_hot(dataframe)
        elif self.enc_type is None or self.enc_type == "none":
            logger.info(f"Encoding type is none, no action taken.")
        else:
            raise Exception("Encoding type not understood")

    def fit_transform(self, dataframe: pd.DataFrame):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = (
                    dataframe.loc[:, c].astype(str).fillna(self.na_value)
                )
        if self.enc_type == "label":
            self._label_encoding(dataframe)
            return self.transform(dataframe)
        elif self.enc_type == "binary":
            self._label_binarization(dataframe)
            return self.transform(dataframe)
        elif self.enc_type == "ohe":
            self._one_hot(dataframe)
            return self.transform(dataframe)
        elif self.enc_type is None or self.enc_type == "none":
            return dataframe[self.cat_feats].values
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe: pd.DataFrame):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = (
                    dataframe.loc[:, c].astype(str).fillna(self.na_value)
                )

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                class_names = [
                    f"{c}__{lbl.classes_[j]}_binary" for j in range(val.shape[1])
                ]
                val = pd.DataFrame(val, columns=class_names, index=dataframe.index)
                dataframe = pd.concat([dataframe, val], axis=1)
            return dataframe

        elif self.enc_type == "ohe":
            val = self.ohe.transform(dataframe[self.cat_feats].values)
            for j in range(val.shape(1)):
                dataframe[self.feat_names[j]] = val[:, j]
            return dataframe

        else:
            raise Exception("Encoding type not understood")


def change_name_func(x):
    return x.lower().replace(", ", "_").replace(" ", "_")


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
    for text in texts.astype("str"):
        if text not in empty_row_values:
            processed_texts.append(text)
        else:
            if replace_text is not None:
                processed_texts.append(replace_text)
    return processed_texts


def load_cat_and_num_feats(df, cat_bool_func, num_bool_func, encode_type=None):
    cat_feats = load_cat_feats(df, cat_bool_func, encode_type)
    num_feats = load_num_feats(df, num_bool_func)
    return cat_feats, num_feats


def load_cat_feats(df, cat_bool_func, encode_type=None):
    """load categorical features from DataFrame and do encoding if specified"""
    cat_cols = get_matching_cols(df, cat_bool_func)
    logger.info(f"{len(cat_cols)} categorical columns")
    if len(cat_cols) == 0:
        return None
    cat_feat_processor = CategoricalFeatures(cat_cols, encode_type)
    return cat_feat_processor.fit_transform(df)


def load_num_feats(df, num_bool_func):
    num_cols = get_matching_cols(df, num_bool_func)
    logger.info(f"{len(num_cols)} numerical columns")
    df = df.copy()
    df[num_cols] = df[num_cols].astype(float)
    df[num_cols] = df[num_cols].fillna(dict(df[num_cols].median()), inplace=False)
    if len(num_cols) == 0:
        return None
    return df[num_cols].values


def get_matching_cols(df, col_match_func):
    return [c for c in df.columns if col_match_func(df, c)]
