"""
From https://github.com/abhishekkrthakur/mlframework/blob/master/src/categorical.py
"""
from sklearn import preprocessing
import numpy as np


class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.df = df
        self.cat_feats = categorical_features
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
        self.feat_names = ohe.get_feature_names(self.cat_feats)
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

