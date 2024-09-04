import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class TorchTabularTextDataset(TorchDataset):
    """
    :obj:`TorchDataset` wrapper for text dataset with categorical features
    and numerical features

    Parameters:
        encodings (:class:`transformers.BatchEncoding`):
            The output from encode_plus() and batch_encode() methods (tokens, attention_masks, etc) of
            a transformers.PreTrainedTokenizer
        categorical_feats (:class:`numpy.ndarray`, of shape :obj:`(n_examples, categorical feat dim)`, `optional`, defaults to :obj:`None`):
            An array containing the preprocessed categorical features
        numerical_feats (:class:`numpy.ndarray`, of shape :obj:`(n_examples, numerical feat dim)`, `optional`, defaults to :obj:`None`):
            An array containing the preprocessed numerical features
        labels (:class: list` or `numpy.ndarray`, `optional`, defaults to :obj:`None`):
            The labels of the training examples
        df (:class:`pandas.DataFrame`, `optional`, defaults to :obj:`None`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            TabularConfig instance specifying the configs for TabularFeatCombiner

    """

    def __init__(
        self,
        encodings,
        categorical_feats,
        numerical_feats,
        labels=None,
        df=None,
        label_list=None,
    ):
        self.df = df
        self.encodings = encodings
        self.cat_feats = categorical_feats
        self.numerical_feats = numerical_feats
        self.labels = labels
        self.label_list = (
            label_list
            if label_list is not None
            else [i for i in range(len(np.unique(labels)))]
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = (
            torch.tensor(self.labels[idx]) if self.labels is not None else None
        )
        item["cat_feats"] = (
            torch.tensor(self.cat_feats.iloc[idx]).float()
            if self.cat_feats is not None
            else torch.zeros(0)
        )
        item["numerical_feats"] = (
            torch.tensor(self.numerical_feats[idx]).float()
            if self.numerical_feats is not None
            else torch.zeros(0)
        )
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

    def get_labels(self):
        """returns the label names for classification"""
        return self.label_list
