import torch
from torch.utils.data import Dataset as TorchDataset


class TorchTextDataset(TorchDataset):
    def __init__(self,
                 df,
                 encodings,
                 categorical_feats,
                 numerical_feats,
                 labels,
                 class_weights=None
                 ):
        """
        TorchDataset Wrapper for text dataset with categorical features
        and numerical features
        """
        self.df = df
        self.encodings = encodings
        self.cat_feats = categorical_feats
        self.numerical_feats = numerical_feats
        self.labels = labels
        self.class_weights = class_weights

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['cat_feats'] = torch.tensor(self.cat_feats[idx]).float() \
            if self.cat_feats is not None else torch.zeros(0)
        item['numerical_feats'] = torch.tensor(self.numerical_feats[idx]).float()\
            if self.numerical_feats is not None else torch.zeros(0)
        return item

    def __len__(self):
        return len(self.labels)
