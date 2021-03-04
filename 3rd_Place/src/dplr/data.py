from torch.utils.data import DataLoader, Dataset as TorchDataset
from typing import NamedTuple, Dict, List
from collections import defaultdict
import torch
import numpy as np
import pandas as pd


def ConcatDict(list_of_dict: List[Dict[str, torch.Tensor]]):
    tensor_dict = defaultdict(list)
    for observation in list_of_dict:
        for tensor_name, tensor in observation.items():
            if tensor.dim() <= 1:
                tensor.unsqueeze_(0)
            tensor_dict[tensor_name].append(tensor)
    return {tensor_name: torch.cat(tensors)
            for tensor_name, tensors in tensor_dict.items()}


class Dataset(TorchDataset):
    def __init__(self, device: str = 'cpu',  **datasets):
        self.datasets = datasets
        self.device = device
        self._lens = list(map(len, self.datasets.values()))
        assert len(np.unique(self._lens)) == 1

    def __len__(self):
        return self._lens[0]

    def _to_torch(self, dataset: np.ndarray):
        return torch.from_numpy(np.asarray(dataset)).to(device=self.device)

    def __getitem__(self, index):
        return {name: self._to_torch(data[index])
                for name, data in self.datasets.items()}

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame,
                       device: str = 'cpu',
                       **split_features):
        datasets = {features_name: data.loc[:, features].to_numpy()
                    for features_name, features in split_features.items()}
        return cls(device=device, **datasets)


class DataBunch(NamedTuple):
    train_dl: DataLoader
    valid_dl: DataLoader


def create_dl(data, device='cpu', batch_size=512,
              **feature_groups):
    ds = Dataset.from_dataframe(data, device=device, **feature_groups)
    dl = DataLoader(ds, batch_size=batch_size)
    return ds, dl
