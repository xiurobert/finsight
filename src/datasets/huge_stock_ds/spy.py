import torch
import torch.utils.data as data
import os
import pandas as pd


class SPYDataset(data.Dataset):
    def __init__(self, data_dir: str, features: list[str], label: str):
        """
        Constructs a SPYDataset. If hidden columns are specified, they will be dropped from the dataframe

        :param data_dir: The data directory. This should contain the folders Stocks and ETFs
        :param features: The feature columns to be used
        :param label: The label column to be used
        """
        self.data_dir = data_dir
        self.features = features
        self.label = label
        self._file_path = os.path.join(self.data_dir, "ETFs", "spy.us.txt")
        self.data = pd.read_csv(self._file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: Fix this soon
        features = self.data.iloc[idx][self.features].values
        label = self.data.iloc[idx][[self.label]].values
        return torch.Tensor(list(features)), label[0]
