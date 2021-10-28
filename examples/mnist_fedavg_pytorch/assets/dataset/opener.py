import os

import pandas as pd
import torch

import substratools as tools


class MNISTOpener(tools.Opener):
    def get_X(self, folders):
        data = self._get_data(folders, True)
        return self._get_X(data)

    def get_y(self, folders):
        data_y = self._get_data(folders, False)
        return self._get_y(data_y)

    def save_predictions(self, y_pred, path):
        with open(path, 'wb') as f:
            # Convert from tensor to numpy to dataframe to csv
            y_pred_np = y_pred.numpy()
            y_pred_df = pd.DataFrame(y_pred_np)
            y_pred_df.to_csv(f, index=False)

    def get_predictions(self, path):
        # Convert from csv to dataframe to numpy to tensor
        y_pred_df = pd.read_csv(path)
        y_pred_np = y_pred_df.to_numpy()
        return torch.from_numpy(y_pred_np)

    def fake_X(self, n_samples=None):
        data = self._fake_data(n_samples)
        return self._get_X(data)

    def fake_y(self, n_samples=None):
        data = self._fake_data(n_samples)
        return self._get_y(data)

    @classmethod
    def _get_X(cls, data):
        X = [torch.from_numpy(sample)[None, :].float() for batch in data for sample in batch]
        return X

    @classmethod
    def _get_y(cls, data):
        y = [torch.from_numpy(sample).long() for batch in data for sample in batch]
        return y

    @classmethod
    def _fake_data(cls, n_samples=None):
        return pd.DataFrame()

    @classmethod
    def _get_data(cls, folders, x_or_y=True):

        # print("folders:")
        # print(folders)

        if x_or_y:
            suffix = 'x'
        else:
            suffix = 'y'

        # find csv files
        paths = []
        for folder in folders:
            paths += [os.path.join(folder, f) for f in os.listdir(folder) if f[-6:] == '_%s.csv' % suffix]

        # print("paths:")
        # print(paths)

        # load data into a list of dataframes
        data = []
        for path in paths:
            data.append(pd.read_csv(path).to_numpy())

        return data
