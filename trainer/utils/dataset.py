import h5py
import torch
from torch.utils.data import Dataset


class ObjectDataset(Dataset):
    """Dataset for Electron training

    Args:
        Dataset (Dataset): _description_
    """

    def __init__(self, h5_paths, start, limit, x_dim, y_dim):
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self._archives = None

        y = self.archives[0]["data"][start : (start + limit), 0:y_dim]
        x = self.archives[0]["data"][start : (start + limit), y_dim : (y_dim + x_dim)]
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
        print("Dataset created.")
        print("x_max: ", self.x_train.max())
        print("x_min: ", self.x_train.min())
        print("y_max: ", self.y_train.max())
        print("y_min: ", self.y_train.min())
        print("Nan in x":, torch.isnan(self.x_train).sum())
        print("Nan in y":, torch.isnan(self.y_train).sum())
        print("Inf in x":, torch.isinf(self.x_train).sum())
        print("Inf in y":, torch.isinf(self.y_train).sum())

    @property
    def archives(self):
        if self._archives is None:
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
