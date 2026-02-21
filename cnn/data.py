from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZPixelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        # Ensure expected dtypes
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

        if mean is not None and std is not None:
            std = np.where(std == 0, 1.0, std)
            self.X = (self.X - mean.astype(np.float32)) / std.astype(np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)