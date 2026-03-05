from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import random


class NPZPatchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None, is_train: bool = False):
        # Allow arrays of any shape, but expect (N, C, H, W) for CNN patches
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.is_train = is_train

        # Normalization (per channel)
        # Expected mean/std shape: (C,) -> expand to (1, C, 1, 1) for broadcasting
        # Fallback if standard MLP: (C,) -> expand to (1, C)
        if mean is not None and std is not None:
            std = np.where(std == 0, 1.0, std)
            
            # Broadcast appropriately based on dimensions of X
            if self.X.ndim == 4:
                mean_b = mean.reshape(1, -1, 1, 1)
                std_b = std.reshape(1, -1, 1, 1)
            else:
                mean_b = mean.reshape(1, -1)
                std_b = std.reshape(1, -1)
                
            self.X = (self.X - mean_b.astype(np.float32)) / std_b.astype(np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        
        # Spatial Augmentations
        if self.is_train and x.ndim == 3: # (C, H, W)
            # Random horizontal flip
            if random.random() > 0.5:
                # flip along W axis (axis=2)
                x = np.flip(x, axis=2)
            
            # Random vertical flip
            if random.random() > 0.5:
                # flip along H axis (axis=1)
                x = np.flip(x, axis=1)
                
            # Random 90 degree rotations (0, 1, 2, or 3 times)
            k = random.randint(0, 3)
            if k > 0:
                # rotate in spatial axes (1 and 2)
                x = np.rot90(x, k=k, axes=(1, 2))
                
        # Must make a copy since negative strides from flip/rot are not supported by torch.from_numpy
        if self.is_train and x.ndim == 3:
            x = x.copy()
            
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)