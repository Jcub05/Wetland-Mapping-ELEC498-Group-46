import torch
import torch.nn as nn


class PixelMLP(nn.Module):
    """
    Baseline deep model for per-pixel embeddings: input (B, 64) -> logits (B, 6)
    This is the correct starting point for your current NPZ dataset shape.
    """
    def __init__(self, in_features: int = 64, num_classes: int = 6, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)