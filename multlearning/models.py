from __future__ import annotations

import torch
from torch import nn


class TanhMLP(nn.Module):
    """
    Fully connected tanh network.

    nn.Linear modules intentionally retain PyTorch's default initialization,
    including default bias initialization.
    """
    def __init__(
        self,
        input_dim: int,
        hidden1: int = 256,
        hidden2: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden1),
            nn.Linear(hidden1, hidden2),
            nn.Linear(hidden2, num_classes),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for layer in self.layers:
            z = torch.tanh(layer(z))
        return z

    def logits(self, x):
        z = x
        for layer in self.layers[:-1]:
            z = torch.tanh(layer(z))
        return self.layers[-1](z)

    def forward_with_cache(self, x: torch.Tensor):
        z_list = [x]
        s_list = []
        z = x
        for layer in self.layers:
            s = layer(z)
            z = torch.tanh(s)
            s_list.append(s)
            z_list.append(z)
        return z_list, s_list

    @property
    def input_dim(self):
        return self.layers[0].in_features
