from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str):
    table = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unknown dtype: {name}")
    return table[name]


def bipolar_targets(
    y: torch.Tensor,
    n_classes: int,
    value: float = 1.0,
) -> torch.Tensor:
    out = torch.full(
        (y.shape[0], n_classes),
        -float(value),
        device=y.device,
        dtype=torch.float32,
    )
    out.scatter_(1, y[:, None], float(value))
    return out


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class DeviceTimer:
    def __init__(self, device: torch.device):
        self.device = device

    def __enter__(self):
        sync_device(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        sync_device(self.device)
        self.elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
