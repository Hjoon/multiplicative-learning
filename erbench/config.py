from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json


@dataclass
class ExperimentConfig:
    # Experiment
    dataset: str = "mnist"              # mnist | fashionmnist | cifar10
    optimizer: str = "er"               # er(Alg.1) | er0 | er1 | pcg32_10 | sgd | momentum | rmsprop | adam
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    deterministic: bool = False

    # Data
    data_root: str = "./data"
    num_workers: int = 4
    batch_size: int = 4096               # -1 = full batch
    eval_batch_size: int = 4096
    input_scale: str = "unit"            # unit=[0,1], centered=[-1,1]
    target_value: float = 1.0             # bipolar +/- target_value
    drop_last: bool = True

    # Model: input dim comes from dataset
    hidden1: int = 256
    hidden2: int = 64
    num_classes: int = 10

    # Training
    epochs: float = 10.0
    max_steps: Optional[int] = None
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0

    # Evaluation / logging
    # Evaluation schedule:
    #   sparse    = step 0, 1..9, 10..90 by 10, >=100 by 100, and final
    #   endpoints = step 0 and final only (used by speed-sensitive stability runs)
    eval_schedule: str = "sparse"        # sparse | endpoints
    # Legacy fields are retained only for backward-compatible config loading.
    eval_every: int = 20
    eval_every_epochs: Optional[float] = None

    # Expensive SVD diagnostics. If diag_every_epochs is set, it takes
    # precedence over diag_every. Cheap ER layer diagnostics are logged every step.
    # Expensive matrix/SVD diagnostic schedule:
    #   periodic = step 1 plus diag_every periodic checkpoints
    #   sparse   = steps 1..9, 10..90 by 10, >=100 by 100, and final
    diag_schedule: str = "periodic"       # periodic | sparse
    diag_every: int = 100
    diag_every_epochs: Optional[float] = None
    log_er_layers: bool = False
    log_every: int = 100

    # ER
    er_solver: str = "ridge"             # ridge | pinv
    ridge_lambda: float = 0.1            # relative coefficient when ridge_relative=True
    ridge_relative: bool = True
    ridge_strategy: str = "auto"          # auto | primal | dual
    solve_dtype: str = "float32"
    er_mix: float = 0.01
    secant_clip: Optional[float] = None
    secant_small: float = 1e-3
    secant_small_series: bool = True
    forward_recompute: bool = True
    cholesky_retries: int = 5

    # Output/cache
    output_root: str = "./results"
    run_name: Optional[str] = None         # cosmetic only; not part of run identity
    save_model: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)

    def save_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
