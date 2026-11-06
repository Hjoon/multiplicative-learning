from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils import bipolar_targets, sync_device


@dataclass
class DataBundle:
    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    input_dim: int
    num_classes: int


@dataclass
class ResidentData:
    """Fully materialized train/test tensors that stay on the target device."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    train_target: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_target: torch.Tensor
    input_dim: int
    num_classes: int

    @property
    def n_train(self) -> int:
        return int(self.train_y.numel())

    @property
    def n_test(self) -> int:
        return int(self.test_y.numel())

    @property
    def resident_bytes(self) -> int:
        tensors = (
            self.train_x, self.train_y, self.train_target,
            self.test_x, self.test_y, self.test_target,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))


class ScaleTensor:
    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, x: torch.Tensor):
        if self.mode == "unit":
            return x
        if self.mode == "centered":
            return 2.0 * x - 1.0
        raise ValueError(f"Unknown input_scale={self.mode}")


def load_data(cfg) -> DataBundle:
    """
    Paper datasets only.

    MNIST:
        1 x 28 x 28 -> 784

    Fashion-MNIST:
        1 x 28 x 28 -> 784

    CIFAR-10 RGB:
        3 x 32 x 32 -> 3072

    Data files are shared through cfg.data_root (paper default: PROJECT/data).
    """
    name = cfg.dataset.lower()

    if name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            ScaleTensor(cfg.input_scale),
            transforms.Lambda(lambda x: x.flatten()),
        ])
        train = datasets.MNIST(
            cfg.data_root, train=True, download=True, transform=tfm
        )
        test = datasets.MNIST(
            cfg.data_root, train=False, download=True, transform=tfm
        )
        return DataBundle(train, test, 784, 10)

    if name == "fashionmnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            ScaleTensor(cfg.input_scale),
            transforms.Lambda(lambda x: x.flatten()),
        ])
        train = datasets.FashionMNIST(
            cfg.data_root, train=True, download=True, transform=tfm
        )
        test = datasets.FashionMNIST(
            cfg.data_root, train=False, download=True, transform=tfm
        )
        return DataBundle(train, test, 784, 10)

    if name == "cifar10":
        tfm = transforms.Compose([
            transforms.ToTensor(),                  # [3,32,32], RGB retained
            ScaleTensor(cfg.input_scale),
            transforms.Lambda(lambda x: x.flatten()),
        ])
        train = datasets.CIFAR10(
            cfg.data_root, train=True, download=True, transform=tfm
        )
        test = datasets.CIFAR10(
            cfg.data_root, train=False, download=True, transform=tfm
        )
        return DataBundle(train, test, 3072, 10)

    raise ValueError(
        f"Unknown dataset={cfg.dataset!r}. "
        "Supported datasets are 'mnist', 'fashionmnist', and 'cifar10'."
    )


def _materialize_dataset(
    dataset,
    *,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_workers: int,
    staging_batch_size: int,
):
    """Stream a torchvision dataset once into preallocated device tensors."""
    n_total = len(dataset)
    pin = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=min(max(1, int(staging_batch_size)), n_total),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=pin,
        persistent_workers=(int(num_workers) > 0),
    )

    X = torch.empty((n_total, input_dim), device=device, dtype=dtype)
    y_all = torch.empty((n_total,), device=device, dtype=torch.long)

    offset = 0
    for x, y in loader:
        n = int(y.numel())
        X[offset:offset+n].copy_(
            x.to(device=device, dtype=dtype, non_blocking=pin)
        )
        y_all[offset:offset+n].copy_(
            y.to(device=device, dtype=torch.long, non_blocking=pin)
        )
        offset += n

    if offset != n_total:
        raise RuntimeError(f"Dataset materialization stopped at {offset}/{n_total}")
    return X, y_all


def make_resident_data(
    cfg,
    bundle: DataBundle,
    device: torch.device,
    dtype: torch.dtype,
) -> ResidentData:
    """
    Materialize the complete train and test sets on ``device`` once per run.

    After this function returns, optimization/evaluation never transfers samples
    from CPU to GPU. ``num_workers`` is used only during this one-time preload.
    """
    staging_bs = max(int(getattr(cfg, "eval_batch_size", 4096)), 4096)

    train_x, train_y = _materialize_dataset(
        bundle.train_dataset,
        input_dim=bundle.input_dim,
        device=device,
        dtype=dtype,
        num_workers=cfg.num_workers,
        staging_batch_size=staging_bs,
    )
    test_x, test_y = _materialize_dataset(
        bundle.test_dataset,
        input_dim=bundle.input_dim,
        device=device,
        dtype=dtype,
        num_workers=cfg.num_workers,
        staging_batch_size=staging_bs,
    )

    train_target = bipolar_targets(
        train_y, bundle.num_classes, cfg.target_value
    ).to(dtype=dtype)
    test_target = bipolar_targets(
        test_y, bundle.num_classes, cfg.target_value
    ).to(dtype=dtype)

    # Ensure the one-time preload is complete before the experiment wall clock
    # starts. Optimization batches thereafter use only device-resident indexing.
    sync_device(device)

    return ResidentData(
        train_x=train_x,
        train_y=train_y,
        train_target=train_target,
        test_x=test_x,
        test_y=test_y,
        test_target=test_target,
        input_dim=bundle.input_dim,
        num_classes=bundle.num_classes,
    )


def resident_steps_per_epoch(cfg, n_train: int) -> int:
    bs_cfg = int(cfg.batch_size)
    if bs_cfg < 0 or bs_cfg >= n_train:
        return 1
    if bs_cfg <= 0:
        raise ValueError("batch_size must be positive or -1 for full batch")
    if bool(cfg.drop_last):
        return max(1, n_train // bs_cfg)
    return max(1, int(math.ceil(n_train / bs_cfg)))


def make_shuffle_generator(seed: int, device: torch.device):
    # Keep permutations on the resident device. CUDA generators are explicit so
    # each run has an isolated deterministic stream independent of other jobs.
    if device.type == "cuda":
        g = torch.Generator(device=device)
    else:
        g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def resident_train_batches(data: ResidentData, cfg, generator):
    """Yield one deterministic shuffled pass over resident training tensors."""
    n = data.n_train
    bs_cfg = int(cfg.batch_size)

    if bs_cfg < 0 or bs_cfg >= n:
        yield data.train_x, data.train_y, data.train_target
        return
    if bs_cfg <= 0:
        raise ValueError("batch_size must be positive or -1 for full batch")

    perm = torch.randperm(n, device=data.train_x.device, generator=generator)
    stop = (n // bs_cfg) * bs_cfg if bool(cfg.drop_last) else n

    for start in range(0, stop, bs_cfg):
        idx = perm[start:min(start + bs_cfg, n)]
        if idx.numel() < bs_cfg and bool(cfg.drop_last):
            break
        yield (
            data.train_x.index_select(0, idx),
            data.train_y.index_select(0, idx),
            data.train_target.index_select(0, idx),
        )


# Legacy CPU DataLoader helper retained for exploratory code that explicitly
# wants it. The paper runner uses make_resident_data/resident_train_batches.
def make_loaders(cfg, bundle: DataBundle):
    bs = len(bundle.train_dataset) if int(cfg.batch_size) < 0 else int(cfg.batch_size)
    eval_bs = min(int(cfg.eval_batch_size), len(bundle.train_dataset))
    pin = torch.cuda.is_available()
    g = torch.Generator().manual_seed(cfg.seed + 777)

    common = dict(
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(int(cfg.num_workers) > 0),
    )
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=bs,
        shuffle=True,
        generator=g,
        drop_last=bool(cfg.drop_last) and bs < len(bundle.train_dataset),
        **common,
    )
    train_eval_loader = DataLoader(
        bundle.train_dataset,
        batch_size=eval_bs,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=min(eval_bs, len(bundle.test_dataset)),
        shuffle=False,
        **common,
    )
    return train_loader, train_eval_loader, test_loader
