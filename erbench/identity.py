from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json

from .config import ExperimentConfig


CACHE_SCHEMA = "er-suite-2026-08-v1"


INT_FIELDS = {
    "seed", "num_workers", "batch_size", "eval_batch_size",
    "hidden1", "hidden2", "num_classes", "max_steps",
    "eval_every", "diag_every", "log_every", "cholesky_retries",
}
FLOAT_FIELDS = {
    "target_value", "epochs", "lr", "momentum", "weight_decay",
    "eval_every_epochs", "diag_every_epochs", "ridge_lambda",
    "er_mix", "secant_clip", "secant_small",
}
BOOL_FIELDS = {
    "deterministic", "drop_last", "ridge_relative",
    "secant_small_series", "forward_recompute",
    "save_model", "log_er_layers",
}


def _normalize_scalar_types(d: dict) -> dict:
    out = dict(d)
    for k in INT_FIELDS:
        if k in out and out[k] is not None:
            out[k] = int(out[k])
    for k in FLOAT_FIELDS:
        if k in out and out[k] is not None:
            out[k] = float(out[k])
    for k in BOOL_FIELDS:
        if k in out and out[k] is not None:
            out[k] = bool(out[k])
    if "dataset" in out:
        out["dataset"] = str(out["dataset"]).lower()
    if "optimizer" in out:
        out["optimizer"] = str(out["optimizer"]).lower()
    return out


# These fields do not change the mathematical/training result.
# eval_schedule is intentionally identity-bearing because it changes the
# recorded experiment artifact (trajectory vs endpoints-only).
NON_IDENTITY_FIELDS = {
    "device",
    "data_root",
    "num_workers",
    "output_root",
    "run_name",
    "log_every",
    "eval_every",
    "eval_every_epochs",
}


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def training_code_hash() -> str:
    """
    Hash only files that affect experiment execution.
    Plotting/notebooks/README changes do not invalidate cached runs.
    """
    names = [
        "config.py", "data.py", "models.py", "er.py",
        "train.py", "utils.py", "identity.py",
    ]
    h = hashlib.sha256()
    root = _package_root()
    for name in names:
        p = root / name
        h.update(name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def canonical_config(cfg: ExperimentConfig) -> dict:
    d = _normalize_scalar_types(asdict(cfg))
    for k in NON_IDENTITY_FIELDS:
        d.pop(k, None)

    opt = str(d["optimizer"]).lower()

    # max_steps is authoritative when supplied; epochs is then descriptive only.
    if d.get("max_steps") is not None:
        d.pop("epochs", None)


    # Remove parameters that the selected optimizer cannot use. This means
    # irrelevant config differences do not create duplicate cache entries.
    if opt in ("er", "er0", "er1", "pcg32_10"):
        for k in ("lr", "momentum", "weight_decay"):
            d.pop(k, None)
    else:
        for k in (
            "er_solver", "ridge_lambda", "ridge_relative", "ridge_strategy",
            "solve_dtype", "er_mix", "secant_clip", "secant_small",
            "secant_small_series", "forward_recompute", "cholesky_retries",
            "diag_every", "diag_every_epochs", "log_er_layers",
        ):
            d.pop(k, None)
        if opt != "momentum":
            d.pop("momentum", None)

    return d


def experiment_payload(cfg: ExperimentConfig) -> dict:
    return {
        "cache_schema": CACHE_SCHEMA,
        "training_code_hash": training_code_hash(),
        "config": canonical_config(cfg),
    }


def experiment_id(cfg: ExperimentConfig) -> str:
    payload = experiment_payload(cfg)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()[:20]
    c = payload["config"]
    return f"{c['dataset']}-{c['optimizer']}-s{c['seed']}-{digest}"


def run_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.output_root) / "runs" / experiment_id(cfg)


def status_path(cfg: ExperimentConfig) -> Path:
    return run_dir(cfg) / "status.json"


def is_complete(cfg: ExperimentConfig) -> bool:
    rd = run_dir(cfg)
    required = [rd / "status.json", rd / "config.json", rd / "eval.csv", rd / "summary.json"]
    if not all(p.exists() for p in required):
        return False
    try:
        status = json.loads((rd / "status.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return status.get("status") == "completed"
