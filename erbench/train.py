from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import (
    load_data,
    make_resident_data,
    resident_steps_per_epoch,
    make_shuffle_generator,
    resident_train_batches,
)
from .models import TanhMLP
from .er import er_step
from .identity import (
    experiment_id,
    experiment_payload,
    run_dir as identity_run_dir,
    is_complete,
)
from .utils import set_seed, resolve_device, resolve_dtype, sync_device


def make_optimizer(model, cfg):
    name = cfg.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.0, weight_decay=cfg.weight_decay
        )
    if name in ("momentum", "sgdm"):
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=cfg.lr, momentum=0.0, weight_decay=cfg.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if name in ("er", "er0", "er1", "pcg32_10"):
        return None
    raise ValueError(f"Unknown optimizer={cfg.optimizer}")


def _write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        if path.exists():
            path.unlink()
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


@torch.no_grad()
def evaluate_resident(model, x, y, target, batch_size: int):
    """
    Evaluate device-resident tensors without per-minibatch host synchronization.
    Loss/correct counts stay on the GPU and are copied to CPU together once per
    split.
    """
    model.eval()
    loss_sum = torch.zeros((), device=x.device, dtype=x.dtype)
    correct = torch.zeros((), device=x.device, dtype=torch.long)
    n = int(y.numel())
    bs = min(max(1, int(batch_size)), n)

    for start in range(0, n, bs):
        xb = x[start:start + bs]
        yb = y[start:start + bs]
        tb = target[start:start + bs]
        pred = model(xb)
        loss_sum.add_(0.5 * (pred - tb).square().sum())
        correct.add_((pred.argmax(1) == yb).sum())

    stats = torch.stack(
        (loss_sum, correct.to(dtype=x.dtype))
    ).detach().cpu()
    return {
        "loss": float(stats[0]) / (n * target.shape[1]),
        "accuracy": float(stats[1]) / n,
        "n": n,
    }


def _gradient_step(model, optimizer, x, y, target, collect_metrics: bool):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    old_norm_sq = None
    old_params = None
    if collect_metrics:
        old_norm_sq = sum(p.detach().square().sum() for p in model.parameters())
        old_params = [p.detach().clone() for p in model.parameters()]

    pred = model(x)
    loss = 0.5 * (pred - target).square().mean()
    loss.backward()

    if collect_metrics:
        acc = (pred.argmax(1) == y).float().mean()
        grad_norm_sq = sum(
            p.grad.detach().square().sum()
            for p in model.parameters()
            if p.grad is not None
        )

    optimizer.step()

    if not collect_metrics:
        return {
            "batch_loss_before": float("nan"),
            "batch_acc_before": float("nan"),
            "batch_loss_after": "",
            "batch_acc_after": "",
            "grad_norm": "",
            "relative_update_norm": "",
        }

    upd_norm_sq = sum(
        (p.detach() - old).square().sum()
        for p, old in zip(model.parameters(), old_params)
    )
    rel_update = torch.sqrt(upd_norm_sq) / torch.sqrt(old_norm_sq).clamp_min(1e-12)

    metrics_cpu = torch.stack((
        loss.detach(),
        acc.detach(),
        torch.sqrt(grad_norm_sq).detach(),
        rel_update.detach(),
    )).cpu()
    return {
        "batch_loss_before": float(metrics_cpu[0]),
        "batch_acc_before": float(metrics_cpu[1]),
        "batch_loss_after": "",
        "batch_acc_after": "",
        "grad_norm": float(metrics_cpu[2]),
        "relative_update_norm": float(metrics_cpu[3]),
    }


def _parameters_finite(model):
    """Reduce all parameter finite checks on-device, then synchronize once."""
    checks = [torch.isfinite(p).all() for p in model.parameters()]
    if not checks:
        return True
    return bool(torch.stack(checks).all().item())


def _period_steps(step_frequency, epoch_frequency, steps_per_epoch):
    if epoch_frequency is not None:
        return max(1, int(round(float(epoch_frequency) * steps_per_epoch)))
    if step_frequency is None or int(step_frequency) <= 0:
        return None
    return int(step_frequency)


def _is_log_decade_step(step: int) -> bool:
    """Sparse schedule: 1..9, 10..90 by 10, then 100,200,..."""
    step = int(step)
    if 1 <= step <= 9:
        return True
    if 10 <= step <= 99:
        return step % 10 == 0
    if step >= 100:
        return step % 100 == 0
    return False


def _should_record_step(cfg, step: int, target_steps: int, periodic_steps=None) -> bool:
    if int(step) == int(target_steps):
        return True
    mode = str(getattr(cfg, "eval_schedule", "sparse")).lower()
    if mode == "endpoints":
        return False
    if mode != "sparse":
        raise ValueError(f"Unknown eval_schedule={mode!r}")
    return _is_log_decade_step(step)


def _is_er_optimizer(name: str) -> bool:
    return str(name).lower() in {"er", "er0", "er1", "pcg32_10"}


def _acquire_lock(lock_dir: Path, timeout_s: float = 600.0):
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            lock_dir.mkdir()
            (lock_dir / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "time": time.time()}),
                encoding="utf-8",
            )
            return
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for run lock: {lock_dir}")
            time.sleep(1.0)


def _release_lock(lock_dir: Path):
    shutil.rmtree(lock_dir, ignore_errors=True)


def _load_cached(cfg):
    rd = identity_run_dir(cfg)
    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    return rd, None, {"summary": summary, "cached": True}


def run_experiment(
    cfg: ExperimentConfig,
    *,
    force: bool = False,
    lock_timeout: float = 600.0,
):
    """
    Run one experiment in the global content-addressed cache.

    The complete train/test datasets are materialized on the target device before
    the experiment wall clock starts. Wall time is measured with exactly one
    device synchronization at the start and one at the end; no per-step timers
    are used. Evaluation and explicit stability checks may synchronize internally,
    and their cost is included in the measured experiment wall time.
    """
    rid = experiment_id(cfg)
    rd = identity_run_dir(cfg)
    lock_dir = Path(cfg.output_root) / ".locks" / f"{rid}.lock"

    if is_complete(cfg) and not force:
        print(f"[cache hit] {rid}")
        return _load_cached(cfg)

    _acquire_lock(lock_dir, timeout_s=lock_timeout)
    try:
        if is_complete(cfg) and not force:
            print(f"[cache hit after wait] {rid}")
            return _load_cached(cfg)

        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True, exist_ok=True)

        _write_json(rd / "status.json", {
            "status": "running",
            "run_id": rid,
            "started_at_unix": time.time(),
            "pid": os.getpid(),
        })
        cfg.save_json(rd / "config.json")
        _write_json(rd / "identity.json", experiment_payload(cfg))

        set_seed(cfg.seed, cfg.deterministic)
        device = resolve_device(cfg.device)
        dtype = resolve_dtype(cfg.dtype)

        bundle = load_data(cfg)
        cfg.num_classes = bundle.num_classes
        resident = make_resident_data(cfg, bundle, device, dtype)

        model = TanhMLP(
            input_dim=bundle.input_dim,
            hidden1=cfg.hidden1,
            hidden2=cfg.hidden2,
            num_classes=bundle.num_classes,
        ).to(device=device, dtype=dtype)
        optimizer = make_optimizer(model, cfg)

        train_rows = []
        eval_rows = []
        er_layer_rows = []

        # For stability runs, track the true maximum applied ER update entirely
        # on-device and materialize it only once at the end.
        run_max_er_update = (
            torch.zeros((), device=device, dtype=dtype)
            if (_is_er_optimizer(cfg.optimizer) and cfg.log_er_layers)
            else None
        )

        n_train = resident.n_train
        steps_per_epoch = resident_steps_per_epoch(cfg, n_train)
        target_steps = (
            int(cfg.max_steps)
            if cfg.max_steps is not None
            else max(1, int(math.ceil(float(cfg.epochs) * steps_per_epoch)))
        )

        eval_period = _period_steps(cfg.eval_every, cfg.eval_every_epochs, steps_per_epoch)
        diag_period = _period_steps(cfg.diag_every, cfg.diag_every_epochs, steps_per_epoch)
        shuffle_generator = make_shuffle_generator(cfg.seed + 777, device)

        # Exactly one timing sync at the beginning. Data preload/model creation are
        # intentionally outside the measured experiment wall clock.
        sync_device(device)
        wall_t0 = time.perf_counter()

        # Step 0 = initialization, before any optimization update.
        train0 = evaluate_resident(
            model, resident.train_x, resident.train_y, resident.train_target,
            cfg.eval_batch_size,
        )
        test0 = evaluate_resident(
            model, resident.test_x, resident.test_y, resident.test_target,
            cfg.eval_batch_size,
        )
        for split, m in (("train", train0), ("test", test0)):
            eval_rows.append({
                "step": 0,
                "epoch": 0.0,
                "equiv_epoch": 0.0,
                "samples_seen": 0,
                "split": split,
                **m,
            })

        step = 0
        samples_seen = 0
        last_eval_step = 0
        model.train()

        try:
            while step < target_steps:
                for x, y, target in resident_train_batches(
                    resident, cfg, shuffle_generator
                ):
                    if step >= target_steps:
                        break

                    next_step = step + 1
                    next_samples = samples_seen + y.numel()
                    log_this_step = _should_record_step(
                        cfg, next_step, target_steps, periodic_steps=eval_period
                    )
                    diag_mode = str(
                        getattr(cfg, "diag_schedule", "periodic")
                    ).lower()
                    if diag_mode == "sparse":
                        do_diag = (
                            _is_er_optimizer(cfg.optimizer)
                            and (
                                _is_log_decade_step(next_step)
                                or next_step == target_steps
                            )
                        )
                    elif diag_mode == "periodic":
                        do_diag = (
                            _is_er_optimizer(cfg.optimizer)
                            and diag_period is not None
                            and (
                                next_step == 1
                                or next_step % diag_period == 0
                                or next_step == target_steps
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unknown diag_schedule={diag_mode!r}"
                        )
                    collect_step_metrics = bool(log_this_step)

                    if _is_er_optimizer(cfg.optimizer):
                        result = er_step(
                            model, x, y, target, cfg,
                            do_diagnostics=do_diag,
                            collect_metrics=collect_step_metrics,
                        )
                        if (
                            run_max_er_update is not None
                            and result.max_relative_update_applied_tensor is not None
                        ):
                            run_max_er_update = torch.maximum(
                                run_max_er_update,
                                result.max_relative_update_applied_tensor.detach(),
                            )

                        if collect_step_metrics:
                            row = {
                                "step": next_step,
                                "optimizer": cfg.optimizer,
                                "batch_size_actual": y.numel(),
                                "batch_loss_before": result.batch_loss_before,
                                "batch_acc_before": result.batch_acc_before,
                                "batch_loss_after": result.batch_loss_after,
                                "batch_acc_after": result.batch_acc_after,
                                "grad_norm": "",
                                "relative_update_norm": (
                                    float(result.max_relative_update_applied_tensor.item())
                                    if result.max_relative_update_applied_tensor is not None
                                    else float("nan")
                                ),
                            }
                        else:
                            row = None

                        # Diagnostics are independent of full train/test metric
                        # logging. Endpoint-only Fig.4 must still persist the
                        # step 1, 250, 500, ... layer/SVD diagnostic rows.
                        if (cfg.log_er_layers and log_this_step) or do_diag:
                            for lr in result.layer_rows:
                                er_layer_rows.append({
                                    "step": next_step,
                                    "epoch": next_step / steps_per_epoch,
                                    "equiv_epoch": next_samples / n_train,
                                    "samples_seen": next_samples,
                                    **lr,
                                })
                    else:
                        g = _gradient_step(
                            model, optimizer, x, y, target,
                            collect_metrics=collect_step_metrics,
                        )
                        row = {
                            "step": next_step,
                            "optimizer": cfg.optimizer,
                            "batch_size_actual": y.numel(),
                            **g,
                        } if collect_step_metrics else None

                    step = next_step
                    samples_seen = next_samples

                    if log_this_step and row is not None:
                        row.update({
                            "epoch": step / steps_per_epoch,
                            "equiv_epoch": samples_seen / n_train,
                            "samples_seen": samples_seen,
                        })
                        train_rows.append(row)

                    # Ordinary runs avoid a forced GPU->CPU synchronization on
                    # every optimization step. Stability runs retain exact-step
                    # failure detection because failure step is an outcome.
                    if (log_this_step or cfg.log_er_layers) and not _parameters_finite(model):
                        raise FloatingPointError(
                            f"Non-finite parameter detected at step={step}"
                        )

                    if log_this_step and row is not None:
                        print(
                            f"[{rid}] step={step}/{target_steps} "
                            f"epoch={step/steps_per_epoch:.3f} "
                            f"samples={samples_seen} "
                            f"batch_loss={float(row['batch_loss_before']):.5f}"
                        )

                        train_m = evaluate_resident(
                            model,
                            resident.train_x,
                            resident.train_y,
                            resident.train_target,
                            cfg.eval_batch_size,
                        )
                        test_m = evaluate_resident(
                            model,
                            resident.test_x,
                            resident.test_y,
                            resident.test_target,
                            cfg.eval_batch_size,
                        )
                        for split, m in (("train", train_m), ("test", test_m)):
                            eval_rows.append({
                                "step": step,
                                "epoch": step / steps_per_epoch,
                                "equiv_epoch": samples_seen / n_train,
                                "samples_seen": samples_seen,
                                "split": split,
                                **m,
                            })
                        last_eval_step = step
                        model.train()

            if last_eval_step != step:
                train_m = evaluate_resident(
                    model,
                    resident.train_x,
                    resident.train_y,
                    resident.train_target,
                    cfg.eval_batch_size,
                )
                test_m = evaluate_resident(
                    model,
                    resident.test_x,
                    resident.test_y,
                    resident.test_target,
                    cfg.eval_batch_size,
                )
                for split, m in (("train", train_m), ("test", test_m)):
                    eval_rows.append({
                        "step": step,
                        "epoch": step / steps_per_epoch,
                        "equiv_epoch": samples_seen / n_train,
                        "samples_seen": samples_seen,
                        "split": split,
                        **m,
                    })

            # Exactly one timing sync at the end.
            sync_device(device)
            experiment_wall_seconds = time.perf_counter() - wall_t0
            mean_wall_ms_per_step = 1000.0 * experiment_wall_seconds / max(step, 1)
            max_er_update_value = (
                float(run_max_er_update.item())
                if run_max_er_update is not None else None
            )

            # Run-level mean is repeated on sparse train rows for convenient CSV
            # analysis; it is not a per-step timer.
            for row in train_rows:
                row["mean_wall_ms_per_step"] = mean_wall_ms_per_step

            _write_csv(rd / "train_steps.csv", train_rows)
            _write_csv(rd / "eval.csv", eval_rows)
            _write_csv(rd / "er_layers.csv", er_layer_rows)

            if cfg.save_model:
                torch.save(model.state_dict(), rd / "model.pt")

            final_train = [r for r in eval_rows if r["split"] == "train"][-1]
            final_test = [r for r in eval_rows if r["split"] == "test"][-1]
            summary = {
                "status": "completed",
                "run_id": rid,
                "run_dir": str(rd),
                "device": str(device),
                "input_dim": bundle.input_dim,
                "steps": step,
                "epochs": step / steps_per_epoch,
                "equiv_epochs": samples_seen / n_train,
                "samples_seen": samples_seen,
                "final_train_loss": final_train["loss"],
                "final_train_accuracy": final_train["accuracy"],
                "final_test_loss": final_test["loss"],
                "final_test_accuracy": final_test["accuracy"],
                "experiment_wall_seconds": experiment_wall_seconds,
                "mean_wall_ms_per_step": mean_wall_ms_per_step,
                "max_applied_relative_update": max_er_update_value,
                "wall_timing_scope": (
                    "timer boundaries synchronized once each; endpoint/sparse "
                    "evaluation and explicit stability finite checks are included"
                ),
                "data_resident_on_device": True,
                "resident_data_bytes": resident.resident_bytes,
                "resident_data_mib": resident.resident_bytes / (1024.0 ** 2),
            }
            _write_json(rd / "summary.json", summary)
            _write_json(rd / "status.json", {
                "status": "completed",
                "run_id": rid,
                "finished_at_unix": time.time(),
            })
            return rd, model, {
                "train_steps": train_rows,
                "eval": eval_rows,
                "er_layers": er_layer_rows,
                "summary": summary,
                "cached": False,
            }

        except Exception as e:
            _write_csv(rd / "train_steps.csv", train_rows)
            _write_csv(rd / "eval.csv", eval_rows)
            _write_csv(rd / "er_layers.csv", er_layer_rows)
            failure_max_update = None
            if run_max_er_update is not None:
                try:
                    failure_max_update = float(run_max_er_update.item())
                except Exception:
                    failure_max_update = None
            _write_json(rd / "failure.json", {
                "status": "failed",
                "run_id": rid,
                "step": step,
                "max_applied_relative_update": failure_max_update,
                "samples_seen": samples_seen,
                "exception_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
            _write_json(rd / "status.json", {
                "status": "failed",
                "run_id": rid,
                "failed_at_unix": time.time(),
                "step": step,
                "samples_seen": samples_seen,
                "exception_type": type(e).__name__,
                "message": str(e),
            })
            raise

    finally:
        _release_lock(lock_dir)
