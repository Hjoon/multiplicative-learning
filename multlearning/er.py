from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json

import torch



def _work_dtype(name: str):
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("solve_dtype must be float32 or float64")


def stable_secant_gain(
    s: torch.Tensor,
    clip: Optional[float] = 10.0,
    small: float = 1e-3,
    use_small_series: bool = True,
):
    """
    Stable ER secant gain g(s)=s/tanh(s).

    Two independent stabilization mechanisms:
      1) small-s series:
            s/tanh(s) = 1 + s^2/3 - s^4/45 + O(s^6)
      2) large-s gain clipping:
            g <- min(g, clip)

    Setting use_small_series=False is the small-s ablation. Exact zero still
    receives the mathematical limiting value 1 to avoid an artificial 0/0 NaN.
    """
    abs_s = s.abs()
    tanh_s = torch.tanh(s)

    if use_small_series:
        s2 = s.square()
        series = 1.0 + s2 / 3.0 - s2.square() / 45.0
        safe_den = torch.where(abs_s < small, torch.ones_like(tanh_s), tanh_s)
        direct = s / safe_den
        gain = torch.where(abs_s < small, series, direct)
    else:
        safe_den = torch.where(tanh_s == 0, torch.ones_like(tanh_s), tanh_s)
        gain = s / safe_den
        gain = torch.where(s == 0, torch.ones_like(gain), gain)

    if clip is not None:
        gain = gain.clamp(min=1.0, max=float(clip))
    return gain


def _ridge_lambda_for(A: torch.Tensor, lambda_base: float, relative: bool):
    if not relative:
        return torch.as_tensor(lambda_base, dtype=A.dtype, device=A.device)

    # Relative to the average diagonal scale of the UNNORMALIZED Gram A^T A.
    # This is equivalent to ||A||_F^2 / d. If the objective is written using
    # G=(A^T A)/B, both G and lambda are divided by B.
    d = max(A.shape[1], 1)
    scale = A.square().sum() / d
    eps = torch.finfo(A.dtype).eps
    return float(lambda_base) * scale.clamp_min(eps)


def ridge_lstsq(
    A: torch.Tensor,
    B: torch.Tensor,
    lambda_base: float,
    relative: bool = True,
    strategy: str = "auto",
    solve_dtype: str = "float32",
    retries: int = 5,
    collect_info: bool = True,
):
    """
    Solve argmin_X ||A X - B||_F^2 + lambda ||X||_F^2.

    primal: (A^T A + λI)X = A^T B
    dual:   X = A^T (A A^T + λI)^(-1) B
    auto:   use the smaller square system.

    On the hot path (collect_info=False), ``cholesky_ex.info`` is never read
    back on the CPU. This removes a host synchronization from every local ER
    solve. Retry/fallback bookkeeping is retained only at sparse logging or
    diagnostic steps, where scalar materialization is already expected.
    """
    out_dtype = A.dtype
    wdtype = _work_dtype(solve_dtype)
    Aw = A.to(wdtype)
    Bw = B.to(wdtype)
    n, d = Aw.shape

    if strategy == "auto":
        route = "primal" if d <= n else "dual"
    elif strategy in ("primal", "dual"):
        route = strategy
    else:
        raise ValueError(f"Unknown ridge strategy={strategy}")

    lam = _ridge_lambda_for(Aw, lambda_base, relative)
    system_dim = d if route == "primal" else n

    def build_system(used_lam):
        if route == "primal":
            system = Aw.T @ Aw
            system.diagonal().add_(used_lam)
            rhs = Aw.T @ Bw
        else:
            system = Aw @ Aw.T
            system.diagonal().add_(used_lam)
            rhs = Bw
        return system, rhs

    # Fast asynchronous training path.
    if not collect_info:
        system, rhs = build_system(lam)
        chol, _ = torch.linalg.cholesky_ex(system, check_errors=False)
        sol = torch.cholesky_solve(rhs, chol)
        X = sol if route == "primal" else Aw.T @ sol
        return X.to(out_dtype), {
            "route": route,
            "lambda_used": "",
            "retries": "",
            "system_dim": int(system_dim),
            "fallback": False,
        }

    # Sparse diagnostic path: preserve retry/fallback behavior.
    used_lam = lam
    max_attempts = max(1, int(retries))
    for attempt in range(max_attempts):
        system, rhs = build_system(used_lam)
        chol, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if bool(torch.all(info == 0).item()):
            sol = torch.cholesky_solve(rhs, chol)
            X = sol if route == "primal" else Aw.T @ sol
            return X.to(out_dtype), {
                "route": route,
                "lambda_used": float(used_lam.detach().cpu()),
                "retries": attempt,
                "system_dim": int(system_dim),
                "fallback": False,
            }
        if attempt + 1 < max_attempts:
            used_lam = used_lam * 10.0

    # Preserve the old fallback semantics: solve the last system attempted,
    # rather than adding one extra decade of ridge after the retry loop.
    if route == "primal":
        X = torch.linalg.solve(system, rhs)
    else:
        X = Aw.T @ torch.linalg.solve(system, rhs)

    return X.to(out_dtype), {
        "route": route,
        "lambda_used": float(used_lam.detach().cpu()),
        "retries": max_attempts,
        "system_dim": int(system_dim),
        "fallback": True,
    }



PCG32_10_ITERS = 10


def pcg32_10_lstsq(
    A: torch.Tensor,
    B: torch.Tensor,
    lambda_base: float,
    relative: bool = True,
    strategy: str = "auto",
    collect_info: bool = True,
):
    """
    Fixed 10-iteration FP32 Jacobi-preconditioned CG approximation to the
    same ridge objective as ``ridge_lstsq``.

    This function is used only for the ER local WEIGHT regression when
    optimizer="pcg32_10". Backward transport remains on the ordinary exact
    ridge/pinv path selected by cfg.er_solver.
    """
    out_dtype = A.dtype
    Aw = A.to(torch.float32)
    Bw = B.to(torch.float32)
    n, d = Aw.shape

    if strategy == "auto":
        route = "primal" if d <= n else "dual"
    elif strategy in ("primal", "dual"):
        route = strategy
    else:
        raise ValueError(f"Unknown ridge strategy={strategy}")

    lam = _ridge_lambda_for(Aw, lambda_base, relative)
    tiny = torch.finfo(torch.float32).tiny

    if route == "primal":
        rhs = Aw.T @ Bw
        diag = Aw.square().sum(dim=0) + lam

        def Aop(V):
            return Aw.T @ (Aw @ V) + lam * V

        def finish(U):
            return U

        system_dim = d
    else:
        rhs = Bw
        diag = Aw.square().sum(dim=1) + lam

        def Aop(V):
            return Aw @ (Aw.T @ V) + lam * V

        def finish(U):
            return Aw.T @ U

        system_dim = n

    # Multiple RHS are advanced simultaneously, with one alpha/beta per RHS.
    U = torch.zeros_like(rhs)
    R = rhs.clone()
    Z = R / diag[:, None].clamp_min(tiny)
    P = Z.clone()
    rz = (R * Z).sum(dim=0)

    for _ in range(PCG32_10_ITERS):
        AP = Aop(P)
        pAp = (P * AP).sum(dim=0)
        active = (rz.abs() > tiny) & (pAp.abs() > tiny)
        alpha = torch.where(active, rz / pAp, torch.zeros_like(rz))

        U = U + P * alpha[None, :]
        R = R - AP * alpha[None, :]
        Z = R / diag[:, None].clamp_min(tiny)
        rz_new = (R * Z).sum(dim=0)

        active_beta = active & (rz.abs() > tiny)
        beta = torch.where(active_beta, rz_new / rz, torch.zeros_like(rz_new))
        P = Z + P * beta[None, :]
        rz = rz_new

    X = finish(U)
    return X.to(out_dtype), {
        "route": f"pcg-{route}",
        "lambda_used": float(lam.detach().cpu()) if collect_info else "",
        "retries": 0,
        "system_dim": int(system_dim),
        "fallback": False,
        "pcg_iterations": PCG32_10_ITERS,
    }


def pinv_lstsq(A: torch.Tensor, B: torch.Tensor, solve_dtype: str = "float32", collect_info: bool = True):
    wdtype = _work_dtype(solve_dtype)
    X = torch.linalg.pinv(A.to(wdtype)) @ B.to(wdtype)
    return X.to(A.dtype), {
        "route": "pinv",
        "lambda_used": 0.0,
        "retries": 0,
        "system_dim": int(min(A.shape)),
        "fallback": False,
    }


def solve_lstsq(A, B, cfg, collect_info: bool = True):
    if cfg.er_solver == "pinv":
        return pinv_lstsq(A, B, cfg.solve_dtype, collect_info=collect_info)
    if cfg.er_solver == "ridge":
        return ridge_lstsq(
            A, B,
            lambda_base=cfg.ridge_lambda,
            relative=cfg.ridge_relative,
            strategy=cfg.ridge_strategy,
            solve_dtype=cfg.solve_dtype,
            retries=cfg.cholesky_retries,
            collect_info=collect_info,
        )
    raise ValueError(f"Unknown er_solver={cfg.er_solver}")


def affine_delta_solve(z: torch.Tensor, delta_s: torch.Tensor, cfg, collect_info: bool = True):
    """
    Solve delta_s ~= z @ dW + db with an unregularized intercept via centering.
    dW uses manuscript convention [in_features, out_features].
    """
    z_mean = z.mean(dim=0, keepdim=True)
    s_mean = delta_s.mean(dim=0, keepdim=True)
    zc = z - z_mean
    sc = delta_s - s_mean

    if cfg.optimizer.lower() == "pcg32_10":
        dW, info = pcg32_10_lstsq(
            zc, sc,
            lambda_base=cfg.ridge_lambda,
            relative=cfg.ridge_relative,
            strategy=cfg.ridge_strategy,
            collect_info=collect_info,
        )
    else:
        dW, info = solve_lstsq(zc, sc, cfg, collect_info=collect_info)
    db = (s_mean - z_mean @ dW).squeeze(0)

    info = dict(info)
    if collect_info:
        residual = z @ dW + db - delta_s
        rel_resid = residual.norm() / delta_s.norm().clamp_min(1e-12)
        info["solve_residual"] = float(rel_resid.detach().cpu())
    else:
        info["solve_residual"] = ""
    return dW, db, info


def backward_transport(layer, delta_s: torch.Tensor, cfg, collect_info: bool = True):
    # PyTorch weight [out,in] is manuscript W^T.
    A = layer.weight
    X_T, info = solve_lstsq(A, delta_s.T, cfg, collect_info=collect_info)
    return X_T.T, info


@torch.no_grad()
def matrix_diagnostics(A: torch.Tensor, ridge_lambda: float = 0.0, relative: bool = True):
    """
    Expensive SVD diagnostic for the local regression design matrix.
    """
    # Paper runs and diagnostics use the same FP32 precision.
    Aw = A.detach().to(torch.float32)
    s = torch.linalg.svdvals(Aw)
    smax = s.max()
    smin = s.min()
    eps = torch.finfo(Aw.dtype).eps
    tol = smax * max(Aw.shape) * eps
    rank = int((s > tol).sum().item())
    cond = float((smax / smin.clamp_min(eps)).cpu())

    if ridge_lambda > 0:
        lam = _ridge_lambda_for(Aw, ridge_lambda, relative)
        eff_cond = float(((smax.square() + lam) / (smin.square() + lam)).cpu())
        lam_value = float(lam.cpu())
    else:
        eff_cond = float((smax.square() / smin.square().clamp_min(eps)).cpu())
        lam_value = 0.0

    return {
        "sv_max": float(smax.cpu()),
        "sv_min": float(smin.cpu()),
        "singular_values_json": json.dumps(s.detach().cpu().tolist()),
        "rank": rank,
        "rank_fraction": rank / min(Aw.shape),
        "condition": cond,
        "ridge_effective_condition": eff_cond,
        "lambda_effective": lam_value,
    }


@dataclass
class ERStepResult:
    batch_loss_before: float
    batch_acc_before: float
    batch_loss_after: float
    batch_acc_after: float
    layer_rows: List[Dict[str, Any]]
    max_relative_update_applied_tensor: Optional[torch.Tensor] = None


@torch.no_grad()
def er_step(
    model,
    x,
    y,
    target,
    cfg,
    do_diagnostics: bool = False,
    collect_metrics: bool = True,
):
    """
    One stabilized ER update.

    Optimizer aliases select the within-iteration ER variant:

      ``er0``  Algorithm 0.  Compute the complete backward correction first,
               form the corrected hidden activities
                   Z_l^corr = tanh(S_l + Delta S_l),
               and keep those corrected activities fixed as the regressors for
               all local weight solves in the current iteration.

      ``er1``  Algorithm 1.  Compute the complete backward correction first,
               then update layers from input to output, recomputing the
               *realized* post-activity after each lower-layer update.  Higher
               layers therefore regress on the newly realized activation.

      ``er``   Backward-compatible alias for Algorithm 1.

      ``pcg32_10`` uses the Algorithm-1 ordering, but replaces only the local
               weight ridge solve by the fixed 10-step PCG approximation.

    ``forward_recompute=False`` is retained only for backward compatibility of
    legacy ``er``/``er1`` runs; it is not used to define Algorithm 0.

    No wall-clock timing or CUDA synchronization is performed here. Expensive
    scalar diagnostics are materialized only when ``collect_metrics`` or
    ``do_diagnostics`` is requested.
    """
    # Batch metrics and layer/solver diagnostics are intentionally decoupled.
    # Fig.2/3 can log a batch point without forcing a CPU read of every local
    # solve's Cholesky info. Layer solve metadata is materialized only when it
    # will actually be saved (diagnostics, or explicit ER-layer logging).
    collect_layer_info = bool(
        do_diagnostics or (collect_metrics and getattr(cfg, "log_er_layers", False))
    )
    opt_name = str(cfg.optimizer).lower()
    algorithm0 = opt_name == "er0"

    z_list, s_list = model.forward_with_cache(x)
    pred = z_list[-1]

    if collect_metrics:
        loss_before_t = 0.5 * (pred - target).square().mean()
        acc_before_t = (pred.argmax(1) == y).float().mean()
    else:
        loss_before_t = acc_before_t = None

    # ------------------------------------------------------------
    # Shared backward correction.  All Delta S_l are computed from
    # the same pre-update forward state for both Algorithm 0 and 1.
    # ------------------------------------------------------------
    L = len(model.layers)
    delta_s_list = [None] * L
    backward_infos = [None] * L
    delta_z = target - pred

    for l in range(L - 1, -1, -1):
        gain = stable_secant_gain(
            s_list[l],
            clip=cfg.secant_clip,
            small=cfg.secant_small,
            use_small_series=cfg.secant_small_series,
        )
        delta_s = gain * delta_z
        delta_s_list[l] = delta_s

        if l > 0:
            delta_z, binfo = backward_transport(
                model.layers[l], delta_s, cfg, collect_info=collect_layer_info
            )
            backward_infos[l] = binfo

    layer_rows = []
    track_update_max = bool(getattr(cfg, "log_er_layers", False))
    need_update_norms = bool(collect_metrics or collect_layer_info or track_update_max)
    step_max_update = (
        torch.zeros((), device=x.device, dtype=x.dtype)
        if need_update_norms else None
    )

    def update_norms(layer, dW):
        old_w_norm = layer.weight.norm().clamp_min(1e-12)
        rel_update_raw = dW.T.norm() / old_w_norm
        rel_update_applied = float(cfg.er_mix) * rel_update_raw
        return rel_update_raw, rel_update_applied

    def make_layer_row(
        l, layer, dW, winfo, solve_input, rel_update_raw, rel_update_applied
    ):
        if not collect_layer_info:
            return None

        rel_cpu = torch.stack(
            (rel_update_raw, rel_update_applied)
        ).detach().cpu()
        row = {
            "layer": l + 1,
            "relative_update_norm_raw": float(rel_cpu[0]),
            "relative_update_norm_applied": float(rel_cpu[1]),
            "weight_solve_residual": winfo["solve_residual"],
            "weight_solver_route": winfo["route"],
            "weight_system_dim": winfo["system_dim"],
            "weight_lambda_used": winfo["lambda_used"],
            "weight_fallback": int(winfo["fallback"]),
            "weight_pcg_iterations": winfo.get("pcg_iterations", ""),
            "er_algorithm": 0 if algorithm0 else 1,
        }

        if backward_infos[l] is not None:
            row.update({
                "backward_solver_route": backward_infos[l]["route"],
                "backward_system_dim": backward_infos[l]["system_dim"],
                "backward_lambda_used": backward_infos[l]["lambda_used"],
            })

        if do_diagnostics:
            row.update(matrix_diagnostics(
                solve_input,
                ridge_lambda=(
                    cfg.ridge_lambda if cfg.er_solver == "ridge" else 0.0
                ),
                relative=cfg.ridge_relative,
            ))
        return row

    if algorithm0:
        # --------------------------------------------------------
        # Algorithm 0: precompute the corrected state and hold it
        # fixed throughout all local solves in this ER iteration.
        #
        # Zcorr[0] = X
        # Zcorr[l] = tanh(S_l + Delta S_l), l=1,...,L-1
        # --------------------------------------------------------
        z_corr = [x]
        for l in range(L - 1):
            z_corr.append(torch.tanh(s_list[l] + delta_s_list[l]))

        # Compute every proposal before applying any of them.  This makes the
        # fixed-corrected-state interpretation explicit and guards against a
        # future local-solve implementation accidentally depending on an
        # already-updated neighboring layer.
        proposals = []
        for l, layer in enumerate(model.layers):
            solve_input = z_corr[l]
            dW, db, winfo = affine_delta_solve(
                solve_input, delta_s_list[l], cfg, collect_info=collect_layer_info
            )
            if need_update_norms:
                rel_raw, rel_applied = update_norms(layer, dW)
                step_max_update = torch.maximum(
                    step_max_update, rel_applied.detach()
                )
            else:
                rel_raw = rel_applied = None
            row = make_layer_row(
                l, layer, dW, winfo, solve_input, rel_raw, rel_applied
            )
            if row is not None:
                layer_rows.append(row)
            proposals.append((dW, db))

        for layer, (dW, db) in zip(model.layers, proposals):
            layer.weight.add_(float(cfg.er_mix) * dW.T)
            if layer.bias is not None:
                layer.bias.add_(float(cfg.er_mix) * db)

        pred_after = model(x) if collect_metrics else None

    else:
        # --------------------------------------------------------
        # Algorithm 1 (default): update lower -> upper layers and
        # use the realized activation after each update as the next
        # local regression input.
        # --------------------------------------------------------
        z_current = x

        for l, layer in enumerate(model.layers):
            solve_input = z_current if cfg.forward_recompute else z_list[l]
            dW, db, winfo = affine_delta_solve(
                solve_input, delta_s_list[l], cfg, collect_info=collect_layer_info
            )

            if need_update_norms:
                rel_raw, rel_applied = update_norms(layer, dW)
                step_max_update = torch.maximum(
                    step_max_update, rel_applied.detach()
                )
            else:
                rel_raw = rel_applied = None
            row = make_layer_row(
                l, layer, dW, winfo, solve_input, rel_raw, rel_applied
            )
            if row is not None:
                layer_rows.append(row)

            layer.weight.add_(float(cfg.er_mix) * dW.T)
            if layer.bias is not None:
                layer.bias.add_(float(cfg.er_mix) * db)

            if cfg.forward_recompute:
                z_current = torch.tanh(layer(z_current))
            else:
                z_current = z_list[l + 1]

        if collect_metrics:
            pred_after = z_current if cfg.forward_recompute else model(x)
        else:
            pred_after = None

    if collect_metrics:
        loss_after_t = 0.5 * (pred_after - target).square().mean()
        acc_after_t = (pred_after.argmax(1) == y).float().mean()
        metrics_cpu = torch.stack((
            loss_before_t, acc_before_t, loss_after_t, acc_after_t
        )).detach().cpu()
        loss_before = float(metrics_cpu[0])
        acc_before = float(metrics_cpu[1])
        loss_after = float(metrics_cpu[2])
        acc_after = float(metrics_cpu[3])
    else:
        loss_before = acc_before = loss_after = acc_after = float("nan")

    return ERStepResult(
        batch_loss_before=loss_before,
        batch_acc_before=acc_before,
        batch_loss_after=loss_after,
        batch_acc_after=acc_after,
        layer_rows=layer_rows,
        max_relative_update_applied_tensor=step_max_update,
    )
