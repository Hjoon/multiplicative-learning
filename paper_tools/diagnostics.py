"""Paper sensitivity and layerwise diagnostic figures, without notebook execution."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import style
from .data import panel_runs as read_panel_runs, long_eval as read_long_eval

MODEL_COLORS={**style.COLORS, 'er':style.COLORS['er1']}
MAIN_LINEWIDTH=1.44
GUIDE_LINEWIDTH=.5
MARKER_SIZE=2.4
FIG3_SIZE=style.FIGSIZE
FIG4_SIZE=style.FIGSIZE
PLOT_EXCLUDED_OPTIMIZERS={'pcg32_10'}
TEMPORAL_BIN_START=2000.
TEMPORAL_BINS_PER_DECADE=10


def _set_positive_step_symlog(ax, **kwargs):
    style.xstyle(ax, max(float(ax.get_xlim()[1]), 1))


def _save(fig, output, name):
    for ax in fig.axes:
        if ax.get_xscale()=='symlog':style.xstyle(ax,max(float(ax.get_xlim()[1]),1))
    fig.tight_layout(pad=.8,w_pad=.7,h_pad=.8)
    style.save_publication(fig,output,name)

def mean_se(df, x, y):
    out = (
        df.groupby(x)[y]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    out["se"] = (
        out["std"].fillna(0)
        / np.sqrt(out["count"].clip(lower=1))
    )
    return out

def _exclude_plot_optimizers(df):
    if df is None or getattr(df, "empty", True) or "optimizer" not in df.columns:
        return df
    return df[~df["optimizer"].astype(str).str.lower().isin(PLOT_EXCLUDED_OPTIMIZERS)].copy()

def _nonnegative_step(df):
    """Keep initialization (step=0) and all later optimization steps."""
    if df is None or getattr(df, "empty", True):
        return df
    return df[df["step"].astype(float) >= 0].copy()

def temporal_bin_learning_curve(
    df,
    value_col,
    *,
    start=TEMPORAL_BIN_START,
    bins_per_decade=TEMPORAL_BINS_PER_DECADE,
):
    if df is None or df.empty:
        return df.copy()

    if "run_id" not in df.columns:
        raise ValueError("temporal_bin_learning_curve requires run_id")

    pieces = []
    log_start = np.log10(float(start))

    for run_id, g in df.groupby("run_id", sort=False):
        g = g.sort_values("step").copy()
        g["step"] = g["step"].astype(float)

        early = g[g.step <= start][["run_id", "step", value_col]].copy()
        pieces.append(early)

        late = g[g.step > start].copy()
        if late.empty:
            continue

        final_step = float(late.step.max())
        final_rows = late[np.isclose(late.step, final_step)]
        interior = late[~np.isclose(late.step, final_step)].copy()

        if not interior.empty:
            interior["_time_bin"] = np.floor(
                (np.log10(interior.step) - log_start) * bins_per_decade
            ).astype(int)

            agg = (
                interior.groupby("_time_bin", as_index=False)[value_col]
                .mean()
            )
            # Deterministic geometric bin center: identical x positions across
            # seeds with the same logging schedule.
            agg["step"] = float(start) * 10.0 ** (
                (agg["_time_bin"].astype(float) + 0.5) / bins_per_decade
            )
            agg["run_id"] = run_id
            pieces.append(agg[["run_id", "step", value_col]])

        # Exact terminal point is never swallowed by a temporal bin.
        pieces.append(
            pd.DataFrame({
                "run_id": [run_id],
                "step": [final_step],
                value_col: [float(final_rows[value_col].mean())],
            })
        )

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["run_id", "step"]).reset_index(drop=True)

def learning_curve_mean_se(df, value_col):
    binned = temporal_bin_learning_curve(df, value_col)
    return binned, mean_se(binned, "step", value_col)

def plot_sensitivity(results, output, checksums):
    def panel_runs(panel, completed_only=True):
        return read_panel_runs(results, panel, completed_only, checksums)
    def long_eval(panel, completed_only=True):
        return read_long_eval(results, panel, completed_only, checksums)
    def load_er_layers(run_dir):
        return pd.read_csv(Path(run_dir)/'er_layers.csv')
    def save_figure(fig, ignored_name):
        _save(fig, output, 'Fig3')
    style.publication_style()
    plt.rcParams["figure.dpi"]=120
    # 3. Figure 3: (a) batch, (b) mix, (c) relative ridge — one figure
    #
    # Fig.3 is an MNIST experiment in the current sweep definition.
    # Existing sweep results are only FILTERED here; no recomputation is required.
    # Adam is intentionally excluded from all Fig.3 comparisons.
    #
    # Style policy:
    #   - SGD reference: original SGD blue, SOLID.
    #   - ER variants: ER red, DASHED; dash pattern + alpha distinguish variants.
    #   - Every main Fig.3 curve uses MAIN_LINEWIDTH, the same final linewidth as
    #     the thick Fig.2 mean/reference curves.
    #   - step=0 is retained on the compressed positive-symlog axis.
    
    fig, axes = plt.subplots(1, 3, figsize=FIG3_SIZE, sharey=True, constrained_layout=False)
    ax_a, ax_b, ax_c = axes
    
    # Reusable variant styles.  All non-reference curves remain dashed.
    FIG3_VARIANT_STYLES = [
        {"ls": (0, (1.2, 1.4)),       "alpha": .58, "lw": MAIN_LINEWIDTH},
        {"ls": (0, (5.0, 2.0)),       "alpha": .74, "lw": MAIN_LINEWIDTH},
        {"ls": (0, (6.0, 2.0, 1, 2)), "alpha": .96, "lw": MAIN_LINEWIDTH},
        {"ls": (0, (2.5, 1.2)),       "alpha": .84, "lw": MAIN_LINEWIDTH},
    ]
    SGD_REFERENCE_STYLE = {
        "color": MODEL_COLORS["sgd"],
        "ls": "-",
        "alpha": 1.0,
        "lw": MAIN_LINEWIDTH,
    }
    
    def _plot_fig3_variant(
        ax,
        part,
        *,
        label,
        style,
        full_cohort_only=False,
    ):
        _, st = learning_curve_mean_se(part, "accuracy")
    
        if full_cohort_only:
            # For an intentionally unstable setting (rho=1), do not let late
            # survivor-only averages masquerade as an unconditional trajectory.
            # Keep only checkpoints at which every available run in this setting
            # still contributes.
            n_runs = int(part["run_id"].nunique())
            st = st[st["count"].astype(int) == n_runs].copy()
    
        ax.fill_between(
            st.step,
            st["mean"] - st.se,
            st["mean"] + st.se,
            color=MODEL_COLORS["er"],
            alpha=.065 * style["alpha"],
            lw=0,
        )
        ax.plot(
            st.step,
            st["mean"],
            color=MODEL_COLORS["er"],
            ls=style["ls"],
            alpha=style["alpha"],
            lw=style["lw"],
            label=label,
        )
    
    def _plot_fig3_sgd_reference(ax, d, *, label="SGD", batch_size=None):
        part = d[d.optimizer == "sgd"]
        if batch_size is not None:
            part = part[part.batch_size == batch_size]
        if part.empty:
            return
        _, st = learning_curve_mean_se(part, "accuracy")
        ax.fill_between(
            st.step,
            st["mean"] - st.se,
            st["mean"] + st.se,
            color=SGD_REFERENCE_STYLE["color"],
            alpha=.08,
            lw=0,
        )
        ax.plot(
            st.step,
            st["mean"],
            color=SGD_REFERENCE_STYLE["color"],
            ls=SGD_REFERENCE_STYLE["ls"],
            alpha=SGD_REFERENCE_STYLE["alpha"],
            lw=SGD_REFERENCE_STYLE["lw"],
            label=label,
        )
    
    # ------------------------------------------------------------
    # (a) ER batch sweep: B = 512, 2048, 4096, full only
    #     SGD B=4096 is the sole optimization reference.
    # ------------------------------------------------------------
    d = long_eval("fig3a")
    d = _nonnegative_step(d[d.split == "test"])
    
    BATCH_ORDER = [512, 2048, 4096, -1]
    
    # Give the standard B=4096 curve the strongest ER emphasis.
    BATCH_STYLE_INDEX = {
        512: 0,
        2048: 1,
        4096: 2,
        -1: 3,
    }
    
    for b in BATCH_ORDER:
        part = d[(d.optimizer == "er") & (d.batch_size == b)]
        if part.empty:
            continue
    
        label = "Full" if b == -1 else f"$n$={b}"
        _plot_fig3_variant(
            ax_a,
            part,
            label=label,
            style=FIG3_VARIANT_STYLES[BATCH_STYLE_INDEX[b]],
        )
    
    _plot_fig3_sgd_reference(
        ax_a,
        d,
        label="SGD",
        batch_size=4096,
    )
    
    ax_a.set(
        title="(a) Batch size",
        xlabel="Iteration",
    )
    _set_positive_step_symlog(ax_a, mark_initial=True)
    ax_a.set_ylim(0, 1.02)
    ax_a.axhline(0.1, c="k", ls=":", lw=GUIDE_LINEWIDTH, zorder=0)
    ax_a.axhline(1.0, c="k", ls=":", lw=GUIDE_LINEWIDTH, zorder=0)
    ax_a.grid(alpha=.15)
    ax_a.legend(loc="lower right", ncol=1, frameon=True)
    
    # ------------------------------------------------------------
    # (b) ER mixing sensitivity: mix = 1.0, 0.1, 0.01
    #     B=4096 for every curve. SGD default is the sole reference.
    #
    # rho=1.0 is intentionally included even though it is unstable.
    # Failed runs are loaded through their finite history. To avoid survivor bias,
    # the rho=1 aggregate is truncated once the full available cohort is no longer
    # represented at a checkpoint.
    # ------------------------------------------------------------
    d = long_eval("fig3b", completed_only=False)
    d = _nonnegative_step(d[d.split == "test"])
    
    MIX_VALUES = [1.0, 0.1, 0.01]
    MIX_STYLE_INDEX = {
        1.0: 0,
        0.1: 1,
        0.01: 2,  # manuscript default emphasized
    }
    
    for val in MIX_VALUES:
        part = d[
            (d.optimizer == "er")
            & np.isclose(d.er_mix.astype(float), val)
        ].copy()
    
        if part.empty:
            raise RuntimeError(
                f"Fig.3(b): no plotted data found for rho={val:g}. "
                "The manuscript figure requires rho in {1.0, 0.1, 0.01}."
            )
    
        n_runs = int(part["run_id"].nunique())
        failed_runs = (
            part.loc[part["run_status"].astype(str) != "completed", "run_id"]
            .drop_duplicates()
        )
        n_failed = int(len(failed_runs))
    
        if n_runs < 10:
            warnings.warn(
                f"Fig.3(b) rho={val:g}: only {n_runs}/10 run directories with "
                "evaluation history were resolved. Check the fig3b manifest/cache."
            )
    
        if np.isclose(val, 1.0):
            part = part[part["step"].astype(float) <= 300].copy()
    
        _plot_fig3_variant(
            ax_b,
            part,
            label=rf"$\rho$={val:g}",
            style=FIG3_VARIANT_STYLES[MIX_STYLE_INDEX[val]],
            full_cohort_only=np.isclose(val, 1.0),
        )
    
    _plot_fig3_sgd_reference(ax_b, d, label="SGD")
    
    ax_b.set(
        title="(b) Mixing",
        xlabel="Iteration",
    )
    _set_positive_step_symlog(ax_b, mark_initial=True)
    ax_b.set_ylim(0, 1.02)
    ax_b.axhline(0.1, c="k", ls=":", lw=GUIDE_LINEWIDTH, zorder=0)
    ax_b.axhline(1.0, c="k", ls=":", lw=GUIDE_LINEWIDTH, zorder=0)
    ax_b.grid(alpha=.15)
    ax_b.legend(loc="lower right", ncol=1, frameon=True)
    
    # ------------------------------------------------------------
    # (c) relative ridge: lambda_rel = 1e-3, 1e-1, 1e1 only
    #     B=4096 for every curve. SGD default is the sole reference.
    # ------------------------------------------------------------
    d = long_eval("fig3c")
    d = _nonnegative_step(d[d.split == "test"])
    
    RIDGE_VALUES = [1e-3, 1e-1, 1e1]
    RIDGE_STYLE_INDEX = {
        1e-3: 0,
        1e-1: 2,  # default / central setting emphasized
        1e1: 1,
    }
    
    for val in RIDGE_VALUES:
        part = d[
            (d.optimizer == "er")
            & np.isclose(d.ridge_lambda.astype(float), val)
        ]
        if part.empty:
            continue
    
        _plot_fig3_variant(
            ax_c,
            part,
            label=rf"$\lambda_{{\rm rel}}$={val:g}",
            style=FIG3_VARIANT_STYLES[RIDGE_STYLE_INDEX[val]],
        )
    
    _plot_fig3_sgd_reference(ax_c, d, label="SGD")
    
    ax_c.set(
        title="(c) Relative ridge",
        xlabel="Iteration",
    )
    _set_positive_step_symlog(ax_c, mark_initial=True)
    ax_c.set_ylim(0, 1.02)
    ax_c.axhline(0.1, c="k", ls=":", lw=GUIDE_LINEWIDTH, zorder=0)
    ax_c.axhline(1.0, c="k", ls=":", lw=GUIDE_LINEWIDTH, zorder=0)
    ax_c.grid(alpha=.15)
    ax_c.legend(loc="lower right", ncol=1, frameon=True)
    
    # Reserve a left band for the shared ylabel.  The label stays fully inside
    # the fixed canvas; the extra band also leaves a visible gap from y ticks.
    fig.supylabel("Test accuracy", x=0.020, y=0.52)
    fig.tight_layout(rect=(0.030, 0.0, 1.0, 1.0), pad=0.20, w_pad=0.55, h_pad=0.35)
    save_figure(fig, "Fig3_abc")
    


def plot_stability(results, output, checksums):
    def panel_runs(panel, completed_only=True):
        return read_panel_runs(results, panel, completed_only, checksums)
    def long_eval(panel, completed_only=True):
        return read_long_eval(results, panel, completed_only, checksums)
    def load_er_layers(run_dir):
        return pd.read_csv(Path(run_dir)/'er_layers.csv')
    def save_figure(fig, ignored_name):
        _save(fig, output, 'FigA3')
    style.publication_style()
    plt.rcParams["figure.dpi"]=120
    # 7. Figure 4: all stability subplots in one figure
    #
    # Time-dependent panels show the raw logged iterations on a positive-only symlog x-axis.
    
    runs = _exclude_plot_optimizers(panel_runs("fig4"))
    # Fig.4 is intentionally a single representative mechanistic diagnostic:
    # MNIST, B=1024, seed 0. Other datasets are covered statistically by
    # Table 1 rather than by repeating the expensive SVD diagnostics.
    if not runs.empty:
        runs = runs[runs.dataset.astype(str) == "mnist"].copy()
    
    parts = []
    for _, meta in runs.iterrows():
        q = load_er_layers(meta.run_dir)
        q["run_id"] = meta.run_id
        q["seed"] = meta.seed
        parts.append(q)
    
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not d.empty:
        d = d[d["step"].astype(float) >= 0].copy()
    
    # Fig.4 uses one explicit sparse diagnostic schedule:
    #   1..9, 10..90 by 10, 100.. by 100, plus the exact final step.
    # Step 0 is the pre-update initialization in eval.csv; layer-solve/update
    # diagnostics are defined from the first ER update onward.
    _diag_steps = sorted(
        d["step"].dropna().astype(int).unique()
    ) if not d.empty else []
    
    if runs.empty:
        raise RuntimeError("No Fig.4 run found.")
    
    _fig4_terminal = int(runs.iloc[0].max_steps) if "max_steps" in runs.columns else 3000
    _expected_diag_steps = (
        list(range(1, min(9, _fig4_terminal) + 1))
        + list(range(10, min(90, _fig4_terminal) + 1, 10))
        + list(range(100, _fig4_terminal + 1, 100))
    )
    if _fig4_terminal > 0 and _fig4_terminal not in _expected_diag_steps:
        _expected_diag_steps.append(_fig4_terminal)
    _expected_diag_steps = sorted(set(_expected_diag_steps))
    
    _missing_diag = [x for x in _expected_diag_steps if x not in _diag_steps]
    if _missing_diag:
        raise RuntimeError(
            "Fig.4 diagnostic history does not match the sparse schedule. "
            f"Missing steps={_missing_diag[:20]}; found={_diag_steps[:30]}. "
            "Apply fig4_sparse_diagnostics.patch and rerun only "
            "sweeps/fig4_stability.json."
        )
    
    fig, axes = plt.subplots(
        2,
        3,
        figsize=FIG4_SIZE,
        gridspec_kw={"height_ratios":[1.0, .95]},
    )
    
    ax_a, ax_b, ax_c = axes[0]
    ax_d1, ax_d2, ax_d3 = axes[1]
    
    # ------------------------------------------------------------
    # (a) ridge-effective condition
    # ------------------------------------------------------------
    svd = d.dropna(subset=["ridge_effective_condition"]).copy()
    svd["log10_eff_cond"] = np.log10(
        svd["ridge_effective_condition"].clip(lower=1)
    )
    
    for layer in sorted(svd.layer.unique()):
        part = svd[svd.layer==layer]
        st = mean_se(part, "step", "log10_eff_cond")
        ax_a.fill_between(
            st.step,
            st["mean"]-st.se,
            st["mean"]+st.se,
            alpha=.12,
        )
        ax_a.plot(
            st.step,
            st["mean"],
            lw=MAIN_LINEWIDTH,
            marker="o",
            ms=MARKER_SIZE,
            label=f"l{int(layer)}",
        )
    
    ax_a.set(
        title="(a) Ridge-effective condition",
        xlabel="Iteration",
        ylabel=r"$\log_{10}\,\kappa_\lambda$",
    )
    
    # ------------------------------------------------------------
    # (b) local solve residual
    # ------------------------------------------------------------
    for layer in sorted(d.layer.unique()):
        part = d[d.layer==layer]
        st = mean_se(part, "step", "weight_solve_residual")
        ax_b.fill_between(
            st.step,
            st["mean"]-st.se,
            st["mean"]+st.se,
            alpha=.12,
        )
        ax_b.plot(
            st.step,
            st["mean"],
            lw=MAIN_LINEWIDTH,
            marker="o",
            ms=MARKER_SIZE,
            label=f"l{int(layer)}",
        )
    
    ax_b.set(
        title="(b) Local solve residual",
        xlabel="Iteration",
        ylabel="Relative residual",
    )
    ax_b.set_yscale("log")
    
    # ------------------------------------------------------------
    # (c) actual applied relative update
    # ------------------------------------------------------------
    for layer in sorted(d.layer.unique()):
        part = d[d.layer==layer]
        st = mean_se(part, "step", "relative_update_norm_applied")
        ax_c.fill_between(
            st.step,
            st["mean"]-st.se,
            st["mean"]+st.se,
            alpha=.12,
        )
        ax_c.plot(
            st.step,
            st["mean"],
            lw=MAIN_LINEWIDTH,
            marker="o",
            ms=MARKER_SIZE,
            label=rf"$\ell = {int(layer)}$",
        )
    
    ax_c.set(
        title="(c) Applied relative update",
        xlabel="Iteration",
        ylabel=r"$||\Delta W||_F/||W||_F$",
    )
    ax_c.set_yscale("log")
    
    for ax in [ax_a, ax_b, ax_c]:
        _set_positive_step_symlog(ax, mark_initial=False)
        ax.grid(alpha=.15)
    
    ax_c.legend(frameon=True)
    
    # ------------------------------------------------------------
    # (d) normalized singular spectra: initial / middle / final
    # Snapshot positions are selected in raw optimization-step coordinates.
    # ------------------------------------------------------------
    import json as _json
    
    if svd.empty:
        raise RuntimeError("Fig.4 has no SVD diagnostic rows.")
    
    min_step = float(svd.step.min())
    max_step = float(svd.step.max())
    targets = [
        min_step,
        min_step + 0.5 * (max_step - min_step),
        max_step,
    ]
    titles = [
        "(d1) Initial",
        "(d2) Middle",
        "(d3) Final",
    ]
    
    for ax, target, title in zip(
        [ax_d1, ax_d2, ax_d3],
        targets,
        titles,
    ):
        for layer in sorted(svd.layer.unique()):
            candidates = svd[svd.layer==layer].copy()
            spectra = []
    
            for rid, g in candidates.groupby("run_id"):
                row = g.iloc[
                    np.argmin(
                        np.abs(g.step.to_numpy(dtype=float) - target)
                    )
                ]
                vals = np.array(
                    _json.loads(row.singular_values_json),
                    dtype=float,
                )
                vals = vals / max(vals[0], 1e-300)
                spectra.append(vals)
    
            if not spectra:
                continue
    
            n = min(map(len, spectra))
            arr = np.stack([x[:n] for x in spectra])
            mean = arr.mean(0)
            se = (
                arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
                if arr.shape[0] > 1
                else np.zeros(n)
            )
            x = np.arange(1, n+1)
    
            ax.fill_between(
                x,
                np.maximum(mean-se, 1e-16),
                mean+se,
                alpha=.12,
            )
            ax.plot(
                x,
                mean,
                lw=MAIN_LINEWIDTH,
                label=rf"$\ell = {int(layer)}$",
            )
    
        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.05)
        ax.set_title(title)
        ax.set_xlabel("Singular-value index")
        ax.grid(alpha=.15)
    
    ax_d1.set_ylabel(r"Normalized singular value $\sigma_i/\sigma_1$")
    #ax_d3.legend(frameon=False)
    
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0), pad=0.25, w_pad=0.65, h_pad=0.55)
    save_figure(fig, "Fig4_abcd_stability")
    


