# Multilayer Multiplicative Learning

Scalar reference code for *Multilayer Multiplicative Learning from
Observation-Prediction Ratios via Pseudoinverse Target Propagation*.
The full individual-run paper campaign completed on 2026-11-08.
Its configuration, metrics and runtime define the paper reproduction baseline.

Current completion and validation record: [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Install and check

Use Python 3.10-3.12 and a matching PyTorch/torchvision installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[paper,test]'
python -m pytest -q
python run.py smoke --output results/smoke
```

The `paper` extra adds Matplotlib, pandas and SciPy. Omit it for training only.
The tested paper environment is recorded in `validation/environment.json`, with
exact analysis package versions in `requirements-paper.txt`. GPU measurements
used PyTorch 2.5.1, CUDA 12.1 and an RTX 4090. Choose a PyTorch build suitable for
your platform; the package-version file does not install a GPU driver.

## Generate experiment data

Run from the repository root. CPU is the default; specify `--device cuda:0` for
a GPU. Data are downloaded by torchvision into `data/` on first use and reside
on the selected device throughout each run.

```bash
python run.py sweep --config configs/fig2_optimizer_comparison.json --dry-run
python run.py sweep --config configs/fig2_optimizer_comparison.json --device cuda:0
```

Run the remaining paper sweeps with the same command and the appropriate JSON:

| Configuration | Purpose |
| --- | --- |
| `fig2_optimizer_comparison.json` | 180 trials; three datasets, six optimizers, ten seeds |
| `mse_ce.json` | 240 paired objective-control trials |
| `tp_dtp.json` | 60 TP/DTP trials |
| `fig3a_batch_main.json` | Batch sensitivity |
| `fig3b_step_mix.json` | Mixing sensitivity, including unstable settings |
| `fig3c_ridge.json` | Ridge sensitivity |
| `fig4_stability.json` | Layerwise diagnostic trajectory |
| `table_stability_ablation.json` | 1,200 solver-stress trials, including numerical failures |

`compute_solver.json` is exploratory and is not part of the adopted paper
baseline. `--limit 1` runs one full-budget trial; it does not shorten its budget.
`smoke` runs twelve tiny synthetic method/objective checks without downloading
a dataset. Smoke results are not scientific measurements.

Each JSON defines `base`, `cases`, optional coupled `setups`, and a Cartesian
`grid`. Effective settings and seeds are saved in `results/manifests/` and each
run directory. Trials execute sequentially in separate processes. Completed
trials are cached; interrupted original-core trials restart, while objective
and TP/DTP trials resume from epoch-boundary checkpoints. Never load untrusted
checkpoints. No server dispatcher or notebook is needed.

## Recreate the paper figures and tables

After all eight paper sweeps finish:

```bash
python -m paper_tools tables --results results --output results/paper
python -m paper_tools render --results results --output results/paper
```

`tables` writes per-seed and aggregate CSVs, paired tests, and LaTeX table bodies.
`render` writes those tables plus vector PDF and PNG versions of all six
experimental figures. Naming follows the current manuscript:

- `Fig2`: main optimizer comparison; `Fig3`: sensitivity.
- `FigA1`: six-method accuracy/MSE; `FigA2`: MSE versus CE.
- `FigA3`: layerwise numerical stability; `FigA4`: TP versus DTP.

Fig1 is a conceptual diagram, not a data-derived plot, and is not generated.
Mean/SE, temporal bins, paired confidence intervals and failure handling match
the adopted manuscript. The CLI preserves unstable partial trajectories where
required; it does not silently substitute survivor-only curves. Input file checksums and run provenance are saved with each rendering.

## Verify against the adopted paper baseline

```bash
python verify.py --output results
# Exact endpoint regression on the recorded runtime/data:
python verify.py --output results --tolerance-pp 0
```

The default is `validation/paper_reference.json.gz`: 1,861 unique paper trials,
including 338 expected numerical failures. It identifies the execution core and
keeps the original-core MSE runs separate from objective-control MSE runs.
Missing runs, wrong budgets, unexpected failure types, ambiguous matches and
endpoint differences cause a nonzero exit. The report is written to
`results/verification.json`. The default 0.5-percentage-point tolerance is a
screening threshold, not a statistical equivalence test.

This is a frozen regression baseline for the paper campaign, rather than an
independent rerun. See `VALIDATION.md` for the completed campaign and checks.
Timing is hardware-dependent; compare it separately from accuracy.

## Minimal local preservation

```bash
python -m paper_tools archive --results results \
  --destination results/archives/paper_scalar_minimal.tar.gz
```

The archive contains only paper manifests/configs, status and compact failure
records, metric trajectories, the one required layer-diagnostic CSV, timing
summaries, environment metadata, reference values and checksums. It excludes
checkpoints, model weights, datasets, `train_steps.csv`, full tracebacks,
exploratory runs, paper PDFs and generated figure attachments. It is stored
locally, not published as a paper-result or release attachment. Existing archive
files are never overwritten.

`data/` and `results/` remain Git-ignored. The repository contains source,
configuration and small regression-reference metadata, not the full campaign
outputs. A local minimal archive can reconstruct the result tree for verification
and replotting without training or checkpoint loading.

## License and citation

The software is available under the MIT license; see `LICENSE` and `NOTICE`.
There are no additional scientific-use restrictions. Citation is requested for
attribution, not as a license condition. `CITATION.cff` records the software and
manuscript authors. Third-party dependencies and datasets retain their own terms.
