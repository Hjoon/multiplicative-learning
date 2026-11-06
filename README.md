# Multilayer Multiplicative Learning

Minimal, sequential code for generating experimental data. This directory is
self-contained: copy it as the root of the repository you share. No parent
checkout, historical results, notebook, server account, or cluster is required.

Use Python 3.10–3.12 and install a matching PyTorch/torchvision pair:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest -q
python run.py smoke --output results/smoke
python run.py sweep --config configs/fig2_optimizer_comparison.json --dry-run
python run.py sweep --config configs/fig2_optimizer_comparison.json --device cuda:0
python run.py sweep --config configs/mse_ce.json --device cuda:0
python run.py sweep --config configs/tp_dtp.json --device cuda:0
python run.py export
```

Run commands from this directory. CPU is the default; pass `--device cuda:0`
for a GPU. `--limit 1` runs the first complete trial, with its full scientific
budget. `--dry-run` validates/expands configurations without downloading data or
creating output files. `smoke` uses tiny synthetic data and 6 updates for each
of 12 method/objective combinations; it is not a scientific experiment.

## Experiment configurations

| JSON in `configs/` | Purpose |
| --- | --- |
| `fig2_optimizer_comparison.json` | MNIST, Fashion-MNIST, CIFAR-10; ER-0/ER-1 and four gradient methods; 10 seeds; 180 runs |
| `mse_ce.json` | Paired MSE/CE gradient controls on all three datasets; 240 runs |
| `tp_dtp.json` | Absolute TP and difference-corrected TP; 60 runs |
| `fig3a_batch_main.json` | Batch-size sensitivity |
| `fig3b_step_mix.json` | ER mixing sensitivity |
| `fig3c_ridge.json` | Ridge sensitivity |
| `fig4_stability.json` | Stability trajectories and diagnostics |
| `table_stability_ablation.json` | Direct pseudoinverse versus ridge; includes failed runs |
| `compute_solver.json` | Exact versus approximate local solver comparison |

Each JSON preserves the source experiment settings. Effective defaults are
expanded into the manifest and each run's config, including architecture,
learning rate, batch size, seed, and step budget. JSON `grid` entries form a
Cartesian product; `setups` groups coupled parameters. No parameter tuning is
performed automatically. Figure filenames are historical experiment identifiers;
consult the actual dataset/config values rather than older manuscript captions.

## Outputs and execution

Datasets are downloaded by torchvision into `data/` on the first real run.
Entire flattened train/test sets reside on the selected device. Full sweeps are
expensive; dataset memory plus the ER solve must fit on that device.

- `results/manifests/<config>.json`: resolved configurations and source-based IDs.
- `results/runs/<id>/`: effective config, status, trajectory, and diagnostics.
- `results/accuracy.csv`: long-form accuracy data (`run_id`, dataset, optimizer,
  seed, loss, status, step, split, accuracy).
- `results/manifests/<config>-failures.json`: caught failures for configurations
  that explicitly request `continue_on_error` (numerical stability experiments).

Per-run original MSE experiments write `eval.csv`, `train_steps.csv`, optional
`er_layers.csv`, `summary.json`, `config.json`, and `identity.json`. Objective
controls and TP/dTP write `record.json`, `metrics.json`, and `checkpoint.pt`.
Completed runs are cached. Interrupted original experiments restart; controls
resume at saved epoch boundaries. Do not load untrusted checkpoint files.
Smoke artifacts are omitted from the CSV export. Failed original runs may have
partial trajectories: always filter on status for final-accuracy comparisons,
and count failures separately for stability analysis.

The two small execution cores are retained to preserve the original diagnostic
implementation and the later objective/target controls. ER mathematical code is
copied from the source implementations. CE trains and evaluates pre-tanh logits;
MSE and ER use tanh outputs. Seeds and model initialization follow the original
data-preload ordering. Sequential execution replaces server packing; floating
point differences from packed GPU execution are possible. Wall times from the
two runners have different logging/checkpoint overheads and should not be pooled.

## Figures and reproducibility

Figure notebooks are not required for running these experiments and are omitted.
This release exports the raw trajectories needed for independent plots. A small
plot script or cleaned notebook can be added later to reproduce exact publication
styling and statistical summaries. This code-only package does not yet reproduce
the publication PDFs automatically.

Software smoke tests do not establish numerical reproduction of the paper.
Full-budget experiments, comparison against the final paper tables/figures, and
GPU validation remain necessary. No historical result data or dataset is bundled.

## Comparing against paper reference values

`validation/paper_reference.json` contains anonymized per-seed endpoints from
the results used for the current manuscript: 180 Fig.2 trials, 240 MSE/CE
controls, 60 TP/dTP trials, and 1,381 sensitivity/stability trials (including
316 recorded numerical failures). These are reference measurements, not fresh
reproduction outputs. Original configurations are retained for matching.

```bash
python verify.py --output results --tolerance-pp 0.5
```

The checker writes `results/verification.json` and exits nonzero while any
reference is missing, pending, has a different failure status/step budget, or
exceeds the chosen endpoint tolerance. It reports best and final test accuracy
deltas separately. The default 0.5 percentage-point tolerance is a screening
threshold, not a statistical equivalence test; use zero for exact endpoint
agreement. Every matching execution backend is reported. Solver timing,
analytical memory estimates, paired significance tables and target-crossing
statistics require separate analysis and are not certified by endpoint checks.
The compute-solver sweep has no bundled paper reference and is exploratory.
