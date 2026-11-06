# Validation

Validated on 2026-11-06 using Python 3.12.15, PyTorch 2.5.1,
torchvision 0.20.1, NumPy 2.5.3, pytest 9.1.1, macOS CPU.
Exact tested dependencies are recorded in `requirements-tested.txt`.

`python -m pytest -q`: 4 passed. Checks cover:

- expansion and unique identities for all nine supplied experiment JSON files;
- CE evaluation under deliberately saturated tanh outputs;
- seven original optimizer/solver methods with synthetic resident datasets,
  cached reruns, and CSV export;
- all twelve control method/objective combinations, checkpoint writing and
  completed-run cache reuse, using six synthetic updates each.

No real datasets were downloaded. Full-budget accuracy, GPU behavior, historical
result equivalence, and exact publication figure regeneration were not validated.
Shared source/config SHA-256 digests are in `source_checksums.json`.

The execution sandbox blocked OpenMP shared-memory creation; tests were run
outside that sandbox with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

## Shannon GPU validation (2026-11-06)

Five tests passed on the Shannon server with PyTorch 2.5.1 and torchvision
0.20.1. Twelve synthetic CUDA method/objective combinations completed on an
RTX 4090. TP/dTP uses the original sparse/logarithmic evaluation schedule.

A fresh MNIST ER-0 seed-0 run completed all 25,000 optimization steps:

| Metric | Paper reference | Fresh run |
| --- | --- | --- |
| Final test accuracy | 88.91% | 88.91% |
| Best test accuracy | 89.58% | 89.58% |

All 538 train/test evaluation rows have matching step/split indices. Across
those rows, the maximum accuracy difference is 0.01 percentage points and the
maximum MSE difference is 1.27e-7. Endpoint agreement does not imply bitwise
trajectory agreement.

The remaining full sweeps are running separately. The endpoint checker initially
matched this one completed reference and marked 1,860 references missing; it did
not declare the paper reproduced. See live server logs for current progress.

## Long-sweep recovery

The first Fig.2 sweep stopped after 61 completed trials because sequential
DataLoader construction in one process exhausted file descriptors (`Too many
open files`, followed by `Pin memory thread exited unexpectedly`). This was an
execution-resource failure during dataset preload, not a numerical ER failure.

The CLI now starts each trial in a separate subprocess, releasing DataLoader
workers, file descriptors and CUDA resources on exit. Training, data ordering,
hyperparameters and scientific source identities are unchanged. Six server tests
passed, including reuse of a completed cached trial through the isolated process.
Completed runs and control checkpoints were preserved for restart.
