# Paper reproduction validation

The paper uses the completed individual-run campaign. `validation/paper_baseline.json` freezes that version,
its paper-asset hashes and its regression-reference identity. Exact package
versions and runtime context are in `validation/environment.json`.

## Completed campaign

The campaign completed on 2026-11-08 at 23:54 KST. All nine sweep processes
finished; explicitly configured stress-test failures were retained as outcomes.

| Scope | Unique trials | Completed | Expected numerical failures |
| --- | ---: | ---: | ---: |
| Eight adopted paper sweeps | 1,861 | 1,523 | 338 |
| Including exploratory compute-solver sweep | 1,901 | 1,563 | 338 |

The 338 paper failures comprise 318 direct-pinv linear-algebra failures in the
solver-stability ablation and 20 nonfinite failures in the undamped mixing
stress cases. Ridge has zero failures in the 600-trial solver-stability cohort.
These recorded outcomes are not missing jobs or server execution errors.

The original full-sweep DataLoader descriptor-exhaustion problem was corrected
by isolating each trial in its own process. Completed runs/checkpoints were
preserved during recovery. The final paper data come from the completed campaign.

## Adopted baseline and verification

`validation/paper_reference.json.gz` contains settings, seed, execution core,
reference run ID, completion/failure state, final step and best/final accuracy.
The default verifier compares against this paper baseline. Its full-campaign
exact-endpoint check reports 1,523 matching completed trials and 338 expected
failures, with no missing reference trials. Reference capture is not an independent
rerun: it freezes the campaign used for the manuscript.

Same-GPU repeat experiments confirmed full trajectory repeatability for
representative Adam MSE, Adam CE and pinv runs.

Scientific code, seeds, budgets and mathematical hyperparameters were not changed
while preparing these documentation, verification and plotting tools. Figures and
tables are recomputed from the adopted individual-run metrics. Hardware/backend
changes can affect per-seed trajectories and stress-test failures, so the execution
context is part of the recorded baseline.

## Reproduction tools

`python -m paper_tools render` regenerates all six experimental figures, aggregate
CSVs and manuscript LaTeX table bodies without notebook execution, parent-directory
imports or server scripts. FigA3 is numerical
stability and FigA4 is TP/DTP, matching the current manuscript labels. The Fig1
conceptual diagram is outside the numerical reproduction pipeline.

`python -m paper_tools archive` preserves the metric/configuration tree required
for those plots and endpoint checks. It omits checkpoints, datasets, paper and
figure attachments, redundant step logs and private runtime paths. Member and
archive checksums are verified before the local archive is finalized.

## Software checks

Twelve server tests cover experiment expansion, objective semantics, synthetic
training/cache behavior, process isolation, strict/ambiguous/backend-aware
verification, failure-type checks, and the minimal archive's integrity/exclusion
rules. Full CLI rendering and all 33 active experimental table rows are checked
against the adopted manuscript statistics. See `validation/tool_checks.json` for
the final check results and source hashes.
