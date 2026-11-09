# Reproduction release status

Updated: 2026-11-09

## Completed

- Minimal training and sweep CLI with JSON experiment configurations; each trial executes in an isolated process.
- The eight paper sweeps finished: 1,861 unique trials, comprising 1,523 completed trials and 338 expected numerical failures in explicitly configured stress cases.
- Manuscript tables, numerical statements and experimental figures use the individual-run paper campaign.
- Standalone `paper_tools` CLI regenerates six experimental figures and the manuscript table statistics without notebooks or another checkout.
- The frozen paper reference, runtime environment and source checksums define one reproduction baseline.
- MIT license, notice and software citation metadata are included.
- A minimal local archive is retained on the workstation and validation server. It contains configurations, metrics, compact status records, runtime metadata and checksums. Checkpoints, datasets, paper PDFs and generated figures are excluded. The archive and all campaign outputs are Git-ignored and are not release attachments.

## Validation

- Server software tests: 12 passed, most recently checked on 2026-11-09.
- Exact endpoint regression: all 1,523 completed trials match; all 338 expected failures match; no missing reference trials.
- Six regenerated experimental figures match the manuscript reference images pixel for pixel; all 33 active experimental table rows were checked.
- The minimal archive was restored in a temporary directory: member checksums, endpoint verification, figure rendering and table statistics passed. The temporary directory was removed.

Reference capture and archive restoration validate preservation and regeneration of the paper campaign; they are not independent training replications. Timing depends on hardware, and different numerical environments can change individual trajectories.

## Public repository scope

The repository includes source code, experiment configurations, reproduction instructions, license/citation metadata and compact validation references. Full experiment outputs and paper-result attachments are not published with the code.

## Remaining release administration

Use the resulting Git commit as the code version for submission. If a permanent citation identifier is required later, archive that version and add the assigned identifier to `CITATION.cff`. No DOI or release tag has been assigned.
