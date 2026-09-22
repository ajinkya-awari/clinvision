# Release Gate — ClinVision Public Repository

## Status: RELEASED, COMPLETE WITH LIMITATIONS

This repository is public and its contents are approved for release under the terms
below. It is not a diagnostic, triage, or patient-facing system, and no such claim is
made anywhere in it.

## What was explicitly approved for publication

- Source code, tests, and the synthetic contract-validation layer (25/25 tests passing).
- The training/evaluation script for the real BLIP-2 run (`scripts/train_blip2_iu_openi.py`).
- Aggregate, non-identifying metrics from two real runs against the Open-i (Indiana
  University) chest X-ray collection — see [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md).
- Dataset name, licence terms, sample counts, split description, model identifier and
  exact revision, exact dependency versions, and run hashes.

## What was explicitly excluded, and why

- Any image, report text, generated caption, or raw model prediction from the real
  dataset — the dataset's CC BY-NC-ND 4.0 licence prohibits redistributing derivatives
  of the licensed images/reports, and generated captions are plausibly closer to a
  derivative of the source images than original code or aggregate statistics are.
- Any trained model checkpoint — never saved to disk in either run, for the same reason.
- Any internal AI-assisted-development control files, prompts, or session logs — this
  repository's history was deliberately curated to contain only the public-facing
  research artifact.

## Known limitations (do not claim beyond these)

- Single-seed, small-sample (n=40 eval) pilot — not a statistically powered benchmark.
- No claim of diagnostic quality, clinical usefulness, or generalization beyond this
  exact dataset/split/model configuration.
- Two runs (see `EVIDENCE_LEDGER.md`) produce slightly different metric values due to
  differing metric-library implementations (`evaluate` wrapper vs. direct libraries) —
  both are legitimate; neither should be quoted without noting which run it's from.

## Licence position

CC BY-NC-ND 4.0 restricts distributing *Adapted Material* of the dataset's images and
reports. Publishing original code, the dataset's name, sample counts, split
description, and aggregate scalar metrics is standard ML-research practice and is not
"Adapted Material" of the licensed images/text under the Creative Commons legal code's
definition. This is a defensible compliance position consistent with universal
practice for reporting benchmark results on restricted datasets — it is not a
substitute for formal legal review.
