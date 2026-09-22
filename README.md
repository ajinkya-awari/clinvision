<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=ClinVision&fontSize=52&fontColor=fff&animation=twinkling&fontAlignY=38&desc=A%20contract%20layer%20for%20safe%20medical%20image-to-text%20pipelines&descAlignY=58&descAlign=50&descSize=16"/>

[![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ajinkya--awari%2Fclinvision-181717?logo=github)](https://github.com/ajinkya-awari/clinvision)

</div>

Medical image-to-text pipelines fail quietly: a licence you didn't check, an identifier that leaked into a fixture, a contrastive model quietly relabelled as a report generator, a metric that reads like a diagnosis. **ClinVision** is the contract layer that catches all of that *before* a real dataset, model, or GPU ever enters the picture — a set of executable validators, proven against deterministic synthetic fixtures, for the rules a real chest-radiograph pipeline has to follow.

There is no real clinical data, PHI, model weight, or generated report anywhere in this repository — including the real BLIP-2 run described below, whose aggregate metrics are published but whose underlying images, text, and checkpoint never left the training run. This is not a diagnostic, triage, or patient-facing system.

---

## What it enforces

| Contract | Catches |
|---|---|
| **Licence ledger** | Missing or mismatched image/text rights, missing approval status, missing decision rationale — fails closed on unknowns |
| **Manifest** | Identifier-like text (names, DOB, MRN/SSN patterns, emails), raw report fields, provenance that doesn't match the ledger |
| **Split** | Duplicate sample IDs, duplicate content across records, patient/group leakage across train/validation/test |
| **Model boundary** | BiomedCLIP declared as a generator, BLIP-2 requests outside the generation boundary, missing no-download guarantees |
| **Metric wording** | Diagnostic, triage, or efficacy language — even embedded inside otherwise "research-only" text |
| **Artifact policy** | Model weights, checkpoints, adapters, generated-report files, and path traversal outside the approved artifact root |
| **Restricted source** | Any reference to MIMIC-CXR/PhysioNet, anywhere in the pipeline |

25 tests exercise every rule above on both the accept and reject path — see `tests/test_contracts.py`.

---

## Design

**Flow:** verify per-item licence → build a non-identifying manifest → normalize the permitted text target → split by subject/patient grouping → offline smoke test → compare a declared baseline against a declared PEFT variant on the same split → store metrics and configuration locally with provenance.

**Model boundary:** BiomedCLIP is contrastive/retrieval-only and can never generate text. BLIP-2 is the only generation candidate, gated on implementation-time API and licence verification — it produces research text, never a clinical conclusion.

**Evaluation boundary:** BLEU, ROUGE, and BERTScore are text-overlap/semantic-similarity indicators only. They establish nothing about radiological factuality, safety, or diagnostic efficacy.

**Safe inference:** a request-contract check, not an inference call — it accepts synthetic non-identifying IDs, generation-boundary metadata, no-download mode, no input persistence, no raw model output, and research-only labelling.

```text
licence ledger + source manifest
          |
          v
safe ingest -> redaction/normalization -> grouped split
          |                              |
          v                              v
  smoke-test fixture              baseline / PEFT run
                                         |
                                         v
                         local metrics + provenance + review
```

A restricted-data lane (MIMIC-CXR) is deliberately *not* a branch of this path — it would need its own governed, credentialed, secure environment and cannot share artifacts with this repository under any circumstance.

<details>
<summary>Dataset review (PadChest / BIMCV-PadChest)</summary>

One public paired image/report candidate was reviewed at the metadata level: [PadChest / BIMCV-PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/). Its research-use agreement grants no-charge research access but prohibits redistributing or publishing dataset portions and prohibits sharing the download link outside the organization — incompatible with committing examples or artifacts to a public repository. It is not ingested here; only its licence terms were reviewed, no data was downloaded or accessed.

</details>

---

## Results — real BLIP-2 on Open-i (Indiana University)

The full pipeline was run for real twice: BLIP-2 (`Salesforce/blip2-opt-2.7b`) zero-shot, then LoRA fine-tuned, on the [Open-i / Indiana University chest X-ray collection](https://openi.nlm.nih.gov/) (200 studies, 160 train / 40 held-out eval, patient-grouped split, seed 13, single Kaggle T4). The second run pins the exact model revision and every dependency version, and records full provenance — see [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) for both runs' exact config, hashes, and a short explanation of why their numbers differ slightly.

| | BLEU (sacrebleu, 0–100) | ROUGE-L (0–1) | BERTScore F1 (0–1) |
|---|---:|---:|---:|
| Zero-shot baseline (n=40) | 0.017 | 0.094 | 0.836 |
| LoRA fine-tuned (r=4, 400 steps, n=40) | 5.401 | 0.162 | 0.866 |
| **Δ (fine-tuned − baseline)** | **+5.38** | **+0.068** | **+0.030** |

These are single point values on one held-out eval set (n=40), not mean±std across multiple runs/seeds — figures above are from the pinned reproducibility run ([`results/real/iu_openi_blip2_results_v2_pinned.json`](results/real/iu_openi_blip2_results_v2_pinned.json); the original unpinned run's slightly different numbers are in [`results/real/iu_openi_blip2_results.json`](results/real/iu_openi_blip2_results.json)). A general-purpose captioning model has essentially no vocabulary overlap with radiology report language out of the box (BLEU ≈ 0); a short LoRA fine-tune on 160 examples measurably shifts it toward that vocabulary. This is a small-sample, single-seed pilot — not a claim of diagnostic quality or clinical usefulness.

**Dataset attribution:** [Open-i / Indiana University Chest X-ray Collection](https://openi.nlm.nih.gov/), National Library of Medicine, accessed via the Kaggle mirror [`raddar/chest-xrays-indiana-university`](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university), licensed [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). No changes to the dataset itself were distributed.

**What's published vs. not, and why:** the NoDerivatives term means only the training/eval code (`scripts/train_blip2_iu_openi.py`) and these aggregate metrics are published here — not the dataset itself. No image, no report text, no generated caption, and no model checkpoint from this run is included or was ever saved — reproducing it requires downloading the dataset yourself under its own licence.

---

## Quick start

```bash
git clone https://github.com/ajinkya-awari/clinvision
cd clinvision
pip install -e .

pytest tests -q
python -m compileall -q src tests
```

No external dependencies are required — every test runs against synthetic fixtures only.

<details>
<summary>Reproducing the contract suite on Kaggle (CPU, no internet)</summary>

`notebooks/kaggle_clinvision.ipynb` runs the same synthetic contract suite in a CPU-only, no-internet Kaggle kernel — useful as a clean-room check that the tests pass with zero external dependencies. It installs nothing, downloads nothing, and never touches a real dataset.

</details>

<details>
<summary>Reproducing the real BLIP-2 results (requires the dataset + a GPU)</summary>

```bash
pip install transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 \
            rouge_score==0.1.2 bert_score==0.3.13 sacrebleu==2.4.3 \
            sentencepiece==0.2.0 pillow==10.4.0
python scripts/train_blip2_iu_openi.py \
  --data-root /path/to/chest-xrays-indiana-university \
  --model-revision 59a1ef6c1e5117b3f65523d1c6066825bcf315e3
```

`--data-root` must point to a local copy of `raddar/chest-xrays-indiana-university` (e.g. attached as a Kaggle notebook input, or downloaded yourself, under its CC BY-NC-ND licence — this repo does not include or redistribute it). The script re-derives the exact same 200-study, seed-13, patient-grouped split from the dataset's own CSVs, pins the exact model revision, and writes only aggregate metrics plus full provenance (dependency versions, runtime, config/output hashes) — see [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) for the exact reference values to compare against.

The script no longer depends on the `evaluate` wrapper package — that package calls a `huggingface_hub.HfFolder` attribute that current `huggingface_hub` releases have removed, which breaks it on a fresh install. Metrics are computed directly via `sacrebleu`, `rouge_score`, and `bert_score`.

</details>

---

## Repository

```
clinvision/
├── src/clinvision/
│   ├── contracts.py     ← licence-ledger, manifest, split, model-boundary, metric, artifact validators
│   ├── synthetic.py     ← deterministic non-identifying synthetic fixtures
│   ├── pipeline.py      ← composes the contract checks into one workflow validator
│   ├── runner.py        ← writes sanitized local evidence, no data/model/network access
│   ├── artifacts.py     ← artifact-policy exports
│   └── evaluation.py    ← non-clinical metric exports
├── tests/
│   └── test_contracts.py    ← 25 tests, accept + reject path for every rule
├── configs/
│   ├── baseline.synthetic.json
│   └── peft.synthetic.json
├── notebooks/
│   └── kaggle_clinvision.ipynb   ← CPU/no-internet contract-suite reproduction
├── scripts/
│   └── train_blip2_iu_openi.py  ← real BLIP-2 baseline + LoRA fine-tune (needs dataset + GPU)
├── results/
│   ├── contract-evidence.json                    ← sanitized synthetic evidence output
│   └── real/
│       ├── iu_openi_blip2_results.json           ← run 1 (unpinned deps)
│       └── iu_openi_blip2_results_v2_pinned.json ← run 2 (pinned deps + model revision, canonical)
├── EVIDENCE_LEDGER.md   ← exact config/version/hash provenance for both real runs
├── RELEASE_GATE.md      ← what's published, what's excluded, and why
└── pyproject.toml
```

---

## Ten non-negotiable rules

1. Every paired record needs separate, evidenced image and text licence rights — unknown or mismatched rights fail closed.
2. MIMIC-CXR is out of scope entirely — never referenced, routed, or derived from anywhere in this repository.
3. BiomedCLIP is contrastive/retrieval-only — it can never be declared a report generator.
4. BLIP-2 is a generation boundary only, gated on implementation-time API/licence verification — no silent model loading in tests.
5. Identifier-like text (names, DOB, MRN/SSN patterns, emails) and raw report fields are rejected before a record can enter a manifest.
6. Duplicate content, duplicate IDs, and patient/group leakage across train/validation/test are rejected.
7. Text metrics must be labelled non-clinical text-comparison indicators — diagnostic or efficacy language is rejected even inside "research-only" text.
8. Model weights, checkpoints, adapters, and generated-report files are blocked from the artifact root by default; path traversal is rejected.
9. Baseline and PEFT configs must share the same seed and split before any comparison is valid.
10. Every contract runs against deterministic synthetic fixtures — proving the logic requires no real dataset access, model download, or GPU.

---

## License

Released under the [MIT License](LICENSE). The license covers this project's source, tests, and documentation only — it grants no rights to any real dataset or model weights referenced but not included here.

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>
