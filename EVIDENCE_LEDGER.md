# Evidence Ledger — Real BLIP-2 Run on Open-i (Indiana University)

Two real runs exist for this result. **Run 2 is the canonical reproducibility run** —
every dependency and the model revision pinned exactly, full provenance captured,
hash-verified. **Run 1 is a historical compatibility run**, kept only for lineage — it
used unpinned dependencies and the now-broken `evaluate` wrapper library. Both are kept
as-is; neither is overwritten. Their metric values differ because the underlying
metric-library implementations differ (see "Why the two runs differ" below) — **the two
runs must not be averaged, combined, or compared as if they came from the same
evaluation pipeline.** All figures below are single-run aggregate results on n=40
held-out studies, not mean±std across multiple runs or seeds.

## Run 2 — pinned reproducibility run (CANONICAL)

| Field | Value |
|---|---|
| Run timestamp (UTC) | `2026-09-22T16:46:30.606918Z` |
| Dataset | `raddar/chest-xrays-indiana-university` (Open-i / Indiana University, CC BY-NC-ND 4.0) |
| Model | `Salesforce/blip2-opt-2.7b` |
| Model revision (exact HF commit) | `59a1ef6c1e5117b3f65523d1c6066825bcf315e3` |
| Seed | `13` |
| Split | 160 train / 40 held-out eval, patient-grouped, `iu-openi-v1-200` |
| LoRA config | r=4, alpha=8, dropout=0.05, target_modules=[q_proj, v_proj] |
| Training steps | 400 / 400 (3 epochs, lr=1e-4) |
| Python | `3.12.13` |
| Torch | `2.10.0+cu128` (CUDA 12.8) |
| GPU | Tesla T4 |
| numpy | `2.0.2` |
| transformers | `5.17.0` |
| peft | `0.21.0` |
| accelerate | `1.15.0` |
| Protocol config SHA-256 | `79c938c96f950105569611b39f278417e6f8fa53ee7a4f89934d0962c3a84a0d` |
| Output (results JSON) SHA-256 | `f64afd7185fbdbf8facf3f15e652fd77655b216a7185ab223e26a8fdbee5ebd5` |
| Full results file | [`results/real/iu_openi_blip2_results_v2_pinned.json`](results/real/iu_openi_blip2_results_v2_pinned.json) |

**Metric definitions:** BLEU = `sacrebleu.corpus_bleu(preds, [refs]).score` (0–100 scale) · ROUGE-L = `rouge_score.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)`, F-measure mean (0–1 scale) · BERTScore F1 = `bert_score.score(preds, refs, lang="en")`, mean across examples (0–1 scale).

| | BLEU | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|
| Zero-shot baseline (n=40) | 0.017 | 0.094 | 0.836 |
| LoRA fine-tuned (n=40) | 5.401 | 0.162 | 0.866 |
| **Difference** | **+5.384** | **+0.068** | **+0.030** |

## Run 1 — historical compatibility run (not canonical, first publication)

| Field | Value |
|---|---|
| Model | `Salesforce/blip2-opt-2.7b` (unpinned revision — resolved to whatever HF Hub served as `main` on run day) |
| Dependencies | unpinned latest `transformers`/`peft`/`accelerate` at run time; exact versions not captured |
| Metrics library | `evaluate.load("sacrebleu"/"rouge"/"bertscore")` — this wrapper now fails on a fresh install against current `huggingface_hub` (see below) |
| Full results file | [`results/real/iu_openi_blip2_results.json`](results/real/iu_openi_blip2_results.json) |

| | BLEU | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|
| Zero-shot baseline (n=40) | 0.017 | 0.092 | 0.836 |
| LoRA fine-tuned (n=40) | 6.032 | 0.140 | 0.846 |

## Why the two runs differ

Run 1 used the `evaluate` wrapper package for all three metrics; Run 2 calls `sacrebleu`, `rouge_score`, and `bert_score` directly. The direct libraries and `evaluate`'s wrappers around them use slightly different default tokenization/normalization and (for BERTScore) baseline rescaling settings, which produces the small differences above. Both are legitimate measurements of the same underlying model behavior — the qualitative conclusion (near-zero BLEU zero-shot, meaningful lift after a short LoRA fine-tune) holds in both.

## Why `evaluate` was dropped

`evaluate==0.4.3` (and 0.4.6, the latest release as of this writing) calls `huggingface_hub.HfFolder`, an attribute removed from current `huggingface_hub` releases (Run 2 resolved `huggingface_hub==1.11.0`). Installing `evaluate` alongside a current `transformers` therefore raises `AttributeError: module 'huggingface_hub.hf_api' has no attribute 'HfFolder'` at metric-load time. `scripts/train_blip2_iu_openi.py` now calls the underlying metric libraries directly and no longer depends on `evaluate`.

## Reproducing this exact run

```bash
pip install transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 \
            rouge_score==0.1.2 bert_score==0.3.13 sacrebleu==2.4.3 \
            sentencepiece==0.2.0 pillow==10.4.0
python scripts/train_blip2_iu_openi.py \
  --data-root /path/to/chest-xrays-indiana-university \
  --model-revision 59a1ef6c1e5117b3f65523d1c6066825bcf315e3
```

The script recomputes and prints its own `protocol_config_sha256` and `output_sha256` — compare them against the values above to confirm an exact match.

## What is not published

No image, report text, generated caption, raw prediction, or model checkpoint from either run is included in this repository or was ever saved to disk outside the ephemeral Kaggle session that produced it — consistent with the dataset's CC BY-NC-ND (NoDerivatives) licence term.

## Status

This result is **not clinically validated, not diagnostic, and not deployment-ready**. It is a research pilot on a small held-out sample (n=40), reported here as-is.
