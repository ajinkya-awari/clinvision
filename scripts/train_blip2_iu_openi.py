"""Real BLIP-2 zero-shot baseline + LoRA fine-tune on the Open-i (Indiana University)
chest X-ray collection.

This script does not embed, ship, or redistribute any dataset content. It expects
the Kaggle dataset `raddar/chest-xrays-indiana-university` (CC BY-NC-ND 4.0) to be
available locally (e.g. attached as a Kaggle notebook input, or downloaded yourself
under that licence) and reads `indiana_reports.csv` / `indiana_projections.csv` plus
`images/images_normalized/` from --data-root at runtime.

Selection is fully deterministic given the same data root and --seed: 200 studies
(one frontal image + findings text per patient study), split 160 train / 40 held-out
eval by patient study ID, so no patient appears in both splits.

Metrics (BLEU via sacrebleu, ROUGE-L via rouge_score, BERTScore via bert_score) are
computed by calling those libraries directly rather than through the `evaluate`
wrapper package, which as of late 2025 depends on a `huggingface_hub.HfFolder`
attribute removed from current `huggingface_hub` releases and will raise
`AttributeError` on a fresh install.

Only aggregate BLEU / ROUGE-L / BERTScore, plus exact model/dependency/runtime
provenance, are ever written out. No image, no report text, no generated caption,
no raw prediction, and no model checkpoint is saved anywhere, consistent with the
dataset's NoDerivatives licence term.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Exact revision verified working as of 2026-09-22 (see results/real/EVIDENCE_LEDGER.md).
DEFAULT_MODEL_REVISION = "59a1ef6c1e5117b3f65523d1c6066825bcf315e3"


def build_manifest(data_root: Path, seed: int, n_total: int, n_train: int) -> list[dict]:
    with open(data_root / "indiana_reports.csv", newline="", encoding="utf-8") as f:
        reports = {row["uid"]: row for row in csv.DictReader(f)}
    with open(data_root / "indiana_projections.csv", newline="", encoding="utf-8") as f:
        proj_rows = list(csv.DictReader(f))

    frontal_by_uid: dict[str, str] = {}
    for row in proj_rows:
        if row["projection"] == "Frontal" and row["uid"] not in frontal_by_uid:
            frontal_by_uid[row["uid"]] = row["filename"]

    candidates = [
        uid
        for uid, rep in reports.items()
        if uid in frontal_by_uid and len((rep.get("findings") or "").strip()) > 20
    ]
    candidates.sort(key=int)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = sorted(candidates[:n_total], key=int)
    train_uids = set(sorted(selected[:n_train], key=int))

    manifest = []
    for uid in selected:
        stable_id = "cv-real-" + hashlib.sha256(f"iu-openi-{uid}".encode()).hexdigest()[:16]
        manifest.append(
            {
                "uid_hash": stable_id,
                "filename": frontal_by_uid[uid],
                "findings": reports[uid]["findings"].strip(),
                "split": "train" if uid in train_uids else "validation",
            }
        )
    return manifest


def collect_runtime_info() -> dict:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": __import__("numpy").__version__,
    }


def collect_dependency_versions() -> dict:
    import accelerate
    import peft
    import transformers

    return {
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
    }


@torch.no_grad()
def generate_caption(processor, model, device, image: Image.Image, max_new_tokens: int = 48) -> str:
    inputs = processor(images=image, text="a chest x-ray showing", return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def run_scoring(processor, model, device, img_root: Path, records: list[dict], rouge_scorer_obj, bertscore_fn) -> dict:
    import sacrebleu

    preds, refs = [], []
    t0 = time.time()
    for rec in records:
        image = Image.open(img_root / rec["filename"]).convert("RGB")
        preds.append(generate_caption(processor, model, device, image))
        refs.append(rec["findings"])

    bleu_score = sacrebleu.corpus_bleu(preds, [refs]).score
    rouge_scores = [rouge_scorer_obj.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]
    rougeL_score = statistics.mean(rouge_scores)
    _, _, bert_f1 = bertscore_fn(preds, refs, lang="en", verbose=False)
    bert_f1_list = bert_f1.tolist()

    return {
        "n": len(records),
        "bleu": bleu_score,
        "rougeL": rougeL_score,
        "bertscore_f1_mean": statistics.mean(bert_f1_list),
        "bertscore_f1_std": statistics.pstdev(bert_f1_list) if len(bert_f1_list) > 1 else 0.0,
        "elapsed_sec": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with indiana_*.csv and images/images_normalized/")
    parser.add_argument("--output", type=Path, default=Path("results/real/iu_openi_blip2_results.json"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-total", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=160)
    parser.add_argument("--model-id", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION, help="Exact HF Hub commit SHA to pin. Pass '' to use the current main branch (not recommended for reproducibility).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = build_manifest(args.data_root, args.seed, args.n_total, args.n_train)
    train_recs = [r for r in manifest if r["split"] == "train"]
    eval_recs = [r for r in manifest if r["split"] == "validation"]
    img_root = args.data_root / "images" / "images_normalized"

    from bert_score import score as bertscore_fn
    from peft import LoraConfig, get_peft_model
    from rouge_score import rouge_scorer
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from torch.optim import AdamW

    revision = args.model_revision or None
    runtime_info = collect_runtime_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(args.model_id, revision=revision)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_id, revision=revision, torch_dtype=torch.float16).to(device)

    rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    baseline_scores = run_scoring(processor, model, device, img_root, eval_recs, rouge_scorer_obj, bertscore_fn)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    dependency_versions = collect_dependency_versions()
    optimizer = AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=args.lr)
    peft_model.train()

    step = 0
    for _ in range(args.epochs):
        order = list(range(len(train_recs)))
        random.shuffle(order)
        for idx in order:
            if step >= args.max_steps:
                break
            rec = train_recs[idx]
            image = Image.open(img_root / rec["filename"]).convert("RGB")
            target = rec["findings"][:400]
            inputs = processor(images=image, text=target, return_tensors="pt").to(device, torch.float16)
            input_ids = inputs["input_ids"]
            outputs = peft_model(pixel_values=inputs["pixel_values"], input_ids=input_ids, labels=input_ids)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
        if step >= args.max_steps:
            break

    peft_model.train(False)
    peft_scores = run_scoring(processor, peft_model, device, img_root, eval_recs, rouge_scorer_obj, bertscore_fn)

    protocol_config = {
        "dataset": "raddar/chest-xrays-indiana-university",
        "dataset_licence": "CC BY-NC-ND 4.0",
        "model": args.model_id,
        "model_revision": revision,
        "seed": args.seed,
        "n_total": args.n_total,
        "n_train": len(train_recs),
        "n_eval": len(eval_recs),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "target_modules": ["q_proj", "v_proj"]},
        "training": {"epochs": args.epochs, "lr": args.lr, "max_steps": args.max_steps},
    }
    protocol_config_sha256 = hashlib.sha256(json.dumps(protocol_config, sort_keys=True).encode()).hexdigest()

    results = {
        "status": "IU_OPENI_REAL_BLIP2_BASELINE_PEFT_COMPLETE",
        "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_config": protocol_config,
        "protocol_config_sha256": protocol_config_sha256,
        "dataset": "raddar/chest-xrays-indiana-university (Open-i / Indiana University, CC BY-NC-ND 4.0)",
        "model": args.model_id,
        "model_revision": revision,
        "dependency_versions": dependency_versions,
        "runtime": runtime_info,
        "seed": args.seed,
        "split_version": "iu-openi-v1-200",
        "n_train": len(train_recs),
        "n_eval": len(eval_recs),
        "lora": protocol_config["lora"],
        "training": {**protocol_config["training"], "steps_run": step},
        "metric_definitions": {
            "bleu": "sacrebleu.corpus_bleu(preds, [refs]).score, 0-100 scale",
            "rougeL": "rouge_score.rouge_scorer RougeScorer([rougeL], use_stemmer=True), F-measure mean, 0-1 scale",
            "bertscore_f1": "bert_score.score(preds, refs, lang=en) F1, mean across examples, 0-1 scale",
        },
        "baseline_zero_shot": baseline_scores,
        "peft_fine_tuned": peft_scores,
        "delta_bleu": peft_scores["bleu"] - baseline_scores["bleu"],
        "delta_rougeL": peft_scores["rougeL"] - baseline_scores["rougeL"],
        "delta_bertscore_f1_mean": peft_scores["bertscore_f1_mean"] - baseline_scores["bertscore_f1_mean"],
        "no_images_saved": True,
        "no_generated_text_saved": True,
        "no_checkpoint_saved": True,
        "no_raw_predictions_saved": True,
        "metrics_are_non_clinical_text_comparison_only": True,
    }
    output_sha256 = hashlib.sha256(json.dumps(results, indent=2, sort_keys=True).encode()).hexdigest()
    results["output_sha256"] = output_sha256

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
