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

Only aggregate BLEU / ROUGE-L / BERTScore are ever written out. No image, no report
text, no generated caption, and no model checkpoint is saved anywhere, consistent
with the dataset's NoDerivatives licence term.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from PIL import Image


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


@torch.no_grad()
def generate_caption(processor, model, device, image: Image.Image, max_new_tokens: int = 48) -> str:
    inputs = processor(images=image, text="a chest x-ray showing", return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def run_scoring(processor, model, device, img_root: Path, records: list[dict], metrics: dict) -> dict:
    preds, refs = [], []
    t0 = time.time()
    for rec in records:
        image = Image.open(img_root / rec["filename"]).convert("RGB")
        preds.append(generate_caption(processor, model, device, image))
        refs.append(rec["findings"])
    bleu = metrics["bleu"].compute(predictions=preds, references=[[r] for r in refs])
    rouge = metrics["rouge"].compute(predictions=preds, references=refs)
    bert = metrics["bertscore"].compute(predictions=preds, references=refs, lang="en")
    return {
        "n": len(records),
        "bleu": bleu["score"],
        "rougeL": rouge["rougeL"],
        "bertscore_f1_mean": statistics.mean(bert["f1"]),
        "bertscore_f1_std": statistics.pstdev(bert["f1"]) if len(bert["f1"]) > 1 else 0.0,
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

    import evaluate
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(args.model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)

    metrics = {
        "bleu": evaluate.load("sacrebleu"),
        "rouge": evaluate.load("rouge"),
        "bertscore": evaluate.load("bertscore"),
    }

    baseline_scores = run_scoring(processor, model, device, img_root, eval_recs, metrics)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
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
    peft_scores = run_scoring(processor, peft_model, device, img_root, eval_recs, metrics)

    results = {
        "status": "IU_OPENI_REAL_BLIP2_BASELINE_PEFT_COMPLETE",
        "dataset": "raddar/chest-xrays-indiana-university (Open-i / Indiana University, CC BY-NC-ND 4.0)",
        "model": args.model_id,
        "seed": args.seed,
        "split_version": "iu-openi-v1-200",
        "n_train": len(train_recs),
        "n_eval": len(eval_recs),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "target_modules": ["q_proj", "v_proj"]},
        "training": {"epochs": args.epochs, "lr": args.lr, "max_steps": args.max_steps, "steps_run": step},
        "baseline_zero_shot": baseline_scores,
        "peft_fine_tuned": peft_scores,
        "delta_bleu": peft_scores["bleu"] - baseline_scores["bleu"],
        "delta_rougeL": peft_scores["rougeL"] - baseline_scores["rougeL"],
        "delta_bertscore_f1_mean": peft_scores["bertscore_f1_mean"] - baseline_scores["bertscore_f1_mean"],
        "no_images_saved": True,
        "no_generated_text_saved": True,
        "no_checkpoint_saved": True,
        "metrics_are_non_clinical_text_comparison_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
