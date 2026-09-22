"""Bounded local synthetic runner.

The runner validates in-memory synthetic contracts and writes a sanitized JSON
evidence record. It does not open images, read reports, import model libraries,
download files, call services, or run inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .contracts import (
    ArtifactManifestEntry,
    BaselineConfig,
    PEFTConfig,
    validate_artifact_manifest,
)
from .pipeline import validate_synthetic_workflow
from .synthetic import build_synthetic_records


@dataclass(frozen=True)
class SyntheticRunResult:
    ok: bool
    path: Path
    issue_count: int


def run_synthetic_contract_pass(output_path: str | Path) -> SyntheticRunResult:
    path = Path(output_path)
    baseline = BaselineConfig(
        name="synthetic-blip2-baseline-contract",
        model_family="BLIP-2",
        task="generation",
        seed=13,
        split_version="synthetic-v1",
        no_download=True,
        generation={"max_new_tokens": 32, "num_beams": 1},
    )
    peft = PEFTConfig(
        name="synthetic-lora-peft-contract",
        base_model_config=baseline.name,
        method="lora",
        seed=baseline.seed,
        split_version=baseline.split_version,
        no_download=True,
        publish_adapter=False,
        parameters={"rank": 4, "alpha": 8, "dropout": 0.05},
    )
    records = build_synthetic_records()
    workflow_result = validate_synthetic_workflow(records, baseline, peft)
    evidence = {
        "status": "LOCAL-SYNTHETIC-IMPLEMENTATION-COMPLETE WITH LIMITATIONS",
        "dataset_access": "none",
        "runtime": "contract-validation-only",
        "downloads": "none",
        "external_services": "none",
        "gpu": "none",
        "models": "not_loaded",
        "sample_ids": [record.stable_id for record in records.manifest],
        "split_version": baseline.split_version,
        "seed": baseline.seed,
        "issues": [
            {"code": issue.code.value, "field": issue.field, "record_id": issue.record_id}
            for issue in workflow_result.issues
        ],
    }
    encoded = json.dumps(evidence, indent=2, sort_keys=True)
    checksum = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    manifest = ArtifactManifestEntry(
        path="artifacts/synthetic/contract-evidence.json",
        artifact_type="synthetic-evidence",
        provenance_url="synthetic://local/runner",
        checksum_algorithm="sha256",
        checksum_value=checksum,
    )
    manifest_result = validate_artifact_manifest([manifest])
    all_issues = (*workflow_result.issues, *manifest_result.issues)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8")
    return SyntheticRunResult(ok=not all_issues, path=path, issue_count=len(all_issues))
