"""Deterministic synthetic ClinVision records.

The records in this module are references and metadata only. They contain no
image bytes, report text, patient identifiers, model weights, or generated
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    DatasetProvenance,
    LicenceLedgerEntry,
    ManifestRecord,
    SplitRecord,
    stable_synthetic_id,
)


@dataclass(frozen=True)
class SyntheticRecords:
    ledger: tuple[LicenceLedgerEntry, ...]
    manifest: tuple[ManifestRecord, ...]
    split: tuple[SplitRecord, ...]


def build_synthetic_records() -> SyntheticRecords:
    sample_id = "cv-synth-ledger-001"
    source_item_id = "fixture-001"
    stable_id = stable_synthetic_id(source_item_id)
    ledger = LicenceLedgerEntry(
        sample_id=sample_id,
        source_dataset="synthetic-public-fixture",
        source_item_id=source_item_id,
        licence_name="Synthetic Test Licence",
        licence_url="https://example.invalid/licence",
        licence_evidence="Synthetic fixture created for local contract tests only.",
        allowed_uses=("research",),
        provenance_url="https://example.invalid/provenance/fixture-001",
        reviewed_by="local-synthetic-review",
        reviewed_at="2026-09-15",
        public_data_lane=True,
        image_rights_holder="synthetic-image-holder",
        text_rights_holder="synthetic-text-holder",
        image_license="Synthetic Test Licence",
        text_license="Synthetic Test Licence",
        image_license_evidence="Synthetic image reference has local test rights.",
        text_license_evidence="Synthetic text reference has local test rights.",
        derivative_and_redistribution_rights="compatible synthetic redistribution for local tests only",
        privacy_deidentification_evidence="synthetic non-identifying reference only",
        retention_rule="local synthetic evidence only",
        access_class="public-synthetic",
        checksum_algorithm="sha256",
        checksum_value="not collected - no artifact access",
        approval_status="approved",
        decision_rationale="Synthetic fixture contains only references and generic comparison text.",
    )
    provenance = DatasetProvenance(
        sample_id=ledger.sample_id,
        source_dataset=ledger.source_dataset,
        source_item_id=ledger.source_item_id,
        provenance_url=ledger.provenance_url,
        licence_name=ledger.licence_name,
        licence_url=ledger.licence_url,
        reviewed_at=ledger.reviewed_at,
    )
    manifest = ManifestRecord(
        stable_id=stable_id,
        ledger_sample_id=sample_id,
        image_ref="synthetic://chest-xray-placeholder-001",
        normalized_target="synthetic non-clinical comparison target with no identifying content",
        group_id="synthetic-group-a",
        provenance=provenance,
        metadata={"source_kind": "synthetic"},
    )
    split = SplitRecord(
        stable_id=stable_id,
        content_hash="synthetic-hash-001",
        group_id=manifest.group_id,
        split="train",
    )
    return SyntheticRecords(ledger=(ledger,), manifest=(manifest,), split=(split,))
