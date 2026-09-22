from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinvision.contracts import (
    ArtifactManifestEntry,
    ArtifactPolicy,
    BaselineConfig,
    BiomedCLIPRetrievalRequest,
    BLIP2GenerationRequest,
    DatasetProvenance,
    ErrorCode,
    ErrorTaxonomyEntry,
    EvaluationReportContract,
    LicenceLedgerEntry,
    ManifestRecord,
    MetricRecord,
    PEFTConfig,
    SafeInferenceRequest,
    SplitRecord,
    check_artifact_path,
    stable_synthetic_id,
    validate_baseline_config,
    validate_biomedclip_request,
    validate_blip2_request,
    validate_ledger,
    validate_manifest,
    validate_metrics,
    validate_peft_config,
    validate_report_contract,
    validate_artifact_manifest,
    validate_safe_inference_request,
    validate_split_contract,
    validate_train_eval_overlap,
)
from clinvision.synthetic import build_synthetic_records
from clinvision.runner import run_synthetic_contract_pass


def licence_entry(sample_id="cv-synth-ledger-001"):
    return LicenceLedgerEntry(
        sample_id=sample_id,
        source_dataset="synthetic-public-fixture",
        source_item_id="fixture-001",
        licence_name="Synthetic Test Licence",
        licence_url="https://example.invalid/licence",
        licence_evidence="Synthetic fixture created for local contract tests only.",
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
        decision_rationale="Synthetic fixture contains no image bytes or clinical text.",
        allowed_uses=("research",),
        provenance_url="https://example.invalid/provenance/fixture-001",
        reviewed_by="local-synthetic-review",
        reviewed_at="2026-08-27",
        public_data_lane=True,
    )


def provenance(sample_id="cv-synth-ledger-001"):
    return DatasetProvenance(
        sample_id=sample_id,
        source_dataset="synthetic-public-fixture",
        source_item_id="fixture-001",
        provenance_url="https://example.invalid/provenance/fixture-001",
        licence_name="Synthetic Test Licence",
        licence_url="https://example.invalid/licence",
        reviewed_at="2026-08-27",
    )


def manifest_record(text="synthetic observation text with no identifiers"):
    return ManifestRecord(
        stable_id=stable_synthetic_id("fixture-001"),
        ledger_sample_id="cv-synth-ledger-001",
        image_ref="synthetic://chest-xray-placeholder-001",
        normalized_target=text,
        group_id="synthetic-group-a",
        provenance=provenance(),
        metadata={"source_kind": "synthetic"},
    )


def test_licence_ledger_accepts_complete_public_research_entry():
    assert validate_ledger([licence_entry()]).ok


def test_licence_ledger_rejects_missing_or_incompatible_rights():
    result = validate_ledger(
        [
            LicenceLedgerEntry(
                sample_id="cv-synth-ledger-002",
                source_dataset="synthetic-public-fixture",
                source_item_id="fixture-002",
                licence_name="",
                licence_url="",
                licence_evidence="",
                image_rights_holder="",
                text_rights_holder="",
                image_license="",
                text_license="",
                image_license_evidence="",
                text_license_evidence="",
                derivative_and_redistribution_rights="",
                privacy_deidentification_evidence="",
                retention_rule="",
                access_class="",
                checksum_algorithm="",
                checksum_value="",
                approval_status="",
                decision_rationale="",
                allowed_uses=("view-only",),
                provenance_url="",
                reviewed_by="",
                reviewed_at="",
                public_data_lane=False,
            )
        ]
    )

    assert {issue.code for issue in result.issues} >= {
        ErrorCode.LICENCE_MISSING,
        ErrorCode.LICENCE_INCOMPATIBLE,
        ErrorCode.PROVENANCE_MISSING,
    }


def test_licence_ledger_blocks_restricted_sources():
    entry = licence_entry()
    restricted = LicenceLedgerEntry(
        **{**entry.__dict__, "source_dataset": "MIMIC-CXR", "restricted_source": True}
    )

    result = validate_ledger([restricted])

    assert ErrorCode.RESTRICTED_SOURCE in {issue.code for issue in result.issues}


def test_licence_ledger_rejects_incompatible_image_text_licences():
    entry = LicenceLedgerEntry(
        **{
            **licence_entry().__dict__,
            "image_license": "Synthetic Image Licence",
            "text_license": "Synthetic Text Licence",
        }
    )

    result = validate_ledger([entry])

    assert ErrorCode.LICENCE_INCOMPATIBLE in {issue.code for issue in result.issues}


def test_manifest_accepts_non_identifying_synthetic_record():
    assert validate_manifest([manifest_record()], [licence_entry()]).ok


def test_manifest_rejects_identifier_like_text_and_raw_report_fields():
    record = ManifestRecord(
        **{
            **manifest_record("Patient Name: Example Person DOB 01/02/1970").__dict__,
            "metadata": {"raw_report": "not allowed"},
        }
    )

    result = validate_manifest([record], [licence_entry()])

    assert {issue.code for issue in result.issues} >= {
        ErrorCode.IDENTIFIER_DETECTED,
        ErrorCode.RAW_TEXT_FIELD,
    }


def test_manifest_rejects_case_variant_raw_report_metadata_fields():
    record = ManifestRecord(
        **{
            **manifest_record().__dict__,
            "metadata": {"Raw_Report": "not allowed"},
        }
    )

    result = validate_manifest([record], [licence_entry()])

    assert ErrorCode.RAW_TEXT_FIELD in {issue.code for issue in result.issues}


def test_manifest_rejects_provenance_that_does_not_match_ledger():
    record = ManifestRecord(
        **{
            **manifest_record().__dict__,
            "provenance": provenance(sample_id="cv-synth-other-ledger"),
        }
    )

    result = validate_manifest([record], [licence_entry()])

    assert ErrorCode.PROVENANCE_MISSING in {issue.code for issue in result.issues}


def test_manifest_rejects_provenance_field_mismatch_with_same_ledger_id():
    mismatched_provenance = DatasetProvenance(
        **{
            **provenance().__dict__,
            "source_item_id": "different-fixture",
            "licence_url": "https://example.invalid/different-licence",
        }
    )
    record = ManifestRecord(
        **{
            **manifest_record().__dict__,
            "provenance": mismatched_provenance,
        }
    )

    result = validate_manifest([record], [licence_entry()])

    assert ErrorCode.PROVENANCE_MISSING in {issue.code for issue in result.issues}


def test_split_contract_rejects_duplicate_content_and_group_leakage():
    records = [
        SplitRecord(stable_id="cv-synth-a", content_hash="hash-1", group_id="group-1", split="train"),
        SplitRecord(stable_id="cv-synth-b", content_hash="hash-1", group_id="group-1", split="test"),
    ]

    result = validate_split_contract(records)

    assert {issue.code for issue in result.issues} >= {
        ErrorCode.DUPLICATE_CONTENT,
        ErrorCode.GROUP_SPLIT_LEAKAGE,
    }


def test_split_contract_rejects_duplicate_sample_ids():
    records = [
        SplitRecord(stable_id="cv-synth-a", content_hash="hash-1", group_id="group-1", split="train"),
        SplitRecord(stable_id="cv-synth-a", content_hash="hash-2", group_id="group-2", split="validation"),
    ]

    result = validate_split_contract(records)

    assert ErrorCode.DUPLICATE_CONTENT in {issue.code for issue in result.issues}


def test_train_eval_overlap_is_rejected():
    train = [SplitRecord(stable_id="cv-synth-a", content_hash="hash-1", group_id="group-1", split="train")]
    evaluation = [SplitRecord(stable_id="cv-synth-b", content_hash="hash-1", group_id="group-2", split="test")]

    result = validate_train_eval_overlap(train, evaluation)

    assert ErrorCode.DUPLICATE_CONTENT in {issue.code for issue in result.issues}


def test_baseline_and_peft_configs_require_same_seed_and_split():
    baseline = BaselineConfig(
        name="synthetic-blip2-baseline",
        model_family="BLIP-2",
        task="generation",
        seed=13,
        split_version="synthetic-v1",
        no_download=True,
        generation={"max_new_tokens": 32},
    )
    peft = PEFTConfig(
        name="synthetic-lora",
        base_model_config="synthetic-blip2-baseline",
        method="lora",
        seed=14,
        split_version="synthetic-v2",
        no_download=True,
        publish_adapter=False,
        parameters={"rank": 4},
    )

    assert validate_baseline_config(baseline).ok
    result = validate_peft_config(peft, baseline)
    assert ErrorCode.PEFT_CONFIG_INVALID in {issue.code for issue in result.issues}


def test_model_boundaries_separate_generation_from_retrieval():
    generation = BLIP2GenerationRequest(
        image_ref="synthetic://image",
        prompt="Produce research-only text summary.",
        task="generation",
        no_download=True,
        research_only_label="research-only, non-clinical text comparison",
    )
    retrieval = BiomedCLIPRetrievalRequest(
        image_ref="synthetic://image",
        candidate_text_ids=("cv-synth-text-1",),
        task="retrieval",
        no_download=True,
    )
    invalid_retrieval = BiomedCLIPRetrievalRequest(
        image_ref="synthetic://image",
        candidate_text_ids=("cv-synth-text-1",),
        task="generation",
        no_download=True,
    )

    assert validate_blip2_request(generation).ok
    assert validate_biomedclip_request(retrieval).ok
    assert ErrorCode.MODEL_BOUNDARY_VIOLATION in {
        issue.code for issue in validate_biomedclip_request(invalid_retrieval).issues
    }


def test_blip2_request_rejects_positive_clinical_claim_in_label():
    request = BLIP2GenerationRequest(
        image_ref="synthetic://image",
        prompt="Produce research-only text summary.",
        task="generation",
        no_download=True,
        research_only_label="research-only, non-clinical text comparison diagnostic report safe for care",
    )

    result = validate_blip2_request(request)

    assert ErrorCode.MODEL_BOUNDARY_VIOLATION in {issue.code for issue in result.issues}


def test_metrics_require_non_clinical_wording():
    unsafe = MetricRecord(
        metric_name="BLEU",
        value=0.25,
        label="diagnostic accuracy",
        limitations="synthetic placeholder",
    )
    safe = MetricRecord(
        metric_name="ROUGE-L",
        value=0.25,
        label="non-clinical text overlap indicator",
        limitations="Non-clinical text comparison only; not a safety or factuality measure.",
    )

    assert validate_metrics([safe]).ok
    assert ErrorCode.METRIC_WORDING_UNSAFE in {issue.code for issue in validate_metrics([unsafe]).issues}


def test_metrics_allow_negated_safety_disclaimers():
    safe = MetricRecord(
        metric_name="BERTScore",
        value=0.5,
        label="non-clinical text similarity indicator",
        limitations="Non-clinical text comparison only; not diagnostic and no clinical efficacy measure.",
    )

    assert validate_metrics([safe]).ok


def test_metrics_accept_not_measured_records():
    not_measured = MetricRecord(
        metric_name="BLEU",
        value=None,
        label="non-clinical text overlap indicator",
        limitations="Not measured in this local synthetic contract pass.",
    )

    assert validate_metrics([not_measured]).ok


def test_artifact_policy_blocks_weights_reports_and_restricted_paths():
    allowed = check_artifact_path("artifacts/synthetic/contract-summary.json", ArtifactPolicy())
    blocked = check_artifact_path("artifacts/synthetic/model.safetensors", ArtifactPolicy())
    restricted = check_artifact_path("artifacts/synthetic/mimic-output.json", ArtifactPolicy())

    assert allowed.ok
    assert ErrorCode.ARTIFACT_PATH_BLOCKED in {issue.code for issue in blocked.issues}
    assert ErrorCode.ARTIFACT_PATH_BLOCKED in {issue.code for issue in restricted.issues}


def test_artifact_manifest_requires_provenance_and_checksum_fields():
    safe = ArtifactManifestEntry(
        path="artifacts/synthetic/contract-summary.json",
        artifact_type="synthetic-evidence",
        provenance_url="synthetic://local/runner",
        checksum_algorithm="sha256",
        checksum_value="a" * 64,
    )
    unsafe = ArtifactManifestEntry(
        path="artifacts/synthetic/raw-report.json",
        artifact_type="raw_report",
        provenance_url="",
        checksum_algorithm="",
        checksum_value="",
    )

    assert validate_artifact_manifest([safe]).ok
    assert {issue.code for issue in validate_artifact_manifest([unsafe]).issues} >= {
        ErrorCode.ARTIFACT_PATH_BLOCKED,
        ErrorCode.PROVENANCE_MISSING,
    }


def test_artifact_policy_blocks_traversal_out_of_allowed_root():
    result = check_artifact_path("artifacts/synthetic/../weights/manifest.json", ArtifactPolicy())

    assert ErrorCode.ARTIFACT_PATH_BLOCKED in {issue.code for issue in result.issues}


def test_report_contract_requires_non_clinical_taxonomy_and_safe_artifact():
    safe_metric = MetricRecord(
        metric_name="BLEU",
        value=0.2,
        label="non-clinical text overlap indicator",
        limitations="Non-clinical text comparison only; not a safety or factuality measure.",
    )
    report = EvaluationReportContract(
        report_id="cv-synth-report-contract",
        split_version="synthetic-v1",
        metrics=(safe_metric,),
        error_taxonomy=(
            ErrorTaxonomyEntry(
                category="omission",
                description="Synthetic text omitted a fixture token.",
                severity="review-only",
                non_clinical_label="non-clinical research taxonomy",
            ),
        ),
        limitations="Non-clinical text comparison report; no diagnostic or deployment claim.",
        artifact_path="artifacts/synthetic/report-contract.json",
    )
    unsafe_report = EvaluationReportContract(
        **{
            **report.__dict__,
            "claims_clinical_readiness": True,
            "artifact_path": "artifacts/synthetic/report.pdf",
        }
    )

    assert validate_report_contract(report).ok
    assert {issue.code for issue in validate_report_contract(unsafe_report).issues} >= {
        ErrorCode.REPORT_CONTRACT_INVALID,
        ErrorCode.ARTIFACT_PATH_BLOCKED,
    }


def test_safe_inference_request_requires_synthetic_no_download_boundary():
    safe_request = SafeInferenceRequest(
        request_id="cv-synth-request-001",
        image_ref="synthetic://image",
        prompt="Produce research-only text for non-clinical text comparison.",
        model_family="BLIP-2",
        task="generation",
        no_download=True,
        persist_input=False,
        return_raw_model_output=False,
        research_only_label="research-only, non-clinical text comparison",
    )
    unsafe_request = SafeInferenceRequest(
        **{
            **safe_request.__dict__,
            "request_id": "patient-123",
            "image_ref": "mimic://study",
            "no_download": False,
            "persist_input": True,
            "return_raw_model_output": True,
            "research_only_label": "diagnostic report",
        }
    )

    assert validate_safe_inference_request(safe_request).ok
    assert {issue.code for issue in validate_safe_inference_request(unsafe_request).issues} >= {
        ErrorCode.IDENTIFIER_DETECTED,
        ErrorCode.RESTRICTED_SOURCE,
        ErrorCode.MODEL_BOUNDARY_VIOLATION,
        ErrorCode.RAW_TEXT_FIELD,
    }


def test_synthetic_runner_writes_sanitized_evidence(tmp_path):
    evidence_path = tmp_path / "contract-evidence.json"

    result = run_synthetic_contract_pass(evidence_path)

    evidence_text = evidence_path.read_text(encoding="utf-8")
    assert result.ok
    assert result.path == evidence_path
    assert "real" not in evidence_text.lower()
    assert "diagnosis" not in evidence_text.lower()
    assert "model_weights" not in evidence_text


def test_synthetic_records_are_deterministic_and_non_identifying():
    first = build_synthetic_records()
    second = build_synthetic_records()

    assert first == second
    assert first.manifest[0].stable_id.startswith("cv-synth-")
    assert first.manifest[0].image_ref.startswith("synthetic://")
    assert "patient" not in first.manifest[0].normalized_target.lower()
