"""Synthetic-only safety contracts for ClinVision.

These validators make the project gates executable without downloading data,
loading model weights, or handling clinical records. Callers pass already
constructed dictionaries or dataclasses; this module does not read external
files, access networks, import model libraries, or execute training/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from pathlib import PurePath
from typing import Iterable, Mapping, Sequence


class ErrorCode(str, Enum):
    LICENCE_MISSING = "licence_missing"
    LICENCE_INCOMPATIBLE = "licence_incompatible"
    PROVENANCE_MISSING = "provenance_missing"
    RESTRICTED_SOURCE = "restricted_source"
    IDENTIFIER_DETECTED = "identifier_detected"
    RAW_TEXT_FIELD = "raw_text_field"
    MANIFEST_REFERENCE_MISSING = "manifest_reference_missing"
    DUPLICATE_CONTENT = "duplicate_content"
    GROUP_SPLIT_LEAKAGE = "group_split_leakage"
    SPLIT_VALUE_INVALID = "split_value_invalid"
    BASELINE_CONFIG_INVALID = "baseline_config_invalid"
    PEFT_CONFIG_INVALID = "peft_config_invalid"
    MODEL_BOUNDARY_VIOLATION = "model_boundary_violation"
    METRIC_WORDING_UNSAFE = "metric_wording_unsafe"
    REPORT_CONTRACT_INVALID = "report_contract_invalid"
    ARTIFACT_PATH_BLOCKED = "artifact_path_blocked"


@dataclass(frozen=True)
class ValidationIssue:
    code: ErrorCode
    message: str
    record_id: str | None = None
    field: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: tuple[ValidationIssue, ...] = ()

    @classmethod
    def from_issues(cls, issues: Iterable[ValidationIssue]) -> "ValidationResult":
        issue_tuple = tuple(issues)
        return cls(ok=not issue_tuple, issues=issue_tuple)


@dataclass(frozen=True)
class LicenceLedgerEntry:
    sample_id: str
    source_dataset: str
    source_item_id: str
    licence_name: str
    licence_url: str
    licence_evidence: str
    allowed_uses: tuple[str, ...]
    provenance_url: str
    reviewed_by: str
    reviewed_at: str
    public_data_lane: bool
    restricted_source: bool = False
    image_rights_holder: str = ""
    text_rights_holder: str = ""
    image_license: str = ""
    text_license: str = ""
    image_license_evidence: str = ""
    text_license_evidence: str = ""
    derivative_and_redistribution_rights: str = ""
    privacy_deidentification_evidence: str = ""
    retention_rule: str = ""
    access_class: str = ""
    checksum_algorithm: str = ""
    checksum_value: str = ""
    approval_status: str = ""
    decision_rationale: str = ""


@dataclass(frozen=True)
class DatasetProvenance:
    sample_id: str
    source_dataset: str
    source_item_id: str
    provenance_url: str
    licence_name: str
    licence_url: str
    reviewed_at: str


@dataclass(frozen=True)
class ManifestRecord:
    stable_id: str
    ledger_sample_id: str
    image_ref: str
    normalized_target: str
    group_id: str
    provenance: DatasetProvenance
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitRecord:
    stable_id: str
    content_hash: str
    group_id: str
    split: str


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    model_family: str
    task: str
    seed: int
    split_version: str
    no_download: bool
    generation: Mapping[str, int | float | str]


@dataclass(frozen=True)
class PEFTConfig:
    name: str
    base_model_config: str
    method: str
    seed: int
    split_version: str
    no_download: bool
    publish_adapter: bool
    parameters: Mapping[str, int | float | str]


@dataclass(frozen=True)
class BLIP2GenerationRequest:
    image_ref: str
    prompt: str
    task: str
    no_download: bool
    research_only_label: str


@dataclass(frozen=True)
class SafeInferenceRequest:
    request_id: str
    image_ref: str
    prompt: str
    model_family: str
    task: str
    no_download: bool
    persist_input: bool
    return_raw_model_output: bool
    research_only_label: str


@dataclass(frozen=True)
class BiomedCLIPRetrievalRequest:
    image_ref: str
    candidate_text_ids: tuple[str, ...]
    task: str
    no_download: bool


@dataclass(frozen=True)
class MetricRecord:
    metric_name: str
    value: float | None
    label: str
    limitations: str


@dataclass(frozen=True)
class ErrorTaxonomyEntry:
    category: str
    description: str
    severity: str
    non_clinical_label: str


@dataclass(frozen=True)
class EvaluationReportContract:
    report_id: str
    split_version: str
    metrics: tuple[MetricRecord, ...]
    error_taxonomy: tuple[ErrorTaxonomyEntry, ...]
    limitations: str
    artifact_path: str
    contains_raw_examples: bool = False
    claims_clinical_readiness: bool = False


@dataclass(frozen=True)
class ArtifactPolicy:
    allowed_root: str = "artifacts/synthetic"
    allow_checkpoints: bool = False
    allow_weights: bool = False
    allow_generated_reports: bool = False


@dataclass(frozen=True)
class ArtifactManifestEntry:
    path: str
    artifact_type: str
    provenance_url: str
    checksum_algorithm: str
    checksum_value: str


RESTRICTED_SOURCE_RE = re.compile(r"\b(mimic|physionet)\b", re.IGNORECASE)
NAME_IDENTIFIER_RE = re.compile(r"\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b")
IDENTIFIER_RE = re.compile(
    r"("
    r"\b(?:MRN|SSN|DOB|patient name|accession|medical record)\b|"
    r"\b\d{3}-\d{2}-\d{4}\b|"
    r"\b\d{2}/\d{2}/(?:19|20)\d{2}\b|"
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    r")",
    re.IGNORECASE,
)
RAW_TEXT_FIELDS = frozenset({"raw_report", "report", "findings", "impression", "patient_text"})
VALID_SPLITS = frozenset({"train", "validation", "test"})
RESEARCH_ONLY = "research-only, non-clinical text comparison"
UNSAFE_METRIC_WORDS = re.compile(
    r"\b(diagnos(?:is|tic)|clinical efficacy|triage|patient outcome|radiologist-level|sota|safe for care)\b",
    re.IGNORECASE,
)
NEGATED_SAFETY_PHRASES = (
    "no diagnosis",
    "no diagnostic",
    "not diagnostic",
    "no clinical efficacy",
    "not clinical efficacy",
    "no triage",
    "not triage",
    "no patient outcome",
    "not patient outcome",
    "not safe for care",
)
WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx"})
REPORT_SUFFIXES = frozenset({".html", ".pdf", ".docx"})


def stable_synthetic_id(source: str) -> str:
    """Create a deterministic non-identifying fixture ID from synthetic text."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"cv-synth-{digest}"


def validate_ledger(entries: Sequence[LicenceLedgerEntry]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.sample_id in seen:
            issues.append(_issue(ErrorCode.DUPLICATE_CONTENT, entry.sample_id, "sample_id", "Duplicate ledger sample ID."))
        seen.add(entry.sample_id)
        joined = " ".join(
            [
                entry.sample_id,
                entry.source_dataset,
                entry.source_item_id,
                entry.licence_name,
                entry.licence_url,
                entry.provenance_url,
            ]
        )
        if entry.restricted_source or RESTRICTED_SOURCE_RE.search(joined):
            issues.append(_issue(ErrorCode.RESTRICTED_SOURCE, entry.sample_id, "source_dataset", "Restricted source is blocked."))
        if not entry.licence_name or not entry.licence_url or not entry.licence_evidence:
            issues.append(_issue(ErrorCode.LICENCE_MISSING, entry.sample_id, "licence", "Item-level licence evidence is required."))
        modality_fields = {
            "image_rights_holder": entry.image_rights_holder,
            "text_rights_holder": entry.text_rights_holder,
            "image_license": entry.image_license,
            "text_license": entry.text_license,
            "image_license_evidence": entry.image_license_evidence,
            "text_license_evidence": entry.text_license_evidence,
            "derivative_and_redistribution_rights": entry.derivative_and_redistribution_rights,
            "privacy_deidentification_evidence": entry.privacy_deidentification_evidence,
            "retention_rule": entry.retention_rule,
            "access_class": entry.access_class,
            "checksum_algorithm": entry.checksum_algorithm,
            "checksum_value": entry.checksum_value,
            "approval_status": entry.approval_status,
            "decision_rationale": entry.decision_rationale,
        }
        missing_modality_fields = [name for name, value in modality_fields.items() if not str(value).strip()]
        if missing_modality_fields:
            issues.append(_issue(ErrorCode.LICENCE_MISSING, entry.sample_id, ",".join(missing_modality_fields), "Separate image/text rights, privacy, retention, checksum, approval, and rationale fields are required."))
        if entry.image_license and entry.text_license and entry.image_license != entry.text_license:
            issues.append(_issue(ErrorCode.LICENCE_INCOMPATIBLE, entry.sample_id, "image_license,text_license", "Image and text licences must be explicitly compatible; unknown mixed rights fail closed."))
        if entry.approval_status and entry.approval_status.lower() != "approved":
            issues.append(_issue(ErrorCode.LICENCE_INCOMPATIBLE, entry.sample_id, "approval_status", "Only approved synthetic ledger entries can pass."))
        if not entry.public_data_lane or "research" not in {use.lower() for use in entry.allowed_uses}:
            issues.append(_issue(ErrorCode.LICENCE_INCOMPATIBLE, entry.sample_id, "allowed_uses", "Research use must be explicitly allowed in the public lane."))
        if not entry.provenance_url or not entry.reviewed_at or not entry.reviewed_by:
            issues.append(_issue(ErrorCode.PROVENANCE_MISSING, entry.sample_id, "provenance", "Reviewer, timestamp, and provenance URL are required."))
    return ValidationResult.from_issues(issues)


def validate_manifest(records: Sequence[ManifestRecord], ledger_entries: Sequence[LicenceLedgerEntry]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    ledger_by_id = {entry.sample_id: entry for entry in ledger_entries}
    ledger_ids = set(ledger_by_id)
    for record in records:
        provenance_text = " ".join(
            [
                record.provenance.sample_id,
                record.provenance.source_dataset,
                record.provenance.source_item_id,
                record.provenance.provenance_url,
                record.provenance.licence_name,
                record.provenance.licence_url,
            ]
        )
        text = " ".join(
            [
                record.stable_id,
                record.image_ref,
                record.normalized_target,
                record.group_id,
                provenance_text,
                *record.metadata.values(),
            ]
        )
        if record.ledger_sample_id not in ledger_ids:
            issues.append(_issue(ErrorCode.MANIFEST_REFERENCE_MISSING, record.stable_id, "ledger_sample_id", "Manifest record must reference a licence ledger entry."))
        if record.provenance.sample_id != record.ledger_sample_id:
            issues.append(_issue(ErrorCode.PROVENANCE_MISSING, record.stable_id, "provenance.sample_id", "Manifest provenance must match the referenced ledger sample ID."))
        ledger_entry = ledger_by_id.get(record.ledger_sample_id)
        if ledger_entry and not _provenance_matches_ledger(record.provenance, ledger_entry):
            issues.append(_issue(ErrorCode.PROVENANCE_MISSING, record.stable_id, "provenance", "Manifest provenance fields must match the referenced ledger entry."))
        if not record.stable_id.startswith("cv-synth-"):
            issues.append(_issue(ErrorCode.IDENTIFIER_DETECTED, record.stable_id, "stable_id", "Stable ID must be synthetic and non-identifying."))
        if _has_identifier_like_text(text):
            issues.append(_issue(ErrorCode.IDENTIFIER_DETECTED, record.stable_id, "normalized_target", "Potential identifier detected."))
        if RESTRICTED_SOURCE_RE.search(text):
            issues.append(_issue(ErrorCode.RESTRICTED_SOURCE, record.stable_id, "source", "Restricted source reference is blocked."))
        forbidden = RAW_TEXT_FIELDS.intersection(key.lower() for key in record.metadata.keys())
        if forbidden:
            issues.append(_issue(ErrorCode.RAW_TEXT_FIELD, record.stable_id, ",".join(sorted(forbidden)), "Raw report-style fields are not allowed in the manifest."))
    return ValidationResult.from_issues(issues)


def validate_split_contract(records: Sequence[SplitRecord]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    hash_to_id: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    for record in records:
        if record.stable_id in seen_ids:
            issues.append(_issue(ErrorCode.DUPLICATE_CONTENT, record.stable_id, "stable_id", "Duplicate sample IDs are not allowed."))
        seen_ids.add(record.stable_id)
        if record.split not in VALID_SPLITS:
            issues.append(_issue(ErrorCode.SPLIT_VALUE_INVALID, record.stable_id, "split", "Split must be train, validation, or test."))
        existing_id = hash_to_id.get(record.content_hash)
        if existing_id and existing_id != record.stable_id:
            issues.append(_issue(ErrorCode.DUPLICATE_CONTENT, record.stable_id, "content_hash", "Duplicate content cannot cross records."))
        hash_to_id[record.content_hash] = record.stable_id
        existing_split = group_to_split.get(record.group_id)
        if existing_split and existing_split != record.split:
            issues.append(_issue(ErrorCode.GROUP_SPLIT_LEAKAGE, record.stable_id, "group_id", "One group cannot appear in multiple splits."))
        group_to_split[record.group_id] = record.split
    return ValidationResult.from_issues(issues)


def validate_train_eval_overlap(train_records: Sequence[SplitRecord], evaluation_records: Sequence[SplitRecord]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    train_ids = {record.stable_id for record in train_records}
    train_hashes = {record.content_hash for record in train_records}
    train_groups = {record.group_id for record in train_records}
    for record in evaluation_records:
        if record.stable_id in train_ids:
            issues.append(_issue(ErrorCode.DUPLICATE_CONTENT, record.stable_id, "stable_id", "Evaluation sample ID overlaps with train."))
        if record.content_hash in train_hashes:
            issues.append(_issue(ErrorCode.DUPLICATE_CONTENT, record.stable_id, "content_hash", "Evaluation content overlaps with train."))
        if record.group_id in train_groups:
            issues.append(_issue(ErrorCode.GROUP_SPLIT_LEAKAGE, record.stable_id, "group_id", "Evaluation group overlaps with train."))
    return ValidationResult.from_issues(issues)


def validate_baseline_config(config: BaselineConfig) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if config.model_family.lower() != "blip-2" or config.task != "generation":
        issues.append(_issue(ErrorCode.BASELINE_CONFIG_INVALID, config.name, "model_family", "Baseline must declare BLIP-2 generation only."))
    if not config.no_download:
        issues.append(_issue(ErrorCode.BASELINE_CONFIG_INVALID, config.name, "no_download", "Local contract tests must run in no-download mode."))
    if config.seed < 0 or not config.split_version:
        issues.append(_issue(ErrorCode.BASELINE_CONFIG_INVALID, config.name, "seed", "Seed and split version are required."))
    if "max_new_tokens" not in config.generation:
        issues.append(_issue(ErrorCode.BASELINE_CONFIG_INVALID, config.name, "generation", "Generation limits must be explicit."))
    return ValidationResult.from_issues(issues)


def validate_peft_config(config: PEFTConfig, baseline: BaselineConfig) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if config.method.lower() != "lora":
        issues.append(_issue(ErrorCode.PEFT_CONFIG_INVALID, config.name, "method", "Only LoRA is declared for the v1 PEFT comparison."))
    if config.seed != baseline.seed or config.split_version != baseline.split_version:
        issues.append(_issue(ErrorCode.PEFT_CONFIG_INVALID, config.name, "split_version", "PEFT must use the same seed and split as baseline."))
    if not config.no_download or config.publish_adapter:
        issues.append(_issue(ErrorCode.PEFT_CONFIG_INVALID, config.name, "artifact", "Local PEFT contracts must not download or publish adapters."))
    return ValidationResult.from_issues(issues)


def validate_blip2_request(request: BLIP2GenerationRequest) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if request.task != "generation":
        issues.append(_issue(ErrorCode.MODEL_BOUNDARY_VIOLATION, request.image_ref, "task", "BLIP-2 boundary only accepts generation requests."))
    if not request.no_download:
        issues.append(_issue(ErrorCode.MODEL_BOUNDARY_VIOLATION, request.image_ref, "no_download", "Local boundary checks must not load/download weights."))
    if RESEARCH_ONLY not in request.research_only_label.lower() or _has_unsafe_positive_claim(request.research_only_label):
        issues.append(_issue(ErrorCode.MODEL_BOUNDARY_VIOLATION, request.image_ref, "research_only_label", "Generated text must carry non-clinical research-only wording."))
    return ValidationResult.from_issues(issues)


def validate_safe_inference_request(request: SafeInferenceRequest) -> ValidationResult:
    issues: list[ValidationIssue] = []
    text = " ".join([request.request_id, request.image_ref, request.prompt, request.research_only_label])
    if not request.request_id.startswith("cv-synth-") or _has_identifier_like_text(text):
        issues.append(_issue(ErrorCode.IDENTIFIER_DETECTED, request.request_id, "request_id", "Inference requests must use synthetic non-identifying IDs and text."))
    if RESTRICTED_SOURCE_RE.search(text):
        issues.append(_issue(ErrorCode.RESTRICTED_SOURCE, request.request_id, "image_ref", "Restricted sources are blocked for safe inference."))
    if request.model_family.lower() != "blip-2" or request.task != "generation" or not request.no_download:
        issues.append(_issue(ErrorCode.MODEL_BOUNDARY_VIOLATION, request.request_id, "model_family", "Safe local inference is BLIP-2 generation boundary validation in no-download mode only."))
    if request.persist_input or request.return_raw_model_output:
        issues.append(_issue(ErrorCode.RAW_TEXT_FIELD, request.request_id, "persistence", "Safe inference must not persist inputs or return raw model output."))
    if RESEARCH_ONLY not in request.research_only_label.lower() or _has_unsafe_positive_claim(request.research_only_label):
        issues.append(_issue(ErrorCode.MODEL_BOUNDARY_VIOLATION, request.request_id, "research_only_label", "Inference output must be labelled research-only and non-clinical."))
    return ValidationResult.from_issues(issues)


def validate_biomedclip_request(request: BiomedCLIPRetrievalRequest) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if request.task not in {"contrastive", "retrieval", "embedding"}:
        issues.append(_issue(ErrorCode.MODEL_BOUNDARY_VIOLATION, request.image_ref, "task", "BiomedCLIP is contrastive/retrieval only and cannot generate reports."))
    if not request.candidate_text_ids:
        issues.append(_issue(ErrorCode.MODEL_BOUNDARY_VIOLATION, request.image_ref, "candidate_text_ids", "Retrieval requires candidate text IDs."))
    if not request.no_download:
        issues.append(_issue(ErrorCode.MODEL_BOUNDARY_VIOLATION, request.image_ref, "no_download", "Local boundary checks must not load/download weights."))
    return ValidationResult.from_issues(issues)


def validate_metrics(records: Sequence[MetricRecord]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    for record in records:
        wording = f"{record.metric_name} {record.label} {record.limitations}"
        if _has_unsafe_positive_claim(wording):
            issues.append(_issue(ErrorCode.METRIC_WORDING_UNSAFE, record.metric_name, "label", "Metrics must be described as non-clinical text comparison only."))
        if record.value is not None and not 0.0 <= record.value <= 1.0:
            issues.append(_issue(ErrorCode.METRIC_WORDING_UNSAFE, record.metric_name, "value", "Metric values must be normalized synthetic placeholders in [0, 1]."))
        if record.value is None and "not measured" not in wording.lower():
            issues.append(_issue(ErrorCode.METRIC_WORDING_UNSAFE, record.metric_name, "value", "Unmeasured metrics must be explicitly labelled not measured."))
        if "non-clinical" not in wording.lower() or "text" not in wording.lower():
            issues.append(_issue(ErrorCode.METRIC_WORDING_UNSAFE, record.metric_name, "limitations", "Metric wording must include non-clinical text-comparison limitations."))
    return ValidationResult.from_issues(issues)


def validate_artifact_manifest(entries: Sequence[ArtifactManifestEntry], policy: ArtifactPolicy | None = None) -> ValidationResult:
    issues: list[ValidationIssue] = []
    blocked_types = {"raw_image", "raw_report", "model_weight", "checkpoint", "adapter", "generated_report"}
    for entry in entries:
        issues.extend(check_artifact_path(entry.path, policy).issues)
        if entry.artifact_type.lower() in blocked_types:
            issues.append(_issue(ErrorCode.ARTIFACT_PATH_BLOCKED, entry.path, "artifact_type", "Raw data, model, checkpoint, adapter, and generated-report artifacts are blocked."))
        if not entry.provenance_url or not entry.checksum_algorithm or not entry.checksum_value:
            issues.append(_issue(ErrorCode.PROVENANCE_MISSING, entry.path, "provenance", "Artifact provenance URL and checksum fields are required."))
        if entry.checksum_algorithm and entry.checksum_algorithm.lower() != "sha256":
            issues.append(_issue(ErrorCode.PROVENANCE_MISSING, entry.path, "checksum_algorithm", "Synthetic artifact manifests require sha256 checksums."))
    return ValidationResult.from_issues(issues)


def validate_report_contract(report: EvaluationReportContract, policy: ArtifactPolicy | None = None) -> ValidationResult:
    issues: list[ValidationIssue] = list(validate_metrics(report.metrics).issues)
    artifact_result = check_artifact_path(report.artifact_path, policy)
    issues.extend(artifact_result.issues)
    wording = f"{report.report_id} {report.limitations}"
    if not report.split_version:
        issues.append(_issue(ErrorCode.REPORT_CONTRACT_INVALID, report.report_id, "split_version", "Report must name the evaluated split version."))
    if report.contains_raw_examples:
        issues.append(_issue(ErrorCode.REPORT_CONTRACT_INVALID, report.report_id, "contains_raw_examples", "Report contract cannot include raw examples by default."))
    if report.claims_clinical_readiness or _has_unsafe_positive_claim(wording):
        issues.append(_issue(ErrorCode.REPORT_CONTRACT_INVALID, report.report_id, "limitations", "Report must not claim clinical readiness or diagnostic value."))
    for entry in report.error_taxonomy:
        entry_words = f"{entry.category} {entry.description} {entry.severity} {entry.non_clinical_label}"
        if "non-clinical" not in entry_words.lower() or _has_unsafe_positive_claim(entry_words):
            issues.append(_issue(ErrorCode.REPORT_CONTRACT_INVALID, report.report_id, "error_taxonomy", "Error taxonomy entries must stay non-clinical."))
    return ValidationResult.from_issues(issues)


def check_artifact_path(path: str, policy: ArtifactPolicy | None = None) -> ValidationResult:
    active_policy = policy or ArtifactPolicy()
    normalized = path.replace("\\", "/").strip()
    parts = PurePath(normalized).parts
    suffix = PurePath(normalized).suffix.lower()
    issues: list[ValidationIssue] = []
    if ".." in parts or PurePath(normalized).is_absolute():
        issues.append(_issue(ErrorCode.ARTIFACT_PATH_BLOCKED, normalized, "path", "Artifact paths must be relative and must not contain traversal segments."))
    if not normalized.startswith(active_policy.allowed_root.rstrip("/") + "/"):
        issues.append(_issue(ErrorCode.ARTIFACT_PATH_BLOCKED, normalized, "path", "Artifacts must stay under the approved synthetic artifact root."))
    if suffix in WEIGHT_SUFFIXES and not active_policy.allow_weights:
        issues.append(_issue(ErrorCode.ARTIFACT_PATH_BLOCKED, normalized, "suffix", "Weights, adapters, and checkpoints are blocked locally."))
    if suffix in REPORT_SUFFIXES and not active_policy.allow_generated_reports:
        issues.append(_issue(ErrorCode.ARTIFACT_PATH_BLOCKED, normalized, "suffix", "Generated report-style artifacts require release review."))
    if RESTRICTED_SOURCE_RE.search(normalized):
        issues.append(_issue(ErrorCode.ARTIFACT_PATH_BLOCKED, normalized, "path", "Restricted-source artifacts are blocked."))
    return ValidationResult.from_issues(issues)


def _issue(code: ErrorCode, record_id: str | None, field: str | None, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, record_id=record_id, field=field)


def _has_unsafe_positive_claim(text: str) -> bool:
    lowered = text.lower()
    scrubbed = lowered
    for phrase in NEGATED_SAFETY_PHRASES:
        scrubbed = scrubbed.replace(phrase, "")
    return bool(UNSAFE_METRIC_WORDS.search(scrubbed))


def _has_identifier_like_text(text: str) -> bool:
    return bool(NAME_IDENTIFIER_RE.search(text) or IDENTIFIER_RE.search(text))


def _provenance_matches_ledger(provenance: DatasetProvenance, ledger_entry: LicenceLedgerEntry) -> bool:
    return (
        provenance.sample_id == ledger_entry.sample_id
        and provenance.source_dataset == ledger_entry.source_dataset
        and provenance.source_item_id == ledger_entry.source_item_id
        and provenance.provenance_url == ledger_entry.provenance_url
        and provenance.licence_name == ledger_entry.licence_name
        and provenance.licence_url == ledger_entry.licence_url
        and provenance.reviewed_at == ledger_entry.reviewed_at
    )
