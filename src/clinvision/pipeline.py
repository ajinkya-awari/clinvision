"""Local synthetic contract orchestration for ClinVision."""

from __future__ import annotations

from .contracts import (
    BaselineConfig,
    PEFTConfig,
    ValidationIssue,
    ValidationResult,
    validate_baseline_config,
    validate_ledger,
    validate_manifest,
    validate_peft_config,
    validate_split_contract,
)
from .synthetic import SyntheticRecords


def validate_synthetic_workflow(records: SyntheticRecords, baseline: BaselineConfig, peft: PEFTConfig) -> ValidationResult:
    issues: list[ValidationIssue] = []
    issues.extend(validate_ledger(records.ledger).issues)
    issues.extend(validate_manifest(records.manifest, records.ledger).issues)
    issues.extend(validate_split_contract(records.split).issues)
    issues.extend(validate_baseline_config(baseline).issues)
    issues.extend(validate_peft_config(peft, baseline).issues)
    return ValidationResult.from_issues(issues)
