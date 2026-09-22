"""Evaluation contract exports for non-clinical text-comparison records."""

from .contracts import MetricRecord, validate_metrics

__all__ = ["MetricRecord", "validate_metrics"]
