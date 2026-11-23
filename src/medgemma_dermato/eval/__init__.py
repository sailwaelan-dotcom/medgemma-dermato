"""Locked evaluation harness for MedGemma dermoscopy outputs."""

from medgemma_dermato.eval.harness import (
    EvalCase,
    EvalReport,
    compare_reports,
    evaluate_predictions,
)

__all__ = [
    "EvalCase",
    "EvalReport",
    "compare_reports",
    "evaluate_predictions",
]
