"""Locked evaluation harness: JSON validity, clinical sensitivity, ECE, and regressions."""

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from medgemma_dermato.schema import (
    DermoscopyAssessment,
    TriageAssessment,
    parse_assessment_json,
    parse_triage_json,
)

TaskMode = Literal["semiology", "triage"]


class EvalCase(BaseModel):
    """Single evaluated case pairing ground truth metadata with raw model output."""

    isic_id: str
    ground_truth_malignant: bool | None = None
    ground_truth_diagnosis: str | None = None
    ground_truth_entity: str | None = None
    ground_truth_is_melanoma: bool | None = None
    ground_truth_semiology: dict[str, Any] | None = None
    raw_model_output: str
    latency_ms: float | None = None


class EvalReport(BaseModel):
    """Aggregated evaluation report for one model/quantization checkpoint."""

    run_name: str
    model_id: str
    quantization: str = "BF16"
    task_mode: str = "semiology"
    total_cases: int
    json_valid_count: int
    json_valid_rate: float
    abstention_count: int
    abstention_rate: float
    malignant_total: int
    malignant_detected: int
    malignant_abstained: int
    malignant_missed_fn: int
    malignant_sensitivity_direct: float | None = None
    malignant_sensitivity_safe_triage: float | None = None
    melanoma_total: int = 0
    melanoma_detected: int = 0
    melanoma_abstained: int = 0
    melanoma_sensitivity_direct: float | None = None
    melanoma_sensitivity_safe_triage: float | None = None
    benign_total: int
    benign_correct: int
    benign_abstained: int
    specificity_direct: float | None = None
    entity_accuracy: float | None = None
    ece_calibration_error: float | None = None
    semiology_agreement: dict[str, float] = Field(default_factory=dict)
    mean_latency_ms: float | None = None
    parse_errors_sample: list[str] = Field(default_factory=list)


def _compute_ece(
    confidences: list[float],
    correctness: list[bool],
    num_bins: int = 10,
) -> float | None:
    """Compute 10-bin Expected Calibration Error over non-abstained predictions."""
    n = len(confidences)
    if n == 0:
        return None
    ece = 0.0
    for b in range(num_bins):
        low = b / num_bins
        high = (b + 1) / num_bins
        indices = [
            i
            for i, c in enumerate(confidences)
            if (low <= c <= high if b == num_bins - 1 else low <= c < high)
        ]
        if not indices:
            continue
        bin_acc = sum(1.0 for i in indices if correctness[i]) / len(indices)
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        ece += (len(indices) / n) * abs(bin_acc - bin_conf)
    return round(ece, 4)


def evaluate_predictions(
    cases: list[EvalCase],
    run_name: str,
    model_id: str,
    quantization: str = "BF16",
    abstention_threshold: float = 0.60,
    task_mode: TaskMode = "semiology",
) -> EvalReport:
    """Score raw model outputs against ground truth labels, entities, and semiology."""
    total_cases = len(cases)
    json_valid_count = 0
    abstention_count = 0

    malignant_total = 0
    malignant_detected = 0
    malignant_abstained = 0
    malignant_missed_fn = 0

    melanoma_total = 0
    melanoma_detected = 0
    melanoma_abstained = 0

    benign_total = 0
    benign_correct = 0
    benign_abstained = 0

    entity_total = 0
    entity_correct = 0

    calib_confidences: list[float] = []
    calib_correctness: list[bool] = []
    semiology_hits: dict[str, int] = defaultdict(int)
    semiology_counts: dict[str, int] = defaultdict(int)
    latencies: list[float] = []
    parse_errors_sample: list[str] = []

    for case in cases:
        if case.latency_ms is not None:
            latencies.append(case.latency_ms)

        if case.ground_truth_malignant is True:
            malignant_total += 1
        elif case.ground_truth_malignant is False:
            benign_total += 1

        if case.ground_truth_is_melanoma is True:
            melanoma_total += 1

        if case.ground_truth_entity is not None:
            entity_total += 1

        parsed: TriageAssessment | DermoscopyAssessment | None
        err: str | None
        if task_mode == "triage":
            parsed, err = parse_triage_json(case.raw_model_output)
        else:
            parsed, err = parse_assessment_json(case.raw_model_output)

        if parsed is None:
            if len(parse_errors_sample) < 10:
                parse_errors_sample.append(f"{case.isic_id}: {err}")
            if case.ground_truth_malignant is True:
                malignant_missed_fn += 1
            continue

        json_valid_count += 1
        is_abstained = (
            parsed.abstain
            or parsed.confidence < abstention_threshold
            or parsed.research_triage_label == "indeterminate"
        )

        if is_abstained:
            abstention_count += 1
            if case.ground_truth_malignant is True:
                malignant_abstained += 1
            elif case.ground_truth_malignant is False:
                benign_abstained += 1
            if case.ground_truth_is_melanoma is True:
                melanoma_abstained += 1
        else:
            pred_malignant = parsed.research_triage_label == "suspicious_malignant_pattern"
            if case.ground_truth_malignant is True:
                if pred_malignant:
                    malignant_detected += 1
                    if case.ground_truth_is_melanoma is True:
                        melanoma_detected += 1
                else:
                    malignant_missed_fn += 1
                calib_confidences.append(parsed.confidence)
                calib_correctness.append(pred_malignant)
            elif case.ground_truth_malignant is False:
                if not pred_malignant:
                    benign_correct += 1
                calib_confidences.append(parsed.confidence)
                calib_correctness.append(not pred_malignant)

        if task_mode == "triage" and case.ground_truth_entity is not None:
            if (
                isinstance(parsed, TriageAssessment)
                and parsed.entity_category == case.ground_truth_entity
            ):
                entity_correct += 1

        if task_mode == "semiology" and case.ground_truth_semiology:
            if isinstance(parsed, DermoscopyAssessment):
                sem_dict = parsed.semiology.model_dump()
                for key, gt_val in case.ground_truth_semiology.items():
                    if key in sem_dict:
                        semiology_counts[key] += 1
                        if sem_dict[key] == gt_val:
                            semiology_hits[key] += 1

    json_valid_rate = round(json_valid_count / total_cases, 4) if total_cases > 0 else 0.0
    abstention_rate = (
        round(abstention_count / json_valid_count, 4) if json_valid_count > 0 else 0.0
    )

    sens_direct = (
        round(malignant_detected / malignant_total, 4) if malignant_total > 0 else None
    )
    sens_safe = (
        round((malignant_detected + malignant_abstained) / malignant_total, 4)
        if malignant_total > 0
        else None
    )
    mel_sens_direct = (
        round(melanoma_detected / melanoma_total, 4) if melanoma_total > 0 else None
    )
    mel_sens_safe = (
        round((melanoma_detected + melanoma_abstained) / melanoma_total, 4)
        if melanoma_total > 0
        else None
    )
    spec_direct = round(benign_correct / benign_total, 4) if benign_total > 0 else None
    ent_acc = (
        round(entity_correct / entity_total, 4)
        if task_mode == "triage" and entity_total > 0
        else None
    )
    ece = _compute_ece(calib_confidences, calib_correctness)

    semiology_agreement = {
        field: round(semiology_hits[field] / count, 4)
        for field, count in sorted(semiology_counts.items())
        if count > 0
    }
    mean_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

    return EvalReport(
        run_name=run_name,
        model_id=model_id,
        quantization=quantization,
        task_mode=task_mode,
        total_cases=total_cases,
        json_valid_count=json_valid_count,
        json_valid_rate=json_valid_rate,
        abstention_count=abstention_count,
        abstention_rate=abstention_rate,
        malignant_total=malignant_total,
        malignant_detected=malignant_detected,
        malignant_abstained=malignant_abstained,
        malignant_missed_fn=malignant_missed_fn,
        malignant_sensitivity_direct=sens_direct,
        malignant_sensitivity_safe_triage=sens_safe,
        melanoma_total=melanoma_total,
        melanoma_detected=melanoma_detected,
        melanoma_abstained=melanoma_abstained,
        melanoma_sensitivity_direct=mel_sens_direct,
        melanoma_sensitivity_safe_triage=mel_sens_safe,
        benign_total=benign_total,
        benign_correct=benign_correct,
        benign_abstained=benign_abstained,
        specificity_direct=spec_direct,
        entity_accuracy=ent_acc,
        ece_calibration_error=ece,
        semiology_agreement=semiology_agreement,
        mean_latency_ms=mean_latency,
        parse_errors_sample=parse_errors_sample,
    )


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def compare_reports(reports: list[EvalReport], min_malignant_sensitivity: float = 0.95) -> str:
    """Render a Markdown comparison table and flag sensitivity regressions."""
    if not reports:
        return "_Aucun rapport d'évaluation disponible._"

    lines = [
        "| Run | Quant | JSON valide | Abstention | Sens. Maligne | Sens. Triage Sûr | "
        "Sens. Mélanome | Spécificité | Acc. Entité | ECE | Latence (ms) | Statut |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    warnings: list[str] = []
    baseline = reports[0]

    for idx, rep in enumerate(reports):
        status_flags: list[str] = []
        sens_check = rep.malignant_sensitivity_safe_triage
        if sens_check is not None and sens_check < min_malignant_sensitivity:
            status_flags.append("SOUS_SEUIL")
            warnings.append(
                f"- **{rep.run_name} ({rep.quantization})** : sensibilité triage sûr "
                f"({_fmt_pct(sens_check)}) inférieure à l'objectif "
                f"({_fmt_pct(min_malignant_sensitivity)})."
            )

        if (
            idx > 0
            and baseline.malignant_sensitivity_safe_triage is not None
            and sens_check is not None
        ):
            drop = baseline.malignant_sensitivity_safe_triage - sens_check
            if drop > 0.02:
                status_flags.append("RÉGRESSION")
                base_pct = _fmt_pct(baseline.malignant_sensitivity_safe_triage)
                curr_pct = _fmt_pct(sens_check)
                warnings.append(
                    f"- **{rep.run_name} ({rep.quantization})** : régression de "
                    f"{drop * 100:.1f} pts vs baseline {baseline.run_name} "
                    f"({base_pct} -> {curr_pct})."
                )

        status_str = ", ".join(status_flags) if status_flags else "OK"
        ece_str = (
            f"{rep.ece_calibration_error:.4f}" if rep.ece_calibration_error is not None else "N/A"
        )
        lat_str = f"{rep.mean_latency_ms:.1f}" if rep.mean_latency_ms is not None else "N/A"
        lines.append(
            f"| {rep.run_name} | {rep.quantization} | {_fmt_pct(rep.json_valid_rate)} | "
            f"{_fmt_pct(rep.abstention_rate)} | {_fmt_pct(rep.malignant_sensitivity_direct)} | "
            f"{_fmt_pct(rep.malignant_sensitivity_safe_triage)} | "
            f"{_fmt_pct(rep.melanoma_sensitivity_safe_triage)} | "
            f"{_fmt_pct(rep.specificity_direct)} | {_fmt_pct(rep.entity_accuracy)} | "
            f"{ece_str} | {lat_str} | {status_str} |"
        )

    if warnings:
        lines.append("")
        lines.append("### Alertes de régression clinique")
        lines.extend(warnings)

    return "\n".join(lines)
