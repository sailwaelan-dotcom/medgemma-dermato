"""Zero-leakage Late Fusion & Safety Disagreement Gate combining System 1 + MedGemma."""

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from medgemma_dermato.eval.calibration import (
    RiskScoreCase,
    ScoreKey,
    apply_temperature,
    compute_auprc,
    compute_auroc,
    compute_binary_ece,
    compute_brier_score,
    compute_roc_curve_points,
)


class FusedCase(BaseModel):
    """Aligned System 1 + System 2 continuous risk scores for one ISIC case."""

    isic_id: str
    ground_truth_malignant: bool | None = None
    ground_truth_is_melanoma: bool | None = None
    ground_truth_entity: str | None = None
    p_sys1_raw: float = Field(ge=0.0, le=1.0)
    p_sys1_cal: float = Field(ge=0.0, le=1.0)
    p_medgemma_raw: float = Field(ge=0.0, le=1.0)
    p_medgemma_cal: float = Field(ge=0.0, le=1.0)
    p_fused: float = Field(ge=0.0, le=1.0)
    disagreement: float = Field(ge=0.0, le=1.0)
    sys1_predicted_entity: str
    medgemma_predicted_entity: str


class FusionOperatingPoint(BaseModel):
    """Clinical, calibration, and safety-gate metrics at one fusion operating point."""

    label: str
    policy_name: str
    weight_sys1: float = Field(ge=0.0, le=1.0)
    threshold: float
    disagreement_delta: float | None = None
    n_binary: int
    malignant_total: int
    benign_total: int
    melanoma_total: int
    tp: int
    fn: int
    tn: int
    fp: int
    melanoma_tp: int
    sensitivity: float
    melanoma_sensitivity: float | None = None
    specificity: float
    ppv: float | None = None
    npv: float | None = None
    flagged_rate: float
    disagreement_triggered_count: int = 0
    disagreement_rescued_tp: int = 0
    disagreement_extra_fp: int = 0
    auroc: float
    auprc: float
    ece: float
    brier: float


def align_score_cases(
    sys1_cases: list[RiskScoreCase],
    medgemma_cases: list[RiskScoreCase],
) -> list[tuple[RiskScoreCase, RiskScoreCase]]:
    """Inner-join System 1 and MedGemma score lists by isic_id with ground-truth checks."""
    med_by_id = {c.isic_id: c for c in medgemma_cases}
    aligned: list[tuple[RiskScoreCase, RiskScoreCase]] = []
    for s1 in sys1_cases:
        s2 = med_by_id.get(s1.isic_id)
        if s2 is None:
            continue
        if s1.ground_truth_malignant != s2.ground_truth_malignant:
            raise ValueError(
                f"Ground-truth malignancy mismatch for {s1.isic_id}: "
                f"{s1.ground_truth_malignant} vs {s2.ground_truth_malignant}"
            )
        aligned.append((s1, s2))
    if not aligned:
        raise ValueError("No overlapping isic_id found between System 1 and MedGemma cases")
    return aligned


def fit_optimal_blend_weight(
    y_true: list[bool],
    p_sys1_cal: list[float],
    p_medgemma_cal: list[float],
    grid_steps: int = 101,
) -> float:
    """Find blend weight w in [0, 1] minimizing binary NLL of w*p_sys1 + (1-w)*p_medgemma."""
    if not (len(y_true) == len(p_sys1_cal) == len(p_medgemma_cal)):
        raise ValueError("Input lists must have identical lengths")
    if not y_true:
        raise ValueError("y_true must not be empty")

    y = np.asarray(y_true, dtype=np.float64)
    p1 = np.clip(np.asarray(p_sys1_cal, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p2 = np.clip(np.asarray(p_medgemma_cal, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    best_w = 0.5
    best_nll = float("inf")
    for w in np.linspace(0.0, 1.0, grid_steps):
        pf = np.clip(w * p1 + (1.0 - w) * p2, 1e-6, 1.0 - 1e-6)
        nll = float(-np.mean(y * np.log(pf) + (1.0 - y) * np.log(1.0 - pf)))
        if nll < best_nll:
            best_nll = nll
            best_w = float(w)
    return round(best_w, 4)


def build_fused_cases(
    aligned_pairs: list[tuple[RiskScoreCase, RiskScoreCase]],
    weight_sys1: float,
    temp_sys1: float = 1.0,
    temp_medgemma: float = 1.0,
    sys1_score_key: ScoreKey = "p_risk",
    medgemma_score_key: ScoreKey = "p_risk",
    combine_mode: Literal["prob", "logit"] = "prob",
) -> list[FusedCase]:
    """Build fused probabilities from temperature-calibrated System 1 and MedGemma scores."""
    if not (0.0 <= weight_sys1 <= 1.0):
        raise ValueError(f"weight_sys1 must be in [0, 1], got {weight_sys1}")

    raw_s1 = [float(getattr(s1, sys1_score_key)) for s1, _ in aligned_pairs]
    raw_s2 = [float(getattr(s2, medgemma_score_key)) for _, s2 in aligned_pairs]
    cal_s1 = apply_temperature(raw_s1, temp_sys1)
    cal_s2 = apply_temperature(raw_s2, temp_medgemma)

    fused: list[FusedCase] = []
    for (s1, s2), r1, c1, r2, c2 in zip(
        aligned_pairs, raw_s1, cal_s1, raw_s2, cal_s2, strict=True
    ):
        if combine_mode == "logit":
            c1_clamped = min(max(c1, 1e-6), 1.0 - 1e-6)
            c2_clamped = min(max(c2, 1e-6), 1.0 - 1e-6)
            z = weight_sys1 * math.log(c1_clamped / (1.0 - c1_clamped)) + (
                1.0 - weight_sys1
            ) * math.log(c2_clamped / (1.0 - c2_clamped))
            z = min(max(z, -40.0), 40.0)
            p_f = 1.0 / (1.0 + math.exp(-z))
        else:
            p_f = weight_sys1 * c1 + (1.0 - weight_sys1) * c2

        fused.append(
            FusedCase(
                isic_id=s1.isic_id,
                ground_truth_malignant=s1.ground_truth_malignant,
                ground_truth_is_melanoma=s1.ground_truth_is_melanoma,
                ground_truth_entity=s1.ground_truth_entity,
                p_sys1_raw=round(r1, 6),
                p_sys1_cal=round(c1, 6),
                p_medgemma_raw=round(r2, 6),
                p_medgemma_cal=round(c2, 6),
                p_fused=round(min(max(p_f, 0.0), 1.0), 6),
                disagreement=round(abs(c1 - c2), 6),
                sys1_predicted_entity=s1.predicted_entity_argmax,
                medgemma_predicted_entity=s2.predicted_entity_argmax,
            )
        )
    return fused


def select_fusion_threshold_for_sensitivity(
    val_cases: list[FusedCase],
    target_sensitivity: float = 0.95,
    disagreement_delta: float | None = None,
) -> float:
    """Select highest fused threshold achieving recall >= target_sensitivity on val."""
    if not (0.0 < target_sensitivity <= 1.0):
        raise ValueError(f"target_sensitivity must be in (0, 1], got {target_sensitivity}")

    positives = [c for c in val_cases if c.ground_truth_malignant is True]
    if not positives:
        raise ValueError("Cannot select fusion threshold without positive validation cases")

    needed = min(max(math.ceil(target_sensitivity * len(positives)), 1), len(positives))
    if disagreement_delta is None:
        scores = sorted((c.p_fused for c in positives), reverse=True)
        return round(float(scores[needed - 1]), 6)

    # Positives already rescued by the disagreement gate are automatically flagged regardless
    # of threshold tau; solve for highest tau covering the remaining required positives.
    gate_rescued = [c for c in positives if c.disagreement >= disagreement_delta]
    remaining_needed = needed - len(gate_rescued)
    if remaining_needed <= 0:
        return 1.0
    uncovered_scores = sorted(
        (c.p_fused for c in positives if c.disagreement < disagreement_delta),
        reverse=True,
    )
    idx = min(remaining_needed - 1, len(uncovered_scores) - 1)
    return round(float(uncovered_scores[idx]), 6)


def evaluate_fusion_operating_point(
    cases: list[FusedCase],
    label: str,
    policy_name: str,
    weight_sys1: float,
    threshold: float,
    disagreement_delta: float | None = None,
) -> FusionOperatingPoint:
    """Evaluate fusion & optional Safety Disagreement Gate at a locked threshold."""
    binary = [c for c in cases if c.ground_truth_malignant is not None]
    y_true = [bool(c.ground_truth_malignant) for c in binary]
    p_fused = [c.p_fused for c in binary]

    tp = fn = tn = fp = 0
    mel_total = mel_tp = 0
    dis_triggered = dis_rescued_tp = dis_extra_fp = 0

    for c in binary:
        by_score = c.p_fused >= threshold
        by_dis = disagreement_delta is not None and c.disagreement >= disagreement_delta
        flagged = by_score or by_dis
        if by_dis:
            dis_triggered += 1
        if c.ground_truth_is_melanoma:
            mel_total += 1
            if flagged:
                mel_tp += 1
        if c.ground_truth_malignant:
            if flagged:
                tp += 1
                if by_dis and not by_score:
                    dis_rescued_tp += 1
            else:
                fn += 1
        else:
            if flagged:
                fp += 1
                if by_dis and not by_score:
                    dis_extra_fp += 1
            else:
                tn += 1

    flagged_all = sum(
        1
        for c in cases
        if (c.p_fused >= threshold)
        or (disagreement_delta is not None and c.disagreement >= disagreement_delta)
    )
    mal_total = tp + fn
    ben_total = tn + fp

    return FusionOperatingPoint(
        label=label,
        policy_name=policy_name,
        weight_sys1=round(weight_sys1, 4),
        threshold=round(threshold, 6),
        disagreement_delta=disagreement_delta,
        n_binary=len(binary),
        malignant_total=mal_total,
        benign_total=ben_total,
        melanoma_total=mel_total,
        tp=tp,
        fn=fn,
        tn=tn,
        fp=fp,
        melanoma_tp=mel_tp,
        sensitivity=round(tp / mal_total, 4) if mal_total > 0 else 0.0,
        melanoma_sensitivity=round(mel_tp / mel_total, 4) if mel_total > 0 else None,
        specificity=round(tn / ben_total, 4) if ben_total > 0 else 0.0,
        ppv=round(tp / (tp + fp), 4) if (tp + fp) > 0 else None,
        npv=round(tn / (tn + fn), 4) if (tn + fn) > 0 else None,
        flagged_rate=round(flagged_all / len(cases), 4) if cases else 0.0,
        disagreement_triggered_count=dis_triggered,
        disagreement_rescued_tp=dis_rescued_tp,
        disagreement_extra_fp=dis_extra_fp,
        auroc=compute_auroc(y_true, p_fused),
        auprc=compute_auprc(y_true, p_fused),
        ece=compute_binary_ece(y_true, p_fused),
        brier=compute_brier_score(y_true, p_fused),
    )


def _compute_pr_curve_points(
    y_true: list[bool],
    y_score: list[float],
) -> list[tuple[float, float]]:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return [(0.0, 1.0), (1.0, 0.0)]
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    sorted_s = s[order]
    pts: list[tuple[float, float]] = [(0.0, 1.0)]
    tp = 0
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        tp += int(np.sum(sorted_y[i:j]))
        pts.append((tp / float(n_pos), tp / float(j)))
        i = j
    return pts


def render_hybrid_roc_pr_svg(
    curves: dict[str, tuple[list[bool], list[float], float, float]],
    title: str = "MSKCC External Hold-Out (n=200) — System 1 vs MedGemma vs Hybrid Fusion",
) -> str:
    """Render side-by-side ROC & Precision-Recall comparison curves as standalone SVG.

    curves maps display_name -> (y_true, scores, auroc, auprc).
    """
    w, h = 980, 450
    palette = ["#64748b", "#2563eb", "#059669", "#dc2626", "#7c3aed"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" style="background:#ffffff;'
        'font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<text x="{w / 2}" y="28" text-anchor="middle" font-size="15" '
        f'font-weight="600" fill="#111827">{title}</text>',
    ]

    pw, ph, ly0 = 375, 315, 58
    lx0, rx0 = 65, 545

    def _draw_grid(x0: int, panel_title: str, x_label: str, y_label: str) -> None:
        lines.append(
            f'<text x="{x0 + pw / 2}" y="48" text-anchor="middle" font-size="13" '
            f'font-weight="600" fill="#374151">{panel_title}</text>'
        )
        for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            xx = x0 + t * pw
            yy = ly0 + (1.0 - t) * ph
            lines.append(
                f'<line x1="{xx:.1f}" y1="{ly0}" x2="{xx:.1f}" y2="{ly0 + ph}" '
                'stroke="#e5e7eb" stroke-width="1"/>'
            )
            lines.append(
                f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" '
                'stroke="#e5e7eb" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{xx:.1f}" y="{ly0 + ph + 16}" text-anchor="middle" '
                f'font-size="10.5" fill="#4b5563">{t:.1f}</text>'
            )
            lines.append(
                f'<text x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" '
                f'font-size="10.5" fill="#4b5563">{t:.1f}</text>'
            )
        lines.append(
            f'<rect x="{x0}" y="{ly0}" width="{pw}" height="{ph}" '
            'fill="none" stroke="#9ca3af" stroke-width="1.2"/>'
        )
        lines.append(
            f'<text x="{x0 + pw / 2}" y="{ly0 + ph + 38}" text-anchor="middle" '
            f'font-size="11.5" fill="#374151">{x_label}</text>'
        )
        lines.append(
            f'<text x="{x0 - 44}" y="{ly0 + ph / 2}" text-anchor="middle" font-size="11.5" '
            f'fill="#374151" transform="rotate(-90 {x0 - 44} {ly0 + ph / 2})">{y_label}</text>'
        )

    _draw_grid(lx0, "ROC Curve", "False Positive Rate (1 - Specificity)", "Sensitivity (TPR)")
    _draw_grid(rx0, "Precision-Recall Curve", "Recall (Sensitivity)", "Precision (PPV)")

    # Reference lines
    y95 = ly0 + (1.0 - 0.95) * ph
    lines.append(
        f'<line x1="{lx0}" y1="{y95:.1f}" x2="{lx0 + pw}" y2="{y95:.1f}" '
        'stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4,3"/>'
    )
    lines.append(
        f'<line x1="{lx0}" y1="{ly0 + ph}" x2="{lx0 + pw}" y2="{ly0}" '
        'stroke="#9ca3af" stroke-width="1" stroke-dasharray="4,4"/>'
    )

    items = list(curves.items())
    for idx, (name, (y_true, scores, auroc, auprc)) in enumerate(items):
        col = palette[idx % len(palette)]
        roc_pts = compute_roc_curve_points(y_true, scores)
        roc_poly = " ".join(
            f"{lx0 + fpr * pw:.1f},{ly0 + (1.0 - tpr) * ph:.1f}" for fpr, tpr in roc_pts
        )
        lines.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="2.4" points="{roc_poly}"/>'
        )

        pr_pts = _compute_pr_curve_points(y_true, scores)
        pr_poly = " ".join(
            f"{rx0 + rec * pw:.1f},{ly0 + (1.0 - prec) * ph:.1f}" for rec, prec in pr_pts
        )
        lines.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="2.4" points="{pr_poly}"/>'
        )

        leg_y = ly0 + ph - 14 - (len(items) - 1 - idx) * 18
        lines.append(
            f'<line x1="{lx0 + 115}" y1="{leg_y}" x2="{lx0 + 135}" y2="{leg_y}" '
            f'stroke="{col}" stroke-width="2.6"/>'
        )
        lines.append(
            f'<text x="{lx0 + 142}" y="{leg_y + 4}" font-size="10.8" fill="#111827">'
            f"{name} (AUC={auroc:.3f})</text>"
        )

        lines.append(
            f'<line x1="{rx0 + 14}" y1="{leg_y}" x2="{rx0 + 34}" y2="{leg_y}" '
            f'stroke="{col}" stroke-width="2.6"/>'
        )
        lines.append(
            f'<text x="{rx0 + 41}" y="{leg_y + 4}" font-size="10.8" fill="#111827">'
            f"{name} (AP={auprc:.3f})</text>"
        )

    lines.append("</svg>")
    return "\n".join(lines)
