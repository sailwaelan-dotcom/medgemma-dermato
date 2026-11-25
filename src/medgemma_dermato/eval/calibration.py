"""Continuous risk calibration, ROC/PR evaluation, and zero-leakage threshold selection."""

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

MALIGNANT_ENTITIES: frozenset[str] = frozenset(
    {
        "melanoma",
        "basal_cell_carcinoma",
        "squamous_cell_carcinoma",
    }
)
INDETERMINATE_ENTITIES: frozenset[str] = frozenset({"actinic_keratosis_or_indeterminate"})
BENIGN_ENTITIES: frozenset[str] = frozenset(
    {
        "nevus",
        "benign_keratosis",
        "dermatofibroma",
        "vascular",
        "other_benign",
    }
)

ScoreKey = Literal["p_risk", "p_malignant"]


class RiskScoreCase(BaseModel):
    """Continuous entity & malignancy risk probabilities extracted for one ISIC case."""

    isic_id: str
    ground_truth_malignant: bool | None = None
    ground_truth_is_melanoma: bool | None = None
    ground_truth_entity: str | None = None
    melanocytic_probs: dict[str, float] = Field(default_factory=dict)
    entity_probs: dict[str, float] = Field(default_factory=dict)
    predicted_entity_argmax: str
    p_malignant: float = Field(ge=0.0, le=1.0)
    p_indeterminate: float = Field(ge=0.0, le=1.0)
    p_risk: float = Field(ge=0.0, le=1.0)
    p_melanoma: float = Field(ge=0.0, le=1.0)
    latency_ms: float | None = None


class OperatingPointMetrics(BaseModel):
    """Clinical & calibration metrics at one operating threshold."""

    label: str
    score_key: ScoreKey
    threshold: float
    temperature: float
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
    auroc: float
    auprc: float
    ece_raw: float
    ece_calibrated: float
    brier_raw: float
    brier_calibrated: float


def _clamp_probs(probs: list[float], eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    return np.clip(arr, eps, 1.0 - eps)


def apply_temperature(probs: list[float], temperature: float) -> list[float]:
    """Scale binary logits by 1 / temperature and map back via sigmoid."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    p = _clamp_probs(probs)
    logits = np.log(p / (1.0 - p))
    scaled = logits / float(temperature)
    scaled = np.clip(scaled, -40.0, 40.0)
    cal = 1.0 / (1.0 + np.exp(-scaled))
    return [float(x) for x in cal]


def compute_auroc(y_true: list[bool], y_score: list[float]) -> float:
    """Compute exact tie-aware ROC AUC via Mann-Whitney U / rank sum."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have identical length")
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires at least one positive and one negative sample")

    # Assign average ranks (1-based) for tied scores.
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    pos_rank_sum = float(np.sum(ranks[y]))
    u_stat = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return round(u_stat / (n_pos * n_neg), 4)


def compute_auprc(y_true: list[bool], y_score: list[float]) -> float:
    """Compute exact Average Precision (area under step-wise precision-recall curve)."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have identical length")
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y))
    if n_pos == 0:
        raise ValueError("AUPRC requires at least one positive sample")

    # Sort descending by score; group ties to avoid optimistic tie ordering.
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    sorted_s = s[order]

    ap = 0.0
    tp = 0
    total = 0
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        group_pos = int(np.sum(sorted_y[i:j]))
        tp += group_pos
        total = j
        if group_pos > 0:
            precision = tp / float(total)
            delta_recall = group_pos / float(n_pos)
            ap += precision * delta_recall
        i = j
    return round(ap, 4)


def compute_binary_ece(
    y_true: list[bool],
    probs: list[float],
    num_bins: int = 10,
) -> float:
    """Compute 10-bin Expected Calibration Error for binary malignancy probabilities."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    n = len(y)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(num_bins):
        low = b / num_bins
        high = (b + 1) / num_bins
        mask = (p >= low) & (p <= high if b == num_bins - 1 else p < high)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return round(float(ece), 4)


def compute_brier_score(y_true: list[bool], probs: list[float]) -> float:
    """Compute Brier score (mean squared probability error)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    return round(float(np.mean((p - y) ** 2)), 4)


def fit_temperature_scaling(
    y_true: list[bool],
    raw_probs: list[float],
    t_min: float = 0.15,
    t_max: float = 10.0,
) -> float:
    """Fit scalar temperature T minimizing binary NLL strictly on validation data."""
    y = np.asarray(y_true, dtype=np.float64)
    p = _clamp_probs(raw_probs)
    logits = np.log(p / (1.0 - p))

    def nll(temp: float) -> float:
        scaled = np.clip(logits / temp, -40.0, 40.0)
        # Stable softplus NLL: log(1 + exp(z)) - y*z
        loss = np.logaddexp(0.0, scaled) - y * scaled
        return float(np.mean(loss))

    # Coarse log-spaced grid followed by golden-section refinement around best bracket.
    grid = np.geomspace(t_min, t_max, 80)
    losses = [nll(float(t)) for t in grid]
    best_idx = int(np.argmin(losses))
    lo = float(grid[max(0, best_idx - 1)])
    hi = float(grid[min(len(grid) - 1, best_idx + 1)])

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):
        c = hi - phi * (hi - lo)
        d = lo + phi * (hi - lo)
        if nll(c) < nll(d):
            hi = d
        else:
            lo = c
    return round(0.5 * (lo + hi), 4)


def select_threshold_for_sensitivity(
    y_true: list[bool],
    scores: list[float],
    target_sensitivity: float = 0.95,
) -> float:
    """Select the highest threshold achieving recall >= target_sensitivity on validation data."""
    if not (0.0 < target_sensitivity <= 1.0):
        raise ValueError(f"target_sensitivity must be in (0, 1], got {target_sensitivity}")
    pos_scores = sorted(
        (float(s) for y, s in zip(y_true, scores, strict=True) if y),
        reverse=True,
    )
    if not pos_scores:
        raise ValueError("Cannot select sensitivity threshold without positive examples")
    needed = math.ceil(target_sensitivity * len(pos_scores))
    needed = min(max(needed, 1), len(pos_scores))
    return round(float(pos_scores[needed - 1]), 6)


def _extract_binary_arrays(
    cases: list[RiskScoreCase],
    score_key: ScoreKey,
) -> tuple[list[RiskScoreCase], list[bool], list[float]]:
    binary_cases = [c for c in cases if c.ground_truth_malignant is not None]
    y_true = [bool(c.ground_truth_malignant) for c in binary_cases]
    scores = [float(getattr(c, score_key)) for c in binary_cases]
    return binary_cases, y_true, scores


def evaluate_operating_point(
    cases: list[RiskScoreCase],
    label: str,
    threshold: float,
    temperature: float = 1.0,
    score_key: ScoreKey = "p_risk",
) -> OperatingPointMetrics:
    """Evaluate clinical sensitivity/specificity/calibration at a fixed threshold & temperature."""
    binary_cases, y_true, raw_scores = _extract_binary_arrays(cases, score_key)
    cal_scores = apply_temperature(raw_scores, temperature)

    tp = fn = tn = fp = 0
    mel_total = mel_tp = 0
    for case, is_mal, score in zip(binary_cases, y_true, cal_scores, strict=True):
        flagged = score >= threshold
        if case.ground_truth_is_melanoma:
            mel_total += 1
            if flagged:
                mel_tp += 1
        if is_mal:
            if flagged:
                tp += 1
            else:
                fn += 1
        else:
            if flagged:
                fp += 1
            else:
                tn += 1

    mal_total = tp + fn
    ben_total = tn + fp
    all_cal = apply_temperature([float(getattr(c, score_key)) for c in cases], temperature)
    flagged_all = sum(1 for s in all_cal if s >= threshold)

    sens = round(tp / mal_total, 4) if mal_total > 0 else 0.0
    spec = round(tn / ben_total, 4) if ben_total > 0 else 0.0
    mel_sens = round(mel_tp / mel_total, 4) if mel_total > 0 else None
    ppv = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    npv = round(tn / (tn + fn), 4) if (tn + fn) > 0 else None
    flagged_rate = round(flagged_all / len(cases), 4) if cases else 0.0

    return OperatingPointMetrics(
        label=label,
        score_key=score_key,
        threshold=round(threshold, 6),
        temperature=round(temperature, 4),
        n_binary=len(binary_cases),
        malignant_total=mal_total,
        benign_total=ben_total,
        melanoma_total=mel_total,
        tp=tp,
        fn=fn,
        tn=tn,
        fp=fp,
        melanoma_tp=mel_tp,
        sensitivity=sens,
        melanoma_sensitivity=mel_sens,
        specificity=spec,
        ppv=ppv,
        npv=npv,
        flagged_rate=flagged_rate,
        auroc=compute_auroc(y_true, cal_scores),
        auprc=compute_auprc(y_true, cal_scores),
        ece_raw=compute_binary_ece(y_true, raw_scores),
        ece_calibrated=compute_binary_ece(y_true, cal_scores),
        brier_raw=compute_brier_score(y_true, raw_scores),
        brier_calibrated=compute_brier_score(y_true, cal_scores),
    )


def compute_roc_curve_points(
    y_true: list[bool],
    y_score: list[float],
) -> list[tuple[float, float]]:
    """Compute (FPR, TPR) points from threshold +inf down to -inf."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return [(0.0, 0.0), (1.0, 1.0)]

    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    sorted_s = s[order]

    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    tp = fp = 0
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        g_pos = int(np.sum(sorted_y[i:j]))
        g_neg = (j - i) - g_pos
        tp += g_pos
        fp += g_neg
        pts.append((fp / float(n_neg), tp / float(n_pos)))
        i = j
    return pts


def render_roc_and_reliability_svg(
    mode_curves: dict[str, tuple[list[bool], list[float], list[float], float]],
    title: str = "MedGemma 1.5 4B QLoRA v1 — MSKCC External Hold-Out (n=200)",
) -> str:
    """Render side-by-side ROC curves and reliability diagram as standalone SVG.

    mode_curves maps mode_name -> (y_true, raw_probs, cal_probs, auroc).
    """
    w, h = 960, 440
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" style="background:#ffffff;'
        'font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<text x="{w / 2}" y="28" text-anchor="middle" font-size="15.5" '
        f'font-weight="600" fill="#111827">{title}</text>',
    ]

    # Left panel: ROC curve
    lx0, ly0, pw, ph = 65, 58, 370, 310
    lines.append(
        f'<text x="{lx0 + pw / 2}" y="48" text-anchor="middle" font-size="13" '
        'font-weight="600" fill="#374151">ROC Curve (p_risk)</text>'
    )
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        xx = lx0 + t * pw
        yy = ly0 + (1.0 - t) * ph
        lines.append(
            f'<line x1="{xx:.1f}" y1="{ly0}" x2="{xx:.1f}" y2="{ly0 + ph}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx0 + pw}" y2="{yy:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{xx:.1f}" y="{ly0 + ph + 16}" text-anchor="middle" '
            f'font-size="10.5" fill="#4b5563">{t:.1f}</text>'
        )
        lines.append(
            f'<text x="{lx0 - 8}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-size="10.5" fill="#4b5563">{t:.1f}</text>'
        )
    # 95% sensitivity target line
    y95 = ly0 + (1.0 - 0.95) * ph
    lines.append(
        f'<line x1="{lx0}" y1="{y95:.1f}" x2="{lx0 + pw}" y2="{y95:.1f}" '
        'stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4,3"/>'
    )
    lines.append(
        f'<line x1="{lx0}" y1="{ly0 + ph}" x2="{lx0 + pw}" y2="{ly0}" '
        'stroke="#9ca3af" stroke-width="1" stroke-dasharray="4,4"/>'
    )
    lines.append(
        f'<rect x="{lx0}" y="{ly0}" width="{pw}" height="{ph}" '
        'fill="none" stroke="#9ca3af" stroke-width="1.2"/>'
    )

    for idx, (mode, (y_true, _raw_p, cal_p, auroc)) in enumerate(mode_curves.items()):
        col = colors[idx % len(colors)]
        pts = compute_roc_curve_points(y_true, cal_p)
        poly = " ".join(f"{lx0 + fpr * pw:.1f},{ly0 + (1.0 - tpr) * ph:.1f}" for fpr, tpr in pts)
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.3" points="{poly}"/>')
        leg_y = ly0 + ph - 18 - (len(mode_curves) - 1 - idx) * 18
        lines.append(
            f'<line x1="{lx0 + 165}" y1="{leg_y}" x2="{lx0 + 185}" y2="{leg_y}" '
            f'stroke="{col}" stroke-width="2.5"/>'
        )
        lines.append(
            f'<text x="{lx0 + 192}" y="{leg_y + 4}" font-size="11" fill="#111827">'
            f"{mode} (AUROC={auroc:.3f})</text>"
        )

    lines.append(
        f'<text x="{lx0 + pw / 2}" y="{ly0 + ph + 38}" text-anchor="middle" '
        'font-size="11.5" fill="#374151">False Positive Rate (1 - Specificity)</text>'
    )
    lines.append(
        f'<text x="18" y="{ly0 + ph / 2}" text-anchor="middle" font-size="11.5" '
        f'fill="#374151" transform="rotate(-90 18 {ly0 + ph / 2})">Sensitivity (TPR)</text>'
    )

    # Right panel: Reliability diagram (calibrated probabilities of each mode)
    rx0 = 535
    lines.append(
        f'<text x="{rx0 + pw / 2}" y="48" text-anchor="middle" font-size="13" '
        'font-weight="600" fill="#374151">Reliability Diagram (Temp-Scaled on Val)</text>'
    )
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        xx = rx0 + t * pw
        yy = ly0 + (1.0 - t) * ph
        lines.append(
            f'<line x1="{xx:.1f}" y1="{ly0}" x2="{xx:.1f}" y2="{ly0 + ph}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<line x1="{rx0}" y1="{yy:.1f}" x2="{rx0 + pw}" y2="{yy:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{xx:.1f}" y="{ly0 + ph + 16}" text-anchor="middle" '
            f'font-size="10.5" fill="#4b5563">{t:.1f}</text>'
        )
        lines.append(
            f'<text x="{rx0 - 8}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-size="10.5" fill="#4b5563">{t:.1f}</text>'
        )
    lines.append(
        f'<line x1="{rx0}" y1="{ly0 + ph}" x2="{rx0 + pw}" y2="{ly0}" '
        'stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="4,4"/>'
    )
    lines.append(
        f'<rect x="{rx0}" y="{ly0}" width="{pw}" height="{ph}" '
        'fill="none" stroke="#9ca3af" stroke-width="1.2"/>'
    )

    for idx, (mode, (y_true, _raw_p, cal_p, _auroc)) in enumerate(mode_curves.items()):
        col = colors[idx % len(colors)]
        y_arr = np.asarray(y_true, dtype=np.float64)
        p_arr = np.asarray(cal_p, dtype=np.float64)
        bin_pts: list[tuple[float, float]] = []
        for b in range(5):
            lo, hi = b / 5.0, (b + 1) / 5.0
            mask = (p_arr >= lo) & (p_arr <= hi if b == 4 else p_arr < hi)
            if int(np.sum(mask)) > 0:
                bin_pts.append((float(np.mean(p_arr[mask])), float(np.mean(y_arr[mask]))))
        if bin_pts:
            poly = " ".join(
                f"{rx0 + px * pw:.1f},{ly0 + (1.0 - py) * ph:.1f}" for px, py in bin_pts
            )
            lines.append(
                f'<polyline fill="none" stroke="{col}" stroke-width="2.1" points="{poly}"/>'
            )
            for px, py in bin_pts:
                lines.append(
                    f'<circle cx="{rx0 + px * pw:.1f}" cy="{ly0 + (1.0 - py) * ph:.1f}" '
                    f'r="3.8" fill="{col}"/>'
                )

    lines.append(
        f'<text x="{rx0 + pw / 2}" y="{ly0 + ph + 38}" text-anchor="middle" '
        'font-size="11.5" fill="#374151">Predicted Calibrated Risk Probability</text>'
    )
    lines.append(
        f'<text x="{rx0 - 44}" y="{ly0 + ph / 2}" text-anchor="middle" font-size="11.5" '
        f'fill="#374151" transform="rotate(-90 {rx0 - 44} {ly0 + ph / 2})">'
        "Observed Malignancy Frequency</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines)
