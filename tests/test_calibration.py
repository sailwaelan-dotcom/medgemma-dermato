"""Unit tests for continuous risk calibration and dermoscopic ROI cropping."""

import numpy as np
from PIL import Image

from medgemma_dermato.eval.calibration import (
    RiskScoreCase,
    apply_temperature,
    compute_auprc,
    compute_auroc,
    compute_binary_ece,
    evaluate_operating_point,
    fit_temperature_scaling,
    render_roc_and_reliability_svg,
    select_threshold_for_sensitivity,
)
from medgemma_dermato.preflight import crop_dermoscopic_roi


def test_auroc_and_auprc_exact_cases() -> None:
    y = [True, True, False, False]
    assert compute_auroc(y, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert compute_auroc(y, [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert compute_auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert compute_auprc(y, [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_temperature_scaling_reduces_overconfidence_ece() -> None:
    # Overconfident model: outputs 0.99 when true frequency is 0.75, 0.01 when true freq is 0.25.
    y_true = [True] * 30 + [False] * 10 + [True] * 10 + [False] * 30
    raw_probs = [0.99] * 40 + [0.01] * 40
    temp = fit_temperature_scaling(y_true, raw_probs)
    assert temp > 2.0
    cal_probs = apply_temperature(raw_probs, temp)
    assert compute_binary_ece(y_true, cal_probs) < compute_binary_ece(y_true, raw_probs)


def test_threshold_selection_and_operating_point() -> None:
    cases = [
        RiskScoreCase(
            isic_id=f"POS_{i}",
            ground_truth_malignant=True,
            ground_truth_is_melanoma=(i < 5),
            ground_truth_entity="melanoma" if i < 5 else "basal_cell_carcinoma",
            predicted_entity_argmax="melanoma",
            p_malignant=0.2 + 0.08 * i,
            p_indeterminate=0.05,
            p_risk=min(1.0, 0.25 + 0.08 * i),
            p_melanoma=0.2,
        )
        for i in range(10)
    ] + [
        RiskScoreCase(
            isic_id=f"NEG_{i}",
            ground_truth_malignant=False,
            ground_truth_is_melanoma=False,
            ground_truth_entity="nevus",
            predicted_entity_argmax="nevus",
            p_malignant=0.02 * i,
            p_indeterminate=0.02,
            p_risk=0.02 + 0.02 * i,
            p_melanoma=0.01,
        )
        for i in range(10)
    ]

    y = [bool(c.ground_truth_malignant) for c in cases]
    s = [c.p_risk for c in cases]
    tau = select_threshold_for_sensitivity(y, s, target_sensitivity=0.90)
    metrics = evaluate_operating_point(cases, "val@90", threshold=tau, temperature=1.0)
    assert metrics.sensitivity >= 0.90
    assert metrics.auroc == 1.0
    assert metrics.specificity == 1.0

    svg = render_roc_and_reliability_svg({"raw_1view": (y, s, s, metrics.auroc)})
    assert svg.startswith("<svg") and "AUROC=1.000" in svg


def test_crop_dermoscopic_roi_crops_vignette_and_preserves_full_frame() -> None:
    # 512x512 image with bright central circular FOV (radius 150) inside black border.
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:512, :512]
    circle = (xx - 256) ** 2 + (yy - 256) ** 2 <= 150**2
    arr[circle] = (180, 140, 130)
    cropped = crop_dermoscopic_roi(Image.fromarray(arr, mode="RGB"))
    assert cropped.width < 380 and cropped.height < 380
    assert cropped.width >= 300 and cropped.height >= 300

    # Borderless full bright image should stay unchanged.
    full = np.full((400, 400, 3), 160, dtype=np.uint8)
    kept = crop_dermoscopic_roi(Image.fromarray(full, mode="RGB"))
    assert kept.size == (400, 400)
