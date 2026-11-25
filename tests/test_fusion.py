"""Unit tests for Edge System 1 post-processing and Late Fusion + Safety Disagreement Gate."""

import numpy as np
from PIL import Image

from medgemma_dermato.eval.calibration import RiskScoreCase
from medgemma_dermato.eval.fusion import (
    align_score_cases,
    build_fused_cases,
    evaluate_fusion_operating_point,
    fit_optimal_blend_weight,
    render_hybrid_roc_pr_svg,
    select_fusion_threshold_for_sensitivity,
)
from medgemma_dermato.system1 import (
    ENTITY_LABELS,
    build_tta4_views_pil,
    logits_to_system1_probs,
    preprocess_batch_numpy,
)


def test_system1_preprocessing_and_logits_conversion() -> None:
    img = Image.new("RGB", (512, 384), color=(180, 120, 110))
    views = build_tta4_views_pil(img)
    assert len(views) == 4
    batch = preprocess_batch_numpy(views, image_size=384)
    assert batch.shape == (4, 3, 384, 384)
    assert batch.dtype == np.float32

    mal_logits = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    ent_logits = np.zeros((4, len(ENTITY_LABELS)), dtype=np.float32)
    ent_logits[:, 0] = 5.0  # melanoma
    probs = logits_to_system1_probs(mal_logits, ent_logits)
    assert probs["predicted_entity"] == "melanoma"
    assert probs["p_malignant"] > 0.85
    assert probs["p_risk"] > 0.85
    assert probs["p_melanoma"] > 0.85


def _make_case(isic_id: str, is_mal: bool, p_risk: float, entity: str) -> RiskScoreCase:
    return RiskScoreCase(
        isic_id=isic_id,
        ground_truth_malignant=is_mal,
        ground_truth_is_melanoma=(entity == "melanoma"),
        ground_truth_entity=entity,
        predicted_entity_argmax=entity,
        p_malignant=p_risk,
        p_indeterminate=0.02,
        p_risk=p_risk,
        p_melanoma=p_risk if entity == "melanoma" else 0.01,
    )


def test_late_fusion_blend_and_safety_disagreement_gate() -> None:
    # System 1 is accurate; MedGemma misses one positive badly (disagreement rescue test).
    sys1 = [
        _make_case("P1", True, 0.90, "melanoma"),
        _make_case("P2", True, 0.85, "melanoma"),
        _make_case("P3", True, 0.80, "basal_cell_carcinoma"),
        _make_case("P4", True, 0.72, "melanoma"),  # MedGemma says 0.05 -> disagreement 0.67!
        _make_case("N1", False, 0.10, "nevus"),
        _make_case("N2", False, 0.12, "benign_keratosis"),
        _make_case("N3", False, 0.15, "nevus"),
        _make_case("N4", False, 0.20, "nevus"),
    ]
    med = [
        _make_case("P1", True, 0.88, "melanoma"),
        _make_case("P2", True, 0.75, "melanoma"),
        _make_case("P3", True, 0.60, "basal_cell_carcinoma"),
        _make_case("P4", True, 0.05, "nevus"),
        _make_case("N1", False, 0.15, "nevus"),
        _make_case("N2", False, 0.18, "nevus"),
        _make_case("N3", False, 0.22, "nevus"),
        _make_case("N4", False, 0.25, "nevus"),
    ]
    aligned = align_score_cases(sys1, med)
    assert len(aligned) == 8

    y = [bool(s1.ground_truth_malignant) for s1, _ in aligned]
    w_opt = fit_optimal_blend_weight(
        y,
        [s1.p_risk for s1, _ in aligned],
        [s2.p_risk for _, s2 in aligned],
    )
    assert w_opt > 0.7  # System 1 dominates cleanly

    fused = build_fused_cases(aligned, weight_sys1=0.50)
    # At threshold 0.50 without gate, P4 has p_fused = 0.385 -> FN.
    op_plain = evaluate_fusion_operating_point(fused, "plain", "equal_50_50", 0.50, threshold=0.50)
    assert op_plain.tp == 3 and op_plain.fn == 1 and op_plain.tn == 4

    # With disagreement gate delta=0.40, P4 (disagreement 0.67) is rescued with 0 extra FP!
    op_gate = evaluate_fusion_operating_point(
        fused, "gated", "equal_50_50_gate", 0.50, threshold=0.50, disagreement_delta=0.40
    )
    assert op_gate.tp == 4 and op_gate.fn == 0
    assert op_gate.disagreement_rescued_tp == 1
    assert op_gate.disagreement_extra_fp == 0

    tau_95 = select_fusion_threshold_for_sensitivity(fused, target_sensitivity=0.95)
    assert tau_95 <= 0.3851

    svg = render_hybrid_roc_pr_svg(
        {"Hybrid": (y, [c.p_fused for c in fused], op_plain.auroc, op_plain.auprc)}
    )
    assert svg.startswith("<svg") and "Precision-Recall Curve" in svg
