"""Tests for the locked evaluation harness and regression reporting."""

import json

from medgemma_dermato.eval.harness import EvalCase, compare_reports, evaluate_predictions


def _make_json_output(
    triage: str,
    confidence: float,
    abstain: bool = False,
    abstention_reason: str | None = None,
    asymmetry: str = "one_axis",
) -> str:
    return json.dumps(
        {
            "image_quality_adequate": True,
            "semiology": {
                "asymmetry": asymmetry,
                "pigment_network": "atypical",
                "dots_and_globules": "irregular",
                "streaks_or_pseudopods": "absent",
                "blue_white_veil": "absent",
                "vascular_structures": "none",
                "regression_structures": "absent",
                "color_count": 2,
                "observed_colors": ["brown", "black"],
            },
            "confidence": confidence,
            "abstain": abstain,
            "abstention_reason": abstention_reason,
            "summary_for_referral_fr": "Synthèse sémiologique de test.",
            "research_triage_label": triage,
        }
    )


def test_evaluate_predictions_metrics_and_calibration() -> None:
    cases = [
        # TP malignant
        EvalCase(
            isic_id="ISIC_1",
            ground_truth_malignant=True,
            ground_truth_semiology={"asymmetry": "one_axis", "color_count": 2},
            raw_model_output=_make_json_output("suspicious_malignant_pattern", 0.90),
            latency_ms=420.0,
        ),
        # Abstained malignant via low confidence threshold (< 0.60)
        EvalCase(
            isic_id="ISIC_2",
            ground_truth_malignant=True,
            ground_truth_semiology={"asymmetry": "two_axes"},
            raw_model_output=_make_json_output("benign_pattern", 0.52),
            latency_ms=400.0,
        ),
        # TN benign
        EvalCase(
            isic_id="ISIC_3",
            ground_truth_malignant=False,
            raw_model_output=_make_json_output("benign_pattern", 0.85),
            latency_ms=380.0,
        ),
        # Invalid JSON on benign case
        EvalCase(
            isic_id="ISIC_4",
            ground_truth_malignant=False,
            raw_model_output="not valid json",
            latency_ms=300.0,
        ),
    ]

    report = evaluate_predictions(
        cases=cases,
        run_name="baseline",
        model_id="google/medgemma-1.5-4b-it",
        quantization="BF16",
        abstention_threshold=0.60,
    )

    assert report.total_cases == 4
    assert report.json_valid_count == 3
    assert report.json_valid_rate == 0.75
    assert report.abstention_count == 1
    assert report.malignant_total == 2
    assert report.malignant_detected == 1
    assert report.malignant_abstained == 1
    assert report.malignant_missed_fn == 0
    assert report.malignant_sensitivity_direct == 0.5
    assert report.malignant_sensitivity_safe_triage == 1.0
    assert report.benign_total == 2
    assert report.benign_correct == 1
    assert report.specificity_direct == 0.5
    assert report.ece_calibration_error is not None
    assert report.semiology_agreement["asymmetry"] == 0.5
    assert report.semiology_agreement["color_count"] == 1.0
    assert report.mean_latency_ms == 375.0
    assert len(report.parse_errors_sample) == 1


def test_compare_reports_flags_regression_and_threshold() -> None:
    baseline = evaluate_predictions(
        cases=[
            EvalCase(
                isic_id="M1",
                ground_truth_malignant=True,
                raw_model_output=_make_json_output("suspicious_malignant_pattern", 0.9),
            )
        ],
        run_name="qlora_bf16",
        model_id="medgemma",
        quantization="BF16",
    )
    degraded = evaluate_predictions(
        cases=[
            EvalCase(
                isic_id="M1",
                ground_truth_malignant=True,
                raw_model_output=_make_json_output("benign_pattern", 0.9),
            )
        ],
        run_name="gguf_q4",
        model_id="medgemma",
        quantization="Q4_K_M",
    )

    md = compare_reports([baseline, degraded], min_malignant_sensitivity=0.95)
    assert "qlora_bf16" in md
    assert "gguf_q4" in md
    assert "SOUS_SEUIL" in md
    assert "RÉGRESSION" in md
