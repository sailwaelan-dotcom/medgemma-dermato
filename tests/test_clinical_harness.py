"""Unit tests for all 6 safety barriers of the Last-Mile Clinical Safety Harness."""

from medgemma_dermato.clinical_harness import (
    ClinicalContext,
    HarnessPolicy,
    LastMileClinicalHarness,
    evaluate_deployment_gate,
)
from medgemma_dermato.schema import PreflightMetrics, PreflightResult, TriageAssessment


def _dummy_preflight(passed: bool) -> PreflightResult:
    return PreflightResult(
        passed=passed,
        issues=[] if passed else ["blurry"],
        metrics=PreflightMetrics(
            width=1024,
            height=768,
            laplacian_var=120.0 if passed else 25.0,
            mean_brightness=150.0,
            specular_ratio=0.01,
            hair_ratio=0.02,
            border_dark_ratio=0.0,
        ),
        recommendation_fr="OK" if passed else "Flou",
    )


def test_green_zone_clear_agreement_passes_cleanly() -> None:
    harness = LastMileClinicalHarness(HarnessPolicy())
    res = harness.evaluate_case(
        case_id="CASE_GREEN",
        p_sys1_raw=0.05,
        p_medgemma_raw=0.04,
        sys1_top_entity="nevus",
        medgemma_top_entity="nevus",
        preflight=_dummy_preflight(True),
        tta_risk_std=0.02,
        medgemma_summary_fr="Aspect compatible avec un naevus typique. Surveillance habituelle.",
    )
    assert res.triage_zone == "GREEN_LOW_RISK_SURVEILLANCE"
    assert res.requires_dermatology_review is False
    assert res.retake_recommended is False
    assert res.audit.text_contradiction_caught is False


def test_preflight_disagreement_tta_and_clinical_red_flags_block_green() -> None:
    harness = LastMileClinicalHarness(HarnessPolicy())

    # 1. Blurry preflight blocks GREEN even when both models output low risk
    r_pf = harness.evaluate_case(
        case_id="CASE_BLUR",
        p_sys1_raw=0.05,
        p_medgemma_raw=0.05,
        sys1_top_entity="nevus",
        preflight=_dummy_preflight(False),
    )
    assert r_pf.triage_zone == "AMBER_UNCERTAIN_TELEEXPERTISE"
    assert r_pf.retake_recommended is True
    assert "PREFLIGHT_FAILED" in r_pf.audit.triggered_rules

    # 2. Cross-model disagreement blocks GREEN
    r_dis = harness.evaluate_case(
        case_id="CASE_DISAGREE",
        p_sys1_raw=0.01,
        p_medgemma_raw=0.55,
        sys1_top_entity="nevus",
        medgemma_top_entity="melanoma",
        preflight=_dummy_preflight(True),
    )
    assert r_dis.triage_zone == "AMBER_UNCERTAIN_TELEEXPERTISE"
    assert "CROSS_MODEL_DISAGREEMENT" in r_dis.audit.triggered_rules

    # 3. Rotational TTA instability blocks GREEN
    r_tta = harness.evaluate_case(
        case_id="CASE_TTA",
        p_sys1_raw=0.04,
        p_medgemma_raw=0.04,
        sys1_top_entity="nevus",
        tta_risk_std=0.18,
    )
    assert r_tta.triage_zone == "AMBER_UNCERTAIN_TELEEXPERTISE"
    assert "TTA_ROTATIONAL_INSTABILITY" in r_tta.audit.triggered_rules

    # 4. Clinical red flag (evolving lesion) blocks GREEN
    r_clin = harness.evaluate_case(
        case_id="CASE_EVOLVING",
        p_sys1_raw=0.04,
        p_medgemma_raw=0.04,
        sys1_top_entity="nevus",
        clinical_context=ClinicalContext(age_years=62, lesion_evolving=True),
    )
    assert r_clin.triage_zone == "AMBER_UNCERTAIN_TELEEXPERTISE"
    assert "CLINICAL_RED_FLAG_EVOLVING" in r_clin.audit.triggered_rules


def test_text_contradiction_blocker_and_regulatory_sanitizer() -> None:
    harness = LastMileClinicalHarness(HarnessPolicy())
    reassuring_triage = TriageAssessment(
        image_quality_adequate=True,
        melanocytic_pattern="non_melanocytic",
        entity_category="benign_keratosis",
        research_triage_label="benign_pattern",
        confidence=0.80,
        abstain=False,
        abstention_reason=None,
        summary_for_referral_fr=(
            "Diagnostic confirmé de lésion bénigne. Pas d'argument majeur de malignité ; "
            "suivi clinique habituel."
        ),
    )
    res = harness.evaluate_case(
        case_id="CASE_CONTRA",
        p_sys1_raw=0.78,
        p_medgemma_raw=0.45,
        sys1_top_entity="melanoma",
        p_melanoma_sys1=0.62,
        medgemma_top_entity="benign_keratosis",
        medgemma_assessment=reassuring_triage,
    )
    assert res.triage_zone == "RED_PRIORITY_DERMATOLOGY"
    assert res.audit.text_contradiction_caught is True
    assert res.audit.regulatory_sanitized is True
    assert "GARDE-FOU HARNESS" in res.safe_referral_summary_fr
    assert "suivi clinique habituel" not in res.safe_referral_summary_fr.lower()


def test_deployment_gate_pass_and_block() -> None:
    passing = evaluate_deployment_gate(
        {
            "json_valid_rate": 1.0,
            "malignant_safe_sensitivity": 0.957,
            "melanoma_safe_sensitivity": 0.974,
            "green_zone_npv": 0.921,
            "green_zone_benign_specificity": 0.304,
        }
    )
    assert passing.decision == "PASS"

    failing = evaluate_deployment_gate(
        {
            "json_valid_rate": 1.0,
            "malignant_safe_sensitivity": 0.88,
            "melanoma_safe_sensitivity": 0.85,
            "green_zone_npv": 0.84,
            "green_zone_benign_specificity": 0.35,
        }
    )
    assert failing.decision == "BLOCK_DEPLOY"
