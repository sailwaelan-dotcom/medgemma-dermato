"""Unit tests for DermoscopyTriageAgent (adaptive routing, ROI recheck, action plans, trace)."""

import numpy as np
from PIL import Image

from medgemma_dermato.agent import (
    AgentConfig,
    DermoscopyTriageAgent,
    simulate_adaptive_routing_on_cohort,
)
from medgemma_dermato.clinical_harness import ClinicalContext, HarnessPolicy
from medgemma_dermato.edge_vlm import EdgeVLMResult
from medgemma_dermato.eval.calibration import RiskScoreCase
from medgemma_dermato.schema import TriageAssessment
from medgemma_dermato.system1 import ENTITY_LABELS, System1Prediction


def _textured_image(size: int = 512, vignette: bool = False) -> Image.Image:
    rng = np.random.default_rng(123)
    base = np.full((size, size, 3), (195.0, 152.0, 132.0), dtype=np.float32)
    texture = rng.normal(0.0, 7.0, size=(size, size, 1))
    arr = np.clip(base + texture, 40, 230).astype(np.uint8)
    if vignette:
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 > int(size * 0.34) ** 2
        arr[mask] = 0
    return Image.fromarray(arr, mode="RGB")


class _MockSystem1:
    def __init__(self, p_risk: float, tta_std: float = 0.01, p_mel: float = 0.01) -> None:
        self.p_risk = p_risk
        self.tta_std = tta_std
        self.p_mel = p_mel
        self.calls = 0

    def predict_image(self, image: Image.Image, tta: bool = True) -> System1Prediction:
        self.calls += 1
        probs = {e: 0.01 for e in ENTITY_LABELS}
        probs["melanoma"] = self.p_mel
        probs["nevus"] = max(0.05, 1.0 - self.p_risk)
        return System1Prediction(
            p_malignant_raw=self.p_risk,
            p_malignant_calibrated=self.p_risk,
            p_risk_raw=self.p_risk,
            p_risk_calibrated=self.p_risk,
            p_indeterminate=0.01,
            p_melanoma=self.p_mel,
            entity_probs=probs,
            predicted_entity="melanoma" if self.p_mel > 0.4 else "nevus",
            tta_views=4 if tta else 1,
            tta_risk_std=self.tta_std,
            tta_risk_range=self.tta_std * 2,
            latency_ms=10.0,
        )


class _SpyVLM:
    def __init__(self, p_risk: float = 0.72) -> None:
        self.p_risk = p_risk
        self.calls = 0

    def healthcheck(self) -> bool:
        return True

    def analyze_image(
        self,
        image: Image.Image,
        clinical_context: dict | None = None,
    ) -> EdgeVLMResult:
        self.calls += 1
        parsed = TriageAssessment(
            image_quality_adequate=True,
            melanocytic_pattern="melanocytic",
            entity_category="melanoma",
            research_triage_label="suspicious_malignant_pattern",
            confidence=0.85,
            abstain=False,
            summary_for_referral_fr="Aspect suspect à adresser.",
        )
        return EdgeVLMResult(
            raw_text=parsed.model_dump_json(),
            parsed=parsed,
            p_risk_estimate=self.p_risk,
            p_melanoma_estimate=self.p_risk,
            logprobs_used=True,
            latency_ms=120.0,
        )


def test_agent_fast_path_skips_vlm_on_ultra_low_risk_clean_case() -> None:
    s1 = _MockSystem1(p_risk=0.02, tta_std=0.01, p_mel=0.01)
    vlm = _SpyVLM(p_risk=0.05)
    agent = DermoscopyTriageAgent(
        system1=s1,
        vlm_client=vlm,
        agent_cfg=AgentConfig(adaptive_routing_enabled=True),
    )

    res = agent.run(_textured_image(512, vignette=False))
    assert res.routing_decision == "FAST_PATH_SYS1_ONLY"
    assert res.vlm_invoked is False
    assert vlm.calls == 0
    assert res.assessment.triage_zone == "GREEN_LOW_RISK_SURVEILLANCE"
    assert len(res.trace) == 6
    assert len(res.action_plan.missing_anamnesis_questions) >= 2


def test_agent_escalates_to_vlm_on_red_flag_or_borderline_and_applies_roi_crop() -> None:
    s1 = _MockSystem1(p_risk=0.35, tta_std=0.04, p_mel=0.22)
    vlm = _SpyVLM(p_risk=0.82)
    agent = DermoscopyTriageAgent(system1=s1, vlm_client=vlm)

    res = agent.run(
        _textured_image(512, vignette=True),
        clinical_context=ClinicalContext(age_years=70, lesion_evolving=True),
    )
    assert res.routing_decision == "HYBRID_FULL_S1_PLUS_S2"
    assert res.vlm_invoked is True
    assert res.roi_crop_applied is True
    assert vlm.calls == 1
    assert s1.calls == 2  # Original + ROI cropped recheck
    assert res.system2_logprobs_used is True
    assert res.assessment.triage_zone == "RED_PRIORITY_DERMATOLOGY"
    assert "FICHE D'ADRESSAGE" in res.action_plan.gp_referral_letter_fr


def test_simulate_adaptive_routing_saves_vlm_compute_without_losing_cancers() -> None:
    s1_cases = [
        RiskScoreCase(
            isic_id="M1",
            ground_truth_malignant=True,
            ground_truth_is_melanoma=True,
            predicted_entity_argmax="melanoma",
            p_malignant=0.75,
            p_indeterminate=0.02,
            p_risk=0.75,
            p_melanoma=0.65,
        ),
        RiskScoreCase(
            isic_id="B1",
            ground_truth_malignant=False,
            ground_truth_is_melanoma=False,
            predicted_entity_argmax="nevus",
            p_malignant=0.02,
            p_indeterminate=0.01,
            p_risk=0.02,
            p_melanoma=0.01,
        ),
    ]
    med_map = {c.isic_id: c for c in s1_cases}
    stats = simulate_adaptive_routing_on_cohort(
        s1_cases, med_map, HarnessPolicy(), AgentConfig(adaptive_routing_enabled=True)
    )
    assert stats["fast_path_count"] == 1
    assert stats["vlm_invoked_count"] == 1
    assert stats["fast_path_vlm_compute_saved_rate"] == 0.5
    assert stats["malignant_safe_sensitivity"] == 1.0
