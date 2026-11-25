"""Unit tests for the zero-PyTorch Edge llama.cpp VLM client."""

from PIL import Image

from medgemma_dermato.edge_vlm import (
    build_openai_multimodal_messages,
    estimate_risk_from_triage,
)
from medgemma_dermato.schema import TriageAssessment


def test_build_openai_multimodal_messages_encodes_jpeg_data_uri() -> None:
    img = Image.new("RGB", (1200, 900), color=(160, 120, 110))
    msgs = build_openai_multimodal_messages(img, clinical_context={"age_approx": 65})
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    img_part = msgs[1]["content"][0]
    assert img_part["type"] == "image_url"
    assert img_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "65 ans" in msgs[1]["content"][1]["text"]


def test_estimate_risk_from_triage_proxy() -> None:
    mal = TriageAssessment(
        image_quality_adequate=True,
        melanocytic_pattern="melanocytic",
        entity_category="melanoma",
        research_triage_label="suspicious_malignant_pattern",
        confidence=0.85,
        abstain=False,
        summary_for_referral_fr="Aspect suspect.",
    )
    p_r, p_m = estimate_risk_from_triage(mal)
    assert p_r == 0.85 and p_m == 0.85

    ben = TriageAssessment(
        image_quality_adequate=True,
        melanocytic_pattern="melanocytic",
        entity_category="nevus",
        research_triage_label="benign_pattern",
        confidence=0.80,
        abstain=False,
        summary_for_referral_fr="Naevus typique.",
    )
    p_rb, p_mb = estimate_risk_from_triage(ben)
    assert p_rb == 0.20 and p_mb < 0.05

    p_none, _ = estimate_risk_from_triage(None)
    assert p_none == 0.55
