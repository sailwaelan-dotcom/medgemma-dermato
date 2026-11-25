"""Unit tests for the zero-PyTorch Edge llama.cpp VLM client and native log-probs parser."""

import math

from PIL import Image

from medgemma_dermato.edge_vlm import (
    build_openai_multimodal_messages,
    estimate_risk_from_triage,
    extract_entity_probs_from_logprobs,
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


def test_extract_entity_probs_from_logprobs_continuous() -> None:
    fake_stream = [
        {"token": '{"entity_category":"', "top_logprobs": []},
        {
            "token": "nev",
            "top_logprobs": [
                {"token": "nev", "logprob": math.log(0.84)},
                {"token": "mel", "logprob": math.log(0.13)},
                {"token": "ben", "logprob": math.log(0.02)},
                {"token": "actin", "logprob": math.log(0.01)},
            ],
        },
        {"token": "us", "top_logprobs": []},
    ]
    extracted = extract_entity_probs_from_logprobs(fake_stream)
    assert extracted is not None
    p_risk, p_mel, ent_probs = extracted
    assert abs(p_mel - 0.13) < 0.005
    assert abs(p_risk - 0.14) < 0.005
    assert ent_probs["nevus"] > 0.80


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
