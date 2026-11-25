"""Unit tests for the zero-PyTorch Edge llama.cpp VLM client and native log-probs parser."""

import math

from PIL import Image

from medgemma_dermato.edge_vlm import (
    EdgeVLMClient,
    build_openai_multimodal_messages,
    estimate_risk_from_triage,
    extract_entity_probs_from_logprobs,
    extract_melanocytic_probs_from_logprobs,
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


def test_extract_melanocytic_probs_and_3branch_marginalization() -> None:
    stream = [
        {"token": '{"melanocytic_pattern":"', "top_logprobs": []},
        {
            "token": "non",
            "top_logprobs": [
                {"token": "non", "logprob": math.log(0.60)},
                {"token": "mel", "logprob": math.log(0.38)},
                {"token": "unc", "logprob": math.log(0.02)},
            ],
        },
        {"token": '_melanocytic","entity_category":"', "top_logprobs": []},
        {
            "token": "ben",
            "top_logprobs": [
                {"token": "ben", "logprob": math.log(0.90)},
                {"token": "bas", "logprob": math.log(0.10)},
            ],
        },
    ]
    mel_probs = extract_melanocytic_probs_from_logprobs(stream)
    assert mel_probs is not None
    assert abs(mel_probs["non_melanocytic"] - 0.60) < 1e-4
    assert abs(mel_probs["melanocytic"] - 0.38) < 1e-4

    greedy_json = TriageAssessment(
        image_quality_adequate=True,
        melanocytic_pattern="non_melanocytic",
        entity_category="benign_keratosis",
        research_triage_label="benign_pattern",
        confidence=0.80,
        abstain=False,
        summary_for_referral_fr="Aspect kératosique.",
    ).model_dump_json()

    calls: list[dict] = []

    class _MockResp:
        def __init__(self, data: dict) -> None:
            self._data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._data

    class _MockHTTP:
        def post(self, url: str, json: dict) -> _MockResp:  # noqa: A002
            calls.append(json)
            if len(calls) == 1:
                return _MockResp(
                    {
                        "choices": [
                            {
                                "message": {"content": greedy_json},
                                "logprobs": {"content": stream},
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                    }
                )
            assistant_txt = json["messages"][-1]["content"]
            if '"melanocytic"' in assistant_txt:
                top_lp = [
                    {"token": "mel", "logprob": math.log(0.80)},
                    {"token": "nev", "logprob": math.log(0.20)},
                ]
            else:
                top_lp = [
                    {"token": "act", "logprob": math.log(0.50)},
                    {"token": "nev", "logprob": math.log(0.50)},
                ]
            return _MockResp({"choices": [{"logprobs": {"content": [{"top_logprobs": top_lp}]}}]})

        def close(self) -> None:
            return None

    client = EdgeVLMClient(marginalize_branches=True, branch_min_prob=0.005)
    client._client = _MockHTTP()  # type: ignore[assignment]
    res = client.analyze_image(Image.new("RGB", (64, 64), color=(120, 120, 120)))

    assert res.branches_probed == 3
    assert len(calls) == 3
    # Expected: 0.60*0.10 + 0.38*0.80 + 0.02*0.50 = 0.06 + 0.304 + 0.01 = 0.374
    assert abs(res.p_risk_estimate - 0.374) < 0.005
    assert abs(res.p_melanoma_estimate - 0.304) < 0.005
