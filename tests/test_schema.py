"""Tests for Pydantic clinical schemas and JSON output parser."""

import json

import pytest
from pydantic import ValidationError

from medgemma_dermato.prompts import build_triage_inference_messages
from medgemma_dermato.schema import (
    DermoscopyAssessment,
    SemiologyCriteria,
    TriageAssessment,
    map_isic_to_entity_category,
    parse_assessment_json,
    parse_triage_json,
    split_thinking_trace,
)


def _valid_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "image_quality_adequate": True,
        "semiology": {
            "asymmetry": "one_axis",
            "pigment_network": "atypical",
            "dots_and_globules": "irregular",
            "streaks_or_pseudopods": "present",
            "blue_white_veil": "absent",
            "vascular_structures": "monomorphic",
            "regression_structures": "absent",
            "color_count": 3,
            "observed_colors": ["light_brown", "dark_brown", "pink"],
        },
        "confidence": 0.85,
        "abstain": False,
        "abstention_reason": None,
        "summary_for_referral_fr": (
            "Réseau pigmentaire atypique focal, avis spécialisé souhaitable."
        ),
        "research_triage_label": "suspicious_malignant_pattern",
    }
    base.update(overrides)
    return base


def test_parse_valid_json_and_fenced_json() -> None:
    raw = json.dumps(_valid_payload())
    parsed, err = parse_assessment_json(raw)
    assert err is None
    assert parsed is not None
    assert parsed.research_triage_label == "suspicious_malignant_pattern"

    fenced = f"Voici l'analyse demandée :\n```json\n{raw}\n```\nFin."
    parsed_fenced, err_fenced = parse_assessment_json(fenced)
    assert err_fenced is None
    assert parsed_fenced is not None
    assert parsed_fenced.confidence == 0.85

    prefixed_braces = f"Note {{preliminary}} avant JSON : {raw}"
    parsed_prefixed, err_prefixed = parse_assessment_json(prefixed_braces)
    assert err_prefixed is None
    assert parsed_prefixed is not None
    assert parsed_prefixed.confidence == 0.85


def test_parse_invalid_json_returns_error() -> None:
    parsed, err = parse_assessment_json("pas de json du tout")
    assert parsed is None
    assert err is not None

    empty_parsed, empty_err = parse_assessment_json("   ")
    assert empty_parsed is None
    assert empty_err is not None


def test_validator_enforces_abstention_invariants() -> None:
    sem = SemiologyCriteria.model_validate(_valid_payload()["semiology"])

    with pytest.raises(ValidationError):
        DermoscopyAssessment(
            image_quality_adequate=False,
            semiology=sem,
            confidence=0.80,
            abstain=False,
            abstention_reason=None,
            summary_for_referral_fr="Test",
            research_triage_label="indeterminate",
        )

    with pytest.raises(ValidationError):
        DermoscopyAssessment(
            image_quality_adequate=True,
            semiology=sem,
            confidence=0.40,
            abstain=False,
            abstention_reason=None,
            summary_for_referral_fr="Test",
            research_triage_label="indeterminate",
        )

    with pytest.raises(ValidationError):
        DermoscopyAssessment(
            image_quality_adequate=True,
            semiology=sem,
            confidence=0.45,
            abstain=True,
            abstention_reason=None,
            summary_for_referral_fr="Test",
            research_triage_label="indeterminate",
        )


def test_parse_medgemma_thinking_trace_and_truncation() -> None:
    raw = json.dumps(_valid_payload())
    with_thought = (
        "<unused94>thought\nLet's draft {bad_json: true} first.\n"
        f"<unused95>```json\n{raw}\n```"
    )
    thought, answer = split_thinking_trace(with_thought)
    assert thought is not None and "bad_json" in thought
    assert raw in answer

    parsed, err = parse_assessment_json(with_thought)
    assert err is None
    assert parsed is not None
    assert parsed.confidence == 0.85

    truncated = "<unused94>thought\nStill reasoning about pigment network..."
    t_parsed, t_err = parse_assessment_json(truncated)
    assert t_parsed is None
    assert t_err is not None and "truncated inside <unused94>thought" in t_err


def test_triage_schema_and_entity_mapping() -> None:
    assert map_isic_to_entity_category("Malignant", "Melanoma", "Melanoma in situ") == "melanoma"
    assert (
        map_isic_to_entity_category(
            "Malignant",
            "Malignant adnexal epithelial proliferations - Follicular",
            "Basal cell carcinoma",
        )
        == "basal_cell_carcinoma"
    )
    assert (
        map_isic_to_entity_category(
            "Malignant",
            "Malignant epidermal proliferations",
            "Squamous cell carcinoma, NOS",
        )
        == "squamous_cell_carcinoma"
    )
    assert map_isic_to_entity_category("Benign", "Benign melanocytic proliferations", "Nevus") == (
        "nevus"
    )
    assert (
        map_isic_to_entity_category(
            "Indeterminate",
            "Indeterminate epidermal proliferations",
            "Solar or actinic keratosis",
        )
        == "actinic_keratosis_or_indeterminate"
    )

    with pytest.raises(ValidationError):
        TriageAssessment(
            image_quality_adequate=True,
            melanocytic_pattern="uncertain",
            entity_category="actinic_keratosis_or_indeterminate",
            research_triage_label="indeterminate",
            confidence=0.70,
            abstain=False,
            abstention_reason=None,
            summary_for_referral_fr="Test",
        )

    valid_triage = json.dumps(
        {
            "image_quality_adequate": True,
            "melanocytic_pattern": "melanocytic",
            "entity_category": "melanoma",
            "research_triage_label": "suspicious_malignant_pattern",
            "confidence": 0.85,
            "abstain": False,
            "abstention_reason": None,
            "summary_for_referral_fr": "Aspect suspect de mélanome.",
        }
    )
    parsed, err = parse_triage_json(valid_triage)
    assert err is None
    assert parsed is not None
    assert parsed.entity_category == "melanoma"

    msgs = build_triage_inference_messages({"age_approx": 55, "sex": "female"})
    user_txt = msgs[1]["content"][1]["text"]
    assert '"properties"' not in user_txt
    assert "$defs" not in user_txt
    for field_name in (
        "image_quality_adequate",
        "melanocytic_pattern",
        "entity_category",
        "research_triage_label",
        "confidence",
        "abstain",
        "abstention_reason",
        "summary_for_referral_fr",
    ):
        assert field_name in user_txt

    template_section = user_txt.split("Format JSON attendu", 1)[1]
    for banned_anchor in (
        "nevus",
        "melanoma",
        "benign_pattern",
        "suspicious_malignant_pattern",
        "0.80",
    ):
        assert banned_anchor not in template_section
