"""Tests for Pydantic clinical schemas and JSON output parser."""

import json

import pytest
from pydantic import ValidationError

from medgemma_dermato.schema import (
    DermoscopyAssessment,
    SemiologyCriteria,
    parse_assessment_json,
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
