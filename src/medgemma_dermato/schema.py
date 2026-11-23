"""Strict Pydantic schemas for pre-analytical QA and semiology outputs."""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

QualityIssue = Literal[
    "low_resolution",
    "blurry",
    "underexposed",
    "overexposed",
    "specular_glare",
    "excessive_hair",
    "poor_framing",
]

PresenceLevel = Literal["absent", "present", "uncertain"]
AsymmetryLevel = Literal["symmetric", "one_axis", "two_axes", "uncertain"]
PigmentNetworkLevel = Literal["typical", "atypical", "absent", "uncertain"]
GlobulesLevel = Literal["regular", "irregular", "absent", "uncertain"]
VascularLevel = Literal["none", "monomorphic", "polymorphic", "arborizing", "uncertain"]
ResearchCategory = Literal["benign_pattern", "suspicious_malignant_pattern", "indeterminate"]
MelanocyticPattern = Literal["melanocytic", "non_melanocytic", "uncertain"]
EntityCategory = Literal[
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "nevus",
    "benign_keratosis",
    "dermatofibroma",
    "vascular",
    "actinic_keratosis_or_indeterminate",
    "other_benign",
]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class PreflightMetrics(BaseModel):
    """Quantitative image quality measurements computed locally on CPU."""

    width: int = Field(ge=1)
    height: int = Field(ge=1)
    laplacian_var: float = Field(ge=0.0)
    mean_brightness: float = Field(ge=0.0, le=255.0)
    specular_ratio: float = Field(ge=0.0, le=1.0)
    hair_ratio: float = Field(ge=0.0, le=1.0)
    border_dark_ratio: float = Field(ge=0.0, le=1.0)


class PreflightResult(BaseModel):
    """Result of pre-analytical quality gate before VLM inference."""

    passed: bool
    issues: list[QualityIssue] = Field(default_factory=list)
    metrics: PreflightMetrics
    recommendation_fr: str


class SemiologyCriteria(BaseModel):
    """Structured dermoscopic semiology criteria (observable features first)."""

    asymmetry: AsymmetryLevel
    pigment_network: PigmentNetworkLevel
    dots_and_globules: GlobulesLevel
    streaks_or_pseudopods: PresenceLevel
    blue_white_veil: PresenceLevel
    vascular_structures: VascularLevel
    regression_structures: PresenceLevel
    color_count: int = Field(ge=1, le=6)
    observed_colors: list[str] = Field(min_length=1)


class DermoscopyAssessment(BaseModel):
    """Contractual VLM output enforcing semiology-first reasoning and abstention."""

    image_quality_adequate: bool
    semiology: SemiologyCriteria
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool
    abstention_reason: str | None = None
    summary_for_referral_fr: str = Field(min_length=1)
    research_triage_label: ResearchCategory

    @model_validator(mode="after")
    def _enforce_safety_invariants(self) -> "DermoscopyAssessment":
        if (not self.image_quality_adequate or self.confidence < 0.5) and not self.abstain:
            raise ValueError(
                "abstain must be True when image_quality_adequate is False or confidence < 0.5"
            )
        if self.abstain and (not self.abstention_reason or not self.abstention_reason.strip()):
            raise ValueError("abstention_reason is required when abstain is True")
        return self


class TriageAssessment(BaseModel):
    """Ground-truth-aligned clinical triage output for SFT and hold-out evaluation."""

    image_quality_adequate: bool
    melanocytic_pattern: MelanocyticPattern
    entity_category: EntityCategory
    research_triage_label: ResearchCategory
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool
    abstention_reason: str | None = None
    summary_for_referral_fr: str = Field(min_length=1)

    @model_validator(mode="after")
    def _enforce_triage_invariants(self) -> "TriageAssessment":
        requires_abstention = (
            not self.image_quality_adequate
            or self.confidence < 0.5
            or self.research_triage_label == "indeterminate"
        )
        if requires_abstention and not self.abstain:
            raise ValueError(
                "abstain must be True when image_quality_adequate is False, "
                "confidence < 0.5, or research_triage_label is indeterminate"
            )
        if self.abstain and (not self.abstention_reason or not self.abstention_reason.strip()):
            raise ValueError("abstention_reason is required when abstain is True")
        return self


def map_isic_to_entity_category(
    diagnosis_1: str | None,
    diagnosis_2: str | None,
    diagnosis_3: str | None,
) -> EntityCategory:
    """Deterministically map ISIC Archive clinical hierarchy to EntityCategory."""
    d1 = (diagnosis_1 or "").strip().lower()
    d2 = (diagnosis_2 or "").strip().lower()
    d3 = (diagnosis_3 or "").strip().lower()

    if "melanoma" in d2 or "melanoma" in d3:
        return "melanoma"
    if "basal cell" in d3 or "adnexal epithelial" in d2:
        return "basal_cell_carcinoma"
    if "squamous cell" in d3 or (d1 == "malignant" and "epidermal" in d2):
        return "squamous_cell_carcinoma"
    if "nevus" in d3 or (d1 == "benign" and "benign melanocytic" in d2):
        return "nevus"
    if (
        any(
            kw in d3
            for kw in ("seborrheic keratosis", "solar lentigo", "lentigo", "lichen planus")
        )
        or (d1 == "benign" and "epidermal" in d2)
        or "flat melanotic" in d2
    ):
        return "benign_keratosis"
    if "dermatofibroma" in d3 or "fibro-histiocytic" in d2:
        return "dermatofibroma"
    if "vascular" in d2 or "vascular" in d3:
        return "vascular"
    if (
        d1 == "indeterminate"
        or "actinic" in d3
        or "atypical melanocytic" in d3
        or "atypical intraepithelial" in d3
    ):
        return "actinic_keratosis_or_indeterminate"
    if d1 == "benign":
        return "other_benign"
    return "actinic_keratosis_or_indeterminate"


_THOUGHT_START = "<unused94>"
_THOUGHT_END = "<unused95>"


def split_thinking_trace(raw_text: str) -> tuple[str | None, str]:
    """Separate MedGemma `<unused94>thought ... <unused95>` trace from final answer."""
    if _THOUGHT_END in raw_text:
        thought_part, answer_part = raw_text.rsplit(_THOUGHT_END, 1)
        cleaned_thought = thought_part
        if cleaned_thought.startswith(_THOUGHT_START):
            cleaned_thought = cleaned_thought[len(_THOUGHT_START) :]
        if cleaned_thought.lstrip().startswith("thought"):
            cleaned_thought = cleaned_thought.lstrip()[len("thought") :].lstrip()
        return cleaned_thought.strip(), answer_part.strip()
    if _THOUGHT_START in raw_text:
        cleaned_thought = raw_text.split(_THOUGHT_START, 1)[1]
        if cleaned_thought.lstrip().startswith("thought"):
            cleaned_thought = cleaned_thought.lstrip()[len("thought") :].lstrip()
        return cleaned_thought.strip(), ""
    return None, raw_text


def _iter_json_objects(text: str) -> list[str]:
    """Scan all `{` positions and return balanced JSON object substrings."""
    decoder = json.JSONDecoder()
    objects: list[str] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(text[start:end])
        idx = end
    return objects


def _parse_model_json[TModel: BaseModel](
    raw_text: str,
    model_cls: type[TModel],
) -> tuple[TModel | None, str | None]:
    """Parse and validate raw model output into a target Pydantic schema."""
    if not raw_text or not raw_text.strip():
        return None, "Empty model output"

    thought, answer = split_thinking_trace(raw_text)
    if thought is not None and not answer.strip():
        return (
            None,
            "Model output truncated inside <unused94>thought block "
            "(increase max_new_tokens or use --disable-thinking)",
        )

    candidates: list[str] = []
    seen: set[str] = set()

    def _add_candidate(item: str) -> None:
        if item and item not in seen:
            seen.add(item)
            candidates.append(item)

    for match in _FENCE_RE.findall(answer):
        stripped = match.strip()
        if stripped:
            for obj_str in _iter_json_objects(stripped):
                _add_candidate(obj_str)
            _add_candidate(stripped)

    for obj_str in _iter_json_objects(answer):
        _add_candidate(obj_str)

    _add_candidate(answer.strip())

    last_error = "No JSON object found in model output"
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            return model_cls.model_validate_json(candidate), None
        except ValidationError as exc:
            last_error = str(exc)

    return None, last_error


def parse_assessment_json(raw_text: str) -> tuple[DermoscopyAssessment | None, str | None]:
    """Parse and validate raw model output into DermoscopyAssessment."""
    return _parse_model_json(raw_text, DermoscopyAssessment)


def parse_triage_json(raw_text: str) -> tuple[TriageAssessment | None, str | None]:
    """Parse and validate raw model output into TriageAssessment."""
    return _parse_model_json(raw_text, TriageAssessment)
