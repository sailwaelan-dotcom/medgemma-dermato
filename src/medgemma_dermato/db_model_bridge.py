"""Bidirectional Format Validator & Anti-Corruption Bridge between SQLite DB and AI Models.

Enforces strict schema versioning, cryptographic SHA-256 payload integrity on storage/hydration,
PII and prompt-injection stripping before model ingestion, and zero-anchoring isolation (prior
model verdicts are NEVER injected into MedGemma's prompt; only deterministic post-inference
longitudinal rules in the Agent/Harness consume `LongitudinalPriorDTO`).
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from medgemma_dermato.clinical_harness import TriageZone

SCHEMA_VERSION: Literal["2.0"] = "2.0"

_CONTROL_OR_TEMPLATE_RE = re.compile(
    r"(<start_of_turn>|<end_of_turn>|<unused\d+>|<bos>|<eos>|[\x00-\x1f\x7f{}\[\]])",
    re.IGNORECASE,
)


class StorageFormatIntegrityError(ValueError):
    """Raised when a SQLite row fails schema validation or SHA-256 integrity verification."""


def sanitize_prompt_text(text: str | None, max_len: int = 48) -> str | None:
    """Strip prompt-injection tokens, braces, and control characters from free-text fields."""
    if text is None:
        return None
    cleaned = _CONTROL_OR_TEMPLATE_RE.sub(" ", str(text))
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return None
    return cleaned[:max_len]


class ValidatedVLMMetadata(BaseModel):
    """Strict SFT-aligned demographic metadata allowed into MedGemma VLM prompts."""

    age_approx: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["male", "female"] | None = None
    anatom_site: str | None = Field(default=None, max_length=48)

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return ONLY SFT-safe demographic keys (zero PII, zero notes, zero prior verdict)."""
        return {
            "age_approx": self.age_approx,
            "sex": self.sex,
            "anatom_site": self.anatom_site,
        }


class LongitudinalPriorDTO(BaseModel):
    """Validated prior exam summary consumed ONLY by deterministic Agent/Harness rules."""

    prior_case_id: str = Field(min_length=1)
    prior_created_utc: str = Field(min_length=1)
    prior_triage_zone: TriageZone
    prior_calibrated_risk: float = Field(ge=0.0, le=1.0)
    prior_p_sys1_cal: float = Field(ge=0.0, le=1.0)
    prior_entity: str | None = None
    days_since_prior: float = Field(ge=0.0)


class ValidatedStorageEnvelope(BaseModel):
    """Versioned, SHA-256 sealed envelope wrapping payloads persisted in SQLite."""

    schema_version: Literal["2.0"] = SCHEMA_VERSION
    payload_sha256: str = Field(min_length=64, max_length=64)
    case_id: str = Field(min_length=1)
    created_utc: str = Field(min_length=1)
    triage_zone: TriageZone
    routing_decision: Literal[
        "FAST_PATH_SYS1_ONLY",
        "HYBRID_FULL_S1_PLUS_S2",
        "DEGRADED_SYS1_ONLY",
    ]
    calibrated_risk_score: float = Field(ge=0.0, le=1.0)
    payload: dict[str, Any]


def compute_payload_sha256(payload: dict[str, Any]) -> str:
    """Compute canonical deterministic SHA-256 digest over a JSON-serializable payload dict."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_clinical_payload_fields(payload: dict[str, Any]) -> tuple[TriageZone, float, str]:
    if not isinstance(payload, dict):
        raise StorageFormatIntegrityError("Payload must be a JSON object.")
    assessment = payload.get("assessment")
    if not isinstance(assessment, dict):
        raise StorageFormatIntegrityError("Missing 'assessment' object in stored payload.")

    zone = assessment.get("triage_zone")
    if zone not in (
        "GREEN_LOW_RISK_SURVEILLANCE",
        "AMBER_UNCERTAIN_TELEEXPERTISE",
        "RED_PRIORITY_DERMATOLOGY",
    ):
        raise StorageFormatIntegrityError(f"Invalid triage_zone: {zone!r}")

    try:
        risk = float(assessment["calibrated_risk_score"])
        p_s1 = float(assessment["p_sys1_cal"])
        p_s2 = float(assessment["p_medgemma_cal"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StorageFormatIntegrityError(f"Missing or non-numeric risk score: {exc}") from exc

    for name, val in (
        ("calibrated_risk_score", risk),
        ("p_sys1_cal", p_s1),
        ("p_medgemma_cal", p_s2),
    ):
        if not (0.0 <= val <= 1.0):
            raise StorageFormatIntegrityError(f"Score {name} out of bounds [0,1]: {val}")

    case_id = str(payload.get("case_id") or "").strip()
    if not case_id:
        raise StorageFormatIntegrityError("Missing case_id in stored payload.")

    return zone, risk, case_id


def validate_for_storage(payload: dict[str, Any]) -> ValidatedStorageEnvelope:
    """Validate and seal an AgentRunResult payload before persisting to SQLite."""
    clean_payload = dict(payload)
    if clean_payload.get("anatom_site") is not None:
        clean_payload["anatom_site"] = sanitize_prompt_text(
            str(clean_payload["anatom_site"]), max_len=64
        )
    if clean_payload.get("patient_ref") is not None:
        clean_payload["patient_ref"] = sanitize_prompt_text(
            str(clean_payload["patient_ref"]), max_len=64
        )
    if clean_payload.get("lesion_label") is not None:
        clean_payload["lesion_label"] = sanitize_prompt_text(
            str(clean_payload["lesion_label"]), max_len=64
        )

    zone, risk, case_id = _validate_clinical_payload_fields(clean_payload)
    created_utc = str(
        clean_payload.get("created_utc")
        or (clean_payload.get("assessment") or {}).get("audit", {}).get("timestamp_utc")
        or datetime.now(UTC).isoformat()
    )
    routing_raw = clean_payload.get("routing_decision")
    if routing_raw not in (
        "FAST_PATH_SYS1_ONLY",
        "HYBRID_FULL_S1_PLUS_S2",
        "DEGRADED_SYS1_ONLY",
    ):
        raise StorageFormatIntegrityError(f"Invalid routing_decision: {routing_raw!r}")
    routing: Literal["FAST_PATH_SYS1_ONLY", "HYBRID_FULL_S1_PLUS_S2", "DEGRADED_SYS1_ONLY"] = (
        routing_raw
    )
    digest = compute_payload_sha256(clean_payload)
    return ValidatedStorageEnvelope(
        schema_version=SCHEMA_VERSION,
        payload_sha256=digest,
        case_id=case_id,
        created_utc=created_utc,
        triage_zone=zone,
        routing_decision=routing,
        calibrated_risk_score=risk,
        payload=clean_payload,
    )


def hydrate_and_validate_from_storage(raw_json: str) -> dict[str, Any]:
    """Hydrate a SQLite `payload_json` row and verify integrity + schema contract."""
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise StorageFormatIntegrityError(f"Corrupted JSON in SQLite payload_json: {exc}") from exc

    if not isinstance(parsed, dict):
        raise StorageFormatIntegrityError("SQLite payload_json is not a JSON object.")

    if "schema_version" in parsed and "payload_sha256" in parsed and "payload" in parsed:
        try:
            env = ValidatedStorageEnvelope.model_validate(parsed)
        except Exception as exc:
            raise StorageFormatIntegrityError(f"Invalid storage envelope: {exc}") from exc
        actual_digest = compute_payload_sha256(env.payload)
        if actual_digest != env.payload_sha256:
            raise StorageFormatIntegrityError(
                f"SQLite SHA-256 integrity mismatch for {env.case_id}: "
                f"expected={env.payload_sha256[:12]}, got={actual_digest[:12]}"
            )
        _validate_clinical_payload_fields(env.payload)
        return env.payload

    # Legacy v1.0 payload validation (prior to envelope introduction)
    _validate_clinical_payload_fields(parsed)
    return parsed


def build_validated_model_inputs(
    prior_case: dict[str, Any] | None,
    current_age: int | None,
    current_sex: str | None,
    current_site: str | None,
) -> tuple[ValidatedVLMMetadata, LongitudinalPriorDTO | None]:
    """Build sanitized VLM demographics + zero-anchoring LongitudinalPriorDTO from DB row."""
    resolved_age = current_age
    resolved_sex_raw = current_sex
    resolved_site = current_site

    prior_dto: LongitudinalPriorDTO | None = None
    if prior_case is not None:
        a = prior_case.get("assessment") or {}
        _validate_clinical_payload_fields(prior_case)
        if resolved_age is None and prior_case.get("age_years") is not None:
            resolved_age = int(prior_case["age_years"])
        if not resolved_sex_raw and prior_case.get("sex"):
            resolved_sex_raw = str(prior_case["sex"])
        if not resolved_site and prior_case.get("anatom_site"):
            resolved_site = str(prior_case["anatom_site"])

        created_str = str(
            prior_case.get("created_utc") or (a.get("audit") or {}).get("timestamp_utc") or ""
        )
        try:
            t_prior = datetime.fromisoformat(created_str)
            if t_prior.tzinfo is None:
                t_prior = t_prior.replace(tzinfo=UTC)
            days = max(0.0, round((datetime.now(UTC) - t_prior).total_seconds() / 86400.0, 2))
        except ValueError:
            days = 0.0

        prior_dto = LongitudinalPriorDTO(
            prior_case_id=str(prior_case["case_id"]),
            prior_created_utc=created_str or datetime.now(UTC).isoformat(),
            prior_triage_zone=a["triage_zone"],
            prior_calibrated_risk=float(a["calibrated_risk_score"]),
            prior_p_sys1_cal=float(a["p_sys1_cal"]),
            prior_entity=a.get("sys1_top_entity"),
            days_since_prior=days,
        )

    sex_norm: Literal["male", "female"] | None = None
    if resolved_sex_raw:
        low = resolved_sex_raw.strip().lower()
        if low in ("male", "female"):
            sex_norm = low  # type: ignore[assignment]

    meta = ValidatedVLMMetadata(
        age_approx=resolved_age,
        sex=sex_norm,
        anatom_site=sanitize_prompt_text(resolved_site, max_len=48),
    )
    return meta, prior_dto
