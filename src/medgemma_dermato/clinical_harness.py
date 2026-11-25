"""Last-Mile Clinical Safety Harness around System 1 (ONNX) + System 2 (MedGemma).

Implements six deterministic, auditable safety barriers designed for Hors-MDR / RUO
decision-support workflows:
  1. Pre-analytical image quality gate (preflight QA)
  2. Calibrated 3-zone routing (GREEN / AMBER / RED) with disagreement, TTA instability,
     and melanoma sub-score escalation guards
  3. Text <-> Probability contradiction blocker + Hors-MDR / RUO phrasing sanitizer
  4. Deterministic clinical red-flag rules (ABCDE Evolution, bleeding, high-risk history)
  5. Immutable per-case AuditTrail JSON
  6. CI/CD deployment regression gate (policy-as-code)
"""

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from medgemma_dermato.eval.calibration import apply_temperature
from medgemma_dermato.schema import PreflightResult, TriageAssessment

TriageZone = Literal[
    "GREEN_LOW_RISK_SURVEILLANCE",
    "AMBER_UNCERTAIN_TELEEXPERTISE",
    "RED_PRIORITY_DERMATOLOGY",
]
RuleSeverity = Literal["info", "warning", "block_green", "force_red"]

RUO_DISCLAIMER_FR = (
    "OUTIL DE RECHERCHE / PRE-TRIAGE HORS-MDR (RUO) — Ne constitue pas un diagnostic "
    "medical ni une prescription. La decision clinique appartient exclusivement au medecin."
)

_REASSURANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"pas\s+d['’]argument\s+majeur\s+de\s+malignit", re.IGNORECASE),
    re.compile(r"suivi\s+clinique\s+habituel", re.IGNORECASE),
    re.compile(r"sans\s+signe\s+de\s+malignit", re.IGNORECASE),
    re.compile(r"aspect\s+(?:clairement\s+)?rassurant", re.IGNORECASE),
    re.compile(r"l[eé]sion\s+(?:clairement\s+)?b[eé]nigne", re.IGNORECASE),
    re.compile(r"b[eé]nin\s*\(", re.IGNORECASE),
)

_PROHIBITED_MDR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdiagnostic\s+confirm[eé]\b", re.IGNORECASE), "orientation semiologique"),
    (re.compile(r"\bdiagnostic\s+certain\b", re.IGNORECASE), "orientation semiologique"),
    (re.compile(r"\bex[eé]r[eè]se\s+obligatoire\b", re.IGNORECASE), "evaluation dermatologique"),
    (re.compile(r"\bprescrire\b", re.IGNORECASE), "discuter"),
    (re.compile(r"\bvous\s+avez\s+un\s+cancer\b", re.IGNORECASE), "aspect suspect a verifier"),
)


class ClinicalContext(BaseModel):
    """Optional clinician-entered patient metadata and clinical red flags."""

    age_years: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = None
    anatom_site: str | None = None
    lesion_evolving: bool = False
    lesion_bleeding_or_ulcerated: bool = False
    new_lesion_after_50: bool = False
    personal_or_family_melanoma_history: bool = False


class HarnessPolicy(BaseModel):
    """Locked validation-calibrated parameters and deterministic safety guardrails."""

    policy_version: str = "last-mile-v1.0"
    sys1_model_id: str = "convnext_tiny_5center_fp32_onnx"
    sys2_model_id: str = "medgemma-1.5-4b-it-qlora-v1"
    weight_sys1: float = Field(default=0.50, ge=0.0, le=1.0)
    temp_sys1: float = Field(default=1.2108, gt=0.0)
    temp_medgemma: float = Field(default=1.3185, gt=0.0)
    tau_green_max: float = Field(default=0.175314, ge=0.0, le=1.0)
    tau_red_min: float = Field(default=0.50, ge=0.0, le=1.0)
    disagreement_delta: float = Field(default=0.35, ge=0.0, le=1.0)
    max_tta_std_for_green: float = Field(default=0.12, ge=0.0, le=1.0)
    melanoma_amber_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    melanoma_red_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    strict_preflight_block: bool = True


class SafetyRuleTrigger(BaseModel):
    """One deterministic safety barrier fired during last-mile evaluation."""

    code: str
    severity: RuleSeverity
    detail_fr: str


class AuditTrail(BaseModel):
    """Immutable audit record attached to every clinical harness inference."""

    harness_version: str = "1.0.0"
    policy_version: str
    timestamp_utc: str
    case_id: str
    image_sha256: str | None = None
    sys1_model_id: str
    sys2_model_id: str
    preflight_passed: bool | None = None
    preflight_issues: list[str] = Field(default_factory=list)
    p_sys1_raw: float
    p_sys1_cal: float
    p_medgemma_raw: float
    p_medgemma_cal: float
    p_fused: float
    disagreement: float
    tta_risk_std: float | None = None
    text_contradiction_caught: bool = False
    regulatory_sanitized: bool = False
    triggered_rules: list[str] = Field(default_factory=list)


class LastMileAssessment(BaseModel):
    """Final clinician-facing output produced by the Last-Mile Clinical Safety Harness."""

    case_id: str
    triage_zone: TriageZone
    requires_dermatology_review: bool
    retake_recommended: bool
    calibrated_risk_score: float = Field(ge=0.0, le=1.0)
    p_sys1_cal: float = Field(ge=0.0, le=1.0)
    p_medgemma_cal: float = Field(ge=0.0, le=1.0)
    disagreement: float = Field(ge=0.0, le=1.0)
    tta_risk_std: float | None = None
    sys1_top_entity: str
    medgemma_top_entity: str | None = None
    safe_referral_summary_fr: str
    ruo_disclaimer_fr: str = RUO_DISCLAIMER_FR
    triggered_rules: list[SafetyRuleTrigger] = Field(default_factory=list)
    audit: AuditTrail


class DeploymentGateCheck(BaseModel):
    """Single policy-as-code check for model release / quantization non-regression."""

    name: str
    passed: bool
    observed: float
    required: str


class DeploymentGateVerdict(BaseModel):
    """Release gate verdict blocking unsafe deployments or quantization regressions."""

    decision: Literal["PASS", "BLOCK_DEPLOY"]
    checks: list[DeploymentGateCheck]


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hex digest of an image file for immutable audit trails."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _contains_reassurance(text: str) -> bool:
    return any(pat.search(text) for pat in _REASSURANCE_PATTERNS)


def _sanitize_regulatory_text(text: str) -> tuple[str, bool]:
    sanitized = text
    changed = False
    for pat, repl in _PROHIBITED_MDR_PATTERNS:
        new_text = pat.sub(repl, sanitized)
        if new_text != sanitized:
            sanitized = new_text
            changed = True
    return sanitized, changed


class LastMileClinicalHarness:
    """Deterministic clinical safety wrapper orchestrating System 1 + System 2 outputs."""

    def __init__(self, policy: HarnessPolicy | None = None) -> None:
        self.policy = policy or HarnessPolicy()

    def evaluate_case(
        self,
        case_id: str,
        p_sys1_raw: float,
        p_medgemma_raw: float,
        sys1_top_entity: str,
        p_melanoma_sys1: float = 0.0,
        p_melanoma_medgemma: float = 0.0,
        medgemma_top_entity: str | None = None,
        medgemma_assessment: TriageAssessment | None = None,
        medgemma_summary_fr: str | None = None,
        preflight: PreflightResult | None = None,
        tta_risk_std: float | None = None,
        clinical_context: ClinicalContext | None = None,
        image_sha256: str | None = None,
    ) -> LastMileAssessment:
        """Apply all 6 safety barriers and return a 3-zone auditable clinical assessment."""
        pol = self.policy
        p_s1_cal = apply_temperature([p_sys1_raw], pol.temp_sys1)[0]
        p_s2_cal = apply_temperature([p_medgemma_raw], pol.temp_medgemma)[0]
        raw_blend = pol.weight_sys1 * p_s1_cal + (1.0 - pol.weight_sys1) * p_s2_cal
        p_fused = min(max(raw_blend, 0.0), 1.0)
        disagreement = abs(p_s1_cal - p_s2_cal)

        triggers: list[SafetyRuleTrigger] = []
        retake_recommended = False

        # Base zone strictly from validation-calibrated fusion risk
        if p_fused >= pol.tau_red_min:
            zone: TriageZone = "RED_PRIORITY_DERMATOLOGY"
            triggers.append(
                SafetyRuleTrigger(
                    code="FUSION_HIGH_RISK",
                    severity="force_red",
                    detail_fr=(
                        f"Score fusionne calibre eleve ({p_fused:.1%} >= {pol.tau_red_min:.1%}) "
                        ": adressage dermatologique prioritaire."
                    ),
                )
            )
        elif p_fused >= pol.tau_green_max:
            zone = "AMBER_UNCERTAIN_TELEEXPERTISE"
            triggers.append(
                SafetyRuleTrigger(
                    code="FUSION_INTERMEDIATE_RISK",
                    severity="block_green",
                    detail_fr=(
                        f"Score fusionne en zone grise ({p_fused:.1%} >= {pol.tau_green_max:.1%}) "
                        ": avis dermatologique / tele-expertise recommande."
                    ),
                )
            )
        else:
            zone = "GREEN_LOW_RISK_SURVEILLANCE"

        # Verrou 1: Pre-analytical image quality gate
        if preflight is not None and not preflight.passed:
            retake_recommended = True
            issue_str = ", ".join(preflight.issues) or "qualite insuffisante"
            if pol.strict_preflight_block:
                triggers.append(
                    SafetyRuleTrigger(
                        code="PREFLIGHT_FAILED",
                        severity="block_green",
                        detail_fr=(
                            f"Controle qualite pre-analytique echoue ({issue_str}) : "
                            "reprise du cliche recommandee ; zone verte bloquee."
                        ),
                    )
                )
            else:
                triggers.append(
                    SafetyRuleTrigger(
                        code="PREFLIGHT_WARNING",
                        severity="warning",
                        detail_fr=f"Qualite d'image degradee ({issue_str}) : reprise conseilee.",
                    )
                )

        # Verrou 2a: Cross-model disagreement gate
        if disagreement >= pol.disagreement_delta:
            triggers.append(
                SafetyRuleTrigger(
                    code="CROSS_MODEL_DISAGREEMENT",
                    severity="block_green",
                    detail_fr=(
                        f"Desaccord significatif System 1 ({p_s1_cal:.1%}) vs "
                        f"MedGemma ({p_s2_cal:.1%}) (ecart={disagreement:.1%}) : "
                        "abstention / tele-expertise forcee."
                    ),
                )
            )

        # Verrou 2b: Rotational TTA epistemic instability gate
        if tta_risk_std is not None and tta_risk_std >= pol.max_tta_std_for_green:
            triggers.append(
                SafetyRuleTrigger(
                    code="TTA_ROTATIONAL_INSTABILITY",
                    severity="block_green",
                    detail_fr=(
                        f"Instabilite sous rotations TTA (ecart-type={tta_risk_std:.3f} >= "
                        f"{pol.max_tta_std_for_green:.2f}) : escalade en zone grise."
                    ),
                )
            )

        # Verrou 2c: Melanoma sub-probability safety floor
        max_mel = max(p_melanoma_sys1, p_melanoma_medgemma)
        if max_mel >= pol.melanoma_red_threshold:
            triggers.append(
                SafetyRuleTrigger(
                    code="MELANOMA_PRIORITY_ALERT",
                    severity="force_red",
                    detail_fr=(
                        f"Sous-probabilite melanome elevee ({max_mel:.1%}) : "
                        "escalade prioritaire."
                    ),
                )
            )
        elif max_mel >= pol.melanoma_amber_threshold:
            triggers.append(
                SafetyRuleTrigger(
                    code="MELANOMA_SUBSCORE_ALERT",
                    severity="block_green",
                    detail_fr=(
                        f"Signal melanome non negligeable ({max_mel:.1%}) : "
                        "zone verte bloquee."
                    ),
                )
            )

        # Verrou 4: Deterministic clinical red flags (patient context)
        if clinical_context is not None:
            if clinical_context.lesion_evolving:
                triggers.append(
                    SafetyRuleTrigger(
                        code="CLINICAL_RED_FLAG_EVOLVING",
                        severity="block_green",
                        detail_fr=(
                            "Critere clinique E (lesion evolutive) : "
                            "avis dermatologique requis."
                        ),
                    )
                )
            if clinical_context.lesion_bleeding_or_ulcerated:
                sev: RuleSeverity = "force_red" if p_fused >= pol.tau_green_max else "block_green"
                triggers.append(
                    SafetyRuleTrigger(
                        code="CLINICAL_RED_FLAG_BLEEDING",
                        severity=sev,
                        detail_fr=(
                            "Saignement ou ulceration signale : "
                            "avis dermatologique rapide requis."
                        ),
                    )
                )
            if clinical_context.new_lesion_after_50:
                triggers.append(
                    SafetyRuleTrigger(
                        code="CLINICAL_RED_FLAG_NEW_AFTER_50",
                        severity="block_green",
                        detail_fr="Lesion recente apres 50 ans : zone verte bloquee par securite.",
                    )
                )
            if clinical_context.personal_or_family_melanoma_history:
                triggers.append(
                    SafetyRuleTrigger(
                        code="CLINICAL_RED_FLAG_MELANOMA_HISTORY",
                        severity="block_green",
                        detail_fr=(
                            "Antecedent personnel/familial de melanome : "
                            "avis specialise conseille."
                        ),
                    )
                )

        # Resolve final zone monotonically (rules can only escalate, never downgrade safety!)
        if any(t.severity == "force_red" for t in triggers):
            zone = "RED_PRIORITY_DERMATOLOGY"
        elif zone == "GREEN_LOW_RISK_SURVEILLANCE" and any(
            t.severity == "block_green" for t in triggers
        ):
            zone = "AMBER_UNCERTAIN_TELEEXPERTISE"

        # Verrou 3: Text <-> Probability Contradiction Blocker + RUO Sanitizer
        raw_summary = (
            medgemma_summary_fr
            if medgemma_summary_fr is not None
            else (medgemma_assessment.summary_for_referral_fr if medgemma_assessment else "")
        ).strip()

        text_contradiction_caught = False
        regulatory_sanitized = False
        safe_summary = raw_summary

        if safe_summary:
            safe_summary, regulatory_sanitized = _sanitize_regulatory_text(safe_summary)
            if regulatory_sanitized:
                triggers.append(
                    SafetyRuleTrigger(
                        code="REGULATORY_TEXT_SANITIZED",
                        severity="info",
                        detail_fr="Formulation diagnostique/prescriptive neutralisee (Hors-MDR).",
                    )
                )

        med_claimed_benign = (
            medgemma_assessment is not None
            and medgemma_assessment.research_triage_label == "benign_pattern"
        )
        if zone != "GREEN_LOW_RISK_SURVEILLANCE" and (
            _contains_reassurance(safe_summary) or med_claimed_benign
        ):
            text_contradiction_caught = True
            triggers.append(
                SafetyRuleTrigger(
                    code="TEXT_RISK_CONTRADICTION",
                    severity="warning",
                    detail_fr=(
                        "Contradiction detectee : texte MedGemma rassurant malgre une "
                        f"escalade {zone} (risque fusionne={p_fused:.1%}). Resume remplace."
                    ),
                )
            )
            zone_fr = (
                "Adressage dermatologique prioritaire (Zone Rouge)"
                if zone == "RED_PRIORITY_DERMATOLOGY"
                else "Tele-expertise / avis dermatologique recommande (Zone Grise)"
            )
            reason_codes = ", ".join(
                t.code for t in triggers if t.severity in ("block_green", "force_red")
            )
            safe_summary = (
                f"[GARDE-FOU HARNESS] {zone_fr}. Risque calibre fusionne={p_fused:.1%} "
                f"(System 1={p_s1_cal:.1%} [{sys1_top_entity}], "
                f"MedGemma={p_s2_cal:.1%} [{medgemma_top_entity or 'n/a'}]). "
                f"Le commentaire brut rassurant a ete neutralise par securite "
                f"(regles: {reason_codes or 'RISK_THRESHOLD'})."
            )
        elif not safe_summary:
            if zone == "RED_PRIORITY_DERMATOLOGY":
                safe_summary = (
                    f"Zone Rouge — Adressage dermatologique prioritaire recommande "
                    f"(risque fusionne calibre={p_fused:.1%}, hypothese S1={sys1_top_entity})."
                )
            elif zone == "AMBER_UNCERTAIN_TELEEXPERTISE":
                safe_summary = (
                    f"Zone Grise — Cas incertain ou signal discordant "
                    f"(risque fusionne calibre={p_fused:.1%}) : avis dermatologique conseille."
                )
            else:
                safe_summary = (
                    f"Zone Verte — Faible risque calibre ({p_fused:.1%}) avec accord "
                    f"inter-modeles ({sys1_top_entity}). Surveillance clinique habituelle "
                    "et reconsultation en cas d'evolution."
                )

        audit = AuditTrail(
            policy_version=pol.policy_version,
            timestamp_utc=datetime.now(UTC).isoformat(),
            case_id=case_id,
            image_sha256=image_sha256,
            sys1_model_id=pol.sys1_model_id,
            sys2_model_id=pol.sys2_model_id,
            preflight_passed=preflight.passed if preflight is not None else None,
            preflight_issues=list(preflight.issues) if preflight is not None else [],
            p_sys1_raw=round(p_sys1_raw, 6),
            p_sys1_cal=round(p_s1_cal, 6),
            p_medgemma_raw=round(p_medgemma_raw, 6),
            p_medgemma_cal=round(p_s2_cal, 6),
            p_fused=round(p_fused, 6),
            disagreement=round(disagreement, 6),
            tta_risk_std=round(tta_risk_std, 6) if tta_risk_std is not None else None,
            text_contradiction_caught=text_contradiction_caught,
            regulatory_sanitized=regulatory_sanitized,
            triggered_rules=[t.code for t in triggers],
        )

        return LastMileAssessment(
            case_id=case_id,
            triage_zone=zone,
            requires_dermatology_review=(zone != "GREEN_LOW_RISK_SURVEILLANCE"),
            retake_recommended=retake_recommended,
            calibrated_risk_score=round(p_fused, 6),
            p_sys1_cal=round(p_s1_cal, 6),
            p_medgemma_cal=round(p_s2_cal, 6),
            disagreement=round(disagreement, 6),
            tta_risk_std=round(tta_risk_std, 6) if tta_risk_std is not None else None,
            sys1_top_entity=sys1_top_entity,
            medgemma_top_entity=medgemma_top_entity,
            safe_referral_summary_fr=safe_summary,
            triggered_rules=triggers,
            audit=audit,
        )


def evaluate_deployment_gate(
    metrics: dict[str, Any],
    min_json_valid_rate: float = 0.99,
    min_safe_sensitivity: float = 0.95,
    min_melanoma_sensitivity: float = 0.95,
    min_green_npv: float = 0.90,
    min_green_benign_specificity: float = 0.20,
) -> DeploymentGateVerdict:
    """Verrou 6: Policy-as-code release gate blocking regressions before Edge deployment."""
    checks = [
        DeploymentGateCheck(
            name="json_valid_rate",
            passed=float(metrics.get("json_valid_rate", 0.0)) >= min_json_valid_rate,
            observed=round(float(metrics.get("json_valid_rate", 0.0)), 4),
            required=f">= {min_json_valid_rate:.2f}",
        ),
        DeploymentGateCheck(
            name="malignant_safe_sensitivity_amber_or_red",
            passed=float(metrics.get("malignant_safe_sensitivity", 0.0)) >= min_safe_sensitivity,
            observed=round(float(metrics.get("malignant_safe_sensitivity", 0.0)), 4),
            required=f">= {min_safe_sensitivity:.2f}",
        ),
        DeploymentGateCheck(
            name="melanoma_safe_sensitivity_amber_or_red",
            passed=float(metrics.get("melanoma_safe_sensitivity", 0.0)) >= min_melanoma_sensitivity,
            observed=round(float(metrics.get("melanoma_safe_sensitivity", 0.0)), 4),
            required=f">= {min_melanoma_sensitivity:.2f}",
        ),
        DeploymentGateCheck(
            name="green_zone_npv",
            passed=float(metrics.get("green_zone_npv", 0.0)) >= min_green_npv,
            observed=round(float(metrics.get("green_zone_npv", 0.0)), 4),
            required=f">= {min_green_npv:.2f}",
        ),
        DeploymentGateCheck(
            name="green_zone_benign_specificity",
            passed=float(metrics.get("green_zone_benign_specificity", 0.0))
            >= min_green_benign_specificity,
            observed=round(float(metrics.get("green_zone_benign_specificity", 0.0)), 4),
            required=f">= {min_green_benign_specificity:.2f}",
        ),
    ]
    decision: Literal["PASS", "BLOCK_DEPLOY"] = (
        "PASS" if all(c.passed for c in checks) else "BLOCK_DEPLOY"
    )
    return DeploymentGateVerdict(decision=decision, checks=checks)
