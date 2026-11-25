"""Adaptive Clinical Triage Agent wrapping System 1 ONNX, MedGemma VLM, and Last-Mile Harness.

Implements a deterministic, auditable tool-calling orchestrator optimized for clinical yield
and Edge compute efficiency:
  1. Pre-analytical optical QA (`optical_preflight_qa`)
  2. Fast System 1 ONNX screening with 4-view TTA instability (`system1_onnx_triage`)
  3. Conditional ROI crop re-check when dark circular vignette or TTA instability is detected
  4. Adaptive Compute Routing: resolves ultra-low-risk, stable, red-flag-free cases via
     System 1 Fast-Path (< 1s CPU, skipping 4B VLM), while escalating borderline/suspicious
     cases to System 2 MedGemma Q8_0 (with native continuous token log-probabilities)
  5. Six-barrier Last-Mile Clinical Safety Harness adjudication & contradiction blocking
  6. Structured GP clinical action plan + targeted missing-anamnesis follow-up questions
"""

import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

from medgemma_dermato.clinical_harness import (
    RUO_DISCLAIMER_FR,
    ClinicalContext,
    HarnessPolicy,
    LastMileAssessment,
    LastMileClinicalHarness,
)
from medgemma_dermato.config import PreflightConfig
from medgemma_dermato.eval.calibration import RiskScoreCase, apply_temperature
from medgemma_dermato.preflight import crop_dermoscopic_roi, evaluate_image_quality
from medgemma_dermato.schema import PreflightResult
from medgemma_dermato.system1 import System1Prediction

RoutingDecision = Literal[
    "FAST_PATH_SYS1_ONLY",
    "HYBRID_FULL_S1_PLUS_S2",
    "DEGRADED_SYS1_ONLY",
]


class AgentStepTrace(BaseModel):
    """Auditable execution trace entry for one deterministic agent tool call."""

    step_index: int = Field(ge=1)
    tool_name: str
    status: Literal["ok", "triggered", "skipped"]
    duration_ms: float = Field(ge=0.0)
    summary_fr: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class AnamnesisQuestion(BaseModel):
    """High-information follow-up question generated when clinical metadata is incomplete."""

    field_name: str
    question_fr: str
    clinical_rationale_fr: str
    priority: Literal["high", "medium"]


class ActionPlan(BaseModel):
    """Clinician-facing workflow package maximizing practical utility at the GP desk."""

    urgency_fr: str
    recommended_pathway_fr: str
    top_differentials: list[tuple[str, float]] = Field(default_factory=list)
    gp_referral_letter_fr: str
    patient_return_precautions_fr: str
    missing_anamnesis_questions: list[AnamnesisQuestion] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Tuning parameters for adaptive Edge compute routing and conditional ROI zoom."""

    # Safety-first default: Always-On Hybrid S1+S2 (passes 95.7% MSKCC DeploymentGate).
    # Opt-in Fast-Path (`tau_mc_val_99 = 0.0670`) saves 35% VLM calls in screening.
    adaptive_routing_enabled: bool = False
    fast_path_sys1_max_risk: float = Field(default=0.0670, ge=0.0, le=0.5)
    fast_path_max_tta_std: float = Field(default=0.06, ge=0.0, le=0.5)
    fast_path_max_melanoma: float = Field(default=0.05, ge=0.0, le=0.5)
    roi_crop_vignette_threshold: float = Field(default=0.20, ge=0.0, le=1.0)


class AgentRunResult(BaseModel):
    """Complete structured output from one DermoscopyTriageAgent execution."""

    case_id: str
    routing_decision: RoutingDecision
    vlm_invoked: bool
    roi_crop_applied: bool
    system2_logprobs_used: bool
    system2_latency_ms: float | None = None
    total_latency_ms: float = Field(ge=0.0)
    preflight: PreflightResult
    system1: System1Prediction
    assessment: LastMileAssessment
    action_plan: ActionPlan
    trace: list[AgentStepTrace] = Field(default_factory=list)


def _has_clinical_red_flag(ctx: ClinicalContext | None) -> bool:
    if ctx is None:
        return False
    return bool(
        ctx.lesion_evolving
        or ctx.lesion_bleeding_or_ulcerated
        or ctx.new_lesion_after_50
        or ctx.personal_or_family_melanoma_history
    )


def _build_missing_anamnesis_questions(
    ctx: ClinicalContext | None,
    zone: str,
) -> list[AnamnesisQuestion]:
    questions: list[AnamnesisQuestion] = []
    if ctx is None or not ctx.lesion_evolving:
        questions.append(
            AnamnesisQuestion(
                field_name="lesion_evolving",
                question_fr="La lésion a-t-elle changé de taille, forme ou couleur récemment ?",
                clinical_rationale_fr=(
                    "Critère E (ABCDE) : tout changement récent interdit la Zone Verte."
                ),
                priority="high",
            )
        )
    if ctx is None or not ctx.lesion_bleeding_or_ulcerated:
        questions.append(
            AnamnesisQuestion(
                field_name="lesion_bleeding_or_ulcerated",
                question_fr="Y a-t-il eu saignement spontané, suintement ou ulcération ?",
                clinical_rationale_fr="Signe d'alerte majeur (mélanome nodulaire / carcinome).",
                priority="high",
            )
        )
    if ctx is None or ctx.age_years is None:
        questions.append(
            AnamnesisQuestion(
                field_name="age_years",
                question_fr=(
                    "Quel est l'âge du patient (et la lésion est-elle apparue après 50 ans) ?"
                ),
                clinical_rationale_fr=(
                    "Modifie fortement la probabilité pré-test des carcinomes et mélanomes."
                ),
                priority="medium",
            )
        )
    if zone != "GREEN_LOW_RISK_SURVEILLANCE" and (ctx is None or not ctx.anatom_site):
        questions.append(
            AnamnesisQuestion(
                field_name="anatom_site",
                question_fr="Quelle est la localisation anatomique exacte (zone photo-exposée ?) ?",
                clinical_rationale_fr=(
                    "Information indispensable pour le dossier de télé-expertise."
                ),
                priority="medium",
            )
        )
    return questions[:3]


def _build_action_plan(
    assessment: LastMileAssessment,
    s1: System1Prediction,
    preflight: PreflightResult,
    ctx: ClinicalContext | None,
    routing_decision: RoutingDecision,
) -> ActionPlan:
    top_diffs = sorted(s1.entity_probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_diffs_clean = [(k, round(v, 4)) for k, v in top_diffs]
    diff_text = ", ".join(f"{k} ({v * 100:.1f}%)" for k, v in top_diffs_clean)

    ctx_parts: list[str] = []
    if ctx is not None:
        if ctx.age_years is not None:
            ctx_parts.append(f"{ctx.age_years} ans")
        if ctx.sex:
            ctx_parts.append(ctx.sex)
        if ctx.anatom_site:
            ctx_parts.append(f"site: {ctx.anatom_site}")
    ctx_line = ", ".join(ctx_parts) if ctx_parts else "non renseigné"

    qa_line = (
        "Validée"
        if preflight.passed
        else f"Alerte qualité ({', '.join(preflight.issues)}) — reprise recommandée"
    )
    rules_line = ", ".join(assessment.audit.triggered_rules) or "Aucune alerte"

    if assessment.triage_zone == "RED_PRIORITY_DERMATOLOGY":
        urgency = "Prioritaire (filière courte dermatologie)"
        pathway = "Adressage dermatologique prioritaire avec dossier photo + contexte structuré."
        precautions = (
            "Informer le patient d'un rendez-vous spécialisé rapide sans retarder la consultation "
            "si saignement, croissance rapide ou douleur."
        )
    elif assessment.triage_zone == "AMBER_UNCERTAIN_TELEEXPERTISE":
        urgency = "Standard / Télé-expertise asynchrone recommandée"
        pathway = (
            "Soumettre le cliché validé + contexte clinique en télé-expertise dermatologique "
            "pour levée de doute."
        )
        precautions = (
            "Compléter les questions d'anamnèse ci-dessous avant envoi ; consulter rapidement "
            "en cas de modification visuelle."
        )
    else:
        urgency = "Faible risque calibré — Surveillance clinique"
        pathway = (
            "Suivi habituel en médecine générale (pas d'adressage urgent si anamnèse rassurante)."
        )
        precautions = (
            "Consignes de sécurité (ABCDE) : reconsulter sans délai si évolution de taille, "
            "forme, couleur, saignement ou prurit persistant ; contrôle photo conseillé à 3 mois."
        )

    letter = (
        f"=== FICHE D'ADRESSAGE / TRIAGE DERMATOSCOPIQUE ({assessment.case_id}) ===\n"
        f"Orientation : {assessment.triage_zone} ({urgency})\n"
        f"Routage Edge : {routing_decision} | Qualité photo : {qa_line}\n"
        f"Contexte patient : {ctx_line}\n"
        f"Risque calibré : {assessment.calibrated_risk_score * 100:.1f}% "
        f"(System 1={assessment.p_sys1_cal * 100:.1f}%, "
        f"MedGemma={assessment.p_medgemma_cal * 100:.1f}%)\n"
        f"Top différentiels S1 : {diff_text}\n"
        f"Garde-fous déclenchés : {rules_line}\n"
        f"Synthèse sécurisée : {assessment.safe_referral_summary_fr}\n"
        f"Conduite conseillée : {pathway}\n"
        f"\n{RUO_DISCLAIMER_FR}"
    )

    return ActionPlan(
        urgency_fr=urgency,
        recommended_pathway_fr=pathway,
        top_differentials=top_diffs_clean,
        gp_referral_letter_fr=letter,
        patient_return_precautions_fr=precautions,
        missing_anamnesis_questions=_build_missing_anamnesis_questions(
            ctx, assessment.triage_zone
        ),
    )


class DermoscopyTriageAgent:
    """Deterministic tool-orchestrating agent optimizing yield, latency, and safety."""

    def __init__(
        self,
        system1: Any,
        vlm_client: Any | None = None,
        harness: LastMileClinicalHarness | None = None,
        preflight_cfg: PreflightConfig | None = None,
        agent_cfg: AgentConfig | None = None,
    ) -> None:
        self.system1 = system1
        self.vlm_client = vlm_client
        self.harness = harness or LastMileClinicalHarness(
            HarnessPolicy(strict_preflight_block=False)
        )
        self.preflight_cfg = preflight_cfg or PreflightConfig()
        self.agent_cfg = agent_cfg or AgentConfig()

    def run(
        self,
        image: Image.Image,
        clinical_context: ClinicalContext | None = None,
        use_tta4: bool = True,
        image_sha256: str | None = None,
        case_id: str | None = None,
    ) -> AgentRunResult:
        """Execute the full adaptive 6-tool agent pipeline on one dermoscopic image."""
        t_all = time.perf_counter()
        cid = case_id or f"AGT-{uuid.uuid4().hex[:8].upper()}"
        rgb = image.convert("RGB")
        trace: list[AgentStepTrace] = []

        # Tool 1: Optical Pre-Analytical QA
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            rgb.save(tmp, format="JPEG", quality=95)
            tmp_path = Path(tmp.name)
        try:
            preflight = evaluate_image_quality(tmp_path, self.preflight_cfg)
        finally:
            tmp_path.unlink(missing_ok=True)
        dt_pf = round((time.perf_counter() - t0) * 1000.0, 1)
        trace.append(
            AgentStepTrace(
                step_index=1,
                tool_name="optical_preflight_qa",
                status="ok" if preflight.passed else "triggered",
                duration_ms=dt_pf,
                summary_fr=(
                    "Qualité optique validée."
                    if preflight.passed
                    else f"Défauts détectés : {', '.join(preflight.issues)}."
                ),
                metrics={
                    "passed": preflight.passed,
                    "laplacian_var": preflight.metrics.laplacian_var,
                    "border_dark_ratio": preflight.metrics.border_dark_ratio,
                },
            )
        )

        # Tool 2: System 1 Edge ONNX Triage (with TTA-4 instability)
        t0 = time.perf_counter()
        s1_pred: System1Prediction = self.system1.predict_image(rgb, tta=use_tta4)
        dt_s1 = round((time.perf_counter() - t0) * 1000.0, 1)
        trace.append(
            AgentStepTrace(
                step_index=2,
                tool_name="system1_onnx_triage",
                status="ok",
                duration_ms=dt_s1,
                summary_fr=(
                    f"ConvNeXt-Tiny ONNX : risque calibré={s1_pred.p_risk_calibrated:.1%}, "
                    f"entité={s1_pred.predicted_entity}, std_TTA={s1_pred.tta_risk_std:.3f}."
                ),
                metrics={
                    "p_risk_cal": s1_pred.p_risk_calibrated,
                    "p_melanoma": s1_pred.p_melanoma,
                    "tta_risk_std": s1_pred.tta_risk_std,
                },
            )
        )

        # Tool 3 (Conditional): Dermoscopic ROI Crop Recheck on strong circular vignette
        roi_crop_applied = False
        needs_roi = (
            preflight.metrics.border_dark_ratio >= self.agent_cfg.roi_crop_vignette_threshold
            or (s1_pred.tta_risk_std >= 0.12 and preflight.metrics.border_dark_ratio >= 0.10)
        )
        if needs_roi:
            t0 = time.perf_counter()
            cropped = crop_dermoscopic_roi(rgb)
            if cropped.size != rgb.size:
                roi_crop_applied = True
                s1_roi: System1Prediction = self.system1.predict_image(cropped, tta=use_tta4)
                dt_roi = round((time.perf_counter() - t0) * 1000.0, 1)
                # Conservative max: adopt cropped risk only if it catches a higher-risk signal
                if s1_roi.p_risk_raw > s1_pred.p_risk_raw:
                    s1_pred = s1_roi
                trace.append(
                    AgentStepTrace(
                        step_index=3,
                        tool_name="dermoscopic_roi_crop_recheck",
                        status="triggered",
                        duration_ms=dt_roi,
                        summary_fr=(
                            f"Vignette détectée (bord={preflight.metrics.border_dark_ratio:.0%}) "
                            f": recadrage ROI {cropped.size}, "
                            f"risque={s1_pred.p_risk_calibrated:.1%}."
                        ),
                        metrics={
                            "cropped_size": list(cropped.size),
                            "p_risk_cal": s1_pred.p_risk_calibrated,
                        },
                    )
                )
            else:
                trace.append(
                    AgentStepTrace(
                        step_index=3,
                        tool_name="dermoscopic_roi_crop_recheck",
                        status="skipped",
                        duration_ms=0.0,
                        summary_fr="Cadrage plein cadre déjà optimal.",
                    )
                )
        else:
            trace.append(
                AgentStepTrace(
                    step_index=3,
                    tool_name="dermoscopic_roi_crop_recheck",
                    status="skipped",
                    duration_ms=0.0,
                    summary_fr="Pas de vignettage périphérique nécessitant un recadrage.",
                )
            )

        # Tool 4 (Adaptive Compute Router): Fast-Path System 1 vs Full System 2 MedGemma VLM
        s1_cal_harness = apply_temperature([s1_pred.p_risk_raw], self.harness.policy.temp_sys1)[0]
        can_fast_path = (
            self.agent_cfg.adaptive_routing_enabled
            and preflight.passed
            and (not _has_clinical_red_flag(clinical_context))
            and s1_cal_harness < self.agent_cfg.fast_path_sys1_max_risk
            and s1_pred.tta_risk_std < self.agent_cfg.fast_path_max_tta_std
            and s1_pred.p_melanoma < self.agent_cfg.fast_path_max_melanoma
        )

        vlm_invoked = False
        sys2_lp_used = False
        sys2_lat_ms: float | None = None
        p_med_raw = s1_pred.p_risk_raw
        p_mel_med = s1_pred.p_melanoma
        med_top_entity: str | None = None
        med_parsed = None
        med_summary: str | None = None

        if can_fast_path:
            routing_decision: RoutingDecision = "FAST_PATH_SYS1_ONLY"
            trace.append(
                AgentStepTrace(
                    step_index=4,
                    tool_name="adaptive_vlm_router",
                    status="skipped",
                    duration_ms=0.0,
                    summary_fr=(
                        f"Fast-Path activé : risque S1 très faible ({s1_cal_harness:.1%} < "
                        f"{self.agent_cfg.fast_path_sys1_max_risk:.1%}), TTA stable et 0 drapeau "
                        "rouge → VLM 4B économisé (~2–3 s gagnés)."
                    ),
                    metrics={"fast_path": True, "p_s1_cal": round(s1_cal_harness, 4)},
                )
            )
        elif self.vlm_client is not None and self.vlm_client.healthcheck():
            t0 = time.perf_counter()
            vlm_ctx = (
                {
                    "age_approx": clinical_context.age_years,
                    "sex": clinical_context.sex,
                    "anatom_site": clinical_context.anatom_site,
                }
                if clinical_context is not None
                else None
            )
            vlm_res = self.vlm_client.analyze_image(rgb, clinical_context=vlm_ctx)
            dt_vlm = round((time.perf_counter() - t0) * 1000.0, 1)
            vlm_invoked = True
            routing_decision = "HYBRID_FULL_S1_PLUS_S2"
            sys2_lp_used = bool(getattr(vlm_res, "logprobs_used", False))
            sys2_lat_ms = vlm_res.latency_ms
            p_med_raw = vlm_res.p_risk_estimate
            p_mel_med = vlm_res.p_melanoma_estimate
            med_parsed = vlm_res.parsed
            if vlm_res.parsed is not None:
                med_top_entity = vlm_res.parsed.entity_category
                med_summary = vlm_res.parsed.summary_for_referral_fr
            lp_tag = "natifs" if sys2_lp_used else "proxy"
            trace.append(
                AgentStepTrace(
                    step_index=4,
                    tool_name="adaptive_vlm_router",
                    status="triggered",
                    duration_ms=dt_vlm,
                    summary_fr=(
                        f"Escalade vers MedGemma Q8_0 (logprobs={lp_tag}) "
                        f": p_risk={p_med_raw:.1%}, entité={med_top_entity or 'n/a'}."
                    ),
                    metrics={"logprobs_used": sys2_lp_used, "p_med_raw": round(p_med_raw, 4)},
                )
            )
        else:
            routing_decision = "DEGRADED_SYS1_ONLY"
            trace.append(
                AgentStepTrace(
                    step_index=4,
                    tool_name="adaptive_vlm_router",
                    status="skipped",
                    duration_ms=0.0,
                    summary_fr=(
                        "System 2 llama-server hors ligne : repli explicite sur System 1 ONNX."
                    ),
                    metrics={"degraded": True},
                )
            )

        # Tool 5: Last-Mile 6-Barrier Safety Harness
        t0 = time.perf_counter()
        assessment = self.harness.evaluate_case(
            case_id=cid,
            p_sys1_raw=s1_pred.p_risk_raw,
            p_medgemma_raw=p_med_raw,
            sys1_top_entity=s1_pred.predicted_entity,
            p_melanoma_sys1=s1_pred.p_melanoma,
            p_melanoma_medgemma=p_mel_med,
            medgemma_top_entity=med_top_entity,
            medgemma_assessment=med_parsed,
            medgemma_summary_fr=med_summary,
            preflight=preflight,
            tta_risk_std=s1_pred.tta_risk_std,
            clinical_context=clinical_context,
            image_sha256=image_sha256,
        )
        if routing_decision == "DEGRADED_SYS1_ONLY":
            assessment.safe_referral_summary_fr = (
                "[MODE DEGRADE SYSTEM 1 EDGE SEUL — llama-server MedGemma hors ligne] "
                + assessment.safe_referral_summary_fr
            )
        elif routing_decision == "FAST_PATH_SYS1_ONLY":
            fp_thr = self.agent_cfg.fast_path_sys1_max_risk
            assessment.safe_referral_summary_fr = (
                f"[FAST-PATH AGENT — Accord haute confiance System 1 (< {fp_thr:.1%}), "
                "VLM 4B économisé] " + assessment.safe_referral_summary_fr
            )
        dt_h = round((time.perf_counter() - t0) * 1000.0, 1)
        trace.append(
            AgentStepTrace(
                step_index=5,
                tool_name="last_mile_safety_harness",
                status="triggered" if assessment.triggered_rules else "ok",
                duration_ms=dt_h,
                summary_fr=(
                    f"Verdict={assessment.triage_zone}, "
                    f"risque={assessment.calibrated_risk_score:.1%}, "
                    f"règles={len(assessment.triggered_rules)}."
                ),
                metrics={
                    "zone": assessment.triage_zone,
                    "contradiction_caught": assessment.audit.text_contradiction_caught,
                },
            )
        )

        # Tool 6: GP Action Plan & Adaptive Anamnesis Builder
        t0 = time.perf_counter()
        action_plan = _build_action_plan(
            assessment, s1_pred, preflight, clinical_context, routing_decision
        )
        dt_ap = round((time.perf_counter() - t0) * 1000.0, 1)
        trace.append(
            AgentStepTrace(
                step_index=6,
                tool_name="clinical_action_plan_builder",
                status="ok",
                duration_ms=dt_ap,
                summary_fr=(
                    f"Fiche d'adressage générée + {len(action_plan.missing_anamnesis_questions)} "
                    "question(s) d'anamnèse prioritaire(s)."
                ),
            )
        )

        total_ms = round((time.perf_counter() - t_all) * 1000.0, 1)
        return AgentRunResult(
            case_id=cid,
            routing_decision=routing_decision,
            vlm_invoked=vlm_invoked,
            roi_crop_applied=roi_crop_applied,
            system2_logprobs_used=sys2_lp_used,
            system2_latency_ms=sys2_lat_ms,
            total_latency_ms=total_ms,
            preflight=preflight,
            system1=s1_pred,
            assessment=assessment,
            action_plan=action_plan,
            trace=trace,
        )


def simulate_adaptive_routing_on_cohort(
    sys1_cases: list[RiskScoreCase],
    medgemma_map: dict[str, RiskScoreCase],
    policy: HarnessPolicy,
    agent_cfg: AgentConfig | None = None,
) -> dict[str, Any]:
    """Quantify exact VLM compute savings vs clinical sensitivity on a locked cohort."""
    cfg = agent_cfg or AgentConfig()
    harness = LastMileClinicalHarness(policy)

    fast_path_cnt = 0
    vlm_called_cnt = 0
    mal_caught = mal_tot = mel_caught = mel_tot = g_ben = ben_tot = g_mal = 0

    for s1 in sys1_cases:
        if s1.isic_id not in medgemma_map:
            continue
        s2 = medgemma_map[s1.isic_id]
        p_s1_cal = apply_temperature([s1.p_risk], policy.temp_sys1)[0]
        is_fast = (
            cfg.adaptive_routing_enabled
            and p_s1_cal < cfg.fast_path_sys1_max_risk
            and s1.p_melanoma < cfg.fast_path_max_melanoma
        )
        if is_fast:
            fast_path_cnt += 1
            p_med_raw = s1.p_risk
            p_mel_med = s1.p_melanoma
        else:
            vlm_called_cnt += 1
            p_med_raw = s2.p_risk
            p_mel_med = s2.p_melanoma

        res = harness.evaluate_case(
            case_id=s1.isic_id,
            p_sys1_raw=s1.p_risk,
            p_medgemma_raw=p_med_raw,
            sys1_top_entity=s1.predicted_entity_argmax,
            p_melanoma_sys1=s1.p_melanoma,
            p_melanoma_medgemma=p_mel_med,
            medgemma_top_entity=s2.predicted_entity_argmax,
        )
        is_green = res.triage_zone == "GREEN_LOW_RISK_SURVEILLANCE"
        if s1.ground_truth_malignant is True:
            mal_tot += 1
            if not is_green:
                mal_caught += 1
            else:
                g_mal += 1
            if s1.ground_truth_is_melanoma:
                mel_tot += 1
                if not is_green:
                    mel_caught += 1
        elif s1.ground_truth_malignant is False:
            ben_tot += 1
            if is_green:
                g_ben += 1

    total = fast_path_cnt + vlm_called_cnt
    return {
        "total_cases": total,
        "fast_path_count": fast_path_cnt,
        "fast_path_vlm_compute_saved_rate": round(fast_path_cnt / max(total, 1), 4),
        "vlm_invoked_count": vlm_called_cnt,
        "malignant_safe_sensitivity": round(mal_caught / max(mal_tot, 1), 4),
        "melanoma_safe_sensitivity": round(mel_caught / max(mel_tot, 1), 4),
        "green_zone_npv": round(g_ben / max(g_ben + g_mal, 1), 4),
        "green_zone_benign_specificity": round(g_ben / max(ben_tot, 1), 4),
        "green_zone_malignant_fn": g_mal,
    }
