"""Prompt templates and MedGemma multimodal message builders."""

import json
from typing import Any

from medgemma_dermato.schema import DermoscopyAssessment

SYSTEM_PROMPT = (
    "Tu es un assistant de recherche en sémiologie dermatoscopique (Research Use Only, "
    "non-dispositif médical). Tu ne poses jamais de diagnostic clinique autonome. "
    "Analyse d'abord les critères sémiologiques observables (asymétrie, réseau pigmentaire, "
    "points/globules, stries, voile bleu-blanc, structures vasculaires, régression, couleurs), "
    "calibre honnêtement ta confiance entre 0.0 et 1.0, et active `abstain=true` avec une "
    "`abstention_reason` explicite si la qualité d'image est insuffisante ou si "
    "`confidence < 0.5`. Réponds EXCLUSIVEMENT par un objet JSON valide conforme au schéma "
    "fourni, sans texte autour."
)


def _format_clinical_context(clinical_context: dict[str, Any] | None) -> str:
    if not clinical_context:
        return "Contexte clinique : non fourni."
    parts: list[str] = []
    if clinical_context.get("age_approx") is not None:
        parts.append(f"âge approximatif {clinical_context['age_approx']} ans")
    if clinical_context.get("sex"):
        parts.append(f"sexe {clinical_context['sex']}")
    if clinical_context.get("anatom_site"):
        parts.append(f"localisation {clinical_context['anatom_site']}")
    if not parts:
        return "Contexte clinique : non fourni."
    return "Contexte clinique : " + ", ".join(parts) + "."


def build_inference_messages(
    clinical_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build chat messages for MedGemma multimodal inference."""
    schema_json = json.dumps(DermoscopyAssessment.model_json_schema(), ensure_ascii=False)
    context_line = _format_clinical_context(clinical_context)
    user_text = (
        f"{context_line}\n"
        "Évalue cette image dermatoscopique et produis uniquement un JSON conforme au schéma "
        f"suivant :\n{schema_json}"
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_sft_messages(
    target_assessment: DermoscopyAssessment,
    clinical_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build supervised fine-tuning messages including the target assistant JSON turn."""
    messages = build_inference_messages(clinical_context=clinical_context)
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": target_assessment.model_dump_json(ensure_ascii=False),
                }
            ],
        }
    )
    return messages
