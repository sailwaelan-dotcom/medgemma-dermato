"""Prompt templates, MedGemma multimodal message builders, and SFT label masking."""

import json
from typing import TYPE_CHECKING, Any

from medgemma_dermato.schema import (
    DermoscopyAssessment,
    EntityCategory,
    MelanocyticPattern,
    TriageAssessment,
    map_isic_to_entity_category,
)

if TYPE_CHECKING:
    from medgemma_dermato.data.isic_ingest import ISICRecord

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

TRIAGE_SYSTEM_PROMPT = (
    "Tu es un assistant de recherche en triage dermatoscopique et rédaction de fiche "
    "d'adressage (Research Use Only, non-dispositif médical). Tu ne poses jamais de diagnostic "
    "clinique autonome. Identifie la nature mélanocytaire, l'entité la plus probable, "
    "le niveau de triage de recherche (`benign_pattern`, `suspicious_malignant_pattern`, ou "
    "`indeterminate`), calibre ta confiance entre 0.0 et 1.0, et active `abstain=true` avec "
    "une `abstention_reason` si la lésion est indéterminée, la qualité insuffisante ou "
    "`confidence < 0.5`. Rédige `summary_for_referral_fr` en français médical clair et concis. "
    "Réponds EXCLUSIVEMENT par un objet JSON valide conforme au schéma fourni."
)

_ENTITY_LABELS_FR: dict[EntityCategory, str] = {
    "melanoma": "aspect évocateur d'une lésion mélanocytaire suspecte (type mélanome)",
    "basal_cell_carcinoma": "aspect compatible avec un carcinome basocellulaire",
    "squamous_cell_carcinoma": "aspect compatible avec un carcinome épidermoïde",
    "nevus": "aspect compatible avec un naevus mélanocytaire bénin",
    "benign_keratosis": "aspect compatible avec une lésion kératosique ou lentigo bénin",
    "dermatofibroma": "aspect compatible avec un dermatofibrome bénin",
    "vascular": "aspect compatible avec une lésion vasculaire bénigne",
    "actinic_keratosis_or_indeterminate": (
        "lésion kératosique actinique ou atypique indéterminée"
    ),
    "other_benign": "lésion cutanée d'allure bénigne",
}


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
    """Build chat messages for MedGemma semiology multimodal inference."""
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
    """Build supervised fine-tuning messages for DermoscopyAssessment."""
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


def build_triage_inference_messages(
    clinical_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build chat messages for MedGemma ground-truth triage & referral inference."""
    schema_json = json.dumps(TriageAssessment.model_json_schema(), ensure_ascii=False)
    context_line = _format_clinical_context(clinical_context)
    user_text = (
        f"{context_line}\n"
        "Analyse cette image dermatoscopique et produis uniquement un JSON conforme au schéma "
        f"suivant :\n{schema_json}"
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": TRIAGE_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _resolve_melanocytic_pattern(
    record: "ISICRecord",
    entity: EntityCategory,
) -> MelanocyticPattern:
    if record.melanocytic is True or entity in ("melanoma", "nevus"):
        return "melanocytic"
    if record.melanocytic is False or entity in (
        "basal_cell_carcinoma",
        "squamous_cell_carcinoma",
        "benign_keratosis",
        "dermatofibroma",
        "vascular",
    ):
        return "non_melanocytic"
    d2 = (record.diagnosis_2 or "").lower()
    if "melanocytic" in d2:
        return "melanocytic"
    if "epidermal" in d2 or "adnexal" in d2 or "soft tissue" in d2:
        return "non_melanocytic"
    return "uncertain"


def build_triage_target_from_record(record: "ISICRecord") -> TriageAssessment:
    """Build an exact ground-truth TriageAssessment from ISIC metadata (zero fabricated fields)."""
    entity = map_isic_to_entity_category(
        record.diagnosis_1, record.diagnosis_2, record.diagnosis_3
    )
    melanocytic_pattern = _resolve_melanocytic_pattern(record, entity)
    site_suffix = f" (localisation : {record.anatom_site})" if record.anatom_site else ""
    entity_fr = _ENTITY_LABELS_FR[entity]
    is_histopath = "histopath" in (record.diagnosis_confirm_type or "").lower()

    if record.is_malignant is True:
        conf = 0.85 if is_histopath else 0.80
        summary = (
            f"Cliché dermatoscopique montrant un {entity_fr}{site_suffix}. "
            "Adressage dermatologique prioritaire recommandé pour évaluation spécialisée."
        )
        return TriageAssessment(
            image_quality_adequate=True,
            melanocytic_pattern=melanocytic_pattern,
            entity_category=entity,
            research_triage_label="suspicious_malignant_pattern",
            confidence=conf,
            abstain=False,
            abstention_reason=None,
            summary_for_referral_fr=summary,
        )

    if record.is_malignant is False:
        summary = (
            f"Cliché dermatoscopique montrant un {entity_fr}{site_suffix}. "
            "Pas d'argument majeur de malignité sur cet aspect ; suivi clinique habituel."
        )
        return TriageAssessment(
            image_quality_adequate=True,
            melanocytic_pattern=melanocytic_pattern,
            entity_category=entity,
            research_triage_label="benign_pattern",
            confidence=0.80,
            abstain=False,
            abstention_reason=None,
            summary_for_referral_fr=summary,
        )

    summary = (
        f"Cliché dermatoscopique correspondant à une {entity_fr}{site_suffix}. "
        "Statut indéterminé nécessitant une corrélation clinique ou un avis dermatologique."
    )
    return TriageAssessment(
        image_quality_adequate=True,
        melanocytic_pattern=melanocytic_pattern,
        entity_category=entity,
        research_triage_label="indeterminate",
        confidence=0.45,
        abstain=True,
        abstention_reason="Lésion atypique ou kératose actinique à statut indéterminé.",
        summary_for_referral_fr=summary,
    )


def build_triage_sft_messages(record: "ISICRecord") -> list[dict[str, Any]]:
    """Build Direct-mode SFT messages for ground-truth TriageAssessment."""
    clinical_ctx = {
        "age_approx": record.age_approx,
        "sex": record.sex,
        "anatom_site": record.anatom_site,
    }
    target = build_triage_target_from_record(record)
    messages = build_triage_inference_messages(clinical_context=clinical_ctx)
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<unused94>thought\n<unused95>"
                    + target.model_dump_json(ensure_ascii=False),
                }
            ],
        }
    )
    return messages


def mask_assistant_labels(
    input_ids_row: list[int],
    marker_ids: list[int],
    pad_token_id: int | None = None,
    extra_mask_ids: set[int] | None = None,
) -> list[int]:
    """Mask prompt, marker, pad, and vision tokens with -100 so loss trains only on JSON."""
    labels = list(input_ids_row)
    mask_set = set(extra_mask_ids or ())
    if pad_token_id is not None:
        mask_set.add(pad_token_id)

    marker_len = len(marker_ids)
    last_match_end = 0
    if marker_len > 0 and len(input_ids_row) >= marker_len:
        for i in range(len(input_ids_row) - marker_len, -1, -1):
            if input_ids_row[i : i + marker_len] == marker_ids:
                last_match_end = i + marker_len
                break

    for idx in range(last_match_end):
        labels[idx] = -100

    for idx, tok in enumerate(labels):
        if tok in mask_set:
            labels[idx] = -100

    return labels
