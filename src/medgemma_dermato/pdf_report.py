"""Printable 1-page clinical referral PDF generator (RUO / Hors-MDR) via fpdf2."""

import base64
import io
from typing import Any

from fpdf import FPDF

from medgemma_dermato.clinical_harness import RUO_DISCLAIMER_FR

_ZONE_TITLES: dict[str, tuple[str, tuple[int, int, int]]] = {
    "GREEN_LOW_RISK_SURVEILLANCE": (
        "ZONE VERTE - Faible risque calibre (Surveillance habituelle)",
        (22, 163, 74),
    ),
    "AMBER_UNCERTAIN_TELEEXPERTISE": (
        "ZONE GRISE - Incertitude / Garde-fou active (Tele-expertise recommandee)",
        (217, 119, 6),
    ),
    "RED_PRIORITY_DERMATOLOGY": (
        "ZONE ROUGE - Risque eleve (Adressage dermatologique prioritaire)",
        (220, 38, 38),
    ),
}


def _latin1_safe(text: Any) -> str:
    """Sanitize Unicode typography so built-in Helvetica never throws encoding errors."""
    s = str(text or "")
    replacements = {
        "\u2192": "->",
        "\u2194": "<->",
        "\u2014": "-",
        "\u2013": "-",
        "\u2022": "*",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u20ac": "EUR",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u0394": "Delta",
        "\u00a0": " ",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def render_referral_pdf(case_record: dict[str, Any]) -> bytes:
    """Render a concise, auditable French GP referral dossier PDF from one stored case."""
    a = case_record["assessment"]
    audit = a["audit"]
    action_plan = case_record.get("action_plan") or {}

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # Header & RUO Banner
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(248, 250, 252)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(
        pdf.epw,
        9,
        _latin1_safe("MedGemma Dermato - Fiche d'Adressage & Audit Edge (RUO)"),
        new_x="LMARGIN",
        new_y="NEXT",
        fill=True,
    )

    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(254, 243, 199)
    pdf.set_text_color(146, 64, 14)
    pdf.set_font("Helvetica", "B", 8)
    pdf.multi_cell(pdf.epw, 4.5, _latin1_safe(RUO_DISCLAIMER_FR), fill=True)
    pdf.ln(3)

    # Patient & Case Metadata
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 9.5)
    meta_parts = [
        f"Dossier : {case_record.get('case_id', 'N/A')}",
        f"Date UTC : {str(case_record.get('created_utc', ''))[:19]}",
        f"Ref. Patient : {case_record.get('patient_ref') or 'Non renseignee'}",
        f"Lesion : {case_record.get('lesion_label') or 'Non renseignee'}",
    ]
    pdf.cell(0, 5.5, _latin1_safe("  |  ".join(meta_parts)), new_x="LMARGIN", new_y="NEXT")

    demo = []
    if case_record.get("age_years") is not None:
        demo.append(f"Age : {case_record['age_years']} ans")
    if case_record.get("sex"):
        demo.append(f"Sexe : {case_record['sex']}")
    if case_record.get("anatom_site"):
        demo.append(f"Site : {case_record['anatom_site']}")
    demo.append(f"Routage : {case_record.get('routing_decision', 'N/A')}")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, _latin1_safe("  |  ".join(demo)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # 3-Zone Verdict Box
    zone = str(a.get("triage_zone", "AMBER_UNCERTAIN_TELEEXPERTISE"))
    z_label, (r, g, b) = _ZONE_TITLES.get(zone, (zone, (71, 85, 105)))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 8, _latin1_safe("  " + z_label), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)

    # Calibrated Risk Row
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 9)
    risk_line = (
        f"Risque Fusionne : {float(a['calibrated_risk_score']) * 100:.1f}%   |   "
        f"System 1 (ONNX) : {float(a['p_sys1_cal']) * 100:.1f}% "
        f"({a.get('sys1_top_entity') or 'n/a'})   |   "
        f"MedGemma S2 : {float(a['p_medgemma_cal']) * 100:.1f}%   |   "
        f"Instabilite TTA : {float(a.get('tta_risk_std') or 0.0):.3f}"
    )
    pdf.cell(0, 6, _latin1_safe(risk_line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Embedded Thumbnails (Original + ONNX Occlusion Heatmap)
    orig_b64 = case_record.get("original_b64")
    heat_b64 = case_record.get("heatmap_b64")
    if orig_b64 or heat_b64:
        y_img = pdf.get_y()
        img_w = 52
        if orig_b64:
            pdf.image(io.BytesIO(base64.b64decode(orig_b64)), x=12, y=y_img, w=img_w)
        if heat_b64:
            pdf.image(io.BytesIO(base64.b64decode(heat_b64)), x=70, y=y_img, w=img_w)
        pdf.set_y(y_img + img_w + 2)
        pdf.set_font("Helvetica", "I", 8)
        cap = "Gauche : Cliche dermatoscopique"
        if heat_b64:
            cap += "   |   Droite : Carte d'explicabilite ONNX (occlusion)"
        pdf.cell(0, 4.5, _latin1_safe(cap), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Triggered Guardrails
    rules = [r_item["code"] for r_item in a.get("triggered_rules", [])]
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(
        pdf.epw,
        5.5,
        _latin1_safe("Garde-fous Last-Mile declenches :"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(
        pdf.epw, 4.5, _latin1_safe(", ".join(rules) if rules else "AUCUNE_ALERTE_HARNESS")
    )
    pdf.ln(1.5)

    # Referral Summary & GP Notes
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(
        pdf.epw,
        5.5,
        _latin1_safe("Synthese clinique d'adressage (Hors-MDR) :"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    letter = action_plan.get("gp_referral_letter_fr") or a.get("safe_referral_summary_fr", "")
    pdf.multi_cell(pdf.epw, 4.5, _latin1_safe(letter))

    notes = (case_record.get("clinician_notes") or "").strip()
    if notes:
        pdf.ln(1.5)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(
            pdf.epw,
            5.5,
            _latin1_safe("Notes du medecin generaliste :"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(pdf.epw, 4.5, _latin1_safe(notes))

    qs = action_plan.get("missing_anamnesis_questions") or []
    if qs:
        pdf.ln(1.5)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(
            pdf.epw,
            5.5,
            _latin1_safe("Questions d'anamnese prioritaires :"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("Helvetica", "", 8.5)
        for q in qs[:4]:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 4.2, _latin1_safe(f"* {q['question_fr']}"))

    # Cryptographic Audit Footer
    pdf.ln(3)
    pdf.set_draw_color(203, 213, 225)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(1.5)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(71, 85, 105)
    audit_footer = (
        f"Audit SHA-256 : {audit.get('image_sha256') or 'n/a'} | "
        f"Policy : {audit.get('policy_version')} | "
        f"S1 : {audit.get('sys1_model_id')} | S2 : {audit.get('sys2_model_id')}"
    )
    pdf.multi_cell(pdf.epw, 3.8, _latin1_safe(audit_footer))

    raw = pdf.output()
    return bytes(raw) if isinstance(raw, (bytes, bytearray)) else raw.encode("latin-1")
