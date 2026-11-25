"""Unit tests for 1-page French clinical referral PDF generator (fpdf2)."""

import base64
import io

from PIL import Image

from medgemma_dermato.agent import AgentConfig, DermoscopyTriageAgent
from medgemma_dermato.pdf_report import render_referral_pdf
from tests.test_agent import _MockSystem1, _SpyVLM, _textured_image


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), color=(170, 120, 110)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_render_referral_pdf_produces_valid_pdf_with_french_accents() -> None:
    agent = DermoscopyTriageAgent(
        system1=_MockSystem1(p_risk=0.62, p_mel=0.45),
        vlm_client=_SpyVLM(p_risk=0.75),
        agent_cfg=AgentConfig(adaptive_routing_enabled=False),
    )
    run_res = agent.run(_textured_image(512), case_id="PDF-TEST-01")
    rec = {
        "case_id": run_res.case_id,
        "created_utc": run_res.assessment.audit.timestamp_utc,
        "patient_ref": "PAT-ÉLOÏSE-99",
        "lesion_label": "Lésion dos → contrôle",
        "age_years": 62,
        "sex": "female",
        "anatom_site": "Back",
        "routing_decision": run_res.routing_decision,
        "clinician_notes": "Contrôle dermatoscopique à 3 mois — critères ABCDE vérifiés.",
        "original_b64": _png_b64(),
        "heatmap_b64": _png_b64(),
        "assessment": run_res.assessment.model_dump(),
        "action_plan": run_res.action_plan.model_dump(),
    }
    pdf_bytes = render_referral_pdf(rec)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2500
