"""End-to-end tests for offline FastAPI Edge Cabinet App (SQLite, notes, PDF, compare)."""

import io
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from tests.test_agent import _MockSystem1, _SpyVLM, _textured_image


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    _textured_image(512).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_edge_app_full_lifecycle_analyze_history_pdf_compare_delete(tmp_path: Path) -> None:
    app = create_app(
        system1_override=_MockSystem1(p_risk=0.22, p_mel=0.10),
        vlm_override=_SpyVLM(p_risk=0.68),
        db_path=tmp_path / "edge_test.db",
    )
    client = TestClient(app)

    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["offline_edge"] is True
    assert h.json()["system2_llama_server_ready"] is True

    # 1. Analyze case 1 (baseline)
    r1 = client.post(
        "/api/analyze",
        files={"image": ("j0.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"patient_ref": "PAT-42", "lesion_label": "Dos", "age_years": "58"},
    )
    assert r1.status_code == 200
    c1 = r1.json()
    assert c1["routing_decision"] == "HYBRID_FULL_S1_PLUS_S2"

    # 2. Analyze case 2 (evolving red flag)
    r2 = client.post(
        "/api/analyze",
        files={"image": ("j90.jpg", _jpeg_bytes(), "image/jpeg")},
        data={
            "patient_ref": "PAT-42",
            "lesion_label": "Dos",
            "lesion_evolving": "true",
        },
    )
    assert r2.status_code == 200
    c2 = r2.json()
    assert c2["assessment"]["triage_zone"] == "RED_PRIORITY_DERMATOLOGY"

    # 3. History list + Notes PATCH
    hist = client.get("/api/cases", params={"patient_ref": "PAT-42"})
    assert hist.status_code == 200
    assert hist.json()["count"] == 2

    patched = client.patch(
        f"/api/cases/{c2['case_id']}/notes",
        json={"clinician_notes": "Adressage dermatologique prioritaire."},
    )
    assert patched.status_code == 200
    assert patched.json()["clinician_notes"] == "Adressage dermatologique prioritaire."

    # 4. PDF Export
    pdf_resp = client.get(f"/api/cases/{c2['case_id']}/report.pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF-")

    # 5. Longitudinal Comparison
    cmp_resp = client.get(
        "/api/cases/compare",
        params={"base_id": c1["case_id"], "followup_id": c2["case_id"]},
    )
    assert cmp_resp.status_code == 200
    assert cmp_resp.json()["comparison"]["base_case_id"] == c1["case_id"]

    # 6. GDPR Delete
    del_resp = client.delete(f"/api/cases/{c1['case_id']}")
    assert del_resp.status_code == 200
    assert client.get(f"/api/cases/{c1['case_id']}").status_code == 404


def test_edge_app_graceful_degraded_mode_when_vlm_offline(tmp_path: Path) -> None:
    class _OfflineVLM:
        def healthcheck(self) -> bool:
            return False

    app = create_app(
        system1_override=_MockSystem1(p_risk=0.05, p_mel=0.02),
        vlm_override=_OfflineVLM(),
        db_path=tmp_path / "edge_degraded.db",
    )
    client = TestClient(app)
    r = client.post(
        "/api/analyze",
        files={"image": ("lesion.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["routing_decision"] == "DEGRADED_SYS1_ONLY"
    assert body["system2_status"] == "offline_degraded_s1_only"
