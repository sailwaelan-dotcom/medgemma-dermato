"""Integration tests for the offline Edge FastAPI app and Last-Mile Harness endpoints."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from medgemma_dermato.edge_vlm import EdgeVLMResult
from medgemma_dermato.schema import TriageAssessment
from medgemma_dermato.system1 import ENTITY_LABELS, System1Prediction


class _StubSystem1:
    def __init__(self, p_risk: float = 0.06, tta_std: float = 0.01) -> None:
        self._p = p_risk
        self._std = tta_std

    def predict_image(self, image: Image.Image, tta: bool = True) -> System1Prediction:
        probs = {k: 0.01 for k in ENTITY_LABELS}
        probs["nevus"] = 0.92
        return System1Prediction(
            p_malignant_raw=self._p,
            p_malignant_calibrated=self._p,
            p_risk_raw=self._p,
            p_risk_calibrated=self._p,
            p_indeterminate=0.01,
            p_melanoma=0.01,
            entity_probs=probs,
            predicted_entity="nevus",
            tta_views=4 if tta else 1,
            tta_risk_std=self._std,
            tta_risk_range=self._std * 2.0,
            latency_ms=12.0,
        )


class _StubVLM:
    def __init__(self, online: bool = True, p_risk: float = 0.08) -> None:
        self._online = online
        self._p = p_risk

    def healthcheck(self) -> bool:
        return self._online

    def analyze_image(
        self,
        image: Image.Image,
        clinical_context: dict | None = None,
    ) -> EdgeVLMResult:
        parsed = TriageAssessment(
            image_quality_adequate=True,
            melanocytic_pattern="melanocytic",
            entity_category="nevus",
            research_triage_label="benign_pattern",
            confidence=0.85,
            abstain=False,
            summary_for_referral_fr="Naevus typique sans atypie majeure.",
        )
        return EdgeVLMResult(
            raw_text=parsed.model_dump_json(),
            parsed=parsed,
            p_risk_estimate=self._p,
            p_melanoma_estimate=0.02,
            latency_ms=150.0,
        )


def _jpeg_bytes() -> bytes:
    img = Image.new("RGB", (400, 400), color=(160, 120, 110))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_edge_app_health_and_red_flag_escalation() -> None:
    app = create_app(system1_override=_StubSystem1(0.05), vlm_override=_StubVLM(True, 0.05))
    client = TestClient(app)

    h = client.get("/api/health")
    assert h.status_code == 200
    body = h.json()
    assert body["system1_onnx_ready"] is True
    assert body["system2_llama_server_ready"] is True

    # Low risk + clinical red flag (lesion_evolving=True) -> MUST block GREEN -> AMBER
    resp = client.post(
        "/api/analyze",
        files={"image": ("lesion.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"age_years": "68", "lesion_evolving": "true"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["system2_status"] == "connected_llama_cpp"
    assert payload["assessment"]["triage_zone"] == "AMBER_UNCERTAIN_TELEEXPERTISE"
    assert "CLINICAL_RED_FLAG_EVOLVING" in payload["assessment"]["audit"]["triggered_rules"]
    assert payload["assessment"]["audit"]["image_sha256"]


def test_edge_app_graceful_degraded_mode_when_vlm_offline() -> None:
    app = create_app(system1_override=_StubSystem1(0.75), vlm_override=_StubVLM(online=False))
    client = TestClient(app)
    resp = client.post(
        "/api/analyze",
        files={"image": ("lesion.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["system2_status"] == "offline_degraded_s1_only"
    assert payload["assessment"]["triage_zone"] == "RED_PRIORITY_DERMATOLOGY"
    assert "MODE DEGRADE SYSTEM 1 EDGE SEUL" in payload["assessment"]["safe_referral_summary_fr"]
