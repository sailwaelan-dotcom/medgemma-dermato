"""100% Offline Edge FastAPI application wired to the Last-Mile Clinical Safety Harness."""

import hashlib
import io
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from medgemma_dermato.clinical_harness import (
    RUO_DISCLAIMER_FR,
    ClinicalContext,
    HarnessPolicy,
    LastMileClinicalHarness,
)
from medgemma_dermato.config import load_config
from medgemma_dermato.edge_vlm import EdgeVLMClient
from medgemma_dermato.preflight import evaluate_image_quality
from medgemma_dermato.system1 import OnnxSystem1Classifier, System1Prediction

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


def create_app(
    onnx_model_path: Path | None = None,
    onnx_calibration_path: Path | None = None,
    llama_server_url: str | None = None,
    config_path: Path = Path("configs/project.yaml"),
    system1_override: Any | None = None,
    vlm_override: Any | None = None,
) -> FastAPI:
    """Create the offline Edge FastAPI app with optional test dependency injection."""
    app = FastAPI(
        title="MedGemma Dermato — Edge Clinical Safety Harness (RUO)",
        version="1.0.0",
        description="Offline pre-analytical QA + System 1 ONNX + MedGemma GGUF + 3-Zone Harness.",
    )

    cfg = load_config(config_path) if config_path.exists() else load_config()
    model_path = onnx_model_path or Path(
        os.getenv("SYS1_ONNX_PATH", "checkpoints/system1_v1/system1_convnext_tiny_fp32.onnx")
    )
    cal_path = onnx_calibration_path or Path(
        os.getenv("SYS1_CAL_PATH", "checkpoints/system1_v1/system1_calibration.json")
    )
    vlm_url = llama_server_url or os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

    sys1_clf = system1_override
    if sys1_clf is None and model_path.exists():
        sys1_clf = OnnxSystem1Classifier(model_path=model_path, calibration_path=cal_path)

    vlm_client = vlm_override if vlm_override is not None else EdgeVLMClient(base_url=vlm_url)
    policy = HarnessPolicy(strict_preflight_block=False)
    harness = LastMileClinicalHarness(policy=policy)

    @app.get("/", response_class=HTMLResponse)
    def serve_ui() -> str:
        if not STATIC_INDEX.exists():
            return "<h1>MedGemma Dermato Edge API</h1><p>UI file missing.</p>"
        return STATIC_INDEX.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        sys2_ok = bool(vlm_client.healthcheck()) if vlm_client is not None else False
        return {
            "status": "ok",
            "offline_edge": True,
            "system1_onnx_ready": sys1_clf is not None,
            "system1_model": str(model_path) if sys1_clf is not None else None,
            "system2_llama_server_ready": sys2_ok,
            "system2_url": vlm_url,
            "policy_version": policy.policy_version,
            "ruo_disclaimer_fr": RUO_DISCLAIMER_FR,
        }

    @app.post("/api/analyze")
    async def analyze_lesion(
        image: UploadFile = File(...),
        age_years: int | None = Form(default=None),
        sex: str | None = Form(default=None),
        anatom_site: str | None = Form(default=None),
        lesion_evolving: bool = Form(default=False),
        lesion_bleeding_or_ulcerated: bool = Form(default=False),
        new_lesion_after_50: bool = Form(default=False),
        personal_or_family_melanoma_history: bool = Form(default=False),
        use_tta4: bool = Form(default=True),
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        raw_bytes = await image.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload.")

        image_sha = hashlib.sha256(raw_bytes).hexdigest()
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

        # Verrou 1: Pre-analytical image quality check
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_img.save(tmp, format="JPEG", quality=95)
            tmp_path = Path(tmp.name)
        try:
            preflight = evaluate_image_quality(tmp_path, cfg.preflight)
        finally:
            tmp_path.unlink(missing_ok=True)

        # System 1: Edge ONNX ConvNeXt-Tiny (< 1s CPU, zero PyTorch)
        if sys1_clf is None:
            raise HTTPException(
                status_code=503,
                detail="System 1 ONNX model is not loaded on disk.",
            )
        s1_pred: System1Prediction = sys1_clf.predict_image(pil_img, tta=use_tta4)

        # System 2: Local llama-server MedGemma GGUF (optional graceful degraded mode)
        clinical_ctx = ClinicalContext(
            age_years=age_years,
            sex=sex or None,
            anatom_site=anatom_site or None,
            lesion_evolving=lesion_evolving,
            lesion_bleeding_or_ulcerated=lesion_bleeding_or_ulcerated,
            new_lesion_after_50=new_lesion_after_50,
            personal_or_family_melanoma_history=personal_or_family_melanoma_history,
        )
        vlm_ctx = {
            "age_approx": age_years,
            "sex": sex or None,
            "anatom_site": anatom_site or None,
        }

        sys2_status = "offline_degraded_s1_only"
        sys2_latency_ms: float | None = None
        sys2_logprobs_used = False
        p_med_raw = s1_pred.p_risk_raw
        p_mel_med = s1_pred.p_melanoma
        med_top_entity: str | None = None
        med_parsed = None
        med_summary: str | None = None

        if vlm_client is not None and vlm_client.healthcheck():
            vlm_res = vlm_client.analyze_image(pil_img, clinical_context=vlm_ctx)
            sys2_status = "connected_llama_cpp"
            sys2_latency_ms = vlm_res.latency_ms
            sys2_logprobs_used = vlm_res.logprobs_used
            p_med_raw = vlm_res.p_risk_estimate
            p_mel_med = vlm_res.p_melanoma_estimate
            med_parsed = vlm_res.parsed
            if vlm_res.parsed is not None:
                med_top_entity = vlm_res.parsed.entity_category
                med_summary = vlm_res.parsed.summary_for_referral_fr

        case_id = f"EDGE-{uuid.uuid4().hex[:8].upper()}"
        assessment = harness.evaluate_case(
            case_id=case_id,
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
            clinical_context=clinical_ctx,
            image_sha256=image_sha,
        )

        if sys2_status != "connected_llama_cpp":
            assessment.safe_referral_summary_fr = (
                "[MODE DEGRADE SYSTEM 1 EDGE SEUL — llama-server MedGemma hors ligne] "
                + assessment.safe_referral_summary_fr
            )

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        return {
            "case_id": case_id,
            "system2_status": sys2_status,
            "system2_logprobs_used": sys2_logprobs_used,
            "system2_latency_ms": sys2_latency_ms,
            "total_latency_ms": total_ms,
            "preflight": preflight.model_dump(),
            "system1": s1_pred.model_dump(),
            "assessment": assessment.model_dump(),
        }

    return app


app = create_app()
