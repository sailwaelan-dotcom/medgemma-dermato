"""100% Offline Edge FastAPI app powered by DermoscopyTriageAgent + Last-Mile Safety Harness."""

import hashlib
import io
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from medgemma_dermato.agent import AgentConfig, DermoscopyTriageAgent
from medgemma_dermato.clinical_harness import (
    RUO_DISCLAIMER_FR,
    ClinicalContext,
    HarnessPolicy,
    LastMileClinicalHarness,
)
from medgemma_dermato.config import load_config
from medgemma_dermato.edge_vlm import EdgeVLMClient
from medgemma_dermato.system1 import OnnxSystem1Classifier

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


def create_app(
    onnx_model_path: Path | None = None,
    onnx_calibration_path: Path | None = None,
    llama_server_url: str | None = None,
    config_path: Path = Path("configs/project.yaml"),
    system1_override: Any | None = None,
    vlm_override: Any | None = None,
    agent_cfg: AgentConfig | None = None,
) -> FastAPI:
    """Create the offline Edge FastAPI app backed by DermoscopyTriageAgent."""
    app = FastAPI(
        title="MedGemma Dermato — Adaptive Clinical Triage Agent & Safety Harness (RUO)",
        version="1.1.0",
        description=(
            "Offline preflight QA + System 1 ONNX + Adaptive MedGemma GGUF + 6-Barrier Harness."
        ),
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
    agent = (
        DermoscopyTriageAgent(
            system1=sys1_clf,
            vlm_client=vlm_client,
            harness=harness,
            preflight_cfg=cfg.preflight,
            agent_cfg=agent_cfg or AgentConfig(),
        )
        if sys1_clf is not None
        else None
    )

    @app.get("/", response_class=HTMLResponse)
    def serve_ui() -> str:
        if not STATIC_INDEX.exists():
            return "<h1>MedGemma Dermato Edge Agent API</h1><p>UI file missing.</p>"
        return STATIC_INDEX.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        sys2_ok = bool(vlm_client.healthcheck()) if vlm_client is not None else False
        return {
            "status": "ok",
            "offline_edge": True,
            "agent_enabled": agent is not None,
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
        adaptive_routing: bool = Form(default=False),
    ) -> dict[str, Any]:
        raw_bytes = await image.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload.")

        image_sha = hashlib.sha256(raw_bytes).hexdigest()
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="System 1 ONNX model is not loaded on disk.",
            )

        agent.agent_cfg.adaptive_routing_enabled = adaptive_routing
        clinical_ctx = ClinicalContext(
            age_years=age_years,
            sex=sex or None,
            anatom_site=anatom_site or None,
            lesion_evolving=lesion_evolving,
            lesion_bleeding_or_ulcerated=lesion_bleeding_or_ulcerated,
            new_lesion_after_50=new_lesion_after_50,
            personal_or_family_melanoma_history=personal_or_family_melanoma_history,
        )
        run_res = agent.run(
            image=pil_img,
            clinical_context=clinical_ctx,
            use_tta4=use_tta4,
            image_sha256=image_sha,
        )

        sys2_status = (
            "connected_llama_cpp"
            if run_res.routing_decision == "HYBRID_FULL_S1_PLUS_S2"
            else (
                "fast_path_skipped"
                if run_res.routing_decision == "FAST_PATH_SYS1_ONLY"
                else "offline_degraded_s1_only"
            )
        )
        return {
            "case_id": run_res.case_id,
            "routing_decision": run_res.routing_decision,
            "vlm_invoked": run_res.vlm_invoked,
            "roi_crop_applied": run_res.roi_crop_applied,
            "system2_status": sys2_status,
            "system2_logprobs_used": run_res.system2_logprobs_used,
            "system2_latency_ms": run_res.system2_latency_ms,
            "total_latency_ms": run_res.total_latency_ms,
            "preflight": run_res.preflight.model_dump(),
            "system1": run_res.system1.model_dump(),
            "assessment": run_res.assessment.model_dump(),
            "action_plan": run_res.action_plan.model_dump(),
            "trace": [step.model_dump() for step in run_res.trace],
        }

    return app


app = create_app()
