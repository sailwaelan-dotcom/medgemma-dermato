"""100% Offline Edge FastAPI app powered by DermoscopyTriageAgent + Last-Mile Safety Harness."""

import hashlib
import io
import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image, UnidentifiedImageError

from medgemma_dermato.agent import AgentConfig, ClinicalTriageOrchestrator
from medgemma_dermato.clinical_harness import (
    RUO_DISCLAIMER_FR,
    ClinicalContext,
    HarnessPolicy,
    LastMileClinicalHarness,
)
from medgemma_dermato.config import load_config
from medgemma_dermato.db_model_bridge import (
    StorageFormatIntegrityError,
    build_validated_model_inputs,
)
from medgemma_dermato.edge_vlm import EdgeVLMClient
from medgemma_dermato.explainability import compute_occlusion_heatmap, encode_pil_png_b64
from medgemma_dermato.pdf_report import render_referral_pdf
from medgemma_dermato.storage import LocalCaseStore
from medgemma_dermato.system1 import ENTITY_LABELS, OnnxSystem1Classifier, System1Prediction

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


class DemoSandboxSystem1:
    """Deterministic lightweight fallback used ONLY when ONNX weights are absent on disk."""

    def predict_image(self, image: Image.Image, tta: bool = True) -> System1Prediction:
        import numpy as np

        arr = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)
        mean_r = float(np.mean(arr[:, :, 0]))
        luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        mean_luma = float(np.mean(luma))
        # Heuristic synthetic placeholder risk strictly for UI exploration
        raw_risk = round(min(max((mean_r - mean_luma + 40.0) / 140.0, 0.08), 0.72), 4)
        p_mel = round(raw_risk * 0.45, 4)
        probs = {e: 0.02 for e in ENTITY_LABELS}
        probs["melanoma"] = p_mel
        probs["nevus"] = max(0.10, round(1.0 - raw_risk, 4))
        return System1Prediction(
            p_malignant_raw=raw_risk,
            p_malignant_calibrated=raw_risk,
            p_risk_raw=raw_risk,
            p_risk_calibrated=raw_risk,
            p_indeterminate=0.02,
            p_melanoma=p_mel,
            entity_probs=probs,
            predicted_entity="melanoma" if raw_risk >= 0.45 else "nevus",
            tta_views=4 if tta else 1,
            tta_risk_std=0.02,
            tta_risk_range=0.04,
            latency_ms=12.0,
        )


def create_app(
    onnx_model_path: Path | None = None,
    onnx_calibration_path: Path | None = None,
    llama_server_url: str | None = None,
    config_path: Path = Path("configs/project.yaml"),
    system1_override: Any | None = None,
    vlm_override: Any | None = None,
    agent_cfg: AgentConfig | None = None,
    db_path: Path | str | None = None,
) -> FastAPI:
    """Create the offline Edge FastAPI app backed by DermoscopyTriageAgent + SQLite."""
    app = FastAPI(
        title="MedGemma Dermato — Adaptive Clinical Triage Agent & Safety Harness (RUO)",
        version="2.0.0",
        description=(
            "Offline preflight QA + System 1 ONNX + Adaptive MedGemma GGUF + 6-Barrier Harness "
            "+ Local SQLite Longitudinal Store + ONNX Explainability + PDF Export."
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
    resolved_db = Path(db_path) if db_path is not None else Path(
        os.getenv("MEDGEMMA_DB_PATH", "data/local_cases.db")
    )
    store = LocalCaseStore(db_path=resolved_db)

    demo_mode = False
    sys1_clf = system1_override
    if sys1_clf is None:
        if model_path.exists():
            sys1_clf = OnnxSystem1Classifier(model_path=model_path, calibration_path=cal_path)
        elif os.getenv("MEDGEMMA_STRICT_NO_DEMO", "0") != "1":
            sys1_clf = DemoSandboxSystem1()
            demo_mode = True

    vlm_client = vlm_override if vlm_override is not None else EdgeVLMClient(base_url=vlm_url)
    policy = HarnessPolicy(strict_preflight_block=False)
    harness = LastMileClinicalHarness(policy=policy)
    agent = (
        ClinicalTriageOrchestrator(
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
            "version": "2.0.0",
            "agent_enabled": agent is not None,
            "demo_mode": demo_mode,
            "system1_onnx_ready": sys1_clf is not None and not demo_mode,
            "system1_model": str(model_path) if (sys1_clf is not None and not demo_mode) else None,
            "system2_llama_server_ready": sys2_ok,
            "system2_url": vlm_url,
            "db_path": str(resolved_db),
            "policy_version": policy.policy_version,
            "ruo_disclaimer_fr": RUO_DISCLAIMER_FR,
        }

    @app.post("/api/analyze")
    async def analyze_lesion(
        image: UploadFile = File(...),
        patient_ref: str | None = Form(default=None),
        lesion_label: str | None = Form(default=None),
        age_years: int | None = Form(default=None),
        sex: str | None = Form(default=None),
        anatom_site: str | None = Form(default=None),
        lesion_evolving: bool = Form(default=False),
        lesion_bleeding_or_ulcerated: bool = Form(default=False),
        new_lesion_after_50: bool = Form(default=False),
        personal_or_family_melanoma_history: bool = Form(default=False),
        use_tta4: bool = Form(default=True),
        adaptive_routing: bool = Form(default=False),
        include_heatmap: bool = Form(default=True),
        prior_case_id: str | None = Form(default=None),
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

        prior_record: dict[str, Any] | None = None
        if prior_case_id and prior_case_id.strip():
            try:
                prior_record = store.get_case(prior_case_id.strip())
            except StorageFormatIntegrityError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Bridge DB integrity error: {exc}"
                ) from exc
            if prior_record is None:
                raise HTTPException(
                    status_code=404, detail=f"Prior case {prior_case_id} not found in SQLite."
                )

        validated_meta, prior_dto = build_validated_model_inputs(
            prior_case=prior_record,
            current_age=age_years,
            current_sex=sex,
            current_site=anatom_site,
        )

        clinical_ctx = ClinicalContext(
            age_years=validated_meta.age_approx,
            sex=validated_meta.sex,
            anatom_site=validated_meta.anatom_site,
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
            prior_context=prior_dto,
            adaptive_routing=adaptive_routing,
        )

        original_b64 = encode_pil_png_b64(pil_img, max_side=384)
        heatmap_b64: str | None = None
        roi_b64: str | None = None
        if include_heatmap:
            exp = compute_occlusion_heatmap(
                sys1_clf,
                pil_img,
                baseline_risk_cal=run_res.assessment.p_sys1_cal,
                roi_crop_applied=run_res.roi_crop_applied,
            )
            if exp is not None:
                original_b64 = exp["original_b64"]
                heatmap_b64 = exp["heatmap_b64"]
                roi_b64 = exp["roi_b64"]

        stored = store.save_case(
            run_res=run_res,
            patient_ref=patient_ref or (prior_record.get("patient_ref") if prior_record else None),
            lesion_label=(
                lesion_label or (prior_record.get("lesion_label") if prior_record else None)
            ),
            age_years=validated_meta.age_approx,
            sex=validated_meta.sex,
            anatom_site=validated_meta.anatom_site,
            original_b64=original_b64,
            heatmap_b64=heatmap_b64,
            roi_b64=roi_b64,
        )
        stored["bridge_validated"] = True
        stored["demo_mode"] = demo_mode
        stored["prior_case_id"] = prior_dto.prior_case_id if prior_dto is not None else None

        sys2_status = (
            "connected_llama_cpp"
            if run_res.routing_decision == "HYBRID_FULL_S1_PLUS_S2"
            else (
                "fast_path_skipped"
                if run_res.routing_decision == "FAST_PATH_SYS1_ONLY"
                else "offline_degraded_s1_only"
            )
        )
        stored["system2_status"] = sys2_status
        return stored

    @app.get("/api/cases")
    def list_cases(
        zone: str | None = Query(default=None),
        patient_ref: str | None = Query(default=None),
        q: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        items = store.list_cases(zone=zone, patient_ref=patient_ref, query=q, limit=limit)
        return {"count": len(items), "cases": items}

    @app.get("/api/cases/compare")
    def compare_cases(
        base_id: str = Query(...),
        followup_id: str = Query(...),
    ) -> dict[str, Any]:
        try:
            cmp_res = store.compare_cases(base_id=base_id, followup_id=followup_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        base_case = store.get_case(base_id)
        follow_case = store.get_case(followup_id)
        return {
            "comparison": cmp_res.model_dump(),
            "base_case": base_case,
            "followup_case": follow_case,
        }

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        try:
            rec = store.get_case(case_id)
        except StorageFormatIntegrityError as exc:
            raise HTTPException(
                status_code=422, detail=f"Bridge DB integrity error: {exc}"
            ) from exc
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
        return rec

    @app.patch("/api/cases/{case_id}/notes")
    def update_case_notes(
        case_id: str,
        payload: dict[str, str] = Body(...),
    ) -> dict[str, Any]:
        notes = str(payload.get("clinician_notes") or "")
        updated = store.update_notes(case_id, notes)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
        return updated

    @app.delete("/api/cases/{case_id}")
    def delete_case(case_id: str) -> dict[str, Any]:
        if not store.delete_case(case_id):
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
        return {"deleted": True, "case_id": case_id}

    @app.get("/api/cases/{case_id}/report.pdf")
    def export_case_pdf(case_id: str) -> Response:
        rec = store.get_case(case_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
        pdf_bytes = render_referral_pdf(rec)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{case_id}_dossier.pdf"'},
        )

    return app


app = create_app()
