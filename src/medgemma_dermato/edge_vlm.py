"""Zero-PyTorch OpenAI-compatible HTTP client for local llama.cpp (llama-server) MedGemma GGUF."""

import base64
import io
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from pydantic import BaseModel, Field

from medgemma_dermato.prompts import build_triage_inference_messages
from medgemma_dermato.schema import TriageAssessment, parse_triage_json, split_thinking_trace


class EdgeVLMResult(BaseModel):
    """Structured inference result from local Edge llama-server (MedGemma GGUF)."""

    raw_text: str
    thinking_trace: str | None = None
    parsed: TriageAssessment | None = None
    parse_error: str | None = None
    p_risk_estimate: float = Field(ge=0.0, le=1.0)
    p_melanoma_estimate: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _encode_image_data_uri(image: Image.Image | Path | str, max_side: int = 896) -> str:
    """Convert PIL image or path to resized RGB JPEG data URI for llama-server multimodal API."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            rgb = im.convert("RGB")
    else:
        rgb = image.convert("RGB")

    w, h = rgb.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        rgb = rgb.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            resample=Image.Resampling.BICUBIC,
        )

    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_openai_multimodal_messages(
    image: Image.Image | Path | str,
    clinical_context: dict[str, Any] | None = None,
    max_side: int = 896,
) -> list[dict[str, Any]]:
    """Adapt canonical triage prompt into OpenAI /v1/chat/completions multimodal message format."""
    canonical = build_triage_inference_messages(clinical_context=clinical_context)
    sys_text = canonical[0]["content"][0]["text"]
    user_text = canonical[1]["content"][1]["text"]
    data_uri = _encode_image_data_uri(image, max_side=max_side)
    return [
        {"role": "system", "content": sys_text},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def estimate_risk_from_triage(parsed: TriageAssessment | None) -> tuple[float, float]:
    """Derive conservative (p_risk, p_melanoma) proxy when token log-probs are unavailable."""
    if parsed is None:
        return 0.55, 0.25
    conf = min(max(float(parsed.confidence), 0.05), 0.95)
    if parsed.research_triage_label == "suspicious_malignant_pattern":
        p_risk = conf
        p_mel = conf if parsed.entity_category == "melanoma" else 0.15
        return round(p_risk, 4), round(p_mel, 4)
    if parsed.research_triage_label == "indeterminate" or parsed.abstain:
        return 0.55, 0.20
    return round(1.0 - conf, 4), 0.03


class EdgeVLMClient:
    """HTTP client targeting a local offline llama-server serving MedGemma GGUF + mmproj FP16."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 120.0,
        model_name: str = "medgemma-dermato-v1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model_name = model_name
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> "EdgeVLMClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def healthcheck(self) -> bool:
        """Return True if llama-server responds ready on /health."""
        try:
            resp = self._client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def analyze_image(
        self,
        image: Image.Image | Path | str,
        clinical_context: dict[str, Any] | None = None,
        max_tokens: int = 320,
        temperature: float = 0.0,
    ) -> EdgeVLMResult:
        """Run multimodal MedGemma inference via llama-server and parse TriageAssessment."""
        messages = build_openai_multimodal_messages(image, clinical_context=clinical_context)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 1 if temperature == 0.0 else 40,
            "top_p": 1.0,
        }
        t0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        body = resp.json()
        choices = body.get("choices") or []
        raw_text = ""
        if choices:
            raw_text = str((choices[0].get("message") or {}).get("content") or "")

        usage = body.get("usage") or {}
        thought, _ = split_thinking_trace(raw_text)
        parsed, parse_err = parse_triage_json(raw_text)
        p_risk, p_mel = estimate_risk_from_triage(parsed)

        return EdgeVLMResult(
            raw_text=raw_text,
            thinking_trace=thought,
            parsed=parsed,
            parse_error=parse_err,
            p_risk_estimate=p_risk,
            p_melanoma_estimate=p_mel,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
