"""Zero-PyTorch OpenAI-compatible HTTP client for local llama.cpp (llama-server) MedGemma GGUF.

Supports native continuous token log-probabilities (`logprobs=True, top_logprobs=20`) at the
`entity_category` JSON value token so Edge Q8_0 produces calibrated continuous malignancy &
melanoma probabilities with zero extra inference passes.
"""

import base64
import io
import math
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from pydantic import BaseModel, Field

from medgemma_dermato.eval.calibration import INDETERMINATE_ENTITIES, MALIGNANT_ENTITIES
from medgemma_dermato.prompts import build_triage_inference_messages
from medgemma_dermato.schema import (
    EntityCategory,
    TriageAssessment,
    parse_triage_json,
    split_thinking_trace,
)

# Unique first-subword prefixes for Gemma 3 tokenizer across all 9 EntityCategory values.
_ENTITY_FIRST_TOKEN_PREFIXES: tuple[tuple[str, EntityCategory], ...] = (
    ("mel", "melanoma"),
    ("bas", "basal_cell_carcinoma"),
    ("squ", "squamous_cell_carcinoma"),
    ("nev", "nevus"),
    ("ben", "benign_keratosis"),
    ("der", "dermatofibroma"),
    ("vas", "vascular"),
    ("act", "actinic_keratosis_or_indeterminate"),
    ("oth", "other_benign"),
)


class EdgeVLMResult(BaseModel):
    """Structured inference result from local Edge llama-server (MedGemma GGUF)."""

    raw_text: str
    thinking_trace: str | None = None
    parsed: TriageAssessment | None = None
    parse_error: str | None = None
    p_risk_estimate: float = Field(ge=0.0, le=1.0)
    p_melanoma_estimate: float = Field(ge=0.0, le=1.0)
    logprobs_used: bool = False
    entity_probs: dict[str, float] = Field(default_factory=dict)
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


def _map_token_to_entity(token_str: str) -> EntityCategory | None:
    cleaned = token_str.strip().lstrip('"').lower()
    if not cleaned:
        return None
    if cleaned == "d":
        return "dermatofibroma"
    if cleaned == "v":
        return "vascular"
    if cleaned == "s":
        return "squamous_cell_carcinoma"
    for prefix, entity in _ENTITY_FIRST_TOKEN_PREFIXES:
        if cleaned.startswith(prefix):
            return entity
    return None


def extract_entity_probs_from_logprobs(
    logprobs_content: list[dict[str, Any]] | None,
) -> tuple[float, float, dict[str, float]] | None:
    """Extract continuous (p_risk, p_melanoma, entity_probs) from OpenAI chat logprobs.

    Locates the first generated value token immediately following `"entity_category":"` and
    aggregates probability mass across all 9 EntityCategory first-token candidates in
    `top_logprobs`. Returns None if the token marker or top_logprobs are missing.
    """
    if not logprobs_content:
        return None

    acc = ""
    target_entry: dict[str, Any] | None = None
    marker = '"entity_category":"'

    for idx, entry in enumerate(logprobs_content):
        tok = str(entry.get("token") or "")
        prev_has_marker = marker in acc
        acc += tok
        if not prev_has_marker and marker in acc:
            # Handle rare case where marker ends mid-token vs cleanly at token end
            after = acc.split(marker, 1)[1]
            if after:
                target_entry = entry
            elif idx + 1 < len(logprobs_content):
                target_entry = logprobs_content[idx + 1]
            break

    if target_entry is None:
        return None

    top_items = target_entry.get("top_logprobs") or []
    if not top_items:
        return None

    raw_mass: dict[str, float] = {}
    for cand in top_items:
        ent = _map_token_to_entity(str(cand.get("token") or ""))
        if ent is None:
            continue
        lp = cand.get("logprob")
        if lp is None:
            continue
        prob = math.exp(max(float(lp), -50.0))
        raw_mass[ent] = raw_mass.get(ent, 0.0) + prob

    total_mass = sum(raw_mass.values())
    if total_mass <= 1e-6:
        return None

    entity_probs = {ent: round(mass / total_mass, 6) for ent, mass in raw_mass.items()}
    p_mal = sum(entity_probs.get(e, 0.0) for e in MALIGNANT_ENTITIES)
    p_ind = sum(entity_probs.get(e, 0.0) for e in INDETERMINATE_ENTITIES)
    p_risk = min(max(p_mal + p_ind, 1e-4), 1.0 - 1e-4)
    p_mel = min(max(entity_probs.get("melanoma", 0.0), 0.0), 1.0)
    return round(p_risk, 6), round(p_mel, 6), entity_probs


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
        request_logprobs: bool = True,
        top_logprobs: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model_name = model_name
        self.request_logprobs = request_logprobs
        self.top_logprobs = top_logprobs
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
        """Run multimodal MedGemma inference via llama-server and extract continuous log-probs."""
        messages = build_openai_multimodal_messages(image, clinical_context=clinical_context)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 1 if temperature == 0.0 else 40,
            "top_p": 1.0,
        }
        if self.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs

        t0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        body = resp.json()
        choices = body.get("choices") or []
        raw_text = ""
        lp_content: list[dict[str, Any]] | None = None
        if choices:
            raw_text = str((choices[0].get("message") or {}).get("content") or "")
            lp_content = (choices[0].get("logprobs") or {}).get("content")

        usage = body.get("usage") or {}
        thought, _ = split_thinking_trace(raw_text)
        parsed, parse_err = parse_triage_json(raw_text)

        lp_extracted = extract_entity_probs_from_logprobs(lp_content)
        if lp_extracted is not None:
            p_risk, p_mel, entity_probs = lp_extracted
            logprobs_used = True
        else:
            p_risk, p_mel = estimate_risk_from_triage(parsed)
            entity_probs = {}
            logprobs_used = False

        return EdgeVLMResult(
            raw_text=raw_text,
            thinking_trace=thought,
            parsed=parsed,
            parse_error=parse_err,
            p_risk_estimate=p_risk,
            p_melanoma_estimate=p_mel,
            logprobs_used=logprobs_used,
            entity_probs=entity_probs,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
