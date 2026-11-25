"""Edge-native System 1 vision triage classifier (NumPy/PIL preprocessing + ONNX Runtime).

Zero PyTorch dependency at inference time: designed to run offline on standard clinic PCs
and mobile/edge devices alongside MedGemma System 2.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from medgemma_dermato.eval.calibration import (
    INDETERMINATE_ENTITIES,
    MALIGNANT_ENTITIES,
    apply_temperature,
)

ENTITY_LABELS: tuple[str, ...] = (
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "nevus",
    "benign_keratosis",
    "dermatofibroma",
    "vascular",
    "actinic_keratosis_or_indeterminate",
    "other_benign",
)
ENTITY_TO_INDEX: dict[str, int] = {label: idx for idx, label in enumerate(ENTITY_LABELS)}

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class System1Config(BaseModel):
    """Serialization metadata and validation-calibrated parameters for ONNX System 1."""

    backbone_name: str = "convnext_tiny.fb_in22k_ft_in1k_384"
    image_size: int = Field(default=384, ge=128)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    entity_labels: tuple[str, ...] = ENTITY_LABELS
    temperature_malignant: float = Field(default=1.0, gt=0.0)
    temperature_risk: float = Field(default=1.0, gt=0.0)
    tau_mc_val_90_risk: float = Field(default=0.5, ge=0.0, le=1.0)
    tau_mc_val_95_risk: float = Field(default=0.5, ge=0.0, le=1.0)
    tau_mc_val_90_malignant: float = Field(default=0.5, ge=0.0, le=1.0)
    tau_mc_val_95_malignant: float = Field(default=0.5, ge=0.0, le=1.0)


class System1Prediction(BaseModel):
    """Calibrated output from Edge System 1 classifier for one dermoscopic image."""

    p_malignant_raw: float = Field(ge=0.0, le=1.0)
    p_malignant_calibrated: float = Field(ge=0.0, le=1.0)
    p_risk_raw: float = Field(ge=0.0, le=1.0)
    p_risk_calibrated: float = Field(ge=0.0, le=1.0)
    p_indeterminate: float = Field(ge=0.0, le=1.0)
    p_melanoma: float = Field(ge=0.0, le=1.0)
    entity_probs: dict[str, float]
    predicted_entity: str
    tta_views: int = Field(ge=1)
    latency_ms: float = Field(ge=0.0)


def build_tta4_views_pil(image: Image.Image) -> list[Image.Image]:
    """Return 4 rotation/flip-invariant dermoscopic views (orig, H-flip, V-flip, 90deg)."""
    rgb = image.convert("RGB")
    return [
        rgb,
        rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        rgb.transpose(Image.Transpose.ROTATE_90),
    ]


def preprocess_image_numpy(
    image: Image.Image,
    image_size: int = 384,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Resize RGB image with bicubic interpolation and normalize to CHW float32."""
    rgb = image.convert("RGB")
    resized = rgb.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    norm = (arr - mean_arr) / std_arr
    return np.ascontiguousarray(np.transpose(norm, (2, 0, 1)), dtype=np.float32)


def preprocess_batch_numpy(
    images: list[Image.Image],
    image_size: int = 384,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Stack multiple PIL images into an NCHW float32 batch."""
    if not images:
        raise ValueError("images list must not be empty")
    return np.stack(
        [preprocess_image_numpy(img, image_size=image_size, mean=mean, std=std) for img in images],
        axis=0,
    )


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def logits_to_system1_probs(
    malignancy_logits: np.ndarray,
    entity_logits: np.ndarray,
    entity_labels: tuple[str, ...] = ENTITY_LABELS,
) -> dict[str, Any]:
    """Convert single or multi-view (TTA) raw logits into unified calibrated-ready probabilities."""
    mal_arr = np.asarray(malignancy_logits, dtype=np.float64).reshape(-1)
    ent_arr = np.asarray(entity_logits, dtype=np.float64)
    if ent_arr.ndim == 1:
        ent_arr = ent_arr.reshape(1, -1)
    if ent_arr.shape[0] != mal_arr.shape[0]:
        raise ValueError(
            f"Batch size mismatch: malignancy_logits={mal_arr.shape[0]}, "
            f"entity_logits={ent_arr.shape[0]}"
        )
    if ent_arr.shape[1] != len(entity_labels):
        raise ValueError(
            f"Expected {len(entity_labels)} entity logits, got {ent_arr.shape[1]}"
        )

    p_mal_head = float(np.mean(_sigmoid_np(mal_arr)))
    ent_probs_mean = np.mean(_softmax_np(ent_arr), axis=0)
    entity_probs = {
        label: round(float(prob), 6)
        for label, prob in zip(entity_labels, ent_probs_mean, strict=True)
    }

    p_entity_mal = sum(entity_probs[e] for e in MALIGNANT_ENTITIES if e in entity_probs)
    p_entity_indet = sum(entity_probs[e] for e in INDETERMINATE_ENTITIES if e in entity_probs)
    p_entity_risk = min(1.0, max(0.0, p_entity_mal + p_entity_indet))

    p_malignant = float(np.clip(p_mal_head, 0.0, 1.0))
    p_risk = float(np.clip(0.5 * p_mal_head + 0.5 * p_entity_risk, 0.0, 1.0))
    p_melanoma = float(np.clip(entity_probs.get("melanoma", 0.0), 0.0, 1.0))
    predicted_entity = max(entity_probs.items(), key=lambda kv: kv[1])[0]

    return {
        "p_malignant": round(p_malignant, 6),
        "p_indeterminate": round(float(np.clip(p_entity_indet, 0.0, 1.0)), 6),
        "p_risk": round(p_risk, 6),
        "p_melanoma": round(p_melanoma, 6),
        "entity_probs": entity_probs,
        "predicted_entity": predicted_entity,
    }


class OnnxSystem1Classifier:
    """Offline ONNX Runtime System 1 classifier (CPU / Edge / Mobile compatible)."""

    def __init__(
        self,
        model_path: Path | str,
        calibration_path: Path | str | None = None,
        providers: list[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        if calibration_path is not None and Path(calibration_path).exists():
            raw_cfg = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            self.config = System1Config.model_validate(raw_cfg)
        else:
            self.config = System1Config()

        sess_opts = ort.SessionOptions()
        if intra_op_num_threads is not None:
            sess_opts.intra_op_num_threads = intra_op_num_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_image(self, image: Image.Image, tta: bool = True) -> System1Prediction:
        """Run System 1 inference on one PIL image with optional 4-view TTA."""
        t0 = time.perf_counter()
        views = build_tta4_views_pil(image) if tta else [image.convert("RGB")]
        batch = preprocess_batch_numpy(
            views,
            image_size=self.config.image_size,
            mean=self.config.mean,
            std=self.config.std,
        )
        outputs = self.session.run(None, {self.input_name: batch})
        mal_logits, ent_logits = outputs[0], outputs[1]
        probs = logits_to_system1_probs(
            mal_logits,
            ent_logits,
            entity_labels=self.config.entity_labels,
        )
        p_mal_cal = apply_temperature(
            [probs["p_malignant"]],
            self.config.temperature_malignant,
        )[0]
        p_risk_cal = apply_temperature(
            [probs["p_risk"]],
            self.config.temperature_risk,
        )[0]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return System1Prediction(
            p_malignant_raw=probs["p_malignant"],
            p_malignant_calibrated=round(p_mal_cal, 6),
            p_risk_raw=probs["p_risk"],
            p_risk_calibrated=round(p_risk_cal, 6),
            p_indeterminate=probs["p_indeterminate"],
            p_melanoma=probs["p_melanoma"],
            entity_probs=probs["entity_probs"],
            predicted_entity=probs["predicted_entity"],
            tta_views=len(views),
            latency_ms=latency_ms,
        )
