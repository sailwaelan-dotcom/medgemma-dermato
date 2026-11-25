"""Zero-PyTorch visual occlusion explainability for System 1 ONNX ConvNeXt-Tiny."""

import base64
import io
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from medgemma_dermato.eval.calibration import apply_temperature
from medgemma_dermato.preflight import crop_dermoscopic_roi
from medgemma_dermato.system1 import logits_to_system1_probs, preprocess_image_numpy


def encode_pil_png_b64(image: Image.Image, max_side: int = 384) -> str:
    """Encode a PIL RGB image into a compact base64 PNG string."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        rgb = rgb.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            resample=Image.Resampling.BICUBIC,
        )
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_occlusion_heatmap(
    system1: Any,
    image: Image.Image,
    grid_size: int = 4,
    baseline_risk_cal: float | None = None,
    roi_crop_applied: bool = False,
) -> dict[str, Any] | None:
    """Compute a single-batch ONNX occlusion saliency map in pure NumPy/OpenCV.

    Returns base64 PNG strings (`original_b64`, `heatmap_b64`, `roi_b64`) plus `peak_drop`
    and `latency_ms`. Returns None gracefully if `system1` lacks an ONNX Runtime session.
    """
    session = getattr(system1, "session", None)
    input_name = getattr(system1, "input_name", None)
    cfg = getattr(system1, "config", None)
    if session is None or input_name is None or cfg is None:
        return None

    t0 = time.perf_counter()
    img_size = int(cfg.image_size)
    rgb = image.convert("RGB").resize((img_size, img_size), resample=Image.Resampling.BICUBIC)
    arr_rgb = np.asarray(rgb, dtype=np.float32)

    # Estimate neutral skin background color from central 50% median so occlusion mimics skin.
    q1, q3 = img_size // 4, (3 * img_size) // 4
    fill_rgb = np.median(arr_rgb[q1:q3, q1:q3], axis=(0, 1)).astype(np.float32)

    base_tensor = preprocess_image_numpy(rgb, image_size=img_size, mean=cfg.mean, std=cfg.std)
    if baseline_risk_cal is None:
        base_out = session.run(None, {input_name: base_tensor[None, ...]})
        base_probs = logits_to_system1_probs(
            base_out[0], base_out[1], entity_labels=cfg.entity_labels
        )
        baseline_risk_cal = apply_temperature([base_probs["p_risk"]], cfg.temperature_risk)[0]

    # Build grid_size^2 soft-Gaussian occluded tensors in a single batched ONNX call
    yy, xx = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    step = img_size / float(grid_size)
    sigma = step * 0.65
    mean_arr = np.asarray(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(cfg.std, dtype=np.float32).reshape(1, 1, 3)

    batch_tensors = np.empty((grid_size * grid_size, 3, img_size, img_size), dtype=np.float32)
    centers: list[tuple[int, int]] = []
    idx = 0
    for gy in range(grid_size):
        cy = (gy + 0.5) * step
        for gx in range(grid_size):
            cx = (gx + 0.5) * step
            mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))[..., None]
            occ = arr_rgb * (1.0 - mask) + fill_rgb.reshape(1, 1, 3) * mask
            norm = ((occ / 255.0) - mean_arr) / std_arr
            batch_tensors[idx] = np.transpose(norm, (2, 0, 1))
            centers.append((gy, gx))
            idx += 1

    outputs = session.run(None, {input_name: batch_tensors})
    mal_logits, ent_logits = outputs[0], outputs[1]

    saliency_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for idx, (gy, gx) in enumerate(centers):
        probs_i = logits_to_system1_probs(
            mal_logits[idx : idx + 1],
            ent_logits[idx : idx + 1],
            entity_labels=cfg.entity_labels,
        )
        occ_risk = apply_temperature([probs_i["p_risk"]], cfg.temperature_risk)[0]
        saliency_grid[gy, gx] = max(0.0, float(baseline_risk_cal) - float(occ_risk))

    peak_drop = float(np.max(saliency_grid))
    norm_grid = saliency_grid / max(peak_drop, 1e-6)
    saliency_up = cv2.resize(norm_grid, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    saliency_u8 = np.clip(saliency_up * 255.0, 0, 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(saliency_u8, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = np.clip(arr_rgb * 0.58 + heatmap_rgb * 0.42, 0, 255).astype(np.uint8)

    roi_b64: str | None = None
    if roi_crop_applied:
        roi_img = crop_dermoscopic_roi(image.convert("RGB"))
        roi_b64 = encode_pil_png_b64(roi_img, max_side=384)

    return {
        "original_b64": encode_pil_png_b64(image, max_side=384),
        "heatmap_b64": encode_pil_png_b64(Image.fromarray(blended, mode="RGB"), max_side=384),
        "roi_b64": roi_b64,
        "peak_drop": round(peak_drop, 4),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }
