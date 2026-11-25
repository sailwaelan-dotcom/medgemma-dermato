"""Deterministic CPU pre-analytical quality gate for dermoscopic images."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from medgemma_dermato.config import PreflightConfig
from medgemma_dermato.schema import PreflightMetrics, PreflightResult, QualityIssue

_ISSUE_MESSAGES_FR: dict[QualityIssue, str] = {
    "low_resolution": "résolution insuffisante (reprendre à >= 256 px)",
    "blurry": "cliché flou (stabiliser le dermatoscope et refaire la mise au point)",
    "underexposed": "image sous-exposée (augmenter l'éclairage)",
    "overexposed": "image surexposée (réduire l'intensité lumineuse)",
    "specular_glare": "reflets spéculaires excessifs (ajouter du gel d'immersion ou polariser)",
    "excessive_hair": "pilosité excessive masquant la lésion (dégager le champ)",
    "poor_framing": "cadrage insuffisant ou vignettage trop important",
}


def _to_rgb_array(image: Path | str | Image.Image | np.ndarray) -> np.ndarray:
    """Load or normalize an input image into an RGB uint8 numpy array."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_img:
            return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, :3].astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.astype(np.uint8)
        raise ValueError(f"Unsupported ndarray shape for image: {image.shape}")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _resize_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Resize image preserving aspect ratio so max(height, width) <= max_side."""
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def evaluate_image_quality(
    image: Path | str | Image.Image | np.ndarray,
    config: PreflightConfig | None = None,
) -> PreflightResult:
    """Evaluate technical suitability of a dermoscopic image before VLM inference."""
    cfg = config or PreflightConfig()
    rgb_full = _to_rgb_array(image)
    orig_height, orig_width = rgb_full.shape[:2]

    rgb = _resize_max_side(rgb_full, cfg.eval_max_side)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Foreground ROI excludes dark circular dermoscope vignette corners while closing
    # thin dark interior structures (hairs, dark globules) inside the field of view.
    raw_fg = (gray > 15).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    roi_mask = cv2.morphologyEx(raw_fg, cv2.MORPH_CLOSE, close_kernel) > 0
    total_pixels = int(gray.size)
    roi_pixels = int(np.count_nonzero(roi_mask))
    border_dark_ratio = float(total_pixels - roi_pixels) / float(total_pixels)
    roi_coverage = float(roi_pixels) / float(total_pixels)

    # Erode ROI so the sharp circular vignette border does not inflate Laplacian or hair metrics.
    if roi_pixels > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eroded_roi = cv2.erode(roi_mask.astype(np.uint8), erode_kernel, iterations=1) > 0
        eval_mask = eroded_roi if int(np.count_nonzero(eroded_roi)) >= 100 else roi_mask
    else:
        eval_mask = np.ones_like(gray, dtype=bool)

    # Sharpness via variance of Laplacian on interior ROI.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(np.var(laplacian[eval_mask])) if np.any(eval_mask) else 0.0

    # Brightness inside foreground ROI.
    roi_gray = gray[roi_mask] if roi_pixels > 0 else gray.ravel()
    mean_brightness = float(np.mean(roi_gray))

    # Specular glare: very bright & desaturated pixels inside ROI.
    if roi_pixels > 0:
        s_vals = hsv[:, :, 1][roi_mask]
        v_vals = hsv[:, :, 2][roi_mask]
        specular_count = int(np.count_nonzero((v_vals > 240) & (s_vals < 30)))
        specular_ratio = float(specular_count) / float(roi_pixels)
    else:
        specular_ratio = 0.0

    # Hair artifact estimation via morphological black-hat inside eroded ROI.
    hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, hair_kernel)
    eval_pixels = int(np.count_nonzero(eval_mask))
    if eval_pixels > 0:
        hair_count = int(np.count_nonzero(blackhat[eval_mask] > 25))
        hair_ratio = float(hair_count) / float(eval_pixels)
    else:
        hair_ratio = 0.0

    issues: list[QualityIssue] = []
    if orig_width < cfg.min_width or orig_height < cfg.min_height:
        issues.append("low_resolution")
    if border_dark_ratio > cfg.max_border_dark_ratio or roi_coverage < 0.20:
        issues.append("poor_framing")
    if laplacian_var < cfg.min_laplacian_var:
        issues.append("blurry")
    if mean_brightness < cfg.min_mean_brightness:
        issues.append("underexposed")
    if mean_brightness > cfg.max_mean_brightness:
        issues.append("overexposed")
    if specular_ratio > cfg.max_specular_ratio:
        issues.append("specular_glare")
    if hair_ratio > cfg.max_hair_ratio:
        issues.append("excessive_hair")

    passed = len(issues) == 0
    if passed:
        recommendation_fr = (
            "Qualité technique suffisante pour analyse sémiologique et télé-expertise."
        )
    else:
        details = "; ".join(_ISSUE_MESSAGES_FR[issue] for issue in issues)
        recommendation_fr = f"Cliché refusé au contrôle pré-analytique : {details}."

    metrics = PreflightMetrics(
        width=orig_width,
        height=orig_height,
        laplacian_var=round(laplacian_var, 2),
        mean_brightness=round(mean_brightness, 2),
        specular_ratio=round(specular_ratio, 4),
        hair_ratio=round(hair_ratio, 4),
        border_dark_ratio=round(border_dark_ratio, 4),
    )

    return PreflightResult(
        passed=passed,
        issues=issues,
        metrics=metrics,
        recommendation_fr=recommendation_fr,
    )


def crop_dermoscopic_roi(
    image: Path | str | Image.Image | np.ndarray,
    dark_threshold: int = 15,
    min_dark_ratio: float = 0.05,
    pad_fraction: float = 0.04,
    eval_max_side: int = 512,
) -> Image.Image:
    """Crop out dark circular dermatoscope vignette borders while preserving lesion FOV."""
    rgb_full = _to_rgb_array(image)
    orig_h, orig_w = rgb_full.shape[:2]

    rgb_small = _resize_max_side(rgb_full, eval_max_side)
    small_h, small_w = rgb_small.shape[:2]
    gray = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)

    raw_fg = (gray > dark_threshold).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    roi_mask = cv2.morphologyEx(raw_fg, cv2.MORPH_CLOSE, close_kernel) > 0

    total_pixels = int(gray.size)
    roi_pixels = int(np.count_nonzero(roi_mask))
    if roi_pixels == 0:
        return Image.fromarray(rgb_full, mode="RGB")

    border_dark_ratio = float(total_pixels - roi_pixels) / float(total_pixels)
    roi_coverage = float(roi_pixels) / float(total_pixels)
    if border_dark_ratio < min_dark_ratio or roi_coverage < 0.20:
        return Image.fromarray(rgb_full, mode="RGB")

    ys, xs = np.nonzero(roi_mask)
    y0_s, y1_s = int(ys.min()), int(ys.max()) + 1
    x0_s, x1_s = int(xs.min()), int(xs.max()) + 1

    bbox_coverage = float((y1_s - y0_s) * (x1_s - x0_s)) / float(total_pixels)
    if bbox_coverage >= 0.92:
        return Image.fromarray(rgb_full, mode="RGB")

    scale_y = orig_h / float(small_h)
    scale_x = orig_w / float(small_w)
    y0 = int(round(y0_s * scale_y))
    y1 = int(round(y1_s * scale_y))
    x0 = int(round(x0_s * scale_x))
    x1 = int(round(x1_s * scale_x))

    pad_y = int(round((y1 - y0) * pad_fraction))
    pad_x = int(round((x1 - x0) * pad_fraction))
    y0 = max(0, y0 - pad_y)
    y1 = min(orig_h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(orig_w, x1 + pad_x)

    if (y1 - y0) < 64 or (x1 - x0) < 64:
        return Image.fromarray(rgb_full, mode="RGB")

    return Image.fromarray(rgb_full[y0:y1, x0:x1], mode="RGB")
