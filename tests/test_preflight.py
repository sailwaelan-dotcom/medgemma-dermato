"""Tests for deterministic CPU pre-analytical image quality checks."""

import cv2
import numpy as np

from medgemma_dermato.preflight import evaluate_image_quality


def _make_clean_dermoscopy(size: int = 512) -> np.ndarray:
    """Create a realistic synthetic dermoscopy image that passes all quality gates."""
    rng = np.random.default_rng(123)
    yy, xx = np.ogrid[:size, :size]
    cy, cx = size / 2.0, size / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Warm skin-toned background inside circular field of view.
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = 195
    img[:, :, 1] = 152
    img[:, :, 2] = 132

    # Fine skin micro-texture (moderate high-frequency detail, not dark thin hairs).
    texture = rng.normal(0.0, 7.0, size=(size, size, 1))
    img = np.clip(img.astype(np.float32) + texture, 40, 230).astype(np.uint8)

    # Central pigmented lesion with crisp pigment network dots/globules.
    lesion_mask = dist < (size * 0.19)
    img[lesion_mask, 0] = np.clip(img[lesion_mask, 0].astype(np.int16) - 65, 20, 255).astype(
        np.uint8
    )
    img[lesion_mask, 1] = np.clip(img[lesion_mask, 1].astype(np.int16) - 70, 20, 255).astype(
        np.uint8
    )
    img[lesion_mask, 2] = np.clip(img[lesion_mask, 2].astype(np.int16) - 75, 20, 255).astype(
        np.uint8
    )

    # Subtle dermoscope corner vignette (~12% border pixels).
    vignette = dist > (size * 0.56)
    img[vignette] = 0
    return img


def test_clean_image_passes_preflight() -> None:
    clean = _make_clean_dermoscopy(512)
    result = evaluate_image_quality(clean)
    assert result.passed is True, f"Unexpected issues: {result.issues}, metrics={result.metrics}"
    assert result.issues == []


def test_low_resolution_flagged() -> None:
    small = cv2.resize(_make_clean_dermoscopy(512), (180, 180), interpolation=cv2.INTER_AREA)
    result = evaluate_image_quality(small)
    assert result.passed is False
    assert "low_resolution" in result.issues


def test_blurry_image_flagged() -> None:
    clean = _make_clean_dermoscopy(512)
    blurred = cv2.GaussianBlur(clean, (21, 21), sigmaX=7.0)
    result = evaluate_image_quality(blurred)
    assert result.passed is False
    assert "blurry" in result.issues


def test_underexposed_and_overexposed_flagged() -> None:
    clean = _make_clean_dermoscopy(512)

    dark = np.clip(clean.astype(np.float32) * 0.14, 18, 255).astype(np.uint8)
    dark_res = evaluate_image_quality(dark)
    assert dark_res.passed is False
    assert "underexposed" in dark_res.issues

    bright = clean.copy()
    fg = bright.max(axis=2) > 15
    bright[fg] = np.clip(bright[fg].astype(np.float32) + 85.0, 0, 238).astype(np.uint8)
    bright_res = evaluate_image_quality(bright)
    assert bright_res.passed is False
    assert "overexposed" in bright_res.issues


def test_specular_glare_flagged() -> None:
    img = _make_clean_dermoscopy(512)
    # Paint bright white glare spots covering ~15% of the frame.
    img[160:340, 160:340] = 252
    result = evaluate_image_quality(img)
    assert result.passed is False
    assert "specular_glare" in result.issues


def test_excessive_hair_flagged() -> None:
    img = _make_clean_dermoscopy(512)
    # Draw dense dark hair lines across the center ROI (>18% coverage).
    for offset in range(40, 470, 7):
        cv2.line(img, (40, offset), (470, offset + 25), (32, 24, 20), thickness=2)
        cv2.line(img, (offset, 40), (offset - 20, 470), (35, 26, 22), thickness=2)
    result = evaluate_image_quality(img)
    assert result.passed is False
    assert "excessive_hair" in result.issues


def test_poor_framing_flagged() -> None:
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 70, (185, 140, 120), thickness=-1)
    result = evaluate_image_quality(img)
    assert result.passed is False
    assert "poor_framing" in result.issues
