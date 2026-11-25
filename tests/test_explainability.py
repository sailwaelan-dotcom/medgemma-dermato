"""Unit tests for zero-PyTorch ONNX occlusion visual explainability."""

import base64
import io

import numpy as np
from PIL import Image

from medgemma_dermato.explainability import compute_occlusion_heatmap
from medgemma_dermato.system1 import ENTITY_LABELS, System1Config
from tests.test_agent import _MockSystem1


class _FakeOnnxSession:
    def run(self, _: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = next(iter(inputs.values()))
        n = batch.shape[0]
        # Give slightly higher logit to patch 0 so peak_drop > 0
        mal_logits = np.zeros((n,), dtype=np.float32)
        if n > 1:
            mal_logits[5] = -2.5
        ent_logits = np.zeros((n, len(ENTITY_LABELS)), dtype=np.float32)
        return [mal_logits, ent_logits]


class _FakeSystem1WithSession:
    def __init__(self) -> None:
        self.session = _FakeOnnxSession()
        self.input_name = "pixel_values"
        self.config = System1Config()


def test_occlusion_heatmap_generates_valid_base64_pngs() -> None:
    img = Image.new("RGB", (512, 512), color=(180, 140, 120))
    res = compute_occlusion_heatmap(_FakeSystem1WithSession(), img, grid_size=4)
    assert res is not None
    assert "original_b64" in res and "heatmap_b64" in res
    decoded = base64.b64decode(res["heatmap_b64"])
    im = Image.open(io.BytesIO(decoded))
    assert im.size == (384, 384)
    assert res["peak_drop"] >= 0.0


def test_occlusion_heatmap_returns_none_gracefully_for_mock_without_session() -> None:
    img = Image.new("RGB", (256, 256), color=(150, 130, 110))
    assert compute_occlusion_heatmap(_MockSystem1(p_risk=0.2), img) is None
