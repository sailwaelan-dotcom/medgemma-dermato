"""Centralized configuration models and YAML loader."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Base VLM configuration."""

    base_model_id: str = "google/medgemma-1.5-4b-it"
    image_size: int = Field(default=896, ge=224)
    dtype: Literal["bfloat16", "float16"] = "bfloat16"


class PreflightConfig(BaseModel):
    """Thresholds for deterministic pre-analytical image quality checks."""

    min_width: int = Field(default=256, ge=64)
    min_height: int = Field(default=256, ge=64)
    eval_max_side: int = Field(default=512, ge=128)
    min_laplacian_var: float = Field(default=55.0, gt=0.0)
    min_mean_brightness: float = Field(default=35.0, ge=0.0, le=255.0)
    max_mean_brightness: float = Field(default=225.0, ge=0.0, le=255.0)
    max_specular_ratio: float = Field(default=0.08, ge=0.0, le=1.0)
    max_hair_ratio: float = Field(default=0.18, ge=0.0, le=1.0)
    max_border_dark_ratio: float = Field(default=0.65, ge=0.0, le=1.0)


class DataConfig(BaseModel):
    """ISIC ingestion, licensing, and leakage-free split configuration."""

    commercial_licenses: list[str] = Field(default_factory=lambda: ["CC-0", "CC-BY", "CC-BY-4.0"])
    allowed_image_types: list[str] = Field(default_factory=lambda: ["dermoscopic"])
    excluded_holdout_collections: list[int] = Field(
        default_factory=lambda: [60, 62, 63, 64, 66, 67, 69, 71, 73, 212]
    )
    train_val_collections: list[int] = Field(default_factory=lambda: [251])
    external_test_collections: list[int] = Field(default_factory=lambda: [163])
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def _check_ratios(self) -> "DataConfig":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        return self


class QLoRAConfig(BaseModel):
    """QLoRA fine-tuning hyperparameters aligned with MedGemma official recipe."""

    r: int = Field(default=16, ge=4)
    lora_alpha: int = Field(default=32, ge=4)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: str | list[str] = "all-linear"
    modules_to_save: list[str] = Field(default_factory=list)
    learning_rate: float = Field(default=1.0e-4, gt=0.0)
    num_train_epochs: int = Field(default=2, ge=1)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    save_steps: int = Field(default=50, ge=1)
    max_grad_norm: float = Field(default=0.3, gt=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)


class EvalConfig(BaseModel):
    """Locked evaluation thresholds."""

    abstention_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    target_malignant_sensitivity: float = Field(default=0.95, ge=0.0, le=1.0)


class ProjectConfig(BaseModel):
    """Root project configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_config(path: Path | str = "configs/project.yaml") -> ProjectConfig:
    """Load and validate project configuration from a YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ProjectConfig.model_validate(raw)
