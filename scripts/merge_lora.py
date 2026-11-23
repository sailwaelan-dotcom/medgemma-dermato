"""Merge a trained QLoRA adapter into BF16 base weights for Edge GGUF export."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from medgemma_dermato.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with MedGemma 1.5 4B.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_id = args.base_model or cfg.model.base_model_id

    print(f"Loading base model {model_id} in bfloat16...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Merging LoRA adapter from {args.adapter_path}...", flush=True)
    peft_model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    merged_model = peft_model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(args.output_dir), safe_serialization=True)
    processor.save_pretrained(str(args.output_dir))
    print(f"Merged BF16 model saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
