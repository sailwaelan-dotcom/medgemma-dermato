"""Fine-tune MedGemma 1.5 4B with QLoRA (NF4 + bfloat16) on ground-truth ISIC triage."""

import argparse
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from medgemma_dermato.config import load_config
from medgemma_dermato.data.isic_ingest import ISICRecord, load_records_jsonl
from medgemma_dermato.prompts import build_triage_sft_messages, mask_assistant_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for MedGemma 1.5 4B.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/medgemma_qlora"))
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--download-workers", type=int, default=16)
    return parser.parse_args()


def _ensure_image(record: ISICRecord, image_dir: Path, client: httpx.Client) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / f"{record.isic_id}.jpg"
    if target.exists() and target.stat().st_size > 0:
        return target
    if not record.image_url:
        raise ValueError(f"No image_url for record {record.isic_id}")
    resp = client.get(record.image_url)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    return target


def _prepare_examples(
    records: list[ISICRecord],
    image_dir: Path,
    split_name: str,
    workers: int = 16,
) -> list[dict[str, Any]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        rec
        for rec in records
        if not (
            (image_dir / f"{rec.isic_id}.jpg").exists()
            and (image_dir / f"{rec.isic_id}.jpg").stat().st_size > 0
        )
    ]
    if missing:
        print(
            f"[{split_name}] Downloading {len(missing)} missing images with {workers} workers...",
            flush=True,
        )

        def _dl(rec: ISICRecord) -> None:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                _ensure_image(rec, image_dir, client)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(_dl, missing))

    examples: list[dict[str, Any]] = []
    for rec in records:
        img_path = image_dir / f"{rec.isic_id}.jpg"
        examples.append(
            {
                "isic_id": rec.isic_id,
                "image_path": str(img_path),
                "messages_json": json.dumps(build_triage_sft_messages(rec), ensure_ascii=False),
            }
        )
    print(f"[{split_name}] Ready {len(examples)} examples.", flush=True)
    return examples


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError(
            "QLoRA training for MedGemma 1.5 4B requires a CUDA GPU with native hardware "
            "bfloat16 (SM >= 8.0, i.e. A100 or L4). T4/CPU are unsupported."
        )

    train_records = load_records_jsonl(args.train_manifest)
    val_records = load_records_jsonl(args.val_manifest)

    print(
        f"Preparing {len(train_records)} train and {len(val_records)} val examples...",
        flush=True,
    )
    train_examples = _prepare_examples(
        train_records, args.image_dir, "train", workers=args.download_workers
    )
    val_examples = _prepare_examples(
        val_records, args.image_dir, "val", workers=args.download_workers
    )

    model_id = cfg.model.base_model_id
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=cfg.qlora.r,
        lora_alpha=cfg.qlora.lora_alpha,
        lora_dropout=cfg.qlora.lora_dropout,
        bias="none",
        target_modules=cfg.qlora.target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=cfg.qlora.modules_to_save or None,
    )

    boi_token = processor.tokenizer.special_tokens_map.get("boi_token")
    extra_mask_ids: set[int] = {262144}
    if boi_token:
        boi_id = processor.tokenizer.convert_tokens_to_ids(boi_token)
        if boi_id is not None:
            extra_mask_ids.add(int(boi_id))

    response_marker_ids = processor.tokenizer.encode("<unused95>", add_special_tokens=False)

    def collate_fn(batch_examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[list[Image.Image]] = []
        for ex in batch_examples:
            with Image.open(ex["image_path"]) as img:
                images.append([img.convert("RGB")])
            formatted = processor.apply_chat_template(
                json.loads(ex["messages_json"]),
                add_generation_prompt=False,
                tokenize=False,
            ).strip()
            texts.append(formatted)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        masked_rows = [
            mask_assistant_labels(
                row.tolist(),
                marker_ids=response_marker_ids,
                pad_token_id=processor.tokenizer.pad_token_id,
                extra_mask_ids=extra_mask_ids,
            )
            for row in batch["input_ids"]
        ]
        batch["labels"] = torch.tensor(masked_rows, dtype=batch["input_ids"].dtype)
        return batch

    sft_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    warmup_kwargs: dict[str, float] = (
        {"warmup_ratio": cfg.qlora.warmup_ratio}
        if "warmup_ratio" in sft_fields
        else {"warmup_steps": cfg.qlora.warmup_ratio}
    )

    save_steps = args.save_steps or cfg.qlora.save_steps
    eval_steps = args.eval_steps or save_steps

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs or cfg.qlora.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=(
            args.per_device_batch_size or cfg.qlora.per_device_train_batch_size
        ),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum or cfg.qlora.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.lr or cfg.qlora.learning_rate,
        max_grad_norm=cfg.qlora.max_grad_norm,
        **warmup_kwargs,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_examples else "no",
        eval_steps=eval_steps,
        load_best_model_at_end=bool(val_examples),
        metric_for_best_model="eval_loss" if val_examples else None,
        greater_is_better=False if val_examples else None,
        bf16=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to="none",
    )

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(val_examples) if val_examples else None

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    print("Starting QLoRA fine-tuning...", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved final LoRA adapter to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
