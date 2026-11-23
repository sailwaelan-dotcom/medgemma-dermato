"""Fine-tune MedGemma 1.5 4B with QLoRA (NF4 + bfloat16) using official TRL pattern."""

import argparse
from pathlib import Path
from typing import Any

import httpx
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from medgemma_dermato.config import load_config
from medgemma_dermato.data.isic_ingest import ISICRecord, load_records_jsonl
from medgemma_dermato.prompts import build_sft_messages
from medgemma_dermato.schema import DermoscopyAssessment, SemiologyCriteria


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for MedGemma 1.5 4B.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/medgemma_qlora"))
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
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


def _record_to_target_assessment(record: ISICRecord) -> DermoscopyAssessment:
    """Build a baseline structured target from ISIC diagnosis metadata."""
    if record.is_malignant is True:
        triage = "suspicious_malignant_pattern"
        summary = (
            f"Aspect dermatoscopique suspect ({record.diagnosis_2 or 'lésion maligne'}). "
            "Avis dermatologique spécialisé requis."
        )
        conf = 0.80
    elif record.is_malignant is False:
        triage = "benign_pattern"
        summary = (
            f"Aspect dermatoscopique compatible avec une lésion bénigne "
            f"({record.diagnosis_2 or 'lésion bénigne'}). Surveillance clinique habituelle."
        )
        conf = 0.80
    else:
        triage = "indeterminate"
        summary = "Données diagnostiques insuffisantes : avis spécialisé recommandé."
        conf = 0.45

    return DermoscopyAssessment(
        image_quality_adequate=True,
        semiology=SemiologyCriteria(
            asymmetry="uncertain",
            pigment_network="uncertain",
            dots_and_globules="uncertain",
            streaks_or_pseudopods="uncertain",
            blue_white_veil="uncertain",
            vascular_structures="uncertain",
            regression_structures="uncertain",
            color_count=1,
            observed_colors=["brown"],
        ),
        confidence=conf,
        abstain=(conf < 0.5),
        abstention_reason=(
            "Étiquette clinique indéterminée dans le manifeste." if conf < 0.5 else None
        ),
        summary_for_referral_fr=summary,
        research_triage_label=triage,
    )


def _prepare_examples(records: list[ISICRecord], image_dir: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0) as client:
        for rec in records:
            img_path = _ensure_image(rec, image_dir, client)
            target = _record_to_target_assessment(rec)
            clinical_ctx = {
                "age_approx": rec.age_approx,
                "sex": rec.sex,
                "anatom_site": rec.anatom_site,
            }
            examples.append(
                {
                    "isic_id": rec.isic_id,
                    "image_path": str(img_path),
                    "messages": build_sft_messages(target, clinical_context=clinical_ctx),
                }
            )
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

    train_examples = _prepare_examples(train_records, args.image_dir)
    val_examples = _prepare_examples(val_records, args.image_dir)

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
        modules_to_save=cfg.qlora.modules_to_save,
    )

    boi_token = processor.tokenizer.special_tokens_map.get("boi_token")
    boi_token_id = (
        processor.tokenizer.convert_tokens_to_ids(boi_token) if boi_token else None
    )

    def collate_fn(batch_examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[list[Image.Image]] = []
        for ex in batch_examples:
            with Image.open(ex["image_path"]) as img:
                images.append([img.convert("RGB")])
            formatted = processor.apply_chat_template(
                ex["messages"],
                add_generation_prompt=False,
                tokenize=False,
            ).strip()
            texts.append(formatted)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        if boi_token_id is not None:
            labels[labels == boi_token_id] = -100
        labels[labels == 262144] = -100
        batch["labels"] = labels
        return batch

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=cfg.qlora.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=cfg.qlora.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.qlora.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=cfg.qlora.learning_rate,
        max_grad_norm=cfg.qlora.max_grad_norm,
        warmup_ratio=cfg.qlora.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=cfg.qlora.save_steps,
        eval_strategy="steps" if val_examples else "no",
        eval_steps=cfg.qlora.save_steps,
        bf16=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_examples,
        eval_dataset=val_examples if val_examples else None,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final_adapter"))
    processor.save_pretrained(str(args.output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
