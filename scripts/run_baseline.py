"""Run MedGemma baseline or LoRA adapter inference and score with Eval Harness."""

import argparse
import json
import time
import warnings
from pathlib import Path

import httpx
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from medgemma_dermato.config import load_config
from medgemma_dermato.data.isic_ingest import ISICRecord, load_records_jsonl
from medgemma_dermato.eval.harness import EvalCase, compare_reports, evaluate_predictions
from medgemma_dermato.preflight import evaluate_image_quality
from medgemma_dermato.prompts import build_inference_messages, build_triage_inference_messages
from medgemma_dermato.schema import (
    map_isic_to_entity_category,
    parse_assessment_json,
    parse_triage_json,
    split_thinking_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MedGemma on a locked ISIC manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to test.jsonl manifest.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument(
        "--adapter-path", type=Path, default=None, help="Optional LoRA adapter dir."
    )
    parser.add_argument("--quantization", type=str, default="BF16")
    parser.add_argument("--run-name", type=str, default="medgemma-1.5-4b-baseline")
    parser.add_argument("--task", choices=["triage", "semiology"], default="triage")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Prefill empty <unused94>thought\\n<unused95> block for fast direct JSON output.",
    )
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_id = args.model_id or cfg.model.base_model_id

    if not torch.cuda.is_available():
        warnings.warn("CUDA unavailable: running on CPU will be extremely slow.", stacklevel=1)
        dtype = torch.float32
    elif torch.cuda.get_device_capability(0)[0] < 8:
        major, minor = torch.cuda.get_device_capability(0)
        warnings.warn(
            f"GPU compute capability {major}.{minor} (< 8.0, e.g. T4) lacks native hardware "
            "bfloat16. Falling back to float16; prefer L4 or A100 for MedGemma stability.",
            stacklevel=1,
        )
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    records = load_records_jsonl(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.adapter_path is not None:
        print(f"Loading and merging LoRA adapter from {args.adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
        model = model.merge_and_unload()
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    eval_cases: list[EvalCase] = []

    with (
        httpx.Client(timeout=30.0) as http_client,
        predictions_path.open("w", encoding="utf-8") as pred_file,
    ):
        for idx, rec in enumerate(records, start=1):
            img_path = _ensure_image(rec, args.image_dir, http_client)
            preflight = evaluate_image_quality(img_path, cfg.preflight)

            clinical_ctx = {
                "age_approx": rec.age_approx,
                "sex": rec.sex,
                "anatom_site": rec.anatom_site,
            }
            if args.task == "triage":
                messages = build_triage_inference_messages(clinical_context=clinical_ctx)
            else:
                messages = build_inference_messages(clinical_context=clinical_ctx)

            prompt_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if args.disable_thinking:
                prompt_text = f"{prompt_text}<unused94>thought\n<unused95>"

            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                inputs = processor(
                    text=[prompt_text],
                    images=[[rgb_img]],
                    return_tensors="pt",
                ).to(model.device, dtype=dtype)

            input_len = inputs["input_ids"].shape[-1]
            t0 = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            output_tokens = generated[0][input_len:]
            raw_text = processor.decode(output_tokens, skip_special_tokens=True)

            thinking_trace, _ = split_thinking_trace(raw_text)
            if args.task == "triage":
                parsed, parse_err = parse_triage_json(raw_text)
            else:
                parsed, parse_err = parse_assessment_json(raw_text)

            gt_entity = map_isic_to_entity_category(
                rec.diagnosis_1, rec.diagnosis_2, rec.diagnosis_3
            )
            case = EvalCase(
                isic_id=rec.isic_id,
                ground_truth_malignant=rec.is_malignant,
                ground_truth_diagnosis=rec.diagnosis_3 or rec.diagnosis_2 or rec.diagnosis_1,
                ground_truth_entity=gt_entity,
                ground_truth_is_melanoma=(gt_entity == "melanoma"),
                raw_model_output=raw_text,
                latency_ms=round(latency_ms, 2),
            )
            eval_cases.append(case)

            pred_record = {
                "isic_id": rec.isic_id,
                "preflight": preflight.model_dump(),
                "thinking_trace": thinking_trace,
                "eval_case": case.model_dump(),
            }
            pred_file.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
            pred_file.flush()

            status_tag = "JSON_OK" if parsed is not None else f"PARSE_ERR({(parse_err or '')[:60]})"
            print(
                f"[{idx}/{len(records)}] {rec.isic_id} | {latency_ms:.0f} ms | {status_tag}",
                flush=True,
            )

    report = evaluate_predictions(
        cases=eval_cases,
        run_name=args.run_name,
        model_id=model_id,
        quantization=args.quantization,
        abstention_threshold=cfg.eval.abstention_confidence_threshold,
        task_mode=args.task,
    )

    report_json_path = args.output_dir / "eval_report.json"
    report_json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    markdown = compare_reports(
        [report], min_malignant_sensitivity=cfg.eval.target_malignant_sensitivity
    )
    report_md_path = args.output_dir / "eval_report.md"
    report_md_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
