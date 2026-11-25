"""Evaluate quantized Edge GGUF MedGemma via llama-server on locked MSKCC test set."""

import argparse
import json
from pathlib import Path

import httpx

from medgemma_dermato.config import load_config
from medgemma_dermato.data.isic_ingest import ISICRecord, load_records_jsonl
from medgemma_dermato.edge_vlm import EdgeVLMClient
from medgemma_dermato.eval.harness import (
    EvalCase,
    EvalReport,
    compare_reports,
    evaluate_predictions,
)
from medgemma_dermato.preflight import evaluate_image_quality
from medgemma_dermato.schema import map_isic_to_entity_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-quantization GGUF non-regression eval via llama-server."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--quantization", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--baseline-bf16-report", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast-invalid", type=int, default=5)
    return parser.parse_args()


def _ensure_image(record: ISICRecord, image_dir: Path, client: httpx.Client) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / f"{record.isic_id}.jpg"
    if target.exists() and target.stat().st_size > 100:
        return target
    if not record.image_url:
        raise ValueError(f"Missing image_url for {record.isic_id}")
    resp = client.get(record.image_url)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    return target


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    records = load_records_jsonl(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.output_dir / "predictions.jsonl"
    eval_cases: list[EvalCase] = []
    valid_count = 0

    with (
        EdgeVLMClient(base_url=args.server_url, timeout=180.0, model_name=args.run_name) as vlm,
        httpx.Client(timeout=30.0, follow_redirects=True) as dl_client,
        preds_path.open("w", encoding="utf-8") as pred_file,
    ):
        if not vlm.healthcheck():
            raise RuntimeError(f"llama-server unreachable at {args.server_url}")

        for idx, rec in enumerate(records, start=1):
            img_path = _ensure_image(rec, args.image_dir, dl_client)
            preflight = evaluate_image_quality(img_path, cfg.preflight)
            ctx = {
                "age_approx": rec.age_approx,
                "sex": rec.sex,
                "anatom_site": rec.anatom_site,
            }
            res = vlm.analyze_image(
                img_path,
                clinical_context=ctx,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            if res.parsed is not None and res.parse_error is None:
                valid_count += 1

            gt_entity = map_isic_to_entity_category(
                rec.diagnosis_1, rec.diagnosis_2, rec.diagnosis_3
            )
            case = EvalCase(
                isic_id=rec.isic_id,
                ground_truth_malignant=rec.is_malignant,
                ground_truth_diagnosis=rec.diagnosis_3 or rec.diagnosis_2 or rec.diagnosis_1,
                ground_truth_entity=gt_entity,
                ground_truth_is_melanoma=(gt_entity == "melanoma"),
                raw_model_output=res.raw_text,
                latency_ms=res.latency_ms,
            )
            eval_cases.append(case)
            pred_file.write(
                json.dumps(
                    {
                        "isic_id": rec.isic_id,
                        "preflight": preflight.model_dump(),
                        "thinking_trace": res.thinking_trace,
                        "eval_case": case.model_dump(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            pred_file.flush()

            if idx <= 5 or idx % 25 == 0 or idx == len(records):
                print(
                    f"[{args.quantization}] {idx}/{len(records)} | valid={valid_count}/{idx} | "
                    f"lat={res.latency_ms:.0f}ms | err={res.parse_error}",
                    flush=True,
                )

            if args.fail_fast_invalid > 0 and idx == args.fail_fast_invalid and valid_count == 0:
                raise RuntimeError(
                    f"Fail-fast: 0/{idx} valid JSON outputs from llama-server. "
                    f"Sample raw_text={res.raw_text[:300]!r}"
                )

    report = evaluate_predictions(
        cases=eval_cases,
        run_name=args.run_name,
        model_id=cfg.model.base_model_id,
        quantization=args.quantization,
        task_mode="triage",
    )
    (args.output_dir / "eval_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        f"# Edge GGUF Evaluation — {args.run_name} ({args.quantization})",
        "",
        f"- **Valid JSON rate :** {report.json_valid_rate * 100:.1f}% "
        f"({report.json_valid_count}/{report.total_cases})",
        f"- **Direct malignant sensitivity :** "
        f"{(report.malignant_sensitivity_direct or 0.0) * 100:.1f}%",
        f"- **Safe-triage malignant sensitivity :** "
        f"{(report.malignant_sensitivity_safe_triage or 0.0) * 100:.1f}%",
        f"- **Melanoma safe-triage sensitivity :** "
        f"{(report.melanoma_sensitivity_safe_triage or 0.0) * 100:.1f}%",
        f"- **Benign specificity :** {(report.specificity_direct or 0.0) * 100:.1f}%",
        f"- **Entity accuracy :** {(report.entity_accuracy or 0.0) * 100:.1f}%",
        f"- **ECE :** {report.ece_calibration_error or 0.0:.4f}",
        f"- **Mean latency :** {report.mean_latency_ms or 0.0:.1f} ms",
    ]

    if args.baseline_bf16_report and args.baseline_bf16_report.exists():
        bf16_report = EvalReport.model_validate_json(
            args.baseline_bf16_report.read_text(encoding="utf-8")
        )
        comp_md = compare_reports([bf16_report, report])
        md_lines.extend(["", "## Non-Regression vs BF16 QLoRA v1", "", comp_md])
        print("Non-regression vs BF16:\n" + comp_md, flush=True)

    (args.output_dir / "eval_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved GGUF eval report -> {args.output_dir / 'eval_report.json'}", flush=True)


if __name__ == "__main__":
    main()
