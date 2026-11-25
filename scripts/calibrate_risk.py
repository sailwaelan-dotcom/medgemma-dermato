"""Extract exact chain-rule log-prob risk scores from MedGemma QLoRA and calibrate on val."""

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx
import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from medgemma_dermato.config import load_config
from medgemma_dermato.data.isic_ingest import ISICRecord, load_records_jsonl
from medgemma_dermato.eval.calibration import (
    INDETERMINATE_ENTITIES,
    MALIGNANT_ENTITIES,
    OperatingPointMetrics,
    RiskScoreCase,
    apply_temperature,
    evaluate_operating_point,
    fit_temperature_scaling,
    render_roc_and_reliability_svg,
    select_threshold_for_sensitivity,
)
from medgemma_dermato.preflight import crop_dermoscopic_roi
from medgemma_dermato.prompts import build_triage_inference_messages
from medgemma_dermato.schema import map_isic_to_entity_category, parse_triage_json

MELANOCYTIC_VALUES: tuple[str, ...] = ("melanocytic", "non_melanocytic", "uncertain")
ENTITY_VALUES: tuple[str, ...] = (
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "nevus",
    "benign_keratosis",
    "dermatofibroma",
    "vascular",
    "actinic_keratosis_or_indeterminate",
    "other_benign",
)
BASE_JSON_PREFIX = '{"image_quality_adequate":true,"melanocytic_pattern":"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract exact continuous log-prob risk scores and calibrate strictly on val."
    )
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v1_qlora_calibration"))
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["raw_1view", "raw_tta4", "roi_tta4"],
        default=["raw_1view", "raw_tta4", "roi_tta4"],
    )
    parser.add_argument(
        "--greedy-test-predictions",
        type=Path,
        default=None,
        help="Optional greedy predictions.jsonl on test.jsonl to verify argmax parity.",
    )
    parser.add_argument("--limit", type=int, default=None)
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


def _build_views(rgb_img: Image.Image, mode: str) -> list[Image.Image]:
    base = crop_dermoscopic_roi(rgb_img) if mode == "roi_tta4" else rgb_img
    if mode == "raw_1view":
        return [base]
    return [
        base,
        base.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        base.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        base.transpose(Image.Transpose.ROTATE_90),
    ]


def _encode_suffix_after_boundary(
    tokenizer: Any,
    boundary_str: str,
    suffix_str: str,
) -> list[int]:
    """Encode suffix_str strictly after boundary_str and verify zero token merge across boundary."""
    b_ids = tokenizer.encode(boundary_str, add_special_tokens=False)
    full_ids = tokenizer.encode(boundary_str + suffix_str, add_special_tokens=False)
    if full_ids[: len(b_ids)] != b_ids:
        raise RuntimeError(
            f"Tokenizer merged across boundary {boundary_str!r} + {suffix_str!r}: "
            f"boundary={b_ids[-3:]}, full={full_ids[len(b_ids) - 3 : len(b_ids) + 3]}"
        )
    return full_ids[len(b_ids) :]


def _score_continuation_with_cache(
    model: Any,
    base_logits: torch.Tensor,
    base_cache: Any,
    base_len: int,
    token_ids: list[int],
    device: torch.device,
) -> tuple[float, torch.Tensor, Any]:
    """Compute exact log P(token_ids | prefix) and return (logp, final_logits, updated_cache)."""
    first_log_probs = F.log_softmax(base_logits[0, -1, :].float(), dim=-1)
    logp = float(first_log_probs[token_ids[0]].item())

    ext = torch.tensor([token_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones((1, base_len + len(token_ids)), dtype=torch.long, device=device)
    cache_pos = torch.arange(base_len, base_len + len(token_ids), dtype=torch.long, device=device)
    cache_copy = copy.deepcopy(base_cache)

    with torch.inference_mode():
        out = model(
            input_ids=ext,
            attention_mask=attn_mask,
            past_key_values=cache_copy,
            cache_position=cache_pos,
            use_cache=True,
        )

    if len(token_ids) > 1:
        step_log_probs = F.log_softmax(out.logits[0, :-1, :].float(), dim=-1)
        for idx in range(len(token_ids) - 1):
            logp += float(step_log_probs[idx, token_ids[idx + 1]].item())

    return logp, out.logits, out.past_key_values


def _score_single_view(
    model: Any,
    processor: Any,
    pil_img: Image.Image,
    clinical_ctx: dict[str, Any],
    mel_value_tokens: dict[str, list[int]],
    mel_bridge_tokens: dict[str, list[int]],
    entity_value_tokens: dict[str, list[int]],
    entities_unique_first_token: bool,
    dtype: torch.dtype,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return exact (melanocytic_probs, entity_probs) for one image view via chain rule."""
    messages = build_triage_inference_messages(clinical_context=clinical_ctx)
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    base_prefix = f"{prompt_text}<unused94>thought\n<unused95>{BASE_JSON_PREFIX}"

    inputs = processor(
        text=[base_prefix],
        images=[[pil_img]],
        return_tensors="pt",
    ).to(model.device, dtype=dtype)
    base_len = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        base_out = model(**inputs, use_cache=True)

    base_logits = base_out.logits
    base_cache = base_out.past_key_values

    mel_logps: dict[str, float] = {}
    entity_cond_probs: dict[str, dict[str, float]] = {}

    for m_val in MELANOCYTIC_VALUES:
        full_toks = mel_bridge_tokens[m_val]
        val_tok_len = len(mel_value_tokens[m_val])

        # One tiny KV-cached pass computes BOTH log P(m_val + '","' | x) and logits at entity start!
        ext = torch.tensor([full_toks], dtype=torch.long, device=model.device)
        attn_mask = torch.ones(
            (1, base_len + len(full_toks)), dtype=torch.long, device=model.device
        )
        cache_pos = torch.arange(
            base_len, base_len + len(full_toks), dtype=torch.long, device=model.device
        )
        cache_copy = copy.deepcopy(base_cache)
        with torch.inference_mode():
            branch_out = model(
                input_ids=ext,
                attention_mask=attn_mask,
                past_key_values=cache_copy,
                cache_position=cache_pos,
                use_cache=True,
            )

        # Exact log-prob of m_val + '","' (= first val_tok_len tokens of full_toks)
        first_lp = F.log_softmax(base_logits[0, -1, :].float(), dim=-1)
        m_logp = float(first_lp[full_toks[0]].item())
        if val_tok_len > 1:
            rest_lp = F.log_softmax(branch_out.logits[0, : val_tok_len - 1, :].float(), dim=-1)
            for idx in range(val_tok_len - 1):
                m_logp += float(rest_lp[idx, full_toks[idx + 1]].item())
        mel_logps[m_val] = m_logp

        # Entity conditional distribution P(e | m_val, x)
        ent_start_logits = branch_out.logits[0, -1, :].float()
        if entities_unique_first_token:
            ent_first_lp = F.log_softmax(ent_start_logits, dim=-1)
            raw_ent_logps = {
                e_val: float(ent_first_lp[entity_value_tokens[e_val][0]].item())
                for e_val in ENTITY_VALUES
            }
        else:
            ent_cache_len = base_len + len(full_toks)
            raw_ent_logps = {}
            for e_val in ENTITY_VALUES:
                e_logp, _, _ = _score_continuation_with_cache(
                    model=model,
                    base_logits=branch_out.logits,
                    base_cache=branch_out.past_key_values,
                    base_len=ent_cache_len,
                    token_ids=entity_value_tokens[e_val],
                    device=model.device,
                )
                raw_ent_logps[e_val] = e_logp

        max_e = max(raw_ent_logps.values())
        exp_e = {k: math.exp(v - max_e) for k, v in raw_ent_logps.items()}
        sum_e = sum(exp_e.values())
        entity_cond_probs[m_val] = {k: v / sum_e for k, v in exp_e.items()}

    max_m = max(mel_logps.values())
    exp_m = {k: math.exp(v - max_m) for k, v in mel_logps.items()}
    sum_m = sum(exp_m.values())
    mel_probs = {k: v / sum_m for k, v in exp_m.items()}

    marg_entity_probs = {e_val: 0.0 for e_val in ENTITY_VALUES}
    for m_val, pm in mel_probs.items():
        for e_val, pe_given_m in entity_cond_probs[m_val].items():
            marg_entity_probs[e_val] += pm * pe_given_m

    tot = sum(marg_entity_probs.values())
    marg_entity_probs = {k: v / tot for k, v in marg_entity_probs.items()}
    return mel_probs, marg_entity_probs


def _run_split_mode(
    records: list[ISICRecord],
    split_name: str,
    mode: str,
    model: Any,
    processor: Any,
    image_dir: Path,
    out_jsonl: Path,
    mel_value_tokens: dict[str, list[int]],
    mel_bridge_tokens: dict[str, list[int]],
    entity_value_tokens: dict[str, list[int]],
    entities_unique_first_token: bool,
    dtype: torch.dtype,
) -> list[RiskScoreCase]:
    cases: list[RiskScoreCase] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with (
        httpx.Client(timeout=30.0) as http_client,
        out_jsonl.open("w", encoding="utf-8") as fout,
    ):
        for idx, rec in enumerate(records, start=1):
            img_path = _ensure_image(rec, image_dir, http_client)
            clinical_ctx = {
                "age_approx": rec.age_approx,
                "sex": rec.sex,
                "anatom_site": rec.anatom_site,
            }
            t0 = time.perf_counter()
            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                views = _build_views(rgb_img, mode)

            agg_mel = {m: 0.0 for m in MELANOCYTIC_VALUES}
            agg_ent = {e: 0.0 for e in ENTITY_VALUES}
            for view in views:
                v_mel, v_ent = _score_single_view(
                    model=model,
                    processor=processor,
                    pil_img=view,
                    clinical_ctx=clinical_ctx,
                    mel_value_tokens=mel_value_tokens,
                    mel_bridge_tokens=mel_bridge_tokens,
                    entity_value_tokens=entity_value_tokens,
                    entities_unique_first_token=entities_unique_first_token,
                    dtype=dtype,
                )
                for m in MELANOCYTIC_VALUES:
                    agg_mel[m] += v_mel[m] / len(views)
                for e in ENTITY_VALUES:
                    agg_ent[e] += v_ent[e] / len(views)

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            p_mal = min(1.0, max(0.0, sum(agg_ent[e] for e in MALIGNANT_ENTITIES)))
            p_indet = min(1.0, max(0.0, sum(agg_ent[e] for e in INDETERMINATE_ENTITIES)))
            p_risk = min(1.0, max(0.0, p_mal + p_indet))
            p_mel = min(1.0, max(0.0, agg_ent["melanoma"]))
            pred_ent = max(agg_ent.items(), key=lambda kv: kv[1])[0]
            gt_entity = map_isic_to_entity_category(
                rec.diagnosis_1, rec.diagnosis_2, rec.diagnosis_3
            )

            case = RiskScoreCase(
                isic_id=rec.isic_id,
                ground_truth_malignant=rec.is_malignant,
                ground_truth_is_melanoma=(gt_entity == "melanoma"),
                ground_truth_entity=gt_entity,
                melanocytic_probs={k: round(v, 6) for k, v in agg_mel.items()},
                entity_probs={k: round(v, 6) for k, v in agg_ent.items()},
                predicted_entity_argmax=pred_ent,
                p_malignant=round(p_mal, 6),
                p_indeterminate=round(p_indet, 6),
                p_risk=round(p_risk, 6),
                p_melanoma=round(p_mel, 6),
                latency_ms=latency_ms,
            )
            cases.append(case)
            fout.write(case.model_dump_json() + "\n")
            fout.flush()

            if idx % 20 == 0 or idx == len(records) or idx <= 5:
                print(
                    f"[{split_name}/{mode} {idx}/{len(records)}] {rec.isic_id} | "
                    f"{latency_ms:.0f} ms | pred={pred_ent} | "
                    f"p_risk={p_risk:.3f} (mal={p_mal:.3f}, mel={p_mel:.3f})",
                    flush=True,
                )
    return cases


def _check_greedy_parity(
    test_cases: list[RiskScoreCase],
    greedy_path: Path,
) -> dict[str, Any]:
    greedy_by_id: dict[str, str] = {}
    for line in greedy_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        parsed, _ = parse_triage_json(item["eval_case"]["raw_model_output"])
        if parsed is not None:
            greedy_by_id[item["isic_id"]] = parsed.entity_category

    matched = total = 0
    mismatches: list[dict[str, Any]] = []
    for c in test_cases:
        if c.isic_id not in greedy_by_id:
            continue
        total += 1
        g_ent = greedy_by_id[c.isic_id]
        if c.predicted_entity_argmax == g_ent:
            matched += 1
        elif len(mismatches) < 10:
            mismatches.append(
                {
                    "isic_id": c.isic_id,
                    "greedy": g_ent,
                    "marginal_argmax": c.predicted_entity_argmax,
                    "top2": sorted(c.entity_probs.items(), key=lambda kv: -kv[1])[:2],
                }
            )
    rate = round(matched / total, 4) if total > 0 else 0.0
    return {"matched": matched, "total": total, "agreement_rate": rate, "mismatches": mismatches}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_id = args.model_id or cfg.model.base_model_id

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("Native bfloat16 CUDA GPU (SM >= 8.0) required.")
    dtype = torch.bfloat16

    val_records = load_records_jsonl(args.val_manifest)
    test_records = load_records_jsonl(args.test_manifest)
    if args.limit is not None:
        val_records = val_records[: args.limit]
        test_records = test_records[: args.limit]

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    mel_value_tokens: dict[str, list[int]] = {}
    mel_bridge_tokens: dict[str, list[int]] = {}
    for m_val in MELANOCYTIC_VALUES:
        # Encode full branch string directly so Gemma 3's '","' token never splits across boundary.
        full_branch = _encode_suffix_after_boundary(
            tokenizer, BASE_JSON_PREFIX, f'{m_val}","entity_category":"'
        )
        val_with_sep = _encode_suffix_after_boundary(tokenizer, BASE_JSON_PREFIX, f'{m_val}","')
        if full_branch[: len(val_with_sep)] != val_with_sep:
            raise RuntimeError(f"Unexpected tokenization split inside {m_val!r} branch")
        mel_value_tokens[m_val] = val_with_sep
        mel_bridge_tokens[m_val] = full_branch

    entity_value_tokens: dict[str, list[int]] = {}
    sample_ent_boundary = BASE_JSON_PREFIX + 'melanocytic","entity_category":"'
    for e_val in ENTITY_VALUES:
        entity_value_tokens[e_val] = _encode_suffix_after_boundary(
            tokenizer, sample_ent_boundary, f'{e_val}","'
        )

    first_ent_ids = {e: toks[0] for e, toks in entity_value_tokens.items()}
    entities_unique_first_token = len(set(first_ent_ids.values())) == len(ENTITY_VALUES)
    print(
        f"Tokenizer check OK | entities_unique_first_token={entities_unique_first_token} | "
        f"first_ids={first_ent_ids}",
        flush=True,
    )

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
    summary: dict[str, Any] = {
        "model_id": model_id,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "val_count": len(val_records),
        "test_count": len(test_records),
        "modes": {},
    }
    svg_curves: dict[str, tuple[list[bool], list[float], list[float], float]] = {}

    for mode in args.modes:
        val_cases = _run_split_mode(
            records=val_records,
            split_name="val",
            mode=mode,
            model=model,
            processor=processor,
            image_dir=args.image_dir,
            out_jsonl=args.output_dir / f"val_scores_{mode}.jsonl",
            mel_value_tokens=mel_value_tokens,
            mel_bridge_tokens=mel_bridge_tokens,
            entity_value_tokens=entity_value_tokens,
            entities_unique_first_token=entities_unique_first_token,
            dtype=dtype,
        )
        test_cases = _run_split_mode(
            records=test_records,
            split_name="test",
            mode=mode,
            model=model,
            processor=processor,
            image_dir=args.image_dir,
            out_jsonl=args.output_dir / f"test_scores_{mode}.jsonl",
            mel_value_tokens=mel_value_tokens,
            mel_bridge_tokens=mel_bridge_tokens,
            entity_value_tokens=entity_value_tokens,
            entities_unique_first_token=entities_unique_first_token,
            dtype=dtype,
        )

        mode_summary: dict[str, Any] = {}
        if mode == "raw_1view" and args.greedy_test_predictions is not None:
            parity = _check_greedy_parity(test_cases, args.greedy_test_predictions)
            mode_summary["greedy_parity"] = parity
            print(f"Greedy argmax parity ({mode}): {parity}", flush=True)

        # Fit calibration strictly on val_cases (known binary malignancy)
        val_bin = [c for c in val_cases if c.ground_truth_malignant is not None]
        test_bin = [c for c in test_cases if c.ground_truth_malignant is not None]
        has_both_classes = (
            any(c.ground_truth_malignant for c in val_bin)
            and any(not c.ground_truth_malignant for c in val_bin)
            and any(c.ground_truth_malignant for c in test_bin)
            and any(not c.ground_truth_malignant for c in test_bin)
        )

        if has_both_classes:
            for score_key in ("p_risk", "p_malignant"):
                val_y = [bool(c.ground_truth_malignant) for c in val_bin]
                val_raw = [float(getattr(c, score_key)) for c in val_bin]
                temp = fit_temperature_scaling(val_y, val_raw)
                val_cal = apply_temperature(val_raw, temp)

                tau_90 = select_threshold_for_sensitivity(val_y, val_cal, target_sensitivity=0.90)
                tau_95 = select_threshold_for_sensitivity(val_y, val_cal, target_sensitivity=0.95)

                ops: dict[str, OperatingPointMetrics] = {
                    "val_at_0.50": evaluate_operating_point(
                        val_cases, "val@0.50", 0.50, temp, score_key
                    ),
                    "val_at_90": evaluate_operating_point(
                        val_cases, "val@90%sens", tau_90, temp, score_key
                    ),
                    "val_at_95": evaluate_operating_point(
                        val_cases, "val@95%sens", tau_95, temp, score_key
                    ),
                    "test_at_0.50": evaluate_operating_point(
                        test_cases, "test@0.50", 0.50, temp, score_key
                    ),
                    "test_at_val90": evaluate_operating_point(
                        test_cases, "test@val90%", tau_90, temp, score_key
                    ),
                    "test_at_val95": evaluate_operating_point(
                        test_cases, "test@val95%", tau_95, temp, score_key
                    ),
                }
                mode_summary[score_key] = {
                    "temperature_val": temp,
                    "tau_val_90": tau_90,
                    "tau_val_95": tau_95,
                    "operating_points": {k: v.model_dump() for k, v in ops.items()},
                }

                if score_key == "p_risk":
                    test_y = [bool(c.ground_truth_malignant) for c in test_bin]
                    test_raw = [c.p_risk for c in test_bin]
                    test_cal = apply_temperature(test_raw, temp)
                    svg_curves[mode] = (test_y, test_raw, test_cal, ops["test_at_0.50"].auroc)

        summary["modes"][mode] = mode_summary

    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if svg_curves:
        svg_text = render_roc_and_reliability_svg(svg_curves)
        (args.output_dir / "roc_and_calibration.svg").write_text(svg_text, encoding="utf-8")

    # Render concise Markdown report
    md_lines = [
        "# Calibration & Log-Prob Risk Diagnostic (`qlora_v1` — HIBA val -> MSKCC test)",
        "",
        "| Mode | Score | Temp (`val`) | Val AUROC | **Test AUROC** | Test AUPRC | "
        "ECE raw -> cal | Test @0.50 (Sens / Spec) | **Test @val90% (Sens / Mel / Spec)** | "
        "**Test @val95% (Sens / Mel / Spec)** |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- |",
    ]
    for mode, ms in summary["modes"].items():
        for sk in ("p_risk", "p_malignant"):
            if sk not in ms:
                continue
            d = ms[sk]
            ops = d["operating_points"]
            v50 = ops["val_at_0.50"]
            t50 = ops["test_at_0.50"]
            t90 = ops["test_at_val90"]
            t95 = ops["test_at_val95"]
            md_lines.append(
                f"| `{mode}` | `{sk}` | {d['temperature_val']:.2f} | {v50['auroc']:.3f} | "
                f"**{t50['auroc']:.3f}** | {t50['auprc']:.3f} | "
                f"{t50['ece_raw']:.3f} -> {t50['ece_calibrated']:.3f} | "
                f"{t50['sensitivity'] * 100:.1f}% / {t50['specificity'] * 100:.1f}% | "
                f"**{t90['sensitivity'] * 100:.1f}%** / "
                f"{(t90['melanoma_sensitivity'] or 0) * 100:.1f}% / "
                f"{t90['specificity'] * 100:.1f}% | "
                f"**{t95['sensitivity'] * 100:.1f}%** / "
                f"{(t95['melanoma_sensitivity'] or 0) * 100:.1f}% / "
                f"{t95['specificity'] * 100:.1f}% |"
            )
    if "raw_1view" in summary["modes"] and "greedy_parity" in summary["modes"]["raw_1view"]:
        gp = summary["modes"]["raw_1view"]["greedy_parity"]
        md_lines.extend(
            [
                "",
                f"- **Greedy argmax parity (`raw_1view` vs `predictions.jsonl`):** "
                f"`{gp['matched']}/{gp['total']}` ({gp['agreement_rate'] * 100:.1f}%)",
            ]
        )

    md_text = "\n".join(md_lines) + "\n"
    (args.output_dir / "calibration_report.md").write_text(md_text, encoding="utf-8")
    print("\n" + md_text, flush=True)


if __name__ == "__main__":
    main()
