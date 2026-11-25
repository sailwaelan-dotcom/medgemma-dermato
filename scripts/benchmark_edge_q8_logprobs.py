"""Benchmark native llama-server Q8_0 logprobs + System 1 ONNX + Last-Mile Harness locally.

Zero Colab Compute Units: runs against a local `llama-server` serving MedGemma Q8_0 + mmproj FP16.
Calibrates temperature strictly on `data/isic_v1/val.jsonl` (n=230) and evaluates blindly on
locked `data/isic_v1/test.jsonl` (n=200), proving end-to-end 100% offline Edge performance.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from medgemma_dermato.clinical_harness import (
    HarnessPolicy,
    LastMileClinicalHarness,
    evaluate_deployment_gate,
)
from medgemma_dermato.data.isic_ingest import ISICRecord, load_records_jsonl
from medgemma_dermato.edge_vlm import EdgeVLMClient
from medgemma_dermato.eval.calibration import (
    RiskScoreCase,
    apply_temperature,
    compute_auprc,
    compute_auroc,
    compute_binary_ece,
    evaluate_operating_point,
    fit_temperature_scaling,
    select_threshold_for_sensitivity,
)
from medgemma_dermato.schema import PreflightResult, map_isic_to_entity_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Edge Q8_0 native logprobs on HIBA val + locked MSKCC test."
    )
    parser.add_argument("--val-manifest", type=Path, default=Path("data/isic_v1/val.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/isic_v1/test.jsonl"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument(
        "--sys1-test",
        type=Path,
        default=Path("outputs/system1_fusion/test_scores_sys1.jsonl"),
    )
    parser.add_argument(
        "--sys1-hiba-val",
        type=Path,
        default=Path("outputs/system1_fusion/hiba_val_scores_sys1.jsonl"),
    )
    parser.add_argument(
        "--preflight-source",
        type=Path,
        default=Path("outputs/sprint4_gguf/q8_0/predictions.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/edge_q8_logprobs"),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _prefetch(records: list[ISICRecord], image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        r
        for r in records
        if not (
            (image_dir / f"{r.isic_id}.jpg").exists()
            and (image_dir / f"{r.isic_id}.jpg").stat().st_size > 100
        )
    ]
    if not missing:
        return
    print(f"Prefetching {len(missing)} images into {image_dir}...", flush=True)

    def _dl(r: ISICRecord) -> None:
        if not r.image_url:
            return
        t = image_dir / f"{r.isic_id}.jpg"
        with httpx.Client(timeout=30.0, follow_redirects=True) as c:
            resp = c.get(r.image_url)
            resp.raise_for_status()
            t.write_bytes(resp.content)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_dl, missing))


def _score_split(
    vlm: EdgeVLMClient,
    records: list[ISICRecord],
    image_dir: Path,
    split_name: str,
    out_jsonl: Path,
) -> tuple[list[RiskScoreCase], dict[str, dict[str, Any]]]:
    """Score all records with native llama-server logprobs (resumable if partial file exists)."""
    existing: dict[str, dict[str, Any]] = {}
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["isic_id"]] = row

    cases: list[RiskScoreCase] = []
    meta_map: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()

    with out_jsonl.open("a", encoding="utf-8") as handle:
        for idx, rec in enumerate(records, start=1):
            if rec.isic_id in existing:
                row = existing[rec.isic_id]
                cases.append(RiskScoreCase.model_validate(row["score_case"]))
                meta_map[rec.isic_id] = row
                continue

            img_path = image_dir / f"{rec.isic_id}.jpg"
            ctx = {
                "age_approx": rec.age_approx,
                "sex": rec.sex,
                "anatom_site": rec.anatom_site,
            }
            res = vlm.analyze_image(img_path, clinical_context=ctx, max_tokens=260, temperature=0.0)
            gt_ent = map_isic_to_entity_category(rec.diagnosis_1, rec.diagnosis_2, rec.diagnosis_3)
            ent_probs = res.entity_probs or {}
            p_ind = float(ent_probs.get("actinic_keratosis_or_indeterminate", 0.0))
            p_mal = max(0.0, min(1.0, res.p_risk_estimate - p_ind))
            pred_ent = (
                res.parsed.entity_category
                if res.parsed is not None
                else (max(ent_probs.items(), key=lambda kv: kv[1])[0] if ent_probs else "nevus")
            )
            sc = RiskScoreCase(
                isic_id=rec.isic_id,
                ground_truth_malignant=rec.is_malignant,
                ground_truth_is_melanoma=(gt_ent == "melanoma"),
                ground_truth_entity=gt_ent,
                entity_probs=ent_probs,
                predicted_entity_argmax=pred_ent,
                p_malignant=round(p_mal, 6),
                p_indeterminate=round(p_ind, 6),
                p_risk=round(res.p_risk_estimate, 6),
                p_melanoma=round(res.p_melanoma_estimate, 6),
                latency_ms=res.latency_ms,
            )
            row = {
                "isic_id": rec.isic_id,
                "logprobs_used": res.logprobs_used,
                "json_valid": res.parsed is not None and res.parse_error is None,
                "raw_text": res.raw_text,
                "parsed": res.parsed.model_dump() if res.parsed is not None else None,
                "score_case": sc.model_dump(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            cases.append(sc)
            meta_map[rec.isic_id] = row

            if idx <= 3 or idx % 25 == 0 or idx == len(records):
                elapsed = time.perf_counter() - t0
                lp_cnt = sum(1 for m in meta_map.values() if m["logprobs_used"])
                print(
                    f"[{split_name}] {idx}/{len(records)} | lp_used={lp_cnt}/{idx} | "
                    f"last_p_risk={sc.p_risk:.3f} | lat={res.latency_ms:.0f}ms | {elapsed:.0f}s",
                    flush=True,
                )
    return cases, meta_map


def _load_scores(path: Path) -> dict[str, RiskScoreCase]:
    out: dict[str, RiskScoreCase] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = RiskScoreCase.model_validate_json(line)
            out[c.isic_id] = c
    return out


def _load_preflights(path: Path) -> dict[str, PreflightResult]:
    out: dict[str, PreflightResult] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("preflight"):
                out[row["isic_id"]] = PreflightResult.model_validate(row["preflight"])
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_recs = load_records_jsonl(args.val_manifest)
    test_recs = load_records_jsonl(args.test_manifest)
    if args.limit is not None:
        val_recs = val_recs[: args.limit]
        test_recs = test_recs[: args.limit]

    _prefetch(val_recs + test_recs, args.image_dir)

    with EdgeVLMClient(
        base_url=args.server_url,
        timeout=180.0,
        request_logprobs=True,
        top_logprobs=20,
    ) as vlm:
        if not vlm.healthcheck():
            raise RuntimeError(f"llama-server unreachable at {args.server_url}")
        val_cases, val_meta = _score_split(
            vlm, val_recs, args.image_dir, "HIBA_VAL", args.output_dir / "q8_val_logprobs.jsonl"
        )
        test_cases, test_meta = _score_split(
            vlm, test_recs, args.image_dir, "MSKCC_TEST", args.output_dir / "q8_test_logprobs.jsonl"
        )

    # 1. Calibrate Q8_0 native logprobs temperature strictly on HIBA val (zero leakage)
    val_bin = [c for c in val_cases if c.ground_truth_malignant is not None]
    y_val = [bool(c.ground_truth_malignant) for c in val_bin]
    raw_val = [c.p_risk for c in val_bin]
    temp_q8 = fit_temperature_scaling(y_val, raw_val)
    cal_val = apply_temperature(raw_val, temp_q8)
    tau_q8_val95 = select_threshold_for_sensitivity(y_val, cal_val, 0.95)

    q8_val_op = evaluate_operating_point(
        val_cases, "q8_lp_val@95%", tau_q8_val95, temp_q8, "p_risk"
    )
    q8_test_op = evaluate_operating_point(
        test_cases, "q8_lp_test@val95%", tau_q8_val95, temp_q8, "p_risk"
    )
    q8_test_050 = evaluate_operating_point(test_cases, "q8_lp_test@0.50", 0.50, temp_q8, "p_risk")

    # 2. Evaluate 100% Offline Edge Stack: System 1 ONNX + Q8_0 native logprobs + Last-Mile Harness
    s1_test = _load_scores(args.sys1_test)
    preflights = _load_preflights(args.preflight_source)
    q8_test_map = {c.isic_id: c for c in test_cases}

    pol = HarnessPolicy(
        policy_version="edge-q8-logprobs-v1.0",
        sys2_model_id="medgemma-1.5-4b-qlora-v1-q8_0-logprobs",
        temp_medgemma=temp_q8,
        strict_preflight_block=False,
    )
    harness = LastMileClinicalHarness(pol)

    mal_caught = mal_tot = mel_caught = mel_tot = g_ben = ben_tot = g_mal = c_blk = c_mal = 0
    valid_json_cnt = 0
    lp_used_cnt = 0
    latencies: list[float] = []
    fused_scores_bin: list[float] = []
    y_bin_test: list[bool] = []

    for rec in test_recs:
        cid = rec.isic_id
        s1 = s1_test[cid]
        q8 = q8_test_map[cid]
        meta = test_meta[cid]
        if meta["json_valid"]:
            valid_json_cnt += 1
        if meta["logprobs_used"]:
            lp_used_cnt += 1
        if q8.latency_ms is not None:
            latencies.append(q8.latency_ms)

        from medgemma_dermato.schema import TriageAssessment

        parsed_obj = TriageAssessment.model_validate(meta["parsed"]) if meta.get("parsed") else None
        res = harness.evaluate_case(
            case_id=cid,
            p_sys1_raw=s1.p_risk,
            p_medgemma_raw=q8.p_risk,
            sys1_top_entity=s1.predicted_entity_argmax,
            p_melanoma_sys1=s1.p_melanoma,
            p_melanoma_medgemma=q8.p_melanoma,
            medgemma_top_entity=q8.predicted_entity_argmax,
            medgemma_assessment=parsed_obj,
            preflight=preflights.get(cid),
        )
        if res.audit.text_contradiction_caught:
            c_blk += 1
            if s1.ground_truth_malignant is True:
                c_mal += 1
        is_green = res.triage_zone == "GREEN_LOW_RISK_SURVEILLANCE"
        if s1.ground_truth_malignant is True:
            mal_tot += 1
            y_bin_test.append(True)
            fused_scores_bin.append(res.calibrated_risk_score)
            if not is_green:
                mal_caught += 1
            else:
                g_mal += 1
            if s1.ground_truth_is_melanoma:
                mel_tot += 1
                if not is_green:
                    mel_caught += 1
        elif s1.ground_truth_malignant is False:
            ben_tot += 1
            y_bin_test.append(False)
            fused_scores_bin.append(res.calibrated_risk_score)
            if is_green:
                g_ben += 1

    harness_metrics = {
        "total_cases": len(test_recs),
        "json_valid_rate": round(valid_json_cnt / max(len(test_recs), 1), 4),
        "logprobs_used_rate": round(lp_used_cnt / max(len(test_recs), 1), 4),
        "mean_q8_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
        "fusion_auroc": compute_auroc(y_bin_test, fused_scores_bin),
        "fusion_auprc": compute_auprc(y_bin_test, fused_scores_bin),
        "fusion_ece": compute_binary_ece(y_bin_test, fused_scores_bin),
        "malignant_safe_sensitivity": round(mal_caught / max(mal_tot, 1), 4),
        "melanoma_safe_sensitivity": round(mel_caught / max(mel_tot, 1), 4),
        "green_zone_npv": round(g_ben / max(g_ben + g_mal, 1), 4),
        "green_zone_benign_specificity": round(g_ben / max(ben_tot, 1), 4),
        "green_zone_malignant_fn": g_mal,
        "text_contradictions_caught_total": c_blk,
        "text_contradictions_caught_on_malignant": c_mal,
    }
    gate = evaluate_deployment_gate(harness_metrics)

    summary = {
        "temperature_q8_val": temp_q8,
        "tau_q8_val95": tau_q8_val95,
        "q8_standalone_val": q8_val_op.model_dump(),
        "q8_standalone_test_050": q8_test_050.model_dump(),
        "q8_standalone_test_val95": q8_test_op.model_dump(),
        "edge_hybrid_q8_logprobs_harness": harness_metrics,
        "deployment_gate": gate.model_dump(),
    }
    (args.output_dir / "edge_q8_logprobs_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md = [
        "# Benchmark Local Edge 100% Offline — MedGemma Q8_0 Native Log-Probs + System 1 ONNX",
        "",
        "Exécuté en local sur **NVIDIA RTX 4080 Laptop (`0.00 CU` Colab)** via `llama-server` "
        "(`logprobs: true, top_logprobs: 20`). Température calibrée sur `HIBA val (n=230)` "
        f"(`T={temp_q8:.4f}`), test en aveugle sur **MSKCC 2020 (`n=200`)**.",
        "",
        "| Configuration Edge | AUROC | Sens. Malins | Sens. Mélanome | Spécificité Zone Verte | "
        "NPV Zone Verte | ECE | Latence RTX 4080 | Gate CI/CD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| Q8_0 Log-probs seul (`@val95%`) | {q8_test_op.auroc:.4f} | "
        f"{q8_test_op.sensitivity*100:.1f}% ({q8_test_op.tp}/70) | "
        f"{(q8_test_op.melanoma_sensitivity or 0)*100:.1f}% | "
        f"{q8_test_op.specificity*100:.1f}% | {(q8_test_op.npv or 0)*100:.1f}% | "
        f"{q8_test_op.ece_calibrated:.4f} | {harness_metrics['mean_q8_latency_ms']:.0f} ms | — |",
        f"| **Edge Q8_0 Log-probs + ONNX + Harness** | **{harness_metrics['fusion_auroc']:.4f}** | "
        f"**{harness_metrics['malignant_safe_sensitivity']*100:.1f}% ({mal_caught}/70)** | "
        f"**{harness_metrics['melanoma_safe_sensitivity']*100:.1f}% ({mel_caught}/39)** | "
        f"**{harness_metrics['green_zone_benign_specificity']*100:.1f}% ({g_ben}/115)** | "
        f"**{harness_metrics['green_zone_npv']*100:.1f}%** | "
        f"**{harness_metrics['fusion_ece']:.4f}** | "
        f"{harness_metrics['mean_q8_latency_ms']:.0f} ms | **`{gate.decision}`** |",
        "",
        f"- **Log-probs natifs extraits :** `{harness_metrics['logprobs_used_rate']*100:.1f}%` "
        f"| **JSON valides :** `{harness_metrics['json_valid_rate']*100:.1f}%`",
        f"- **Contradictions texte bloquées :** **{c_blk}** (dont **{c_mal}** vrais cancers)",
        "",
    ]
    (args.output_dir / "edge_q8_logprobs_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n=== EDGE Q8_0 NATIVE LOGPROBS SUMMARY (MSKCC n=200) ===", flush=True)
    print(
        f"Q8_0 standalone logprobs AUROC={q8_test_op.auroc:.4f} (val AUROC={q8_val_op.auroc:.4f})",
        flush=True,
    )
    print(json.dumps(harness_metrics, indent=2), flush=True)
    print(f"DEPLOYMENT GATE: {gate.decision}", flush=True)


if __name__ == "__main__":
    main()
