"""Run local CPU Late Fusion evaluation and optional Edge ONNX latency benchmark."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image

from medgemma_dermato.eval.calibration import (
    RiskScoreCase,
    ScoreKey,
    apply_temperature,
    evaluate_operating_point,
    fit_temperature_scaling,
    select_threshold_for_sensitivity,
)
from medgemma_dermato.eval.fusion import (
    align_score_cases,
    build_fused_cases,
    evaluate_fusion_operating_point,
    fit_optimal_blend_weight,
    render_hybrid_roc_pr_svg,
    select_fusion_threshold_for_sensitivity,
)
from medgemma_dermato.system1 import OnnxSystem1Classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate System 1 + MedGemma Late Fusion locally on locked MSKCC test."
    )
    parser.add_argument(
        "--sys1-mc-val",
        type=Path,
        default=Path("outputs/system1_fusion/val_scores_sys1.jsonl"),
    )
    parser.add_argument(
        "--sys1-hiba-val",
        type=Path,
        default=Path("outputs/system1_fusion/hiba_val_scores_sys1.jsonl"),
    )
    parser.add_argument(
        "--sys1-test",
        type=Path,
        default=Path("outputs/system1_fusion/test_scores_sys1.jsonl"),
    )
    parser.add_argument(
        "--medgemma-val",
        type=Path,
        default=Path("outputs/v1_qlora_calibration/val_scores_raw_1view.jsonl"),
    )
    parser.add_argument(
        "--medgemma-test",
        type=Path,
        default=Path("outputs/v1_qlora_calibration/test_scores_raw_1view.jsonl"),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("outputs/v1_baseline_neutral/eval_report.json"),
    )
    parser.add_argument(
        "--qlora-greedy-report",
        type=Path,
        default=Path("outputs/v1_qlora_bf16/eval_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/system1_fusion"),
    )
    parser.add_argument(
        "--sys1-score-key",
        choices=["p_risk", "p_malignant"],
        default="p_risk",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Optional path to ONNX model to benchmark real local CPU latency.",
    )
    parser.add_argument(
        "--onnx-calibration",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--benchmark-image-dir",
        type=Path,
        default=None,
        help="Optional directory with JPEG images for real ONNX CPU benchmark (or synthetic).",
    )
    return parser.parse_args()


def _load_cases(path: Path) -> list[RiskScoreCase]:
    cases: list[RiskScoreCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s:
                cases.append(RiskScoreCase.model_validate_json(s))
    return cases


def _benchmark_onnx_cpu(
    onnx_model: Path,
    calibration_path: Path | None,
    image_dir: Path | None,
    n_runs: int = 12,
) -> dict[str, Any]:
    clf = OnnxSystem1Classifier(onnx_model, calibration_path=calibration_path)
    imgs: list[Image.Image] = []
    if image_dir and image_dir.exists():
        for p in sorted(image_dir.glob("*.jpg"))[:n_runs]:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
    if not imgs:
        imgs = [Image.new("RGB", (600, 450), color=(140, 100, 95)) for _ in range(n_runs)]

    # Warm-up run
    _ = clf.predict_image(imgs[0], tta=False)
    _ = clf.predict_image(imgs[0], tta=True)

    lat_1view = [clf.predict_image(im, tta=False).latency_ms for im in imgs]
    lat_tta4 = [clf.predict_image(im, tta=True).latency_ms for im in imgs]
    return {
        "model_path": str(onnx_model),
        "model_size_mb": round(onnx_model.stat().st_size / (1024 * 1024), 2),
        "n_runs": len(imgs),
        "single_view_median_ms": round(statistics.median(lat_1view), 1),
        "single_view_mean_ms": round(statistics.mean(lat_1view), 1),
        "tta4_median_ms": round(statistics.median(lat_tta4), 1),
        "tta4_mean_ms": round(statistics.mean(lat_tta4), 1),
    }


def _fmt_pct(val: float | None) -> str:
    return f"{val * 100:.1f}%" if val is not None else "N/A"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_key: ScoreKey = args.sys1_score_key

    sys1_mc_val = _load_cases(args.sys1_mc_val)
    sys1_hiba_val = _load_cases(args.sys1_hiba_val)
    sys1_test = _load_cases(args.sys1_test)
    med_val = _load_cases(args.medgemma_val)
    med_test = _load_cases(args.medgemma_test)

    # 1. Calibrate System 1 strictly on multi-center val (n~1068 binary)
    mc_bin = [c for c in sys1_mc_val if c.ground_truth_malignant is not None]
    y_mc = [bool(c.ground_truth_malignant) for c in mc_bin]
    raw_mc = [float(getattr(c, score_key)) for c in mc_bin]
    temp_s1 = fit_temperature_scaling(y_mc, raw_mc)
    cal_mc = apply_temperature(raw_mc, temp_s1)
    tau_s1_mc90 = select_threshold_for_sensitivity(y_mc, cal_mc, 0.90)
    tau_s1_mc95 = select_threshold_for_sensitivity(y_mc, cal_mc, 0.95)

    s1_mc_val_95 = evaluate_operating_point(
        sys1_mc_val, "sys1_mc_val@95%", tau_s1_mc95, temp_s1, score_key
    )
    s1_test_050 = evaluate_operating_point(
        sys1_test, "sys1_test@0.50", 0.50, temp_s1, score_key
    )
    s1_test_mc90 = evaluate_operating_point(
        sys1_test, "sys1_test@mc_val90%", tau_s1_mc90, temp_s1, score_key
    )
    s1_test_mc95 = evaluate_operating_point(
        sys1_test, "sys1_test@mc_val95%", tau_s1_mc95, temp_s1, score_key
    )

    # Also compute System 1 entity accuracy on MSKCC test
    s1_test_ent_acc = round(
        sum(1 for c in sys1_test if c.predicted_entity_argmax == c.ground_truth_entity)
        / max(len(sys1_test), 1),
        4,
    )

    # 2. Calibrate MedGemma strictly on HIBA val (n=226 binary) — reproduces Option C
    med_bin = [c for c in med_val if c.ground_truth_malignant is not None]
    y_med_val = [bool(c.ground_truth_malignant) for c in med_bin]
    raw_med_val = [c.p_risk for c in med_bin]
    temp_med = fit_temperature_scaling(y_med_val, raw_med_val)
    cal_med_val = apply_temperature(raw_med_val, temp_med)
    tau_med_val95 = select_threshold_for_sensitivity(y_med_val, cal_med_val, 0.95)
    med_test_val95 = evaluate_operating_point(
        med_test, "medgemma_test@val95%", tau_med_val95, temp_med, "p_risk"
    )
    med_test_050 = evaluate_operating_point(
        med_test, "medgemma_test@0.50", 0.50, temp_med, "p_risk"
    )

    # 3. Late Fusion — aligned strictly on HIBA val (unseen by BOTH models) and locked MSKCC test
    aligned_val = align_score_cases(sys1_hiba_val, med_val)
    aligned_test = align_score_cases(sys1_test, med_test)

    val_s1_cal_all = apply_temperature(
        [float(getattr(s1, score_key)) for s1, _ in aligned_val], temp_s1
    )
    val_med_cal_all = apply_temperature([s2.p_risk for _, s2 in aligned_val], temp_med)
    y_align_val = [
        bool(s1.ground_truth_malignant)
        for s1, _ in aligned_val
        if s1.ground_truth_malignant is not None
    ]
    v_s1_bin = [
        p
        for (s1, _), p in zip(aligned_val, val_s1_cal_all, strict=True)
        if s1.ground_truth_malignant is not None
    ]
    v_med_bin = [
        p
        for (s1, _), p in zip(aligned_val, val_med_cal_all, strict=True)
        if s1.ground_truth_malignant is not None
    ]
    w_val_nll = fit_optimal_blend_weight(y_align_val, v_s1_bin, v_med_bin)

    policies: list[tuple[str, float, float | None]] = [
        ("val_nll_blend", w_val_nll, None),
        ("equal_50_50", 0.50, None),
        ("multicenter_75_25", 0.75, None),
        ("multicenter_85_15", 0.85, None),
        ("equal_50_50_safety_gate_0.40", 0.50, 0.40),
        ("multicenter_75_25_safety_gate_0.40", 0.75, 0.40),
    ]

    fusion_results: dict[str, Any] = {}
    curve_data: dict[str, tuple[list[bool], list[float], float, float]] = {}

    # Add standalone curves first
    test_bin_s1 = [c for c in sys1_test if c.ground_truth_malignant is not None]
    y_test = [bool(c.ground_truth_malignant) for c in test_bin_s1]
    s1_test_cal = apply_temperature(
        [float(getattr(c, score_key)) for c in test_bin_s1], temp_s1
    )
    med_by_id = {c.isic_id: c for c in med_test}
    med_test_cal = apply_temperature(
        [med_by_id[c.isic_id].p_risk for c in test_bin_s1], temp_med
    )

    curve_data["MedGemma QLoRA v1 (1-center)"] = (
        y_test,
        med_test_cal,
        med_test_050.auroc,
        med_test_050.auprc,
    )
    curve_data["System 1 ConvNeXt-Tiny (5-center)"] = (
        y_test,
        s1_test_cal,
        s1_test_050.auroc,
        s1_test_050.auprc,
    )

    for policy_name, w_s1, delta in policies:
        f_val = build_fused_cases(
            aligned_val,
            weight_sys1=w_s1,
            temp_sys1=temp_s1,
            temp_medgemma=temp_med,
            sys1_score_key=score_key,
            medgemma_score_key="p_risk",
        )
        f_test = build_fused_cases(
            aligned_test,
            weight_sys1=w_s1,
            temp_sys1=temp_s1,
            temp_medgemma=temp_med,
            sys1_score_key=score_key,
            medgemma_score_key="p_risk",
        )
        tau_val95 = select_fusion_threshold_for_sensitivity(f_val, 0.95, disagreement_delta=delta)
        tau_val90 = select_fusion_threshold_for_sensitivity(f_val, 0.90, disagreement_delta=delta)

        op_val95 = evaluate_fusion_operating_point(
            f_val, f"{policy_name}@val95", policy_name, w_s1, tau_val95, delta
        )
        op_test_val95 = evaluate_fusion_operating_point(
            f_test, f"{policy_name}@test_val95", policy_name, w_s1, tau_val95, delta
        )
        op_test_mc95 = evaluate_fusion_operating_point(
            f_test, f"{policy_name}@test_mc95", policy_name, w_s1, tau_s1_mc95, delta
        )
        op_test_val90 = evaluate_fusion_operating_point(
            f_test, f"{policy_name}@test_val90", policy_name, w_s1, tau_val90, delta
        )
        op_test_050 = evaluate_fusion_operating_point(
            f_test, f"{policy_name}@test_0.50", policy_name, w_s1, 0.50, delta
        )

        fusion_results[policy_name] = {
            "weight_sys1": w_s1,
            "disagreement_delta": delta,
            "tau_hiba_val_95": tau_val95,
            "tau_mc_val_95_ref": tau_s1_mc95,
            "val_at_95": op_val95.model_dump(),
            "test_at_0.50": op_test_050.model_dump(),
            "test_at_val90": op_test_val90.model_dump(),
            "test_at_val95": op_test_val95.model_dump(),
            "test_at_mc95": op_test_mc95.model_dump(),
        }

        if policy_name in ("equal_50_50", "multicenter_75_25"):
            f_test_bin = [c for c in f_test if c.ground_truth_malignant is not None]
            label = (
                "Hybrid Fusion 50/50"
                if policy_name == "equal_50_50"
                else "Hybrid Fusion 75/25"
            )
            curve_data[label] = (
                [bool(c.ground_truth_malignant) for c in f_test_bin],
                [c.p_fused for c in f_test_bin],
                op_test_050.auroc,
                op_test_050.auprc,
            )

    # Optional local CPU ONNX latency benchmark
    onnx_bench: dict[str, Any] | None = None
    if args.onnx_model is not None and args.onnx_model.exists():
        onnx_bench = _benchmark_onnx_cpu(
            args.onnx_model,
            args.onnx_calibration,
            args.benchmark_image_dir,
        )

    baseline_data = (
        json.loads(args.baseline_report.read_text(encoding="utf-8"))
        if args.baseline_report.exists()
        else None
    )
    qlora_greedy_data = (
        json.loads(args.qlora_greedy_report.read_text(encoding="utf-8"))
        if args.qlora_greedy_report.exists()
        else None
    )

    summary = {
        "sys1_score_key": score_key,
        "system1_calibration": {
            "temperature_mc_val": temp_s1,
            "tau_mc_val_90": tau_s1_mc90,
            "tau_mc_val_95": tau_s1_mc95,
            "mc_val_at_95": s1_mc_val_95.model_dump(),
            "test_at_0.50": s1_test_050.model_dump(),
            "test_at_mc90": s1_test_mc90.model_dump(),
            "test_at_mc95": s1_test_mc95.model_dump(),
            "test_entity_accuracy": s1_test_ent_acc,
        },
        "medgemma_calibration": {
            "temperature_hiba_val": temp_med,
            "tau_hiba_val_95": tau_med_val95,
            "test_at_0.50": med_test_050.model_dump(),
            "test_at_val95": med_test_val95.model_dump(),
        },
        "w_val_nll_fitted_on_hiba_val": w_val_nll,
        "fusion_policies": fusion_results,
        "onnx_cpu_benchmark": onnx_bench,
    }
    (args.output_dir / "fusion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    svg = render_hybrid_roc_pr_svg(curve_data)
    (args.output_dir / "hybrid_roc_pr.svg").write_text(svg, encoding="utf-8")

    # Build crisp comparative Markdown report
    md_lines = [
        "# Architecture Hybride System 1 + System 2 — Evaluation Externe MSKCC (n=200)",
        "",
        "Evaluation stricte en aveugle sur le hold-out externe verrouille "
        "**MSKCC Consecutive Biopsies 2020 (`n=200`, 70 malins / 39 melanomes / "
        "115 benins / 15 indetermines)**.",
        "Zero image MSKCC en entrainement ou calibration. Toutes les temperatures "
        "et seuils sont calibres uniquement sur `val`.",
        "",
        "## 1. Tableau comparatif global (MSKCC Externe n=200)",
        "",
        "| Modele / Configuration | AUROC | AUPRC | Sensibilite Malins | "
        "Sensibilite Melanome | Specificite | ECE | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    if baseline_data:
        md_lines.append(
            f"| 1. MedGemma 1.5 Zero-Shot (Neutre) | N/A | N/A | "
            f"{_fmt_pct(baseline_data.get('malignant_sensitivity_safe_triage'))} | "
            f"{_fmt_pct(baseline_data.get('melanoma_sensitivity_safe_triage'))} | "
            f"{_fmt_pct(baseline_data.get('specificity_direct'))} | "
            f"{baseline_data.get('ece_calibration_error', 0.0):.4f} | Baseline greedy |"
        )
    if qlora_greedy_data:
        md_lines.append(
            f"| 2. MedGemma QLoRA v1 (Greedy, 1 centre HIBA) | N/A | N/A | "
            f"{_fmt_pct(qlora_greedy_data.get('malignant_sensitivity_safe_triage'))} | "
            f"{_fmt_pct(qlora_greedy_data.get('melanoma_sensitivity_safe_triage'))} | "
            f"{_fmt_pct(qlora_greedy_data.get('specificity_direct'))} | "
            f"{qlora_greedy_data.get('ece_calibration_error', 0.0):.4f} | Entity acc 22.0% |"
        )

    md_lines.append(
        f"| 3. MedGemma QLoRA v1 (Log-probs `@val95%`) | {med_test_val95.auroc:.4f} | "
        f"{med_test_val95.auprc:.4f} | {_fmt_pct(med_test_val95.sensitivity)} "
        f"({med_test_val95.tp}/{med_test_val95.malignant_total}) | "
        f"{_fmt_pct(med_test_val95.melanoma_sensitivity)} | "
        f"{_fmt_pct(med_test_val95.specificity)} | {med_test_val95.ece_calibrated:.4f} | "
        f"T={temp_med:.2f}, tau={tau_med_val95:.3f} |"
    )
    md_lines.append(
        f"| 4a. System 1 ConvNeXt-Tiny (`@0.50`, 5 centres) | {s1_test_050.auroc:.4f} | "
        f"{s1_test_050.auprc:.4f} | {_fmt_pct(s1_test_050.sensitivity)} "
        f"({s1_test_050.tp}/{s1_test_050.malignant_total}) | "
        f"{_fmt_pct(s1_test_050.melanoma_sensitivity)} | "
        f"{_fmt_pct(s1_test_050.specificity)} | {s1_test_050.ece_calibrated:.4f} | "
        f"Entity acc {_fmt_pct(s1_test_ent_acc)} |"
    )
    md_lines.append(
        f"| 4b. System 1 ConvNeXt-Tiny (`@mc_val95%`, 5 centres) | {s1_test_mc95.auroc:.4f} | "
        f"{s1_test_mc95.auprc:.4f} | {_fmt_pct(s1_test_mc95.sensitivity)} "
        f"({s1_test_mc95.tp}/{s1_test_mc95.malignant_total}) | "
        f"{_fmt_pct(s1_test_mc95.melanoma_sensitivity)} | "
        f"{_fmt_pct(s1_test_mc95.specificity)} | {s1_test_mc95.ece_calibrated:.4f} | "
        f"T={temp_s1:.2f}, tau={tau_s1_mc95:.3f} |"
    )

    for pname in (
        "multicenter_75_25",
        "multicenter_75_25_safety_gate_0.40",
        "equal_50_50",
        "equal_50_50_safety_gate_0.40",
        "val_nll_blend",
    ):
        for op_key, op_tag in (("test_at_mc95", "@mc_val95%"), ("test_at_val95", "@hiba_val95%")):
            op = fusion_results[pname][op_key]
            note = f"w_s1={op['weight_sys1']:.2f}, tau={op['threshold']:.3f}"
            if op["disagreement_delta"] is not None:
                note += (
                    f", gate(d>={op['disagreement_delta']:.2f}): "
                    f"+{op['disagreement_rescued_tp']} TP / +{op['disagreement_extra_fp']} FP"
                )
            md_lines.append(
                f"| 5. Fusion `{pname}` ({op_tag}) | {op['auroc']:.4f} | {op['auprc']:.4f} | "
                f"{_fmt_pct(op['sensitivity'])} ({op['tp']}/{op['malignant_total']}) | "
                f"{_fmt_pct(op['melanoma_sensitivity'])} | {_fmt_pct(op['specificity'])} | "
                f"{op['ece']:.4f} | {note} |"
            )

    if onnx_bench:
        md_lines.extend(
            [
                "",
                "## 2. Benchmark Edge Local CPU (ONNX Runtime — Zero PyTorch)",
                "",
                f"- **Modele ONNX :** `{onnx_bench['model_path']}` "
                f"(**{onnx_bench['model_size_mb']} MB**)",
                "- **Latence 1-vue CPU (mediane / moyenne) :** "
                f"**{onnx_bench['single_view_median_ms']} ms** / "
                f"{onnx_bench['single_view_mean_ms']} ms",
                "- **Latence TTA-4 CPU (mediane / moyenne) :** "
                f"**{onnx_bench['tta4_median_ms']} ms** / "
                f"{onnx_bench['tta4_mean_ms']} ms",
            ]
        )

    md_lines.extend(
        [
            "",
            "## 3. Courbes ROC & Precision-Recall",
            "",
            "![Hybrid ROC & PR](hybrid_roc_pr.svg)",
            "",
        ]
    )
    (args.output_dir / "hybrid_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved fusion report -> {args.output_dir / 'hybrid_comparison.md'}")
    print(
        f"System 1 AUROC={s1_test_050.auroc:.4f} (AUPRC={s1_test_050.auprc:.4f}, "
        f"sens@mc95={s1_test_mc95.sensitivity:.4f}, spec@mc95={s1_test_mc95.specificity:.4f}, "
        f"entity_acc={s1_test_ent_acc:.4f})"
    )
    f75 = fusion_results["multicenter_75_25"]["test_at_mc95"]
    f75g = fusion_results["multicenter_75_25_safety_gate_0.40"]["test_at_mc95"]
    print(
        f"Fusion 75/25 AUROC={f75['auroc']:.4f} (sens@mc95={f75['sensitivity']:.4f}, "
        f"spec@mc95={f75['specificity']:.4f})"
    )
    print(
        f"Fusion 75/25 + SafetyGate: sens={f75g['sensitivity']:.4f}, "
        f"spec={f75g['specificity']:.4f}, rescued_tp={f75g['disagreement_rescued_tp']}, "
        f"extra_fp={f75g['disagreement_extra_fp']}"
    )


if __name__ == "__main__":
    main()
