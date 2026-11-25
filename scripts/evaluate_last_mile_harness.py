"""Evaluate the 3-zone Last-Mile Clinical Safety Harness on locked MSKCC (n=200) and HIBA val."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from medgemma_dermato.clinical_harness import (
    HarnessPolicy,
    LastMileAssessment,
    LastMileClinicalHarness,
    evaluate_deployment_gate,
)
from medgemma_dermato.eval.calibration import RiskScoreCase
from medgemma_dermato.schema import PreflightResult, parse_triage_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Last-Mile 3-zone safety harness on locked MSKCC n=200."
    )
    parser.add_argument(
        "--sys1-test",
        type=Path,
        default=Path("outputs/system1_fusion/test_scores_sys1.jsonl"),
    )
    parser.add_argument(
        "--medgemma-test",
        type=Path,
        default=Path("outputs/v1_qlora_calibration/test_scores_raw_1view.jsonl"),
    )
    parser.add_argument(
        "--medgemma-tta4-test",
        type=Path,
        default=Path("outputs/v1_qlora_calibration/test_scores_raw_tta4.jsonl"),
    )
    parser.add_argument(
        "--medgemma-greedy-preds",
        type=Path,
        default=Path("outputs/v1_qlora_bf16/predictions.jsonl"),
    )
    parser.add_argument(
        "--sys1-hiba-val",
        type=Path,
        default=Path("outputs/system1_fusion/hiba_val_scores_sys1.jsonl"),
    )
    parser.add_argument(
        "--medgemma-val",
        type=Path,
        default=Path("outputs/v1_qlora_calibration/val_scores_raw_1view.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/last_mile_harness"),
    )
    return parser.parse_args()


def _load_scores(path: Path) -> dict[str, RiskScoreCase]:
    out: dict[str, RiskScoreCase] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s:
                c = RiskScoreCase.model_validate_json(s)
                out[c.isic_id] = c
    return out


def _load_greedy_preds(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s:
                row = json.loads(s)
                out[row["isic_id"]] = row
    return out


def _summarize_cohort(
    assessments: list[LastMileAssessment],
    gt_map: dict[str, RiskScoreCase],
    json_valid_rate: float = 1.0,
) -> dict[str, Any]:
    zones = (
        "GREEN_LOW_RISK_SURVEILLANCE",
        "AMBER_UNCERTAIN_TELEEXPERTISE",
        "RED_PRIORITY_DERMATOLOGY",
    )
    counts_all = Counter(a.triage_zone for a in assessments)
    mal_zones: Counter[str] = Counter()
    mel_zones: Counter[str] = Counter()
    ben_zones: Counter[str] = Counter()
    ind_zones: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()

    contradictions_total = 0
    contradictions_on_malignant = 0
    retake_count = 0

    for a in assessments:
        gt = gt_map[a.case_id]
        for t in a.triggered_rules:
            rule_counts[t.code] += 1
        if a.retake_recommended:
            retake_count += 1
        if a.audit.text_contradiction_caught:
            contradictions_total += 1
            if gt.ground_truth_malignant is True:
                contradictions_on_malignant += 1

        if gt.ground_truth_malignant is True:
            mal_zones[a.triage_zone] += 1
            if gt.ground_truth_is_melanoma:
                mel_zones[a.triage_zone] += 1
        elif gt.ground_truth_malignant is False:
            ben_zones[a.triage_zone] += 1
        else:
            ind_zones[a.triage_zone] += 1

    mal_total = sum(mal_zones.values())
    mel_total = sum(mel_zones.values())
    ben_total = sum(ben_zones.values())

    mal_caught = mal_zones["AMBER_UNCERTAIN_TELEEXPERTISE"] + mal_zones["RED_PRIORITY_DERMATOLOGY"]
    mel_caught = mel_zones["AMBER_UNCERTAIN_TELEEXPERTISE"] + mel_zones["RED_PRIORITY_DERMATOLOGY"]
    green_mal = mal_zones["GREEN_LOW_RISK_SURVEILLANCE"]
    green_mel = mel_zones["GREEN_LOW_RISK_SURVEILLANCE"]
    green_ben = ben_zones["GREEN_LOW_RISK_SURVEILLANCE"]
    green_binary = green_mal + green_ben

    red_mal = mal_zones["RED_PRIORITY_DERMATOLOGY"]
    red_ben = ben_zones["RED_PRIORITY_DERMATOLOGY"]
    red_binary = red_mal + red_ben

    metrics = {
        "total_cases": len(assessments),
        "json_valid_rate": round(json_valid_rate, 4),
        "zone_counts_all": {z: counts_all.get(z, 0) for z in zones},
        "zone_counts_malignant": {z: mal_zones.get(z, 0) for z in zones},
        "zone_counts_melanoma": {z: mel_zones.get(z, 0) for z in zones},
        "zone_counts_benign": {z: ben_zones.get(z, 0) for z in zones},
        "zone_counts_indeterminate": {z: ind_zones.get(z, 0) for z in zones},
        "malignant_safe_sensitivity": round(mal_caught / mal_total, 4) if mal_total else 0.0,
        "melanoma_safe_sensitivity": round(mel_caught / mel_total, 4) if mel_total else 0.0,
        "malignant_red_direct_rate": round(red_mal / mal_total, 4) if mal_total else 0.0,
        "green_zone_benign_specificity": round(green_ben / ben_total, 4) if ben_total else 0.0,
        "green_zone_npv": round(green_ben / green_binary, 4) if green_binary else 0.0,
        "green_zone_malignant_fn": green_mal,
        "green_zone_melanoma_fn": green_mel,
        "red_zone_ppv": round(red_mal / red_binary, 4) if red_binary else 0.0,
        "retake_recommended_count": retake_count,
        "text_contradictions_caught_total": contradictions_total,
        "text_contradictions_caught_on_malignant": contradictions_on_malignant,
        "triggered_rule_counts": dict(rule_counts.most_common()),
    }
    return metrics


def _render_3zone_svg(msk_metrics: dict[str, Any]) -> str:
    """Render stacked 3-zone clinical distribution bar chart for MSKCC n=200."""
    w, h = 820, 340
    groups = [
        ("Malignant (n=70)", msk_metrics["zone_counts_malignant"], 70),
        ("Melanoma (n=39)", msk_metrics["zone_counts_melanoma"], 39),
        ("Benign Biopsies (n=115)", msk_metrics["zone_counts_benign"], 115),
        ("Indeterminate (n=15)", msk_metrics["zone_counts_indeterminate"], 15),
    ]
    z_colors = [
        ("GREEN_LOW_RISK_SURVEILLANCE", "#16a34a", "Zone Verte (Surveillance)"),
        ("AMBER_UNCERTAIN_TELEEXPERTISE", "#f59e0b", "Zone Grise (Tele-expertise)"),
        ("RED_PRIORITY_DERMATOLOGY", "#dc2626", "Zone Rouge (Prioritaire)"),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" style="background:#ffffff;'
        'font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<text x="{w / 2}" y="28" text-anchor="middle" font-size="15" font-weight="600" '
        'fill="#111827">Last-Mile 3-Zone Safety Harness — MSKCC External Hold-Out (n=200)</text>',
    ]

    # Legend
    lx = 125
    for _zk, col, lbl in z_colors:
        lines.append(f'<rect x="{lx}" y="44" width="14" height="14" rx="2" fill="{col}"/>')
        lines.append(f'<text x="{lx + 20}" y="55" font-size="11.5" fill="#1f2937">{lbl}</text>')
        lx += 205

    x0, bar_w, bar_h = 205, 560, 36
    y_cursor = 88
    for label, counts, total in groups:
        lines.append(
            f'<text x="{x0 - 12}" y="{y_cursor + 23}" text-anchor="end" '
            f'font-size="12" font-weight="500" fill="#111827">{label}</text>'
        )
        cx = float(x0)
        for zk, col, _ in z_colors:
            cnt = int(counts.get(zk, 0))
            if cnt <= 0 or total <= 0:
                continue
            seg_w = bar_w * (cnt / float(total))
            pct = 100.0 * cnt / float(total)
            lines.append(
                f'<rect x="{cx:.1f}" y="{y_cursor}" width="{seg_w:.1f}" '
                f'height="{bar_h}" fill="{col}"/>'
            )
            if seg_w >= 46:
                lines.append(
                    f'<text x="{cx + seg_w / 2:.1f}" y="{y_cursor + 22}" text-anchor="middle" '
                    f'font-size="11" font-weight="600" fill="#ffffff">{cnt} ({pct:.0f}%)</text>'
                )
            cx += seg_w
        y_cursor += 58

    mal_s = msk_metrics["malignant_safe_sensitivity"] * 100
    mel_s = msk_metrics["melanoma_safe_sensitivity"] * 100
    g_npv = msk_metrics["green_zone_npv"] * 100
    n_blk = msk_metrics["text_contradictions_caught_total"]
    lines.append(
        f'<text x="{w / 2}" y="{h - 14}" text-anchor="middle" font-size="11" fill="#4b5563">'
        f"Malignant Safe Triage (Amber+Red): {mal_s:.1f}% | "
        f"Melanoma Safe Triage: {mel_s:.1f}% | "
        f"Green Zone NPV: {g_npv:.1f}% | "
        f"Text Contradictions Blocked: {n_blk}"
        "</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _run_policy(
    policy: HarnessPolicy,
    sys1_map: dict[str, RiskScoreCase],
    med_map: dict[str, RiskScoreCase],
    med_tta4_map: dict[str, RiskScoreCase],
    greedy_map: dict[str, dict[str, Any]],
) -> tuple[list[LastMileAssessment], dict[str, Any]]:
    harness = LastMileClinicalHarness(policy)
    assessments: list[LastMileAssessment] = []
    valid_json_cnt = 0

    for isic_id, s1 in sys1_map.items():
        if isic_id not in med_map:
            continue
        s2 = med_map[isic_id]
        s2_tta = med_tta4_map.get(isic_id)
        # Approximate TTA sensitivity proxy when only MedGemma 1-view vs 4-view is precomputed
        tta_proxy = abs(s2.p_risk - s2_tta.p_risk) if s2_tta is not None else None

        greedy_row = greedy_map.get(isic_id)
        preflight: PreflightResult | None = None
        parsed_triage = None
        summary_fr: str | None = None
        if greedy_row:
            if greedy_row.get("preflight"):
                preflight = PreflightResult.model_validate(greedy_row["preflight"])
            raw_out = (greedy_row.get("eval_case") or {}).get("raw_model_output") or ""
            parsed_triage, err = parse_triage_json(raw_out)
            if parsed_triage is not None and err is None:
                valid_json_cnt += 1
                summary_fr = parsed_triage.summary_for_referral_fr
        else:
            valid_json_cnt += 1

        res = harness.evaluate_case(
            case_id=isic_id,
            p_sys1_raw=s1.p_risk,
            p_medgemma_raw=s2.p_risk,
            sys1_top_entity=s1.predicted_entity_argmax,
            p_melanoma_sys1=s1.p_melanoma,
            p_melanoma_medgemma=s2.p_melanoma,
            medgemma_top_entity=s2.predicted_entity_argmax,
            medgemma_assessment=parsed_triage,
            medgemma_summary_fr=summary_fr,
            preflight=preflight,
            tta_risk_std=tta_proxy,
        )
        assessments.append(res)

    jvr = valid_json_cnt / max(len(assessments), 1)
    return assessments, _summarize_cohort(assessments, sys1_map, json_valid_rate=jvr)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys1_test = _load_scores(args.sys1_test)
    med_test = _load_scores(args.medgemma_test)
    med_tta4_test = (
        _load_scores(args.medgemma_tta4_test) if args.medgemma_tta4_test.exists() else {}
    )
    greedy_test = _load_greedy_preds(args.medgemma_greedy_preds)

    sys1_val = _load_scores(args.sys1_hiba_val)
    med_val = _load_scores(args.medgemma_val)

    # Ablation across Last-Mile Safety Guardrails (all thresholds strictly from val!)
    pol_plain = HarnessPolicy(
        policy_version="ablation-plain-fusion",
        disagreement_delta=1.0,
        max_tta_std_for_green=1.0,
        melanoma_amber_threshold=1.0,
        melanoma_red_threshold=1.0,
        strict_preflight_block=False,
    )
    pol_guards_no_preflight = HarnessPolicy(
        policy_version="ablation-guards-no-preflight",
        strict_preflight_block=False,
    )
    pol_full = HarnessPolicy(
        policy_version="last-mile-v1.0-full",
        strict_preflight_block=True,
    )

    _, msk_plain = _run_policy(pol_plain, sys1_test, med_test, med_tta4_test, greedy_test)
    _, msk_no_pf = _run_policy(
        pol_guards_no_preflight, sys1_test, med_test, med_tta4_test, greedy_test
    )
    full_assessments, msk_full = _run_policy(
        pol_full, sys1_test, med_test, med_tta4_test, greedy_test
    )
    _, hiba_val_full = _run_policy(pol_full, sys1_val, med_val, {}, {})

    gate_full = evaluate_deployment_gate(msk_full)
    gate_no_pf = evaluate_deployment_gate(msk_no_pf)

    # Save full audit JSONL for all 200 MSKCC cases
    audit_path = args.output_dir / "mskcc_harness_assessments.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for item in full_assessments:
            handle.write(item.model_dump_json() + "\n")

    report = {
        "policy": pol_full.model_dump(),
        "hiba_val_full": hiba_val_full,
        "mskcc_ablation": {
            "1_plain_fusion_3zone": msk_plain,
            "2_fusion_plus_disagreement_melanoma_tta": msk_no_pf,
            "3_full_last_mile_harness_with_preflight": msk_full,
        },
        "deployment_gate_full": gate_full.model_dump(),
        "deployment_gate_no_preflight_block": gate_no_pf.model_dump(),
    }
    (args.output_dir / "harness_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "harness_3zone_chart.svg").write_text(
        _render_3zone_svg(msk_full), encoding="utf-8"
    )

    def _row(name: str, m: dict[str, Any]) -> str:
        zc = m["zone_counts_all"]
        return (
            f"| {name} | {m['malignant_safe_sensitivity']*100:.1f}% "
            f"({70 - m['green_zone_malignant_fn']}/70) | "
            f"{m['melanoma_safe_sensitivity']*100:.1f}% "
            f"({39 - m['green_zone_melanoma_fn']}/39) | "
            f"{m['green_zone_npv']*100:.1f}% | "
            f"{m['green_zone_benign_specificity']*100:.1f}% "
            f"({m['zone_counts_benign']['GREEN_LOW_RISK_SURVEILLANCE']}/115) | "
            f"{zc['GREEN_LOW_RISK_SURVEILLANCE']} / {zc['AMBER_UNCERTAIN_TELEEXPERTISE']} / "
            f"{zc['RED_PRIORITY_DERMATOLOGY']} | "
            f"{m['text_contradictions_caught_total']} "
            f"({m['text_contradictions_caught_on_malignant']} malins) |"
        )

    md_lines = [
        "# Last-Mile Clinical Safety Harness — Benchmark MSKCC Externe (n=200)",
        "",
        "Evaluation deterministe des 6 verrous du harness dernier kilometre sur les **200 biopsies "
        "consecutives MSKCC 2020** (zero fuite : tous les seuils proviennent de `val`).",
        "",
        "## 1. Ablation des verrous du Harness (MSKCC n=200)",
        "",
        "| Configuration Harness | Sensibilite Malins (Gris+Rouge) | Sensibilite Melanome | "
        "NPV Zone Verte | Specificite Zone Verte | Repartition Vert / Gris / Rouge | "
        "Contradictions Texte Bloquees |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _row("1. Fusion 50/50 3-zones seule", msk_plain),
        _row("2. + Desaccord S1/S2 + Signal Melanome + TTA", msk_no_pf),
        _row("3. Harness Complet (+ Verrou Preflight QA)", msk_full),
        "",
        "## 2. Verdict du Deployment Gate (Verrou 6 — Policy-as-Code)",
        "",
        f"- **Harness Complet :** **`{gate_full.decision}`**",
        f"- **Sans blocage strict Preflight :** **`{gate_no_pf.decision}`**",
        "",
        "| Critere Gate (Harness Complet) | Observe | Seuil Requis | Statut |",
        "|---|---:|---:|---|",
    ]
    for chk in gate_full.checks:
        st = "PASS" if chk.passed else "FAIL"
        md_lines.append(f"| `{chk.name}` | `{chk.observed:.4f}` | `{chk.required}` | **{st}** |")

    md_lines.extend(
        [
            "",
            "## 3. Regles declenchees sur MSKCC (Harness Complet)",
            "",
        ]
    )
    for rcode, rcnt in msk_full["triggered_rule_counts"].items():
        md_lines.append(f"- `{rcode}` : **{rcnt}** cas")

    md_lines.extend(
        [
            "",
            "## 4. Repartition 3 Zones (Vert / Gris / Rouge)",
            "",
            "![Last-Mile 3-Zone Chart](harness_3zone_chart.svg)",
            "",
        ]
    )
    (args.output_dir / "harness_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("=== LAST-MILE HARNESS MSKCC (n=200) ===")
    for tag, m in [
        ("Plain 3-zone", msk_plain),
        ("+Disag/Mel/TTA", msk_no_pf),
        ("Full (+Preflight)", msk_full),
    ]:
        print(
            f"{tag:18s} | Mal={m['malignant_safe_sensitivity']*100:5.1f}% "
            f"(FN={m['green_zone_malignant_fn']}) | "
            f"Mel={m['melanoma_safe_sensitivity']*100:5.1f}% "
            f"(FN={m['green_zone_melanoma_fn']}) | "
            f"GreenNPV={m['green_zone_npv']*100:5.1f}% | "
            f"GreenSpec={m['green_zone_benign_specificity']*100:5.1f}% | "
            f"Zones(G/A/R)={m['zone_counts_all']['GREEN_LOW_RISK_SURVEILLANCE']}/"
            f"{m['zone_counts_all']['AMBER_UNCERTAIN_TELEEXPERTISE']}/"
            f"{m['zone_counts_all']['RED_PRIORITY_DERMATOLOGY']} | "
            f"TextBlocks={m['text_contradictions_caught_total']} "
            f"({m['text_contradictions_caught_on_malignant']} mal)"
        )
    print(f"Gate Full: {gate_full.decision} | Gate No-PF: {gate_no_pf.decision}")
    print(
        f"HIBA Val Full: Mal={hiba_val_full['malignant_safe_sensitivity']*100:.1f}%, "
        f"GreenNPV={hiba_val_full['green_zone_npv']*100:.1f}%, "
        f"GreenSpec={hiba_val_full['green_zone_benign_specificity']*100:.1f}%"
    )


if __name__ == "__main__":
    main()
