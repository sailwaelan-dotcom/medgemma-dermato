"""Build a balanced, leakage-free multi-center dataset for MedGemma 1.5 4B QLoRA v2.

Combines 100% of HIBA (`data/isic_v1/{train,val}.jsonl`) with stratified samples across all
remaining 4 cohorts in `data/system1_v1/` (`DERM12345`, `ChileSK`, `MEL-SELF`, `Histopath-294`),
while asserting ZERO overlap with locked external test set `data/isic_v1/test.jsonl` (MSKCC 2020).
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medgemma_dermato.data.isic_ingest import (
    ISICRecord,
    export_records_jsonl,
    load_records_jsonl,
)
from medgemma_dermato.schema import map_isic_to_entity_category

MSK_FORBIDDEN_KEYWORDS = ("memorial sloan", "mskcc", "msk-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build balanced multi-center dataset for QLoRA v2."
    )
    parser.add_argument("--sys1-train", type=Path, default=Path("data/system1_v1/train.jsonl"))
    parser.add_argument("--sys1-val", type=Path, default=Path("data/system1_v1/val.jsonl"))
    parser.add_argument("--hiba-train", type=Path, default=Path("data/isic_v1/train.jsonl"))
    parser.add_argument("--hiba-val", type=Path, default=Path("data/isic_v1/val.jsonl"))
    parser.add_argument("--locked-test", type=Path, default=Path("data/isic_v1/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/qlora_v2"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _is_msk(rec: ISICRecord) -> bool:
    low = (rec.attribution or "").strip().lower()
    return any(kw in low for kw in MSK_FORBIDDEN_KEYWORDS)


def _cohort_key(rec: ISICRecord) -> str:
    return (rec.attribution or "Unknown").strip()[:32]


def _stratified_cap_by_cohort(
    records: list[ISICRecord],
    cap: int,
    rng: random.Random,
) -> list[ISICRecord]:
    if len(records) <= cap:
        return list(records)
    by_cohort: dict[str, list[ISICRecord]] = defaultdict(list)
    for r in records:
        by_cohort[_cohort_key(r)].append(r)
    for grp in by_cohort.values():
        rng.shuffle(grp)

    selected: list[ISICRecord] = []
    cohorts = sorted(by_cohort.keys())
    idx = 0
    while len(selected) < cap:
        progressed = False
        for c in cohorts:
            if idx < len(by_cohort[c]):
                selected.append(by_cohort[c][idx])
                progressed = True
                if len(selected) >= cap:
                    break
        if not progressed:
            break
        idx += 1
    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    hiba_train = load_records_jsonl(args.hiba_train)
    hiba_val = load_records_jsonl(args.hiba_val)
    sys1_train = load_records_jsonl(args.sys1_train)
    sys1_val = load_records_jsonl(args.sys1_val)
    locked_test = load_records_jsonl(args.locked_test)

    hiba_train_ids = {r.isic_id for r in hiba_train}
    hiba_val_ids = {r.isic_id for r in hiba_val}
    test_ids = {r.isic_id for r in locked_test}
    test_patients = {r.patient_id for r in locked_test if r.patient_id}
    test_lesions = {r.lesion_id for r in locked_test if r.lesion_id}

    def _safe_extra(records: list[ISICRecord], seen_ids: set[str]) -> list[ISICRecord]:
        out: list[ISICRecord] = []
        for r in records:
            if r.isic_id in seen_ids or r.isic_id in test_ids:
                continue
            if r.patient_id and r.patient_id in test_patients:
                continue
            if r.lesion_id and r.lesion_id in test_lesions:
                continue
            if _is_msk(r):
                continue
            out.append(r)
        return out

    extra_train_pool = _safe_extra(sys1_train, hiba_train_ids | hiba_val_ids)
    by_entity: dict[str, list[ISICRecord]] = defaultdict(list)
    for r in extra_train_pool:
        ent = map_isic_to_entity_category(r.diagnosis_1, r.diagnosis_2, r.diagnosis_3)
        by_entity[ent].append(r)

    # Keep 100% of minority & hard-mimic entities; cap majority classes cleanly.
    train_caps = {
        "melanoma": 600,
        "basal_cell_carcinoma": 220,
        "nevus": 450,
    }
    train_extra: list[ISICRecord] = []
    for ent, recs in sorted(by_entity.items()):
        if ent in train_caps:
            train_extra.extend(_stratified_cap_by_cohort(recs, train_caps[ent], rng))
        else:
            train_extra.extend(recs)

    train_all = list(hiba_train) + train_extra
    rng.shuffle(train_all)

    # Val: 100% HIBA val (230) + 170 stratified multi-center records from sys1_val
    used_train_ids = {r.isic_id for r in train_all}
    extra_val_pool = _safe_extra(sys1_val, hiba_train_ids | hiba_val_ids | used_train_ids)
    val_by_ent: dict[str, list[ISICRecord]] = defaultdict(list)
    for r in extra_val_pool:
        ent = map_isic_to_entity_category(r.diagnosis_1, r.diagnosis_2, r.diagnosis_3)
        val_by_ent[ent].append(r)

    val_extra: list[ISICRecord] = []
    val_quotas = {
        "benign_keratosis": 35,
        "other_benign": 25,
        "melanoma": 45,
        "squamous_cell_carcinoma": 20,
        "basal_cell_carcinoma": 20,
        "nevus": 25,
    }
    for ent, q in val_quotas.items():
        val_extra.extend(_stratified_cap_by_cohort(val_by_ent.get(ent, []), q, rng))
    val_all = list(hiba_val) + val_extra

    # Strict anti-leakage assertions
    tr_ids = {r.isic_id for r in train_all}
    va_ids = {r.isic_id for r in val_all}
    assert not (tr_ids & va_ids), "Train/Val overlap detected!"
    assert not (tr_ids & test_ids), "Train/MSKCC test overlap detected!"
    assert not (va_ids & test_ids), "Val/MSKCC test overlap detected!"
    assert all(not _is_msk(r) for r in train_all + val_all), "MSKCC attribution detected!"
    assert hiba_val_ids.issubset(va_ids), "HIBA val must be 100% preserved in val!"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_records_jsonl(train_all, args.output_dir / "train.jsonl")
    export_records_jsonl(val_all, args.output_dir / "val.jsonl")

    def _stats(recs: list[ISICRecord]) -> dict[str, Any]:
        ents = Counter(
            map_isic_to_entity_category(r.diagnosis_1, r.diagnosis_2, r.diagnosis_3) for r in recs
        )
        cohorts = Counter(_cohort_key(r) for r in recs)
        return {
            "total": len(recs),
            "malignant": sum(1 for r in recs if r.is_malignant is True),
            "benign": sum(1 for r in recs if r.is_malignant is False),
            "other": sum(1 for r in recs if r.is_malignant is None),
            "entities": dict(ents.most_common()),
            "cohorts": dict(cohorts.most_common()),
        }

    audit = {
        "mode": "qlora-v2-multi-center",
        "overlap_with_mskcc_test": 0,
        "overlap_train_val": 0,
        "hiba_val_preserved": len(hiba_val),
        "train": _stats(train_all),
        "val": _stats(val_all),
    }
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
