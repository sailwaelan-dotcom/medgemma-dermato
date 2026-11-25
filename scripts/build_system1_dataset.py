"""Build a leakage-free, commercial-only, multi-center dermoscopy dataset for System 1.

Strictly excludes ALL MSKCC images/collections (especially locked external test set
`data/isic_v1/test.jsonl` from Collection 163 / 216) and preserves every image of
`data/isic_v1/val.jsonl` strictly inside the validation split so Late Fusion calibration
with MedGemma QLoRA has zero data leakage.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from medgemma_dermato.config import load_config
from medgemma_dermato.data.isic_ingest import (
    ISICRecord,
    export_records_jsonl,
    is_commercial_license,
    load_records_jsonl,
    parse_isic_api_item,
)
from medgemma_dermato.schema import map_isic_to_entity_category

MSK_FORBIDDEN_KEYWORDS = ("memorial sloan", "mskcc", "msk-")
COMMERCIAL_QUERY = (
    'image_type:dermoscopic AND (copyright_license:"CC-0" OR copyright_license:"CC-BY")'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-center non-MSKCC commercial dataset for System 1."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--hiba-train", type=Path, default=Path("data/isic_v1/train.jsonl"))
    parser.add_argument("--hiba-val", type=Path, default=Path("data/isic_v1/val.jsonl"))
    parser.add_argument("--locked-test", type=Path, default=Path("data/isic_v1/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/system1_v1"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _is_msk_attribution(attribution: str | None) -> bool:
    low = (attribution or "").strip().lower()
    return any(kw in low for kw in MSK_FORBIDDEN_KEYWORDS)


def _fetch_collection_records(
    client: httpx.Client,
    collection_id: int,
    query: str,
    max_items: int,
    commercial_licenses: set[str],
) -> list[tuple[ISICRecord, str]]:
    """Fetch commercial dermoscopic records with attribution from one ISIC collection."""
    url: str | None = "https://api.isic-archive.com/api/v2/images/search/"
    params: dict[str, Any] | None = {
        "collections": str(collection_id),
        "query": query,
        "limit": 100,
    }
    collected: list[tuple[ISICRecord, str]] = []
    while url and len(collected) < max_items:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("results", []):
            attr = str(item.get("attribution") or "Unknown")
            if _is_msk_attribution(attr):
                continue
            rec = parse_isic_api_item(item, source_collection_id=collection_id)
            if rec is None:
                continue
            if (
                not is_commercial_license(rec.copyright_license)
                or (rec.copyright_license or "").strip().upper() not in commercial_licenses
            ):
                continue
            if (rec.image_type or "").strip().lower() != "dermoscopic" or not rec.image_url:
                continue
            collected.append((rec, attr))
            if len(collected) >= max_items:
                break
        url = payload.get("next")
        params = None
    return collected


def _fetch_all_collection_ids(client: httpx.Client, collection_id: int) -> set[str]:
    """Fetch all isic_ids in a forbidden collection (e.g. 163) for airtight exclusion."""
    url: str | None = "https://api.isic-archive.com/api/v2/images/search/"
    params: dict[str, Any] | None = {"collections": str(collection_id), "limit": 100}
    ids: set[str] = set()
    while url:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("results", []):
            if item.get("isic_id"):
                ids.add(str(item["isic_id"]))
        url = payload.get("next")
        params = None
    return ids


def _group_split_records(
    records: list[ISICRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[ISICRecord], list[ISICRecord]]:
    """Patient/lesion-grouped split keeping class balance approximately preserved."""
    groups: dict[str, list[ISICRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.group_key].append(rec)

    mal_groups: list[list[ISICRecord]] = []
    ben_groups: list[list[ISICRecord]] = []
    for grp in groups.values():
        if any(r.is_malignant is True for r in grp):
            mal_groups.append(grp)
        else:
            ben_groups.append(grp)

    rng = random.Random(seed)
    rng.shuffle(mal_groups)
    rng.shuffle(ben_groups)

    train_out: list[ISICRecord] = []
    val_out: list[ISICRecord] = []
    for bucket in (mal_groups, ben_groups):
        n_total = sum(len(g) for g in bucket)
        n_val_target = int(round(n_total * val_ratio))
        val_acc = 0
        for g in bucket:
            if val_acc < n_val_target:
                val_out.extend(g)
                val_acc += len(g)
            else:
                train_out.extend(g)
    return train_out, val_out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    commercial_licenses = {lic.strip().upper() for lic in cfg.data.commercial_licenses}
    rng = random.Random(args.seed)

    hiba_train = load_records_jsonl(args.hiba_train)
    hiba_val = load_records_jsonl(args.hiba_val)
    locked_test = load_records_jsonl(args.locked_test)

    forbidden_ids: set[str] = {r.isic_id for r in locked_test}
    forbidden_patients: set[str] = {r.patient_id for r in locked_test if r.patient_id}
    forbidden_lesions: set[str] = {r.lesion_id for r in locked_test if r.lesion_id}

    existing_ids: set[str] = {r.isic_id for r in hiba_train} | {r.isic_id for r in hiba_val}

    print("Fetching full Collection 163 (MSKCC 2020) IDs for airtight exclusion...", flush=True)
    with httpx.Client(timeout=30.0) as client:
        col163_ids = _fetch_all_collection_ids(client, 163)
        forbidden_ids |= col163_ids
        print(f"Total forbidden MSKCC-163 IDs: {len(forbidden_ids)}", flush=True)

        extra_sources: dict[str, list[ISICRecord]] = {}
        attribution_counts: Counter[str] = Counter()
        license_counts: Counter[str] = Counter()

        def _accept(pairs: list[tuple[ISICRecord, str]], source_tag: str) -> list[ISICRecord]:
            kept: list[ISICRecord] = []
            for rec, attr in pairs:
                if rec.isic_id in forbidden_ids or rec.isic_id in existing_ids:
                    continue
                if rec.patient_id and rec.patient_id in forbidden_patients:
                    continue
                if rec.lesion_id and rec.lesion_id in forbidden_lesions:
                    continue
                existing_ids.add(rec.isic_id)
                attribution_counts[f"{source_tag} :: {attr}"] += 1
                license_counts[rec.copyright_license or "UNKNOWN"] += 1
                kept.append(rec)
            return kept

        # 1. DERM12345 (Collection 399, CC-BY, Imperial College / Turkey):
        #    Take ALL malignant (up to 1400) + benign pool prioritizing non-nevus diversity.
        print("Fetching DERM12345 (399) malignant & benign...", flush=True)
        d399_mal = _accept(
            _fetch_collection_records(
                client,
                399,
                f"{COMMERCIAL_QUERY} AND diagnosis_1:Malignant",
                1400,
                commercial_licenses,
            ),
            "DERM12345(399)",
        )
        d399_indet = _accept(
            _fetch_collection_records(
                client,
                399,
                f"{COMMERCIAL_QUERY} AND diagnosis_1:Indeterminate",
                400,
                commercial_licenses,
            ),
            "DERM12345(399)",
        )
        # Fetch rich non-nevus benign lesions + sampled nevi from DERM12345
        d399_ben_all = _accept(
            _fetch_collection_records(
                client,
                399,
                f"{COMMERCIAL_QUERY} AND diagnosis_1:Benign",
                4500,
                commercial_licenses,
            ),
            "DERM12345(399)",
        )
        non_nevus_399 = [
            r
            for r in d399_ben_all
            if map_isic_to_entity_category(r.diagnosis_1, r.diagnosis_2, r.diagnosis_3) != "nevus"
        ]
        nevus_399 = [
            r
            for r in d399_ben_all
            if map_isic_to_entity_category(r.diagnosis_1, r.diagnosis_2, r.diagnosis_3) == "nevus"
        ]
        rng.shuffle(nevus_399)
        d399_ben_kept = non_nevus_399 + nevus_399[: max(0, 1800 - len(non_nevus_399))]
        extra_sources["DERM12345_399"] = d399_mal + d399_indet + d399_ben_kept
        n_nev = len(d399_ben_kept) - len(non_nevus_399)
        print(
            f"  DERM12345 kept: mal={len(d399_mal)}, indet={len(d399_indet)}, "
            f"non_nevus_ben={len(non_nevus_399)}, nevus_ben={n_nev}",
            flush=True,
        )

        # 2. Chile Clonal Seborrheic Keratoses (534, CC-BY, hard benign keratosis mimics)
        print("Fetching Chile Clonal Seborrheic Keratoses (534)...", flush=True)
        c534 = _accept(
            _fetch_collection_records(client, 534, COMMERCIAL_QUERY, 200, commercial_licenses),
            "ChileSK(534)",
        )
        extra_sources["ChileSK_534"] = c534
        print(f"  ChileSK kept: {len(c534)}", flush=True)

        # 3. MEL-SELF Australia (485, CC-BY)
        print("Fetching MEL-SELF Australia (485)...", flush=True)
        m485_mal = _accept(
            _fetch_collection_records(
                client,
                485,
                f"{COMMERCIAL_QUERY} AND diagnosis_1:Malignant",
                100,
                commercial_licenses,
            ),
            "MEL-SELF(485)",
        )
        m485_ben = _accept(
            _fetch_collection_records(
                client,
                485,
                f"{COMMERCIAL_QUERY} AND diagnosis_1:Benign",
                350,
                commercial_licenses,
            ),
            "MEL-SELF(485)",
        )
        extra_sources["MEL_SELF_485"] = m485_mal + m485_ben
        print(f"  MEL-SELF kept: mal={len(m485_mal)}, ben={len(m485_ben)}", flush=True)

        # 4. Histopathologically Confirmed Melanoma & Nevus (294, CC-0, non-MSK biopsy cohort)
        print("Fetching Histopath Melanoma & Nevus (294, CC-0, non-MSK)...", flush=True)
        h294_mal = _accept(
            _fetch_collection_records(
                client,
                294,
                f"{COMMERCIAL_QUERY} AND diagnosis_1:Malignant",
                1000,
                commercial_licenses,
            ),
            "HistopathMelNev(294)",
        )
        h294_ben = _accept(
            _fetch_collection_records(
                client,
                294,
                f"{COMMERCIAL_QUERY} AND diagnosis_1:Benign",
                1000,
                commercial_licenses,
            ),
            "HistopathMelNev(294)",
        )
        extra_sources["Histopath_294"] = h294_mal + h294_ben
        print(f"  Histopath(294) kept: mal={len(h294_mal)}, ben={len(h294_ben)}", flush=True)

    # Assemble splits: HIBA stays strictly pinned to data/isic_v1/{train,val}.jsonl!
    train_all: list[ISICRecord] = list(hiba_train)
    val_all: list[ISICRecord] = list(hiba_val)

    for src_name, recs in extra_sources.items():
        tr, va = _group_split_records(recs, val_ratio=0.15, seed=args.seed)
        train_all.extend(tr)
        val_all.extend(va)
        print(f"  Split {src_name}: train={len(tr)}, val={len(va)}", flush=True)

    # Strict leakage assertions
    train_ids = {r.isic_id for r in train_all}
    val_ids = {r.isic_id for r in val_all}
    test_ids = {r.isic_id for r in locked_test}
    assert not (train_ids & val_ids), "Train/Val ID overlap!"
    assert not (train_ids & forbidden_ids), "Train/MSKCC-163 ID overlap!"
    assert not (val_ids & forbidden_ids), "Val/MSKCC-163 ID overlap!"
    assert not (train_ids & test_ids), "Train/Test ID overlap!"
    assert not (val_ids & test_ids), "Val/Test ID overlap!"
    assert {r.isic_id for r in hiba_val}.issubset(val_ids), "HIBA val not fully preserved in val!"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_records_jsonl(train_all, args.output_dir / "train.jsonl")
    export_records_jsonl(val_all, args.output_dir / "val.jsonl")

    def _split_stats(recs: list[ISICRecord]) -> dict[str, Any]:
        ent = Counter(
            map_isic_to_entity_category(r.diagnosis_1, r.diagnosis_2, r.diagnosis_3) for r in recs
        )
        return {
            "total": len(recs),
            "malignant": sum(1 for r in recs if r.is_malignant is True),
            "benign": sum(1 for r in recs if r.is_malignant is False),
            "other": sum(1 for r in recs if r.is_malignant is None),
            "entities": dict(ent.most_common()),
            "licenses": dict(Counter(r.copyright_license for r in recs)),
        }

    audit = {
        "mode": "multi-center-system1",
        "mskcc_strictly_excluded": True,
        "forbidden_col163_id_count": len(forbidden_ids),
        "overlap_with_locked_test": 0,
        "hiba_val_strictly_preserved_in_val": len(hiba_val),
        "source_counts": {
            "HIBA_251_train": len(hiba_train),
            "HIBA_251_val": len(hiba_val),
            **{k: len(v) for k, v in extra_sources.items()},
        },
        "extra_attributions": dict(attribution_counts.most_common()),
        "splits": {
            "train": _split_stats(train_all),
            "val": _split_stats(val_all),
        },
    }
    audit_path = args.output_dir / "audit_report.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n=== SYSTEM 1 MULTI-CENTER DATASET AUDIT ===")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
