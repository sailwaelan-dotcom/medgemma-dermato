"""Build leakage-free ISIC dataset splits with license and collection auditing."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from medgemma_dermato.config import load_config
from medgemma_dermato.data.isic_ingest import (
    ISICArchiveClient,
    ISICRecord,
    build_cross_hospital_splits,
    build_leakage_free_splits,
    export_records_jsonl,
    filter_records,
    stratified_sample_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ISIC Archive records, audit licenses, and build leakage-free splits."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/isic_v1"))
    parser.add_argument(
        "--mode",
        choices=["cross-hospital", "pooled"],
        default="cross-hospital",
        help="Split strategy: cross-hospital (HIBA->train/val, MSKCC->test) or pooled.",
    )
    parser.add_argument(
        "--commercial-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only CC-0 / CC-BY / CC-BY-4.0 records (default: True).",
    )
    parser.add_argument(
        "--collections",
        type=int,
        nargs="*",
        default=None,
        help="Optional ISIC collection IDs (used in pooled mode).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on raw ISIC records fetched per pool (default: None = full cohort).",
    )
    parser.add_argument(
        "--test-stratified-size",
        type=int,
        default=200,
        help=(
            "Stratified external test size (~35%% malignant, ~57.5%% benign, ~7.5%% indet; 0=all)."
        ),
    )
    parser.add_argument(
        "--skip-exclude-fetch",
        action="store_true",
        help="Skip fetching excluded ISIC 2017/2018/HAM10000 IDs (for fast smoke runs).",
    )
    return parser.parse_args()


def _label_counts(records: list[ISICRecord]) -> dict[str, Any]:
    malignant = sum(1 for r in records if r.is_malignant is True)
    benign = sum(1 for r in records if r.is_malignant is False)
    other = len(records) - malignant - benign
    subtypes = Counter(r.diagnosis_3 or r.diagnosis_2 or "Unknown" for r in records)
    fitzpatrick = Counter(r.fitzpatrick_skin_type or "Unknown" for r in records)
    return {
        "total": len(records),
        "malignant": malignant,
        "benign": benign,
        "other": other,
        "subtypes": dict(subtypes.most_common()),
        "fitzpatrick": dict(fitzpatrick),
    }


def _count_legacy_pre2019_ids(records: list[ISICRecord]) -> int:
    count = 0
    for rec in records:
        if rec.isic_id.startswith("ISIC_"):
            suffix = rec.isic_id[5:]
            if suffix.isdigit() and int(suffix) <= 39999:
                count += 1
    return count


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    allowed_types = set(cfg.data.allowed_image_types)

    with ISICArchiveClient() as client:
        excluded_ids: set[str] = set()
        if not args.skip_exclude_fetch and cfg.data.excluded_holdout_collections:
            print(
                f"Fetching excluded hold-out IDs from collections "
                f"{cfg.data.excluded_holdout_collections}..."
            )
            excluded_ids = client.fetch_collection_isic_ids(
                cfg.data.excluded_holdout_collections
            )
            print(f"Loaded {len(excluded_ids)} excluded hold-out IDs.")

        if args.mode == "cross-hospital":
            raw_tv = list(
                client.iter_images(
                    query="image_type:dermoscopic",
                    collections=cfg.data.train_val_collections,
                    limit=100,
                    max_items=args.max_items,
                )
            )
            raw_test = list(
                client.iter_images(
                    query="image_type:dermoscopic",
                    collections=cfg.data.external_test_collections,
                    limit=100,
                    max_items=args.max_items,
                )
            )
            filtered_tv, audit_tv = filter_records(
                raw_tv,
                commercial_only=args.commercial_only,
                allowed_image_types=allowed_types,
            )
            filtered_test, audit_test = filter_records(
                raw_test,
                commercial_only=args.commercial_only,
                allowed_image_types=allowed_types,
            )
            if args.test_stratified_size > 0 and len(filtered_test) > args.test_stratified_size:
                mal_n = int(round(args.test_stratified_size * 0.35))
                ben_n = int(round(args.test_stratified_size * 0.575))
                ind_n = max(0, args.test_stratified_size - mal_n - ben_n)
                filtered_test = stratified_sample_records(
                    filtered_test,
                    malignant_target=mal_n,
                    benign_target=ben_n,
                    indeterminate_target=ind_n,
                    seed=cfg.data.seed,
                )

            splits = build_cross_hospital_splits(
                train_val_records=filtered_tv,
                external_test_records=filtered_test,
                excluded_holdout_ids=excluded_ids,
                val_ratio=cfg.data.val_ratio / (cfg.data.train_ratio + cfg.data.val_ratio),
                seed=cfg.data.seed,
            )
            filter_audit: dict[str, Any] = {
                "train_val_pool": audit_tv,
                "external_test_pool": audit_test,
            }
        else:
            raw_records = list(
                client.iter_images(
                    query="image_type:dermoscopic",
                    collections=args.collections,
                    limit=100,
                    max_items=args.max_items,
                )
            )
            filtered, filter_audit = filter_records(
                raw_records,
                commercial_only=args.commercial_only,
                allowed_image_types=allowed_types,
            )
            splits = build_leakage_free_splits(
                filtered,
                excluded_holdout_ids=excluded_ids,
                train_ratio=cfg.data.train_ratio,
                val_ratio=cfg.data.val_ratio,
                seed=cfg.data.seed,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_records_jsonl(splits.train, args.output_dir / "train.jsonl")
    export_records_jsonl(splits.val, args.output_dir / "val.jsonl")
    export_records_jsonl(splits.test, args.output_dir / "test.jsonl")

    audit_payload = {
        "mode": args.mode,
        "filter_audit": filter_audit,
        "excluded_holdout_collections": (
            [] if args.skip_exclude_fetch else cfg.data.excluded_holdout_collections
        ),
        "excluded_holdout_id_count": len(excluded_ids),
        "excluded_from_test_count": splits.excluded_from_test_count,
        "legacy_pre2019_id_count_in_test": _count_legacy_pre2019_ids(splits.test),
        "split_counts": {
            "train": _label_counts(splits.train),
            "val": _label_counts(splits.val),
            "test": _label_counts(splits.test),
        },
    }
    audit_path = args.output_dir / "audit_report.json"
    audit_path.write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
