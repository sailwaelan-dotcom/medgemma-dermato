"""ISIC data ingestion, license auditing, and leakage-free splitting."""

from medgemma_dermato.data.isic_ingest import (
    ISICArchiveClient,
    ISICRecord,
    SplitResult,
    build_cross_hospital_splits,
    build_leakage_free_splits,
    export_records_jsonl,
    filter_records,
    is_commercial_license,
    load_records_jsonl,
    parse_isic_api_item,
    verify_split_invariants,
)

__all__ = [
    "ISICArchiveClient",
    "ISICRecord",
    "SplitResult",
    "build_cross_hospital_splits",
    "build_leakage_free_splits",
    "export_records_jsonl",
    "filter_records",
    "is_commercial_license",
    "load_records_jsonl",
    "parse_isic_api_item",
    "verify_split_invariants",
]
