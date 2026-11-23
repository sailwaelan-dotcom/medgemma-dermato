"""ISIC Archive API v2 ingestion, license auditing, and leakage-free splitting."""

import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

_COMMERCIAL_LICENSES = frozenset({"CC-0", "CC-BY", "CC-BY-4.0"})


class ISICRecord(BaseModel):
    """Normalized ISIC image record with license and grouping metadata."""

    isic_id: str
    copyright_license: str | None = None
    attribution: str | None = None
    image_url: str | None = None
    image_type: str | None = None
    diagnosis_1: str | None = None
    diagnosis_2: str | None = None
    diagnosis_3: str | None = None
    melanocytic: bool | None = None
    age_approx: float | int | None = None
    sex: str | None = None
    anatom_site: str | None = None
    patient_id: str | None = None
    lesion_id: str | None = None
    fitzpatrick_skin_type: str | None = None
    diagnosis_confirm_type: str | None = None
    source_collection_id: int | None = None

    @property
    def is_malignant(self) -> bool | None:
        if self.diagnosis_1 is None:
            return None
        normalized = self.diagnosis_1.strip().lower()
        if normalized == "malignant":
            return True
        if normalized == "benign":
            return False
        return None

    @property
    def group_key(self) -> str:
        if self.patient_id and str(self.patient_id).strip():
            return f"patient:{self.patient_id.strip()}"
        if self.lesion_id and str(self.lesion_id).strip():
            return f"lesion:{self.lesion_id.strip()}"
        return f"isic:{self.isic_id}"


class SplitResult(BaseModel):
    """Leakage-free train/val/test split container."""

    train: list[ISICRecord]
    val: list[ISICRecord]
    test: list[ISICRecord]
    excluded_from_test_count: int = 0


def is_commercial_license(license_str: str | None) -> bool:
    """Return True only for permissive licenses allowing commercial derivative weights."""
    if not license_str:
        return False
    normalized = license_str.strip().upper().replace(" ", "-")
    return normalized in _COMMERCIAL_LICENSES


def parse_isic_api_item(
    item: dict[str, Any],
    source_collection_id: int | None = None,
) -> ISICRecord | None:
    """Parse one ISIC Archive API v2 image JSON object into an ISICRecord."""
    isic_id = item.get("isic_id")
    if not isic_id:
        return None

    files = item.get("files") or {}
    full_file = files.get("full") or {}
    image_url = full_file.get("url")

    metadata = item.get("metadata") or {}
    acquisition = metadata.get("acquisition") or {}
    clinical = metadata.get("clinical") or {}

    anatom_site = (
        clinical.get("anatom_site_general")
        or clinical.get("anatom_site_1")
        or clinical.get("anatom_site_2")
        or clinical.get("anatom_site_3")
    )

    return ISICRecord(
        isic_id=str(isic_id),
        copyright_license=item.get("copyright_license"),
        attribution=item.get("attribution"),
        image_url=image_url,
        image_type=acquisition.get("image_type"),
        diagnosis_1=clinical.get("diagnosis_1"),
        diagnosis_2=clinical.get("diagnosis_2"),
        diagnosis_3=clinical.get("diagnosis_3"),
        melanocytic=clinical.get("melanocytic"),
        age_approx=clinical.get("age_approx"),
        sex=clinical.get("sex"),
        anatom_site=anatom_site,
        patient_id=clinical.get("patient_id"),
        lesion_id=clinical.get("lesion_id"),
        fitzpatrick_skin_type=clinical.get("fitzpatrick_skin_type"),
        diagnosis_confirm_type=clinical.get("diagnosis_confirm_type"),
        source_collection_id=source_collection_id,
    )


class ISICArchiveClient:
    """HTTP client for ISIC Archive v2 using cursor pagination."""

    def __init__(
        self,
        base_url: str = "https://api.isic-archive.com/api/v2",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> "ISICArchiveClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.get(url, params=params)
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"ISIC API request failed after {self.max_retries} attempts: {last_exc}")

    def iter_images(
        self,
        query: str = "image_type:dermoscopic",
        collections: list[int] | None = None,
        limit: int = 100,
        max_items: int | None = None,
    ) -> Iterator[ISICRecord]:
        """Iterate ISIC records via cursor pagination (`next` URL)."""
        yielded = 0
        collection_targets: list[int | None] = (
            list(collections) if collections else [None]
        )

        for coll_id in collection_targets:
            next_url: str | None = f"{self.base_url}/images/search/"
            params: dict[str, Any] | None = {"limit": limit}
            if query:
                params["query"] = query
            if coll_id is not None:
                params["collections"] = coll_id

            while next_url:
                payload = self._get_json(next_url, params=params)
                params = None  # Subsequent pages use cursor query string baked into `next`.
                results = payload.get("results") or []
                if not results:
                    break
                for item in results:
                    record = parse_isic_api_item(item, source_collection_id=coll_id)
                    if record is None:
                        continue
                    yield record
                    yielded += 1
                    if max_items is not None and yielded >= max_items:
                        return
                next_url = payload.get("next")

    def fetch_collection_isic_ids(
        self,
        collection_ids: list[int],
        limit: int = 100,
        max_per_collection: int | None = None,
    ) -> set[str]:
        """Fetch all `isic_id` values belonging to specified ISIC collections."""
        ids: set[str] = set()
        for coll_id in collection_ids:
            count = 0
            for record in self.iter_images(
                query="",
                collections=[coll_id],
                limit=limit,
                max_items=max_per_collection,
            ):
                ids.add(record.isic_id)
                count += 1
                if max_per_collection is not None and count >= max_per_collection:
                    break
        return ids


def filter_records(
    records: list[ISICRecord],
    commercial_only: bool = True,
    allowed_image_types: set[str] | None = None,
) -> tuple[list[ISICRecord], dict[str, Any]]:
    """Filter ISIC records by license, image modality, and valid download URL."""
    allowed_types = (
        {t.strip().lower() for t in allowed_image_types}
        if allowed_image_types is not None
        else {"dermoscopic"}
    )

    kept: list[ISICRecord] = []
    license_counts: Counter[str] = Counter()
    rejected_non_commercial = 0
    rejected_image_type = 0
    rejected_missing_url = 0

    for record in records:
        lic_label = record.copyright_license or "UNKNOWN"
        license_counts[lic_label] += 1

        if commercial_only and not is_commercial_license(record.copyright_license):
            rejected_non_commercial += 1
            continue

        img_type = (record.image_type or "").strip().lower()
        if allowed_types and img_type not in allowed_types:
            rejected_image_type += 1
            continue

        if not record.image_url:
            rejected_missing_url += 1
            continue

        kept.append(record)

    audit = {
        "total_input": len(records),
        "kept": len(kept),
        "commercial_only": commercial_only,
        "rejected_non_commercial": rejected_non_commercial,
        "rejected_image_type": rejected_image_type,
        "rejected_missing_url": rejected_missing_url,
        "license_breakdown": dict(license_counts),
    }
    return kept, audit


def verify_split_invariants(split: SplitResult, excluded_holdout_ids: set[str]) -> None:
    """Assert zero ID/patient/lesion leakage across splits and zero excluded IDs in test."""
    split_sets = {
        "train": {r.isic_id for r in split.train},
        "val": {r.isic_id for r in split.val},
        "test": {r.isic_id for r in split.test},
    }
    if split_sets["train"] & split_sets["val"]:
        raise AssertionError("Overlap of isic_id between train and val")
    if split_sets["train"] & split_sets["test"]:
        raise AssertionError("Overlap of isic_id between train and test")
    if split_sets["val"] & split_sets["test"]:
        raise AssertionError("Overlap of isic_id between val and test")

    for field_name in ("patient_id", "lesion_id"):
        groups = {
            name: {
                str(getattr(r, field_name)).strip()
                for r in getattr(split, name)
                if getattr(r, field_name)
            }
            for name in ("train", "val", "test")
        }
        if (
            (groups["train"] & groups["val"])
            or (groups["train"] & groups["test"])
            or (groups["val"] & groups["test"])
        ):
            raise AssertionError(f"Leakage detected on {field_name} across splits")

    contaminated_test = split_sets["test"] & excluded_holdout_ids
    if contaminated_test:
        raise AssertionError(
            f"Excluded hold-out IDs leaked into test split: {sorted(contaminated_test)[:5]}"
        )


def build_leakage_free_splits(
    records: list[ISICRecord],
    excluded_holdout_ids: set[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Split records by patient/lesion group while keeping excluded IDs out of test."""
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("Invalid split ratios")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 to leave room for test")

    # Deduplicate by isic_id preserving order.
    deduped_map: dict[str, ISICRecord] = {}
    for rec in records:
        deduped_map.setdefault(rec.isic_id, rec)
    deduped = list(deduped_map.values())

    grouped: dict[str, list[ISICRecord]] = defaultdict(list)
    for rec in deduped:
        grouped[rec.group_key].append(rec)

    eligible_for_test: list[list[ISICRecord]] = []
    ineligible_for_test: list[list[ISICRecord]] = []
    excluded_count = 0

    for group_records in grouped.values():
        if any(r.isic_id in excluded_holdout_ids for r in group_records):
            ineligible_for_test.append(group_records)
            excluded_count += sum(1 for r in group_records if r.isic_id in excluded_holdout_ids)
        else:
            eligible_for_test.append(group_records)

    rng = random.Random(seed)
    rng.shuffle(eligible_for_test)
    rng.shuffle(ineligible_for_test)

    total_records = len(deduped)
    target_test_records = int(round(total_records * (1.0 - train_ratio - val_ratio)))

    test_records: list[ISICRecord] = []
    remaining_groups: list[list[ISICRecord]] = []
    for grp in eligible_for_test:
        if len(test_records) < target_test_records:
            test_records.extend(grp)
        else:
            remaining_groups.append(grp)

    remaining_groups.extend(ineligible_for_test)
    rng.shuffle(remaining_groups)

    remaining_total = sum(len(g) for g in remaining_groups)
    train_share_of_remaining = train_ratio / (train_ratio + val_ratio)
    target_train_records = int(round(remaining_total * train_share_of_remaining))

    train_records: list[ISICRecord] = []
    val_records: list[ISICRecord] = []
    for grp in remaining_groups:
        if len(train_records) < target_train_records:
            train_records.extend(grp)
        else:
            val_records.extend(grp)

    result = SplitResult(
        train=train_records,
        val=val_records,
        test=test_records,
        excluded_from_test_count=excluded_count,
    )
    verify_split_invariants(result, excluded_holdout_ids)
    return result


def build_cross_hospital_splits(
    train_val_records: list[ISICRecord],
    external_test_records: list[ISICRecord],
    excluded_holdout_ids: set[str],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Build cross-hospital train/val vs external test splits without leakage."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be strictly between 0 and 1")

    test_dedup: dict[str, ISICRecord] = {}
    excluded_count = 0
    for rec in external_test_records:
        if rec.isic_id in excluded_holdout_ids:
            excluded_count += 1
            continue
        test_dedup.setdefault(rec.isic_id, rec)
    test_records = list(test_dedup.values())

    test_ids = {r.isic_id for r in test_records}
    test_patients = {r.patient_id.strip() for r in test_records if r.patient_id}
    test_lesions = {r.lesion_id.strip() for r in test_records if r.lesion_id}

    tv_dedup: dict[str, ISICRecord] = {}
    for rec in train_val_records:
        if rec.isic_id in test_ids:
            continue
        if rec.patient_id and rec.patient_id.strip() in test_patients:
            continue
        if rec.lesion_id and rec.lesion_id.strip() in test_lesions:
            continue
        tv_dedup.setdefault(rec.isic_id, rec)

    grouped: dict[str, list[ISICRecord]] = defaultdict(list)
    for rec in tv_dedup.values():
        grouped[rec.group_key].append(rec)

    groups = list(grouped.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_tv = sum(len(g) for g in groups)
    target_val = int(round(total_tv * val_ratio))

    val_records: list[ISICRecord] = []
    train_records: list[ISICRecord] = []
    for grp in groups:
        if len(val_records) < target_val:
            val_records.extend(grp)
        else:
            train_records.extend(grp)

    result = SplitResult(
        train=train_records,
        val=val_records,
        test=test_records,
        excluded_from_test_count=excluded_count,
    )
    verify_split_invariants(result, excluded_holdout_ids)
    return result


def _sample_grouped_bucket(
    groups: list[list[ISICRecord]],
    target_count: int,
    rng: random.Random,
) -> list[ISICRecord]:
    """Select groups up to target_count while interleaving diagnosis_3 subtypes."""
    if target_count <= 0 or not groups:
        return []
    by_subtype: dict[str, list[list[ISICRecord]]] = defaultdict(list)
    for grp in groups:
        subtype = (grp[0].diagnosis_3 or grp[0].diagnosis_2 or "unknown").strip()
        by_subtype[subtype].append(grp)

    subtype_keys = sorted(by_subtype.keys())
    for key in subtype_keys:
        rng.shuffle(by_subtype[key])

    selected: list[ISICRecord] = []
    # Round-robin across subtypes so rare malignant entities (BCC, SCC) are preserved.
    idx = 0
    while len(selected) < target_count:
        progressed = False
        for key in subtype_keys:
            bucket = by_subtype[key]
            if idx < len(bucket):
                grp = bucket[idx]
                if len(selected) + len(grp) <= target_count or not selected:
                    selected.extend(grp)
                    progressed = True
                    if len(selected) >= target_count:
                        break
        if not progressed:
            break
        idx += 1
    if len(selected) > target_count and len(selected) == 1:
        return selected[:target_count]
    return selected


def stratified_sample_records(
    records: list[ISICRecord],
    malignant_target: int = 70,
    benign_target: int = 115,
    indeterminate_target: int = 15,
    seed: int = 42,
) -> list[ISICRecord]:
    """Deterministically sample grouped records across Malignant/Benign/Indeterminate."""
    deduped_map: dict[str, ISICRecord] = {}
    for rec in records:
        deduped_map.setdefault(rec.isic_id, rec)

    grouped: dict[str, list[ISICRecord]] = defaultdict(list)
    for rec in deduped_map.values():
        grouped[rec.group_key].append(rec)

    mal_groups: list[list[ISICRecord]] = []
    ben_groups: list[list[ISICRecord]] = []
    ind_groups: list[list[ISICRecord]] = []
    for grp in grouped.values():
        if any(r.is_malignant is True for r in grp):
            mal_groups.append(grp)
        elif all(r.is_malignant is False for r in grp):
            ben_groups.append(grp)
        else:
            ind_groups.append(grp)

    rng = random.Random(seed)
    chosen = (
        _sample_grouped_bucket(mal_groups, malignant_target, rng)
        + _sample_grouped_bucket(ben_groups, benign_target, rng)
        + _sample_grouped_bucket(ind_groups, indeterminate_target, rng)
    )
    rng.shuffle(chosen)
    return chosen


def export_records_jsonl(records: list[ISICRecord], path: Path | str) -> None:
    """Write ISICRecord list to a JSONL manifest."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(rec.model_dump_json() + "\n")


def load_records_jsonl(path: Path | str) -> list[ISICRecord]:
    """Load ISICRecord list from a JSONL manifest."""
    in_path = Path(path)
    records: list[ISICRecord] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(ISICRecord.model_validate_json(stripped))
    return records
