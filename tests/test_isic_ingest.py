"""Tests for ISIC Archive parsing, commercial license filtering, and leakage-free splitting."""

from medgemma_dermato.data.isic_ingest import (
    ISICRecord,
    build_cross_hospital_splits,
    build_leakage_free_splits,
    filter_records,
    is_commercial_license,
    parse_isic_api_item,
    stratified_sample_records,
)


def _sample_api_item(
    isic_id: str,
    license_name: str = "CC-0",
    image_type: str = "dermoscopic",
    patient_id: str | None = None,
    lesion_id: str | None = None,
) -> dict[str, object]:
    return {
        "public": True,
        "isic_id": isic_id,
        "copyright_license": license_name,
        "attribution": "Test Clinic",
        "files": {
            "full": {
                "url": f"https://isic-archive.s3.us-east-1.amazonaws.com/images/{isic_id}.jpg",
                "size": 50000,
            }
        },
        "metadata": {
            "acquisition": {
                "pixels_x": 1022,
                "pixels_y": 767,
                "image_type": image_type,
            },
            "clinical": {
                "sex": "female",
                "age_approx": 50,
                "anatom_site_1": "Trunk",
                "diagnosis_1": "Malignant",
                "diagnosis_2": "Melanoma",
                "diagnosis_confirm_type": "histopathology",
                "fitzpatrick_skin_type": "II",
                "melanocytic": True,
                "patient_id": patient_id,
                "lesion_id": lesion_id,
            },
        },
    }


def test_parse_isic_api_item_and_license_filter() -> None:
    cc0 = parse_isic_api_item(_sample_api_item("ISIC_1", "CC-0"))
    ccby = parse_isic_api_item(_sample_api_item("ISIC_2", "CC-BY"))
    ccbync = parse_isic_api_item(_sample_api_item("ISIC_3", "CC-BY-NC"))
    tbp = parse_isic_api_item(
        _sample_api_item("ISIC_4", "CC-BY-4.0", image_type="TBP tile: close-up")
    )
    unknown = parse_isic_api_item(_sample_api_item("ISIC_5", ""))

    assert cc0 is not None and cc0.is_malignant is True
    assert cc0.fitzpatrick_skin_type == "II"
    assert cc0.diagnosis_confirm_type == "histopathology"
    assert is_commercial_license("CC-0") is True
    assert is_commercial_license("CC-BY") is True
    assert is_commercial_license("CC-BY-NC") is False
    assert is_commercial_license(None) is False

    records = [r for r in (cc0, ccby, ccbync, tbp, unknown) if r is not None]
    kept, audit = filter_records(records, commercial_only=True, allowed_image_types={"dermoscopic"})

    assert {r.isic_id for r in kept} == {"ISIC_1", "ISIC_2"}
    assert audit["rejected_non_commercial"] == 2
    assert audit["rejected_image_type"] == 1


def test_leakage_free_splits_respect_patients_lesions_and_excluded_ids() -> None:
    records: list[ISICRecord] = []
    # Patient P1 has 3 images, one in excluded_holdout_ids -> whole group stays out of test.
    for idx in range(3):
        records.append(
            ISICRecord(
                isic_id=f"ISIC_EX_{idx}",
                copyright_license="CC-0",
                image_url=f"https://example.com/{idx}.jpg",
                image_type="dermoscopic",
                patient_id="P_CONTAMINATED",
                lesion_id=f"L_EX_{idx}",
            )
        )

    # 20 clean patients with 2 images each sharing the same lesion_id.
    for p_idx in range(20):
        for img_idx in range(2):
            records.append(
                ISICRecord(
                    isic_id=f"ISIC_{p_idx}_{img_idx}",
                    copyright_license="CC-BY",
                    image_url=f"https://example.com/{p_idx}_{img_idx}.jpg",
                    image_type="dermoscopic",
                    patient_id=f"P_{p_idx}",
                    lesion_id=f"L_{p_idx}",
                    diagnosis_1="Malignant" if p_idx % 4 == 0 else "Benign",
                )
            )

    excluded = {"ISIC_EX_0"}
    split = build_leakage_free_splits(
        records,
        excluded_holdout_ids=excluded,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )

    test_ids = {r.isic_id for r in split.test}
    test_patients = {r.patient_id for r in split.test}
    assert len(split.test) > 0
    assert not (test_ids & excluded)
    assert "P_CONTAMINATED" not in test_patients
    assert split.excluded_from_test_count == 1
    assert len(split.train) + len(split.val) + len(split.test) == len(records)


def test_cross_hospital_splits_isolate_test_and_strip_overlaps() -> None:
    tv_pool = [
        ISICRecord(
            isic_id=f"HIBA_{i}",
            copyright_license="CC-BY",
            image_url=f"https://example.com/h_{i}.jpg",
            image_type="dermoscopic",
            patient_id=f"PAT_HIBA_{i // 2}",
            lesion_id=f"LES_HIBA_{i // 2}",
        )
        for i in range(10)
    ]
    # Add a record sharing patient_id with external test to verify removal from train/val.
    tv_pool.append(
        ISICRecord(
            isic_id="HIBA_LEAK",
            copyright_license="CC-BY",
            image_url="https://example.com/h_leak.jpg",
            image_type="dermoscopic",
            patient_id="PAT_SHARED",
            lesion_id="LES_OTHER",
        )
    )
    ext_test = [
        ISICRecord(
            isic_id="MSK_1",
            copyright_license="CC-BY",
            image_url="https://example.com/m_1.jpg",
            image_type="dermoscopic",
            patient_id="PAT_SHARED",
            lesion_id="LES_MSK_1",
        ),
        ISICRecord(
            isic_id="MSK_EXCLUDED",
            copyright_license="CC-0",
            image_url="https://example.com/m_ex.jpg",
            image_type="dermoscopic",
            patient_id="PAT_MSK_2",
            lesion_id="LES_MSK_2",
        ),
    ]

    split = build_cross_hospital_splits(
        train_val_records=tv_pool,
        external_test_records=ext_test,
        excluded_holdout_ids={"MSK_EXCLUDED"},
        val_ratio=0.20,
        seed=42,
    )

    assert [r.isic_id for r in split.test] == ["MSK_1"]
    assert split.excluded_from_test_count == 1
    tv_ids = {r.isic_id for r in split.train} | {r.isic_id for r in split.val}
    assert "HIBA_LEAK" not in tv_ids
    assert len(tv_ids) == 10


def test_stratified_sample_preserves_subtypes_and_groups() -> None:
    pool: list[ISICRecord] = []
    # 30 malignant split across melanoma, BCC, SCC (2 images per patient).
    subtypes = ["Melanoma in situ", "Basal cell carcinoma", "Squamous cell carcinoma, NOS"]
    for idx in range(15):
        sub = subtypes[idx % 3]
        for j in range(2):
            pool.append(
                ISICRecord(
                    isic_id=f"MAL_{idx}_{j}",
                    copyright_license="CC-BY",
                    image_url=f"https://example.com/mal_{idx}_{j}.jpg",
                    image_type="dermoscopic",
                    diagnosis_1="Malignant",
                    diagnosis_3=sub,
                    patient_id=f"PM_{idx}",
                )
            )
    # 30 benign, 10 indeterminate
    for idx in range(30):
        pool.append(
            ISICRecord(
                isic_id=f"BEN_{idx}",
                copyright_license="CC-BY",
                image_url=f"https://example.com/ben_{idx}.jpg",
                image_type="dermoscopic",
                diagnosis_1="Benign",
                diagnosis_3="Nevus",
                patient_id=f"PB_{idx}",
            )
        )
    for idx in range(10):
        pool.append(
            ISICRecord(
                isic_id=f"IND_{idx}",
                copyright_license="CC-BY",
                image_url=f"https://example.com/ind_{idx}.jpg",
                image_type="dermoscopic",
                diagnosis_1="Indeterminate",
                diagnosis_3="Solar or actinic keratosis",
                patient_id=f"PI_{idx}",
            )
        )

    sampled = stratified_sample_records(
        pool,
        malignant_target=12,
        benign_target=14,
        indeterminate_target=4,
        seed=42,
    )
    mal_sampled = [r for r in sampled if r.is_malignant is True]
    ben_sampled = [r for r in sampled if r.is_malignant is False]
    ind_sampled = [r for r in sampled if r.is_malignant is None]
    assert len(mal_sampled) == 12
    assert len(ben_sampled) == 14
    assert len(ind_sampled) == 4
    assert {r.diagnosis_3 for r in mal_sampled} == set(subtypes)
