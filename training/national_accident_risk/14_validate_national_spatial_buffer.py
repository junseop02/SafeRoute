from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
import shapely

from shapely import STRtree


# ============================================================
# 1. 경로 및 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

RESULTS_DIR = PROCESSED_DIR / "results"

ROAD_PATH = (
    PROCESSED_DIR
    / "national_road_features.parquet"
)

DATASET_PATH = (
    PROCESSED_DIR
    / "national_accident_risk_dataset.parquet"
)

MATCH_PATH = (
    PROCESSED_DIR
    / "national_accident_link_matches.parquet"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "national_spatial_validation_summary.csv"
)

GRID_SIZES = [1000, 2000, 5000]

BUFFER_DISTANCE = 400.0
BATCH_SIZE = 1000

EXPECTED_ROWS = 1562356
EXPECTED_POSITIVE = 207391

SPLIT_NAMES = [
    "train",
    "validation",
    "test",
    "excluded"
]


# ============================================================
# 2. 메모리 확인
# ============================================================

def print_memory(title):

    memory = psutil.virtual_memory()

    print(f"\n===== {title} =====")

    print(
        f"Available RAM: "
        f"{memory.available / 1024**3:.2f} GiB"
    )

    print(f"RAM Usage: {memory.percent}%")


# ============================================================
# 3. 도로 형상 배치 읽기
# ============================================================

def road_batches():

    parquet = pq.ParquetFile(ROAD_PATH)

    metadata = parquet.schema_arrow.metadata or {}

    assert (
        metadata.get(b"geometry_crs")
        == b"EPSG:5179"
    )

    assert (
        parquet.metadata.num_rows
        == EXPECTED_ROWS
    )

    for batch in parquet.iter_batches(

        batch_size=BATCH_SIZE,

        columns=[
            "link_id",
            "geometry_wkb"
        ]

    ):

        yield batch.to_pandas()


# ============================================================
# 4. 도로 형상 변환
# ============================================================

def decode_geometry(values):

    geometries = shapely.from_wkb(values)

    assert not shapely.is_missing(
        geometries
    ).any()

    assert not shapely.is_empty(
        geometries
    ).any()

    assert shapely.is_valid(
        geometries
    ).all()

    return geometries


# ============================================================
# 5. 분할 및 라벨 무결성 검증
# ============================================================

def validate_integrity(
    frame,
    original,
    labels,
    matches
):

    assert len(frame) == EXPECTED_ROWS
    assert len(original) == EXPECTED_ROWS
    assert len(labels) == EXPECTED_ROWS

    assert not frame["link_id"].duplicated().any()
    assert not original["link_id"].duplicated().any()
    assert not labels["link_id"].duplicated().any()

    assert frame["split"].notna().all()

    assert frame["split"].isin(
        SPLIT_NAMES
    ).all()

    assert frame["target"].isin(
        [0, 1]
    ).all()

    assert (
        int(frame["target"].sum())
        == EXPECTED_POSITIVE
    )

    link_index = pd.Index(frame["link_id"])

    original = (
        original
        .set_index("link_id")
        .reindex(link_index)
    )

    labels = (
        labels
        .set_index("link_id")
        .reindex(link_index)
    )

    assert original["split"].notna().all()
    assert labels["target"].notna().all()

    # 원본 라벨과 일치하는지 확인
    label_violations = int(

        (
            frame["target"].to_numpy()
            != labels["target"].to_numpy()
        ).sum()

    )

    # 기존 공간 그룹과 격자 유지 여부
    group_change_violations = int(

        (
            frame["spatial_group"].to_numpy()
            != original["spatial_group"].to_numpy()
        ).sum()

    )

    grid_change_violations = int(

        (
            (
                frame["grid_x"].to_numpy()
                != original["grid_x"].to_numpy()
            )
            |
            (
                frame["grid_y"].to_numpy()
                != original["grid_y"].to_numpy()
            )
        ).sum()

    )

    original_split = original[
        "split"
    ].astype(str).to_numpy()

    final_split = frame[
        "split"
    ].astype(str).to_numpy()

    stored_original_split = frame[
        "original_split"
    ].astype(str).to_numpy()

    original_split_violations = int(

        (
            original_split
            != stored_original_split
        ).sum()

    )

    # Test 도로 추가 및 제거 검사
    test_removed = int(

        (
            (original_split == "test")
            &
            (final_split != "test")
        ).sum()

    )

    test_added = int(

        (
            (original_split != "test")
            &
            (final_split == "test")
        ).sum()

    )

    # 다른 데이터셋으로 이동했는지 확인
    split_change_violations = int(

        (
            (final_split != "excluded")
            &
            (final_split != original_split)
        ).sum()

    )

    # 제외 상태 및 제외 사유 검증
    excluded = final_split == "excluded"

    buffer_flag_violations = int(

        (
            excluded
            != frame["buffer_excluded"].to_numpy()
        ).sum()

    )

    expected_reasons = np.full(
        len(frame),
        "none",
        dtype=object
    )

    expected_reasons[
        excluded & (original_split == "train")
    ] = "within_400m_of_validation_or_test"

    expected_reasons[
        excluded & (original_split == "validation")
    ] = "within_400m_of_test"

    reason_violations = int(

        (
            expected_reasons
            != frame["exclusion_reason"].astype(str).to_numpy()
        ).sum()

    )

    # 실제 학습 및 평가에 사용하는 데이터
    retained = frame.loc[
        ~excluded
    ].copy()

    # 공간 그룹 중복 검사
    group_violations = int(

        (
            retained.groupby(
                "spatial_group",
                observed=True
            )["split"].nunique()
            > 1
        ).sum()

    )

    # 공간 격자 중복 검사
    grid_violations = int(

        (
            retained.groupby(
                ["grid_x", "grid_y"],
                observed=True
            )["split"].nunique()
            > 1
        ).sum()

    )

    # 사고다발지역 중복 검사
    hotspot_splits = matches.merge(

        frame[["link_id", "split"]],

        on="link_id",

        how="left",

        validate="many_to_one"

    )

    assert hotspot_splits["split"].notna().all()

    hotspot_splits = hotspot_splits.loc[
        hotspot_splits["split"] != "excluded"
    ]

    hotspot_violations = int(

        (
            hotspot_splits.groupby(
                "hotspot_id"
            )["split"].nunique()
            > 1
        ).sum()

    )

    results = {

        "label_violations":
            label_violations,

        "group_change_violations":
            group_change_violations,

        "grid_change_violations":
            grid_change_violations,

        "original_split_violations":
            original_split_violations,

        "test_removed":
            test_removed,

        "test_added":
            test_added,

        "split_change_violations":
            split_change_violations,

        "buffer_flag_violations":
            buffer_flag_violations,

        "reason_violations":
            reason_violations,

        "group_violations":
            group_violations,

        "grid_violations":
            grid_violations,

        "hotspot_violations":
            hotspot_violations

    }

    print("\n===== Integrity Validation =====")

    for name, value in results.items():

        print(f"{name}: {value}")

    assert all(
        value == 0
        for value in results.values()
    )

    return results


# ============================================================
# 6. 보존된 Validation 및 Test 형상 읽기
# ============================================================

def load_reference_geometries(
    link_index,
    split_codes
):

    validation_parts = []
    test_parts = []

    seen = np.zeros(
        len(link_index),
        dtype=bool
    )

    for batch in road_batches():

        positions = link_index.get_indexer(
            batch["link_id"]
        )

        assert (positions >= 0).all()

        assert (
            len(np.unique(positions))
            == len(positions)
        )

        assert not seen[positions].any()

        seen[positions] = True

        codes = split_codes[positions]

        validation_mask = codes == 1
        test_mask = codes == 2

        if validation_mask.any():

            validation_parts.append(

                decode_geometry(

                    batch.loc[
                        validation_mask,
                        "geometry_wkb"
                    ].to_numpy()

                )

            )

        if test_mask.any():

            test_parts.append(

                decode_geometry(

                    batch.loc[
                        test_mask,
                        "geometry_wkb"
                    ].to_numpy()

                )

            )

    assert seen.all()

    validation_geometries = np.concatenate(
        validation_parts
    )

    test_geometries = np.concatenate(
        test_parts
    )

    assert len(validation_geometries) == (
        split_codes == 1
    ).sum()

    assert len(test_geometries) == (
        split_codes == 2
    ).sum()

    print("\n===== Reference Geometries =====")

    print(
        "Retained Validation:",
        len(validation_geometries)
    )

    print(
        "Retained Test:",
        len(test_geometries)
    )

    return (
        validation_geometries,
        test_geometries
    )


# ============================================================
# 7. 독립적인 공간 거리 검사
# ============================================================

def count_spatial_violations(
    geometries,
    reference_tree
):

    if len(geometries) == 0:

        return 0, 0

    # 13번의 query_nearest와 다른 검색 방식
    pairs = reference_tree.query(

        geometries,

        predicate="dwithin",

        distance=BUFFER_DISTANCE

    )

    pair_count = pairs.shape[1]

    affected_links = (

        len(np.unique(pairs[0]))

        if pair_count > 0

        else 0

    )

    return pair_count, affected_links


# ============================================================
# 8. 격자별 공간 누수 검증
# ============================================================

def validate_grid(
    grid_size,
    labels,
    matches
):

    start_time = time.perf_counter()

    print(
        f"\n===== Grid Size: "
        f"{grid_size / 1000:g} km ====="
    )

    source_path = (

        PROCESSED_DIR

        / f"national_buffered_split_{grid_size}m.parquet"

    )

    original_path = (

        PROCESSED_DIR

        / f"national_optimized_split_{grid_size}m.parquet"

    )

    frame = pd.read_parquet(source_path)

    original = pd.read_parquet(

        original_path,

        columns=[
            "link_id",
            "spatial_group",
            "grid_x",
            "grid_y",
            "split"
        ]

    )

    integrity_results = validate_integrity(

        frame,

        original,

        labels,

        matches

    )

    del original

    gc.collect()

    link_index = pd.Index(frame["link_id"])

    split_codes = pd.Categorical(

        frame["split"],

        categories=SPLIT_NAMES

    ).codes.copy()

    assert (split_codes >= 0).all()

    (
        validation_geometries,
        test_geometries

    ) = load_reference_geometries(

        link_index,

        split_codes

    )

    validation_tree = STRtree(
        validation_geometries
    )

    test_tree = STRtree(
        test_geometries
    )

    print_memory("After Spatial Index Build")

    pair_counts = {

        "train_validation": 0,

        "train_test": 0,

        "validation_test": 0

    }

    affected_counts = {

        "train_validation": 0,

        "train_test": 0,

        "validation_test": 0

    }

    processed = 0

    print("\n===== Independent Spatial Validation =====")

    for batch in road_batches():

        positions = link_index.get_indexer(
            batch["link_id"]
        )

        assert (positions >= 0).all()

        codes = split_codes[positions]

        # Train과 Validation/Test 검사
        train_mask = codes == 0

        if train_mask.any():

            train_geometries = decode_geometry(

                batch.loc[
                    train_mask,
                    "geometry_wkb"
                ].to_numpy()

            )

            pairs, affected = count_spatial_violations(

                train_geometries,

                validation_tree

            )

            pair_counts[
                "train_validation"
            ] += pairs

            affected_counts[
                "train_validation"
            ] += affected

            pairs, affected = count_spatial_violations(

                train_geometries,

                test_tree

            )

            pair_counts[
                "train_test"
            ] += pairs

            affected_counts[
                "train_test"
            ] += affected

            del train_geometries

        # Validation과 Test 검사
        validation_mask = codes == 1

        if validation_mask.any():

            geometries = decode_geometry(

                batch.loc[
                    validation_mask,
                    "geometry_wkb"
                ].to_numpy()

            )

            pairs, affected = count_spatial_violations(

                geometries,

                test_tree

            )

            pair_counts[
                "validation_test"
            ] += pairs

            affected_counts[
                "validation_test"
            ] += affected

            del geometries

        processed += len(batch)

        if (

            processed % 200000 == 0

            or processed == EXPECTED_ROWS

        ):

            print(

                f"Processed: "
                f"{processed:,} / "
                f"{EXPECTED_ROWS:,}"

                f" | Violations: "
                f"{sum(pair_counts.values()):,}"

                f" | Elapsed: "
                f"{time.perf_counter() - start_time:.1f}s"

            )

    assert processed == EXPECTED_ROWS

    print("\n===== Spatial Distance Results =====")

    for name in pair_counts:

        print(

            f"{name}: "

            f"Pairs={pair_counts[name]:,}"

            f" | Affected Links="
            f"{affected_counts[name]:,}"

        )

    total_violations = sum(
        pair_counts.values()
    )

    spatial_passed = total_violations == 0

    # 분할별 데이터 통계
    results = []

    for split_idx, split_name in enumerate(
        SPLIT_NAMES
    ):

        subset = frame.loc[
            split_codes == split_idx
        ]

        total = len(subset)

        positive = int(
            subset["target"].sum()
        )

        negative = total - positive

        if split_name != "excluded":

            assert positive > 0
            assert negative > 0

        results.append({

            "grid_size_m": grid_size,

            "split": split_name,

            "total_links": total,

            "positive_links": positive,

            "negative_links": negative,

            "positive_ratio":
                positive / total if total else 0.0

        })

    distribution = pd.DataFrame(results)

    print("\n===== Verified Distribution =====")

    print(
        distribution.to_string(index=False)
    )

    retained_count = int(
        (split_codes != 3).sum()
    )

    result = {

        "grid_size_m": grid_size,

        "buffer_distance_m":
            BUFFER_DISTANCE,

        "retained_links":
            retained_count,

        "excluded_links":
            EXPECTED_ROWS - retained_count,

        "retention_ratio":
            retained_count / EXPECTED_ROWS,

        **integrity_results,

        "train_validation_pairs":
            pair_counts["train_validation"],

        "train_test_pairs":
            pair_counts["train_test"],

        "validation_test_pairs":
            pair_counts["validation_test"],

        "train_validation_affected":
            affected_counts["train_validation"],

        "train_test_affected":
            affected_counts["train_test"],

        "validation_test_affected":
            affected_counts["validation_test"],

        "total_spatial_violations":
            total_violations,

        "validation_passed":
            spatial_passed

    }

    print(

        "\nValidation Status:",

        "PASS" if spatial_passed else "FAIL"

    )

    print(

        f"Grid Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f}s"

    )

    del validation_tree
    del test_tree
    del validation_geometries
    del test_geometries
    del frame

    gc.collect()

    return result


# ============================================================
# 9. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Spatial Buffer Validation =====")

    print("Grid Sizes:", GRID_SIZES)
    print("Buffer Distance:", BUFFER_DISTANCE)
    print("Batch Size:", BATCH_SIZE)
    print("Validation Method: STRtree dwithin")

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    if SUMMARY_PATH.exists():

        raise FileExistsError(SUMMARY_PATH)

    labels = pd.read_parquet(

        DATASET_PATH,

        columns=[
            "link_id",
            "target"
        ]

    )

    matches = pd.read_parquet(

        MATCH_PATH,

        columns=[
            "link_id",
            "hotspot_id"
        ]

    )

    assert len(labels) == EXPECTED_ROWS

    assert (
        int(labels["target"].sum())
        == EXPECTED_POSITIVE
    )

    assert len(matches) == 485562

    assert (
        matches["hotspot_id"].nunique()
        == 12752
    )

    assert not matches.duplicated(
        ["link_id", "hotspot_id"]
    ).any()

    results = []

    for grid_size in GRID_SIZES:

        result = validate_grid(

            grid_size,

            labels,

            matches

        )

        results.append(result)

        gc.collect()

        print_memory(

            f"After {grid_size} m Validation"

        )

    comparison = pd.DataFrame(results)

    print("\n===== Final Validation Comparison =====")

    print(
        comparison.to_string(index=False)
    )

    comparison.to_csv(

        SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    print("\n===== Saved Summary =====")

    print("Path:", SUMMARY_PATH)

    all_passed = bool(

        comparison[
            "validation_passed"
        ].all()

    )

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(

        f"Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f}s"

    )

    print(

        "All Spatial Validations:",

        "PASS" if all_passed else "FAIL"

    )

    print("Original Dataset Modified: No")
    print("Grid Size Selected: No")
    print("Final Test Set Frozen: No")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")

    assert all_passed, (
        "Spatial buffer validation failed."
    )


if __name__ == "__main__":
    main()