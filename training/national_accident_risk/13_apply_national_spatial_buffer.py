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

GRID_SIZES = [1000, 2000, 5000]

BUFFER_DISTANCE = 400.0
BATCH_SIZE = 5000

EXPECTED_ROWS = 1562356
EXPECTED_POSITIVE = 207391

SPLIT_NAMES = [
    "train",
    "validation",
    "test"
]

SUMMARY_PATH = (
    RESULTS_DIR
    / "national_spatial_buffer_summary.csv"
)


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
# 3. 도로 원본 데이터 배치 읽기
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
# 4. 도로 ID 위치 확인
# ============================================================

def get_positions(batch, link_index):

    positions = link_index.get_indexer(
        batch["link_id"]
    )

    assert (positions >= 0).all()

    return positions


# ============================================================
# 5. 도로 형상 변환
# ============================================================

def decode_geometry(wkb_values):

    geometries = shapely.from_wkb(
        wkb_values
    )

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
# 6. Validation 및 Test 도로 형상 확보
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

    processed = 0

    for batch in road_batches():

        positions = get_positions(
            batch,
            link_index
        )

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

        processed += len(batch)

        if processed % 200000 == 0:

            print(
                f"Reference Scan: "
                f"{processed:,} / "
                f"{EXPECTED_ROWS:,}"
            )

    assert processed == EXPECTED_ROWS
    assert seen.all()

    validation_geometries = np.concatenate(
        validation_parts
    )

    test_geometries = np.concatenate(
        test_parts
    )

    assert (
        len(validation_geometries)
        == int((split_codes == 1).sum())
    )

    assert (
        len(test_geometries)
        == int((split_codes == 2).sum())
    )

    print("\n===== Reference Geometries =====")

    print(
        "Validation:",
        len(validation_geometries)
    )

    print(
        "Test:",
        len(test_geometries)
    )

    return (
        validation_geometries,
        test_geometries
    )


# ============================================================
# 7. 완충 거리 이내 도로 검색
# ============================================================

def find_nearby_roads(
    geometries,
    reference_tree
):

    excluded = np.zeros(
        len(geometries),
        dtype=bool
    )

    if len(geometries) == 0:
        return excluded

    # 실제 도로 형상 간 최단거리 검색
    indices, distances = (
        reference_tree.query_nearest(

            geometries,

            max_distance=BUFFER_DISTANCE,

            return_distance=True,

            all_matches=False

        )
    )

    if indices.size > 0:

        query_indices = indices[0]

        assert np.isfinite(
            distances
        ).all()

        assert (
            distances <= BUFFER_DISTANCE
        ).all()

        excluded[query_indices] = True

    return excluded


# ============================================================
# 8. 격자별 공간 완충 구역 적용
# ============================================================

def apply_buffer(grid_size):

    start_time = time.perf_counter()

    print(
        f"\n===== Grid Size: "
        f"{grid_size / 1000:g} km ====="
    )

    source_path = (
        PROCESSED_DIR
        / f"national_optimized_split_{grid_size}m.parquet"
    )

    output_path = (
        PROCESSED_DIR
        / f"national_buffered_split_{grid_size}m.parquet"
    )

    frame = pd.read_parquet(
        source_path
    )

    assert len(frame) == EXPECTED_ROWS

    assert (
        int(frame["target"].sum())
        == EXPECTED_POSITIVE
    )

    assert not frame[
        "link_id"
    ].duplicated().any()

    link_index = pd.Index(
        frame["link_id"]
    )

    split_codes = pd.Categorical(

        frame["split"],

        categories=SPLIT_NAMES

    ).codes.copy()

    assert (split_codes >= 0).all()

    # --------------------------------------------------------
    # Validation 및 Test 도로 형상 확보
    # --------------------------------------------------------

    (
        validation_geometries,
        test_geometries
    ) = load_reference_geometries(

        link_index,

        split_codes

    )

    print_memory(
        "After Reference Geometry Load"
    )

    # Validation 제외 기준: Test
    test_tree = STRtree(
        test_geometries
    )

    # Train 제외 기준: 기존 Validation + Test
    holdout_geometries = np.concatenate([

        validation_geometries,

        test_geometries

    ])

    holdout_tree = STRtree(
        holdout_geometries
    )

    print_memory(
        "After Spatial Index Build"
    )

    # 제외 여부 및 사유 저장
    excluded = np.zeros(
        len(frame),
        dtype=bool
    )

    reason_codes = np.zeros(
        len(frame),
        dtype=np.int8
    )

    processed = 0

    print("\n===== Apply Spatial Buffer =====")

    # --------------------------------------------------------
    # 전국 도로 배치별 검사
    # --------------------------------------------------------

    for batch in road_batches():

        positions = get_positions(
            batch,
            link_index
        )

        codes = split_codes[positions]

        # Train 도로 검사
        train_mask = codes == 0

        if train_mask.any():

            train_positions = positions[
                train_mask
            ]

            geometries = decode_geometry(

                batch.loc[
                    train_mask,
                    "geometry_wkb"
                ].to_numpy()

            )

            near_holdout = find_nearby_roads(

                geometries,

                holdout_tree

            )

            removed_positions = (
                train_positions[near_holdout]
            )

            excluded[removed_positions] = True

            reason_codes[removed_positions] = 1

            del geometries

        # Validation 도로 검사
        validation_mask = codes == 1

        if validation_mask.any():

            validation_positions = positions[
                validation_mask
            ]

            geometries = decode_geometry(

                batch.loc[
                    validation_mask,
                    "geometry_wkb"
                ].to_numpy()

            )

            near_test = find_nearby_roads(

                geometries,

                test_tree

            )

            removed_positions = (
                validation_positions[near_test]
            )

            excluded[removed_positions] = True

            reason_codes[removed_positions] = 2

            del geometries

        # Test는 변경하지 않음

        processed += len(batch)

        if (
            processed % 100000 == 0
            or processed == EXPECTED_ROWS
        ):

            available_ram = (
                psutil.virtual_memory().available
                / 1024**3
            )

            print(

                f"Processed: "
                f"{processed:,} / "
                f"{EXPECTED_ROWS:,}"

                f" | Excluded: "
                f"{excluded.sum():,}"

                f" | Elapsed: "
                f"{time.perf_counter() - start_time:.1f}s"

                f" | RAM: "
                f"{available_ram:.2f} GiB"

            )

    assert processed == EXPECTED_ROWS

    # Test 제외 여부 검사
    assert not excluded[
        split_codes == 2
    ].any()

    assert np.array_equal(

        excluded,

        reason_codes > 0

    )

    # 공간 인덱스 메모리 해제
    del test_tree
    del holdout_tree
    del validation_geometries
    del test_geometries
    del holdout_geometries

    gc.collect()

    # ========================================================
    # 9. 최종 분할 상태 저장
    # ========================================================

    final_codes = split_codes.copy()

    final_codes[excluded] = 3

    frame["original_split"] = frame["split"]

    frame["split"] = (
        pd.Categorical.from_codes(

            final_codes,

            categories=[
                "train",
                "validation",
                "test",
                "excluded"
            ]

        )
    )

    frame["buffer_excluded"] = excluded

    frame["exclusion_reason"] = (
        pd.Categorical.from_codes(

            reason_codes,

            categories=[
                "none",
                "within_400m_of_validation_or_test",
                "within_400m_of_test"
            ]

        )
    )

    assert len(frame) == EXPECTED_ROWS

    assert (
        int(frame["target"].sum())
        == EXPECTED_POSITIVE
    )

    assert frame["split"].notna().all()

    assert not frame[
        "link_id"
    ].duplicated().any()

    # ========================================================
    # 10. 분할별 보존율 및 클래스 분포
    # ========================================================

    results = []

    for split_idx, split_name in enumerate(
        SPLIT_NAMES
    ):

        original_mask = (
            split_codes == split_idx
        )

        retained_mask = (
            final_codes == split_idx
        )

        original = frame.loc[
            original_mask
        ]

        retained = frame.loc[
            retained_mask
        ]

        original_total = len(original)
        retained_total = len(retained)

        original_positive = int(
            original["target"].sum()
        )

        retained_positive = int(
            retained["target"].sum()
        )

        original_negative = (
            original_total
            - original_positive
        )

        retained_negative = (
            retained_total
            - retained_positive
        )

        assert retained_positive > 0
        assert retained_negative > 0

        group_summary = retained.groupby(
            "spatial_group"
        )["target"].sum()

        positive_groups = int(
            (group_summary > 0).sum()
        )

        max_positive_group = int(
            group_summary.max()
        )

        results.append({

            "grid_size_m": grid_size,

            "buffer_distance_m":
                BUFFER_DISTANCE,

            "split": split_name,

            "original_links":
                original_total,

            "retained_links":
                retained_total,

            "excluded_links":
                original_total
                - retained_total,

            "original_positive":
                original_positive,

            "retained_positive":
                retained_positive,

            "retained_negative":
                retained_negative,

            "excluded_positive":
                original_positive
                - retained_positive,

            "excluded_negative":
                original_negative
                - retained_negative,

            "retention_ratio":
                retained_total
                / original_total,

            "positive_retention_ratio":
                retained_positive
                / original_positive,

            "negative_retention_ratio":
                retained_negative
                / original_negative,

            "positive_ratio":
                retained_positive
                / retained_total,

            "spatial_groups":
                len(group_summary),

            "positive_groups":
                positive_groups,

            "max_positive_group":
                max_positive_group,

            "max_positive_group_share":
                max_positive_group
                / retained_positive

        })

    result_frame = pd.DataFrame(
        results
    )

    # ========================================================
    # 11. 전체 데이터 수 검증
    # ========================================================

    retained_total = int(
        (~excluded).sum()
    )

    excluded_total = int(
        excluded.sum()
    )

    assert (
        retained_total + excluded_total
        == EXPECTED_ROWS
    )

    assert (

        result_frame["retained_links"].sum()

        == retained_total

    )

    assert (

        result_frame["excluded_links"].sum()

        == excluded_total

    )

    print("\n===== Buffer Result =====")

    print("Original Links:", EXPECTED_ROWS)

    print("Retained Links:", retained_total)

    print("Excluded Links:", excluded_total)

    print(

        f"Overall Retention: "

        f"{retained_total / EXPECTED_ROWS * 100:.4f}%"

    )

    print("\n===== Split Distribution =====")

    print(
        result_frame.to_string(
            index=False
        )
    )

    print("\n===== Exclusion Reasons =====")

    print(

        frame["exclusion_reason"]
        .value_counts()
        .to_string()

    )

    # ========================================================
    # 12. 새로운 파일로 저장
    # ========================================================

    temp_path = output_path.with_suffix(
        ".parquet.tmp"
    )

    frame.to_parquet(

        temp_path,

        index=False

    )

    temp_path.replace(
        output_path
    )

    print("\n===== Saved Output =====")

    print("Path:", output_path)

    print(
        f"File Size: "
        f"{output_path.stat().st_size / 1024**2:.2f} MiB"
    )

    print(
        f"Grid Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f}s"
    )

    return result_frame


# ============================================================
# 13. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Spatial Buffer =====")

    print("Grid Sizes:", GRID_SIZES)

    print(
        "Buffer Distance:",
        BUFFER_DISTANCE
    )

    print("Batch Size:", BATCH_SIZE)

    print("Road Source:", ROAD_PATH)

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    # 기존 파일 덮어쓰기 방지
    output_paths = [

        PROCESSED_DIR
        / f"national_buffered_split_{size}m.parquet"

        for size in GRID_SIZES

    ]

    output_paths.append(
        SUMMARY_PATH
    )

    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    results = []

    for grid_size in GRID_SIZES:

        result = apply_buffer(
            grid_size
        )

        results.append(result)

        gc.collect()

        print_memory(

            f"After {grid_size} m Buffer"

        )

    comparison = pd.concat(

        results,

        ignore_index=True

    )

    print("\n===== Final Buffer Comparison =====")

    print(

        comparison.to_string(
            index=False
        )

    )

    comparison.to_csv(

        SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    print("\n===== Saved Summary =====")

    print("Path:", SUMMARY_PATH)

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(

        f"Elapsed Time: "

        f"{time.perf_counter() - start_time:.2f} seconds"

    )

    print("Original Split Files: Not Modified")
    print("Test Links Removed: 0")
    print("Grid Size Selected: No")
    print("Independent Buffer Validation: Not Started")
    print("Final Test Set Frozen: No")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()