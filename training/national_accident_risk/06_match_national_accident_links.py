from pathlib import Path
from collections import Counter
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from shapely import STRtree


# ============================================================
# 1. 경로 및 환경 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

ROAD_PATH = (
    PROCESSED_DIR
    / "national_road_features.parquet"
)

POLYGON_PATH = (
    PROCESSED_DIR
    / "national_accident_polygons_crs_candidate.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "national_accident_link_matches.parquet"
)

TEMP_PATH = OUTPUT_PATH.with_suffix(".tmp")

BATCH_SIZE = 5000
TARGET_CRS = "EPSG:5179"

THRESHOLDS = [
    0.10,
    0.25,
    0.50,
    0.75,
    0.90
]


# ============================================================
# 2. 저장 스키마
# ============================================================

SCHEMA = pa.schema([
    ("link_id", pa.string()),
    ("hotspot_id", pa.int64()),
    ("accident_year", pa.int32()),
    ("accident_type", pa.string()),
    ("province", pa.string()),
    ("geometry_length_m", pa.float64()),
    ("overlap_length_m", pa.float64()),
    ("overlap_ratio", pa.float64())
])


# ============================================================
# 3. 메모리 확인
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
# 4. 사고다발지역 공간 인덱스 생성
# ============================================================

def prepare_hotspots():

    hotspots = gpd.read_parquet(
        POLYGON_PATH
    ).reset_index(drop=True)

    print("\n===== Hotspot Information =====")

    print("Hotspots:", len(hotspots))
    print("CRS:", hotspots.crs)

    if hotspots.crs != TARGET_CRS:
        raise RuntimeError("Hotspot CRS mismatch.")

    assert len(hotspots) == 12780

    assert not hotspots[
        "hotspot_id"
    ].duplicated().any()

    assert hotspots.geometry.notna().all()
    assert not hotspots.geometry.is_empty.any()
    assert hotspots.geometry.is_valid.all()

    geometries = hotspots.geometry.to_numpy()

    tree = STRtree(geometries)

    print("Spatial Index Size:", len(geometries))

    return hotspots, geometries, tree


# ============================================================
# 5. 배치별 공간 매칭
# ============================================================

def process_batch(
    frame,
    hotspots,
    hotspot_geometries,
    tree,
    writer,
    stats,
    matched_hotspots,
    hotspot_link_counts,
    province_counts,
    year_counts,
    type_counts
):

    stats["processed_links"] += len(frame)

    # WKB를 Geometry로 복원
    road_geometries = shapely.from_wkb(
        frame["geometry_wkb"].to_numpy()
    )

    road_lengths = frame[
        "geometry_length_m"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(road_lengths).all():
        raise RuntimeError("Non-finite road lengths.")

    if (road_lengths <= 0).any():
        raise RuntimeError("Invalid road lengths.")

    # 도로와 사고다발지역 폴리곤 교차 검색
    pairs = tree.query(
        road_geometries,
        predicate="intersects"
    )

    stats["intersect_pairs"] += pairs.shape[1]

    if pairs.shape[1] == 0:
        return

    road_indexes = pairs[0]
    hotspot_indexes = pairs[1]

    # 실제 중첩 Geometry 계산
    intersections = shapely.intersection(
        road_geometries[road_indexes],
        hotspot_geometries[hotspot_indexes]
    )

    overlap_lengths = shapely.length(
        intersections
    )

    if not np.isfinite(overlap_lengths).all():
        raise RuntimeError(
            "Non-finite overlap lengths."
        )

    # 경계점 접촉만 있는 경우 제외
    positive_mask = overlap_lengths > 0

    stats["boundary_only_pairs"] += int(
        (~positive_mask).sum()
    )

    road_indexes = road_indexes[
        positive_mask
    ]

    hotspot_indexes = hotspot_indexes[
        positive_mask
    ]

    overlap_lengths = overlap_lengths[
        positive_mask
    ]

    if len(road_indexes) == 0:
        return

    # 전체 도로 길이 대비 중첩 비율
    overlap_ratios = (
        overlap_lengths
        / road_lengths[road_indexes]
    )

    # 큰 수치 오류는 중단
    if (overlap_ratios > 1.000001).any():
        raise RuntimeError(
            "Overlap ratio exceeds 1."
        )

    # 미세한 부동소수점 오차만 보정
    stats["ratio_rounding_corrections"] += int(
        (overlap_ratios > 1).sum()
    )

    overlap_ratios = np.clip(
        overlap_ratios,
        0,
        1
    )

    stats["positive_overlap_pairs"] += len(
        road_indexes
    )

    # --------------------------------------------------------
    # 매칭 결과 생성
    # --------------------------------------------------------

    selected_hotspots = hotspots.iloc[
        hotspot_indexes
    ]

    result = pd.DataFrame({
        "link_id": (
            frame["link_id"]
            .to_numpy()[road_indexes]
        ),

        "hotspot_id": (
            selected_hotspots["hotspot_id"]
            .to_numpy(dtype=np.int64)
        ),

        "accident_year": (
            selected_hotspots["사고연도"]
            .to_numpy(dtype=np.int32)
        ),

        "accident_type": (
            selected_hotspots["사고유형구분"]
            .to_numpy()
        ),

        "province": (
            selected_hotspots["province"]
            .to_numpy()
        ),

        "geometry_length_m": (
            road_lengths[road_indexes]
        ),

        "overlap_length_m": overlap_lengths,

        "overlap_ratio": overlap_ratios
    })

    # --------------------------------------------------------
    # 매칭 통계 계산
    # --------------------------------------------------------

    unique_road_indexes = np.unique(
        road_indexes
    )

    stats["matched_links"] += len(
        unique_road_indexes
    )

    matched_hotspots.update(
        result["hotspot_id"].tolist()
    )

    hotspot_link_counts.update(
        result["hotspot_id"].tolist()
    )

    # 링크별 매칭된 사고다발지역 수
    counts = np.bincount(
        road_indexes,
        minlength=len(frame)
    )

    stats["multiple_hotspot_links"] += int(
        (counts > 1).sum()
    )

    stats["max_hotspots_per_link"] = max(
        stats["max_hotspots_per_link"],
        int(counts.max())
    )

    # 링크별 최대 개별 폴리곤 중첩 비율
    max_ratios = np.zeros(
        len(frame),
        dtype=np.float64
    )

    np.maximum.at(
        max_ratios,
        road_indexes,
        overlap_ratios
    )

    for threshold in THRESHOLDS:

        key = (
            f"matched_links_"
            f"{int(threshold * 100)}pct"
        )

        stats[key] += int(
            (max_ratios >= threshold).sum()
        )

    # 시도별 고유 링크 매칭 수
    province_counts.update(
        result[
            ["province", "link_id"]
        ]
        .drop_duplicates()["province"]
        .tolist()
    )

    # 연도별 고유 링크 매칭 수
    year_counts.update(
        result[
            ["accident_year", "link_id"]
        ]
        .drop_duplicates()["accident_year"]
        .tolist()
    )

    # 유형별 고유 링크 매칭 수
    type_counts.update(
        result[
            ["accident_type", "link_id"]
        ]
        .drop_duplicates()["accident_type"]
        .tolist()
    )

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    table = pa.Table.from_pandas(
        result,
        schema=SCHEMA,
        preserve_index=False
    )

    writer.write_table(table)

    stats["saved_pairs"] += len(result)


# ============================================================
# 6. 전국 도로 공간 매칭
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Accident Link Matching =====")
    print("Road Source:", ROAD_PATH)
    print("Hotspot Source:", POLYGON_PATH)
    print("Batch Size:", BATCH_SIZE)

    print_memory("Initial Memory")

    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)

    if TEMP_PATH.exists():
        raise FileExistsError(TEMP_PATH)

    (
        hotspots,
        hotspot_geometries,
        tree
    ) = prepare_hotspots()

    print_memory("After Spatial Index")

    road_file = pq.ParquetFile(
        ROAD_PATH
    )

    expected_links = (
        road_file.metadata.num_rows
    )

    metadata = (
        road_file.schema_arrow.metadata or {}
    )

    road_crs = metadata.get(
        b"geometry_crs",
        b""
    ).decode("utf-8")

    geometry_encoding = metadata.get(
        b"geometry_encoding",
        b""
    ).decode("utf-8")

    print("\n===== Road Information =====")
    print("Road Links:", expected_links)
    print("Road CRS:", road_crs)
    print("Geometry Encoding:", geometry_encoding)

    assert expected_links == 1562356
    assert road_crs == TARGET_CRS
    assert geometry_encoding == "WKB"

    stats = Counter()

    matched_hotspots = set()

    hotspot_link_counts = Counter()
    province_counts = Counter()
    year_counts = Counter()
    type_counts = Counter()

    columns = [
        "link_id",
        "geometry_length_m",
        "geometry_wkb"
    ]

    print("\n===== Spatial Matching =====")

    with pq.ParquetWriter(
        TEMP_PATH,
        SCHEMA,
        compression="zstd"
    ) as writer:

        for batch in road_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=columns
        ):

            frame = batch.to_pandas()

            process_batch(
                frame,
                hotspots,
                hotspot_geometries,
                tree,
                writer,
                stats,
                matched_hotspots,
                hotspot_link_counts,
                province_counts,
                year_counts,
                type_counts
            )

            if (
                stats["processed_links"] % 100000
                < len(frame)
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"Processed: "
                    f"{stats['processed_links']:,}"
                    f" / {expected_links:,} | "
                    f"Matched Links: "
                    f"{stats['matched_links']:,} | "
                    f"Pairs: "
                    f"{stats['saved_pairs']:,} | "
                    f"Elapsed: {elapsed:.1f}s",
                    flush=True
                )

    # ========================================================
    # 7. 매칭 결과 검증
    # ========================================================

    print("\n===== Matching Summary =====")

    for key, value in sorted(stats.items()):
        print(f"{key}: {value:,}")

    matched_links = stats["matched_links"]

    unmatched_links = (
        expected_links - matched_links
    )

    print("\nTotal Road Links:", expected_links)
    print("Matched Road Links:", matched_links)
    print("Unmatched Road Links:", unmatched_links)

    print(
        f"Matched Link Ratio: "
        f"{matched_links / expected_links * 100:.4f}%"
    )

    print("\n===== Hotspot Coverage =====")

    print("Total Hotspots:", len(hotspots))

    print(
        "Matched Hotspots:",
        len(matched_hotspots)
    )

    print(
        "Unmatched Hotspots:",
        len(hotspots) - len(matched_hotspots)
    )

    print(
        f"Hotspot Coverage: "
        f"{len(matched_hotspots) / len(hotspots) * 100:.4f}%"
    )

    # 매칭되지 않은 사고다발지역 확인
    unmatched_hotspots = hotspots.loc[
        ~hotspots["hotspot_id"].isin(
            matched_hotspots
        )
    ]

    if not unmatched_hotspots.empty:

        print("\nUnmatched Hotspot Samples:")

        print(
            unmatched_hotspots[
                [
                    "hotspot_id",
                    "사고연도",
                    "사고유형구분",
                    "province",
                    "사고지역위치명"
                ]
            ].head(20).to_string(index=False)
        )

    # ========================================================
    # 8. 중첩 비율 기준별 비교
    # ========================================================

    print("\n===== Overlap Threshold Comparison =====")

    print(
        f"Any Positive Overlap: "
        f"{matched_links:,} "
        f"({matched_links / expected_links * 100:.4f}%)"
    )

    for threshold in THRESHOLDS:

        key = (
            f"matched_links_"
            f"{int(threshold * 100)}pct"
        )

        count = stats[key]

        print(
            f"Overlap >= {threshold * 100:.0f}%: "
            f"{count:,} "
            f"({count / expected_links * 100:.4f}%)"
        )

    # ========================================================
    # 9. 중복 매칭 통계
    # ========================================================

    print("\n===== Multiple Hotspot Matching =====")

    print(
        "Links Matched to Multiple Hotspots:",
        stats["multiple_hotspot_links"]
    )

    print(
        "Maximum Hotspots per Link:",
        stats["max_hotspots_per_link"]
    )

    if matched_links:

        print(
            "Mean Hotspots per Matched Link:",
            stats["positive_overlap_pairs"]
            / matched_links
        )

    hotspot_counts = pd.Series(
        [
            hotspot_link_counts.get(hotspot_id, 0)
            for hotspot_id in hotspots["hotspot_id"]
        ],
        dtype="int64"
    )

    print("\nLinks per Hotspot Statistics:")

    print(
        hotspot_counts.describe().to_string()
    )

    # ========================================================
    # 10. 시도별, 연도별, 유형별 매칭 통계
    # ========================================================

    print("\n===== Matched Links by Hotspot Province =====")

    for key, value in province_counts.most_common():
        print(key, value)

    print("\n===== Matched Links by Accident Year =====")

    for key, value in sorted(year_counts.items()):
        print(key, value)

    print("\n===== Matched Links by Accident Type =====")

    for key, value in type_counts.most_common():
        print(key, value)

    # ========================================================
    # 11. 최종 파일 검증
    # ========================================================

    assert (
        stats["processed_links"]
        == expected_links
    )

    assert (
        stats["positive_overlap_pairs"]
        + stats["boundary_only_pairs"]
        == stats["intersect_pairs"]
    )

    assert (
        stats["saved_pairs"]
        == stats["positive_overlap_pairs"]
    )

    assert matched_links <= expected_links

    if matched_links == 0:
        raise RuntimeError("No matched links found.")

    previous_count = matched_links

    for threshold in THRESHOLDS:

        key = (
            f"matched_links_"
            f"{int(threshold * 100)}pct"
        )

        assert stats[key] <= previous_count

        previous_count = stats[key]

    parquet = pq.ParquetFile(
        TEMP_PATH
    )

    assert (
        parquet.metadata.num_rows
        == stats["saved_pairs"]
    )

    del parquet
    del road_file

    TEMP_PATH.replace(OUTPUT_PATH)

    print("\n===== Saved Output =====")
    print("Path:", OUTPUT_PATH)
    print("Rows:", stats["saved_pairs"])
    print("Columns:", len(SCHEMA))

    print(
        f"File Size: "
        f"{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB"
    )

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(
        f"Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    print("Original Files: Not Modified")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()