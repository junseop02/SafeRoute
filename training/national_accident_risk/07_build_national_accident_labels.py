from pathlib import Path
from collections import Counter
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


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

RESULTS_DIR = PROCESSED_DIR / "results"

ROAD_PATH = (
    PROCESSED_DIR / "national_road_features.parquet"
)

MATCH_PATH = (
    PROCESSED_DIR
    / "national_accident_link_matches.parquet"
)

POLYGON_PATH = (
    PROCESSED_DIR
    / "national_accident_polygons_crs_candidate.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "national_accident_risk_dataset.parquet"
)

TEMP_PATH = OUTPUT_PATH.with_suffix(".tmp")

UNMATCHED_PATH = (
    RESULTS_DIR / "unmatched_hotspots.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR / "national_label_summary.csv"
)

BATCH_SIZE = 50000

THRESHOLDS = [10, 25, 50, 75, 90]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 학습 피처 정의
# ============================================================

BASE_FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_branch"
]

INTERSECTION_FEATURES = [
    "n_intersect_200m",
    "intersection_nodes_200m"
]

FEATURES = BASE_FEATURES + INTERSECTION_FEATURES

TARGET_COLUMNS = [
    "target",
    "target_10",
    "target_25",
    "target_50",
    "target_75",
    "target_90"
]

ROAD_COLUMNS = [
    "link_id",
    "f_node",
    "t_node",
    "lanes",
    "max_spd",
    "road_rank",
    "road_type",
    "length",
    "geometry_length_m",
    "n_branch",
    "n_intersect_200m",
    "intersection_nodes_200m",
    "center_x",
    "center_y"
]


# ============================================================
# 3. 저장 스키마
# ============================================================

SCHEMA = pa.schema(
    [
        ("link_id", pa.string()),
        ("f_node", pa.string()),
        ("t_node", pa.string()),
        ("lanes", pa.float64()),
        ("max_spd", pa.float64()),
        ("road_rank", pa.float64()),
        ("road_type", pa.string()),
        ("length", pa.float64()),
        ("geometry_length_m", pa.float64()),
        ("n_branch", pa.int32()),
        ("n_intersect_200m", pa.int32()),
        ("intersection_nodes_200m", pa.int32()),
        ("center_x", pa.float64()),
        ("center_y", pa.float64()),
        ("hotspot_count", pa.int32()),
        ("max_overlap_length_m", pa.float64()),
        ("max_overlap_ratio", pa.float64()),
        ("first_accident_year", pa.int32()),
        ("last_accident_year", pa.int32()),
        ("target", pa.int8()),
        ("target_10", pa.int8()),
        ("target_25", pa.int8()),
        ("target_50", pa.int8()),
        ("target_75", pa.int8()),
        ("target_90", pa.int8())
    ],
    metadata={
        b"coordinate_crs": b"EPSG:5179",
        b"target_meaning": (
            b"historical_hotspot_membership"
        ),
        b"target_definition": (
            b"positive_polygon_overlap_length"
        )
    }
)


# ============================================================
# 4. 메모리 확인
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
# 5. 사고다발지역 매칭 정보 집계
# ============================================================

def prepare_match_summary():

    matches = pd.read_parquet(MATCH_PATH)

    print("\n===== Match Information =====")
    print("Match Rows:", len(matches))

    required_columns = [
        "link_id",
        "hotspot_id",
        "accident_year",
        "geometry_length_m",
        "overlap_length_m",
        "overlap_ratio"
    ]

    assert matches[required_columns].notna().all().all()

    duplicate_pairs = matches.duplicated(
        subset=["link_id", "hotspot_id"]
    ).sum()

    print("Duplicate Pairs:", duplicate_pairs)

    assert duplicate_pairs == 0

    assert (
        matches["overlap_length_m"] > 0
    ).all()

    assert matches["overlap_ratio"].between(
        0, 1
    ).all()

    assert np.isfinite(
        matches[
            [
                "geometry_length_m",
                "overlap_length_m",
                "overlap_ratio"
            ]
        ].to_numpy()
    ).all()

    # 도로별 사고다발지역 매칭 정보 집계
    summary = matches.groupby(
        "link_id",
        sort=False
    ).agg(
        hotspot_count=("hotspot_id", "nunique"),

        max_overlap_length_m=(
            "overlap_length_m", "max"
        ),

        max_overlap_ratio=(
            "overlap_ratio", "max"
        ),

        first_accident_year=(
            "accident_year", "min"
        ),

        last_accident_year=(
            "accident_year", "max"
        )
    )

    print("\n===== Link Match Summary =====")
    print("Matched Links:", len(summary))

    print(
        "Matched Hotspots:",
        matches["hotspot_id"].nunique()
    )

    print(
        "Multiple Hotspot Links:",
        (summary["hotspot_count"] > 1).sum()
    )

    print(
        "Maximum Hotspots per Link:",
        summary["hotspot_count"].max()
    )

    print("\nHotspot Count Statistics:")

    print(
        summary["hotspot_count"]
        .describe()
        .to_string()
    )

    matched_hotspot_ids = set(
        matches["hotspot_id"]
    )

    expected_pairs = len(matches)

    assert len(summary) == 207391
    assert expected_pairs == 485562
    assert len(matched_hotspot_ids) == 12752

    return (
        summary,
        matched_hotspot_ids,
        expected_pairs
    )


# ============================================================
# 6. 미매칭 사고다발지역 확인
# ============================================================

def prepare_unmatched_hotspots(matched_hotspot_ids):

    hotspots = gpd.read_parquet(
        POLYGON_PATH
    )

    assert not hotspots["hotspot_id"].duplicated().any()

    assert matched_hotspot_ids.issubset(
        set(hotspots["hotspot_id"])
    )

    unmatched = hotspots.loc[
        ~hotspots["hotspot_id"].isin(
            matched_hotspot_ids
        )
    ].copy()

    print("\n===== Unmatched Hotspots =====")
    print("Total Hotspots:", len(hotspots))
    print("Unmatched Hotspots:", len(unmatched))

    print("\nBy Province:")

    print(
        unmatched["province"]
        .value_counts()
        .to_string()
    )

    print("\nBy Accident Type:")

    print(
        unmatched["사고유형구분"]
        .value_counts()
        .to_string()
    )

    assert len(unmatched) == 28

    # 별도 검토용으로 Geometry를 제외한 정보 보존
    unmatched = pd.DataFrame(
        unmatched.drop(
            columns=[hotspots.geometry.name]
        )
    )

    return unmatched


# ============================================================
# 7. 배치별 학습 라벨 생성
# ============================================================

def process_batch(
    frame,
    summary,
    writer,
    stats,
    feature_sums,
    matched_ids
):

    original_rows = len(frame)

    # 도로 피처와 사고다발지역 정보 결합
    frame = frame.merge(
        summary,
        left_on="link_id",
        right_index=True,
        how="left",
        validate="one_to_one",
        sort=False
    )

    assert len(frame) == original_rows

    # 미매칭 도로의 중첩 정보는 0
    frame["hotspot_count"] = (
        frame["hotspot_count"]
        .fillna(0)
        .astype("int32")
    )

    for column in [
        "max_overlap_length_m",
        "max_overlap_ratio"
    ]:

        frame[column] = (
            frame[column].fillna(0)
        )

    # 미매칭 도로의 사고연도는 결측값 유지
    for column in [
        "first_accident_year",
        "last_accident_year"
    ]:

        frame[column] = (
            frame[column].astype("Int32")
        )

    # 기본 라벨: 중첩 길이 > 0
    frame["target"] = (
        frame["max_overlap_length_m"] > 0
    ).astype("int8")

    # 민감도 분석용 라벨
    for threshold in THRESHOLDS:

        column = f"target_{threshold}"

        frame[column] = (
            frame["max_overlap_ratio"]
            >= threshold / 100
        ).astype("int8")

    # --------------------------------------------------------
    # 배치 검증
    # --------------------------------------------------------

    assert frame[FEATURES].notna().all().all()

    assert np.isfinite(
        frame[FEATURES].to_numpy(
            dtype=np.float64
        )
    ).all()

    assert frame["max_overlap_ratio"].between(
        0, 1
    ).all()

    assert (
        frame["target"]
        == (frame["hotspot_count"] > 0)
    ).all()

    positive_mask = frame["target"] == 1
    negative_mask = frame["target"] == 0

    assert frame.loc[
        positive_mask,
        "first_accident_year"
    ].notna().all()

    assert frame.loc[
        positive_mask,
        "last_accident_year"
    ].notna().all()

    assert frame.loc[
        negative_mask,
        "first_accident_year"
    ].isna().all()

    assert frame.loc[
        negative_mask,
        "last_accident_year"
    ].isna().all()

    assert (
        frame.loc[
            positive_mask,
            "first_accident_year"
        ]
        <= frame.loc[
            positive_mask,
            "last_accident_year"
        ]
    ).all()

    assert (
        frame.loc[
            negative_mask,
            "max_overlap_ratio"
        ] == 0
    ).all()

    # 높은 중첩 기준의 양성은 낮은 기준에도 포함
    for previous, current in zip(
        TARGET_COLUMNS[:-1],
        TARGET_COLUMNS[1:]
    ):

        assert (
            frame[current] <= frame[previous]
        ).all()

    # --------------------------------------------------------
    # 전체 통계 누적
    # --------------------------------------------------------

    stats["total_rows"] += len(frame)

    stats["total_hotspot_matches"] += int(
        frame["hotspot_count"].sum()
    )

    for column in TARGET_COLUMNS:

        stats[column] += int(
            frame[column].sum()
        )

    matched_ids.update(
        frame.loc[
            positive_mask,
            "link_id"
        ].tolist()
    )

    for label in [0, 1]:

        subset = frame.loc[
            frame["target"] == label
        ]

        stats[f"class_{label}"] += len(subset)

        for feature in FEATURES:

            feature_sums[(label, feature)] += float(
                subset[feature].sum()
            )

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    frame = frame[SCHEMA.names]

    table = pa.Table.from_pandas(
        frame,
        schema=SCHEMA,
        preserve_index=False
    )

    writer.write_table(table)


# ============================================================
# 8. 전국 학습 데이터셋 생성
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Accident Label Builder =====")
    print("Road Source:", ROAD_PATH)
    print("Match Source:", MATCH_PATH)
    print("Batch Size:", BATCH_SIZE)

    print_memory("Initial Memory")

    for path in [
        OUTPUT_PATH,
        TEMP_PATH,
        UNMATCHED_PATH,
        SUMMARY_PATH
    ]:

        if path.exists():
            raise FileExistsError(path)

    (
        summary,
        matched_hotspot_ids,
        expected_pairs
    ) = prepare_match_summary()

    unmatched = prepare_unmatched_hotspots(
        matched_hotspot_ids
    )

    print_memory("After Match Summary")

    road_file = pq.ParquetFile(
        ROAD_PATH
    )

    expected_rows = (
        road_file.metadata.num_rows
    )

    assert expected_rows == 1562356

    metadata = (
        road_file.schema_arrow.metadata or {}
    )

    assert (
        metadata.get(b"geometry_crs")
        == b"EPSG:5179"
    )

    stats = Counter()
    feature_sums = Counter()
    matched_ids = set()

    print("\n===== Build National Dataset =====")

    with pq.ParquetWriter(
        TEMP_PATH,
        SCHEMA,
        compression="zstd"
    ) as writer:

        for batch in road_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=ROAD_COLUMNS
        ):

            frame = batch.to_pandas()

            process_batch(
                frame,
                summary,
                writer,
                stats,
                feature_sums,
                matched_ids
            )

            if (
                stats["total_rows"] % 100000
                < len(frame)
            ):

                memory = psutil.virtual_memory()

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"Processed: "
                    f"{stats['total_rows']:,}"
                    f" / {expected_rows:,} | "
                    f"Positive: "
                    f"{stats['target']:,} | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"RAM: "
                    f"{memory.available / 1024**3:.2f} GiB",
                    flush=True
                )

    # ========================================================
    # 9. 전체 데이터 검증
    # ========================================================

    expected_targets = {
        "target": 207391,
        "target_10": 203483,
        "target_25": 197053,
        "target_50": 185103,
        "target_75": 171932,
        "target_90": 164323
    }

    assert stats["total_rows"] == expected_rows

    assert (
        stats["class_0"] + stats["class_1"]
        == expected_rows
    )

    assert stats["class_0"] == 1354965
    assert stats["class_1"] == 207391

    assert (
        stats["total_hotspot_matches"]
        == expected_pairs
    )

    assert matched_ids == set(summary.index)

    for column, expected in expected_targets.items():

        assert stats[column] == expected, (
            f"{column} count mismatch."
        )

    # ========================================================
    # 10. 라벨 분포 출력
    # ========================================================

    print("\n===== Target Distribution =====")

    print("Total Rows:", expected_rows)
    print("Class 0:", stats["class_0"])
    print("Class 1:", stats["class_1"])

    print(
        f"Positive Ratio: "
        f"{stats['class_1'] / expected_rows * 100:.4f}%"
    )

    print(
        f"Negative / Positive Ratio: "
        f"{stats['class_0'] / stats['class_1']:.6f}"
    )

    print("\n===== Target Sensitivity Summary =====")

    label_rows = []

    for column in TARGET_COLUMNS:

        positive = stats[column]
        negative = expected_rows - positive

        label_rows.append({
            "target_column": column,
            "total_rows": expected_rows,
            "class_0": negative,
            "class_1": positive,
            "positive_ratio": positive / expected_rows,
            "negative_positive_ratio": negative / positive
        })

    label_summary = pd.DataFrame(label_rows)

    print(
        label_summary.to_string(index=False)
    )

    # ========================================================
    # 11. 클래스별 피처 평균 비교
    # ========================================================

    print("\n===== Feature Means by Target =====")

    feature_rows = []

    for feature in FEATURES:

        mean_0 = (
            feature_sums[(0, feature)]
            / stats["class_0"]
        )

        mean_1 = (
            feature_sums[(1, feature)]
            / stats["class_1"]
        )

        feature_rows.append({
            "feature": feature,
            "target_0_mean": mean_0,
            "target_1_mean": mean_1,
            "difference": mean_1 - mean_0
        })

    print(
        pd.DataFrame(feature_rows)
        .to_string(index=False)
    )

    # ========================================================
    # 12. 학습 피처 구성 출력
    # ========================================================

    print("\n===== Candidate Feature Sets =====")

    print(
        "A - Degree Intersection:",
        BASE_FEATURES + ["n_intersect_200m"]
    )

    print(
        "B - Type101 Intersection:",
        BASE_FEATURES + ["intersection_nodes_200m"]
    )

    print(
        "C - Both Intersections:",
        FEATURES
    )

    print("\n===== Excluded From Model Inputs =====")

    print([
        "link_id",
        "f_node",
        "t_node",
        "road_type",
        "geometry_length_m",
        "center_x",
        "center_y",
        "hotspot_count",
        "max_overlap_length_m",
        "max_overlap_ratio",
        "first_accident_year",
        "last_accident_year"
    ] + TARGET_COLUMNS)

    # ========================================================
    # 13. 최종 파일 검증 및 저장
    # ========================================================

    parquet = pq.ParquetFile(
        TEMP_PATH
    )

    assert (
        parquet.metadata.num_rows
        == expected_rows
    )

    assert (
        parquet.schema_arrow.names
        == SCHEMA.names
    )

    del parquet
    del road_file

    TEMP_PATH.replace(OUTPUT_PATH)

    unmatched.to_csv(
        UNMATCHED_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    label_summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== Saved Output =====")
    print("Dataset:", OUTPUT_PATH)
    print("Rows:", expected_rows)
    print("Columns:", len(SCHEMA))

    print(
        f"File Size: "
        f"{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB"
    )

    print("Label Summary:", SUMMARY_PATH)
    print("Unmatched Hotspots:", UNMATCHED_PATH)

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(
        f"Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    print("Records Removed: 0")
    print("Original Files: Not Modified")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()