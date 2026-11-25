from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
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

DATASET_PATH = (
    PROCESSED_DIR
    / "national_accident_risk_dataset.parquet"
)

ROAD_PATH = (
    PROCESSED_DIR
    / "national_road_features.parquet"
)

MATCH_PATH = (
    PROCESSED_DIR
    / "national_accident_link_matches.parquet"
)

UNMATCHED_PATH = (
    RESULTS_DIR / "unmatched_hotspots.csv"
)

FEATURE_STATS_PATH = (
    RESULTS_DIR / "national_feature_statistics.csv"
)

CORRELATION_PATH = (
    RESULTS_DIR / "national_feature_correlation.csv"
)

RANK_SUMMARY_PATH = (
    RESULTS_DIR / "national_road_rank_summary.csv"
)

VALIDATION_PATH = (
    RESULTS_DIR / "national_dataset_validation.csv"
)

EXPECTED_ROWS = 1562356
EXPECTED_COLUMNS = 25

BATCH_SIZE = 50000


# ============================================================
# 2. 학습 피처 및 라벨 정의
# ============================================================

BASE_FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_branch"
]

FEATURE_SETS = {
    "A": BASE_FEATURES + [
        "n_intersect_200m"
    ],

    "B": BASE_FEATURES + [
        "intersection_nodes_200m"
    ],

    "C": BASE_FEATURES + [
        "n_intersect_200m",
        "intersection_nodes_200m"
    ]
}

FEATURES = FEATURE_SETS["C"]

TARGET_COLUMNS = [
    "target",
    "target_10",
    "target_25",
    "target_50",
    "target_75",
    "target_90"
]

EXPECTED_TARGETS = {
    "target": 207391,
    "target_10": 203483,
    "target_25": 197053,
    "target_50": 185103,
    "target_75": 171932,
    "target_90": 164323
}

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

MATCH_COLUMNS = [
    "hotspot_count",
    "max_overlap_length_m",
    "max_overlap_ratio",
    "first_accident_year",
    "last_accident_year"
]

FORBIDDEN_FEATURES = set([
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


# ============================================================
# 3. 검증 결과 관리
# ============================================================

validation_results = []


def check(name, condition):

    passed = bool(condition)

    validation_results.append({
        "check": name,
        "passed": passed
    })

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {name}")

    if not passed:
        raise RuntimeError(
            f"Validation failed: {name}"
        )


def print_memory(title):

    memory = psutil.virtual_memory()

    print(f"\n===== {title} =====")

    print(
        f"Available RAM: "
        f"{memory.available / 1024**3:.2f} GiB"
    )

    print(f"RAM Usage: {memory.percent}%")


# ============================================================
# 4. 파일 및 스키마 검증
# ============================================================

def validate_metadata():

    dataset_file = pq.ParquetFile(
        DATASET_PATH
    )

    road_file = pq.ParquetFile(
        ROAD_PATH
    )

    print("\n===== Dataset Metadata =====")

    print(
        "Rows:",
        dataset_file.metadata.num_rows
    )

    print(
        "Columns:",
        len(dataset_file.schema_arrow.names)
    )

    check(
        "Expected dataset row count",
        dataset_file.metadata.num_rows
        == EXPECTED_ROWS
    )

    check(
        "Expected dataset column count",
        len(dataset_file.schema_arrow.names)
        == EXPECTED_COLUMNS
    )

    check(
        "Expected road row count",
        road_file.metadata.num_rows
        == EXPECTED_ROWS
    )

    expected_columns = (
        ROAD_COLUMNS
        + MATCH_COLUMNS
        + TARGET_COLUMNS
    )

    check(
        "Dataset column structure",
        dataset_file.schema_arrow.names
        == expected_columns
    )

    metadata = (
        dataset_file.schema_arrow.metadata or {}
    )

    check(
        "Coordinate CRS",
        metadata.get(b"coordinate_crs")
        == b"EPSG:5179"
    )

    check(
        "Target meaning",
        metadata.get(b"target_meaning")
        == b"historical_hotspot_membership"
    )

    check(
        "Target definition",
        metadata.get(b"target_definition")
        == b"positive_polygon_overlap_length"
    )

    return dataset_file, road_file


# ============================================================
# 5. 원본 도로 피처 일치 여부 검증
# ============================================================

def validate_original_features(
    dataset_file,
    road_file
):

    print("\n===== Original Feature Consistency =====")

    dataset_batches = dataset_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=ROAD_COLUMNS
    )

    road_batches = road_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=ROAD_COLUMNS
    )

    # 서로 다른 Parquet 배치 경계에도 대응
    dataset_batch = next(
        dataset_batches, None
    )

    road_batch = next(
        road_batches, None
    )

    dataset_offset = 0
    road_offset = 0
    checked_rows = 0

    mismatches = {
        column: 0
        for column in ROAD_COLUMNS
    }

    while (
        dataset_batch is not None
        and road_batch is not None
    ):

        size = min(
            dataset_batch.num_rows - dataset_offset,
            road_batch.num_rows - road_offset
        )

        left = dataset_batch.slice(
            dataset_offset, size
        )

        right = road_batch.slice(
            road_offset, size
        )

        for i, column in enumerate(ROAD_COLUMNS):

            a = left.column(i)
            b = right.column(i)

            if not a.equals(b):

                a_values = a.to_pandas()
                b_values = b.to_pandas()

                mismatches[column] += int(
                    (a_values != b_values).sum()
                )

        checked_rows += size

        dataset_offset += size
        road_offset += size

        if dataset_offset == dataset_batch.num_rows:

            dataset_batch = next(
                dataset_batches, None
            )

            dataset_offset = 0

        if road_offset == road_batch.num_rows:

            road_batch = next(
                road_batches, None
            )

            road_offset = 0

    check(
        "Both source streams fully consumed",
        dataset_batch is None
        and road_batch is None
    )

    check(
        "All original rows compared",
        checked_rows == EXPECTED_ROWS
    )

    for column, count in mismatches.items():

        check(
            f"Original {column} consistency",
            count == 0
        )

    print("Compared Rows:", checked_rows)


# ============================================================
# 6. 데이터 누수 검사
# ============================================================

def validate_feature_sets():

    print("\n===== Feature Leakage Check =====")

    for name, features in FEATURE_SETS.items():

        forbidden = (
            set(features)
            & FORBIDDEN_FEATURES
        )

        print(f"\nFeature Set {name}:")
        print(features)
        print("Forbidden Features:", sorted(forbidden))

        check(
            f"Feature Set {name}: no forbidden features",
            len(forbidden) == 0
        )

        check(
            f"Feature Set {name}: no duplicate features",
            len(features) == len(set(features))
        )


# ============================================================
# 7. 학습 데이터 읽기
# ============================================================

def load_dataset():

    columns = (
        ["link_id"]
        + FEATURES
        + [
            "geometry_length_m",
            "center_x",
            "center_y"
        ]
        + MATCH_COLUMNS
        + TARGET_COLUMNS
    )

    frame = pd.read_parquet(
        DATASET_PATH,
        columns=columns
    )

    print("\n===== Loaded Dataset =====")
    print("Shape:", frame.shape)

    print(
        "Memory Usage:",
        f"{frame.memory_usage(deep=True).sum() / 1024**2:.2f} MiB"
    )

    return frame


# ============================================================
# 8. 데이터 품질 검사
# ============================================================

def validate_data_quality(frame):

    print("\n===== Data Quality =====")

    check(
        "Expected loaded rows",
        len(frame) == EXPECTED_ROWS
    )

    check(
        "No missing LINK IDs",
        frame["link_id"].notna().all()
    )

    check(
        "No empty LINK IDs",
        frame["link_id"].str.strip().ne("").all()
    )

    duplicate_ids = (
        frame["link_id"].duplicated().sum()
    )

    print("Duplicate LINK IDs:", duplicate_ids)

    check(
        "Unique LINK IDs",
        duplicate_ids == 0
    )

    print("\nFeature Missing Values:")

    print(
        frame[FEATURES]
        .isna()
        .sum()
        .to_string()
    )

    check(
        "No missing model features",
        frame[FEATURES].notna().all().all()
    )

    check(
        "All model features finite",
        np.isfinite(
            frame[FEATURES].to_numpy(
                dtype=np.float64
            )
        ).all()
    )

    check(
        "Positive lanes",
        (frame["lanes"] > 0).all()
    )

    check(
        "Positive speed limits",
        (frame["max_spd"] > 0).all()
    )

    check(
        "Positive road lengths",
        (frame["length"] > 0).all()
    )

    check(
        "Positive finite geometry lengths",
        np.isfinite(frame["geometry_length_m"]).all()
        and (frame["geometry_length_m"] > 0).all()
    )

    check(
        "Nonnegative branch counts",
        (frame["n_branch"] >= 0).all()
    )

    check(
        "Nonnegative degree intersections",
        (frame["n_intersect_200m"] >= 0).all()
    )

    check(
        "Nonnegative type101 intersections",
        (frame["intersection_nodes_200m"] >= 0).all()
    )

    check(
        "Road rank range",
        frame["road_rank"].isin(
            range(101, 109)
        ).all()
    )

    check(
        "Finite center coordinates",
        np.isfinite(
            frame[
                ["center_x", "center_y"]
            ].to_numpy()
        ).all()
    )


# ============================================================
# 9. 라벨 독립 재검증
# ============================================================

def validate_targets(frame):

    print("\n===== Target Validation =====")

    for column in TARGET_COLUMNS:

        check(
            f"{column}: binary values",
            frame[column].isin([0, 1]).all()
        )

        actual = int(frame[column].sum())
        expected = EXPECTED_TARGETS[column]

        print(
            f"{column}: "
            f"{actual:,} / {expected:,}"
        )

        check(
            f"{column}: expected positive count",
            actual == expected
        )

    recalculated = (
        frame["max_overlap_length_m"] > 0
    ).astype("int8")

    check(
        "Primary target consistency",
        np.array_equal(
            frame["target"],
            recalculated
        )
    )

    for threshold in [10, 25, 50, 75, 90]:

        column = f"target_{threshold}"

        recalculated = (
            frame["max_overlap_ratio"]
            >= threshold / 100
        ).astype("int8")

        check(
            f"{column}: overlap consistency",
            np.array_equal(
                frame[column],
                recalculated
            )
        )

    for previous, current in zip(
        TARGET_COLUMNS[:-1],
        TARGET_COLUMNS[1:]
    ):

        check(
            f"{current} nested within {previous}",
            (frame[current] <= frame[previous]).all()
        )

    check(
        "Overlap ratios within range",
        frame["max_overlap_ratio"]
        .between(0, 1)
        .all()
    )

    check(
        "Target and hotspot count consistency",
        (
            frame["target"]
            == (frame["hotspot_count"] > 0)
        ).all()
    )

    positive = frame["target"] == 1
    negative = frame["target"] == 0

    check(
        "Positive overlap lengths",
        (
            frame.loc[
                positive,
                "max_overlap_length_m"
            ] > 0
        ).all()
    )

    check(
        "Negative overlap lengths",
        (
            frame.loc[
                negative,
                "max_overlap_length_m"
            ] == 0
        ).all()
    )

    check(
        "Negative overlap ratios",
        (
            frame.loc[
                negative,
                "max_overlap_ratio"
            ] == 0
        ).all()
    )

    check(
        "Positive accident years present",
        frame.loc[
            positive,
            [
                "first_accident_year",
                "last_accident_year"
            ]
        ].notna().all().all()
    )

    check(
        "Negative accident years missing",
        frame.loc[
            negative,
            [
                "first_accident_year",
                "last_accident_year"
            ]
        ].isna().all().all()
    )

    check(
        "Accident year ordering",
        (
            frame.loc[
                positive,
                "first_accident_year"
            ]
            <= frame.loc[
                positive,
                "last_accident_year"
            ]
        ).all()
    )

    check(
        "Accident years within source period",
        frame.loc[
            positive,
            "first_accident_year"
        ].between(2012, 2024).all()
        and frame.loc[
            positive,
            "last_accident_year"
        ].between(2012, 2024).all()
    )


# ============================================================
# 10. 원본 공간 매칭 결과 재검증
# ============================================================

def validate_match_source(frame):

    print("\n===== Match Source Consistency =====")

    matches = pd.read_parquet(
        MATCH_PATH,
        columns=[
            "link_id",
            "hotspot_id",
            "accident_year",
            "overlap_length_m",
            "overlap_ratio"
        ]
    )

    check(
        "Expected match rows",
        len(matches) == 485562
    )

    check(
        "Unique link-hotspot pairs",
        not matches.duplicated(
            ["link_id", "hotspot_id"]
        ).any()
    )

    check(
        "Expected matched hotspots",
        matches["hotspot_id"].nunique()
        == 12752
    )

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

    positive = frame.loc[
        frame["target"] == 1,
        ["link_id"] + MATCH_COLUMNS
    ].set_index("link_id")

    check(
        "Matched LINK ID count",
        len(summary) == len(positive)
    )

    check(
        "Matched LINK ID set consistency",
        summary.index.difference(
            positive.index
        ).empty
        and positive.index.difference(
            summary.index
        ).empty
    )

    summary = summary.reindex(
        positive.index
    )

    for column in MATCH_COLUMNS:

        a = summary[column].to_numpy(
            dtype=np.float64
        )

        b = positive[column].to_numpy(
            dtype=np.float64
        )

        check(
            f"Match source {column} consistency",
            np.array_equal(a, b)
        )

    check(
        "Total hotspot relationship count",
        int(frame["hotspot_count"].sum())
        == len(matches)
    )

    unmatched = pd.read_csv(
        UNMATCHED_PATH,
        encoding="utf-8-sig"
    )

    check(
        "Expected unmatched hotspot count",
        len(unmatched) == 28
    )

    check(
        "Unique unmatched hotspot IDs",
        unmatched["hotspot_id"].nunique()
        == 28
    )

    check(
        "No matched-unmatched hotspot overlap",
        not unmatched["hotspot_id"].isin(
            matches["hotspot_id"]
        ).any()
    )


# ============================================================
# 11. 피처 분포 분석
# ============================================================

def analyze_features(frame):

    print("\n===== Feature Statistics =====")

    statistics = frame[FEATURES].describe(
        percentiles=[
            0.01,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    ).T

    print(
        statistics.to_string()
    )

    print("\n===== Feature Unique Values =====")

    print(
        frame[FEATURES]
        .nunique()
        .to_string()
    )

    print("\n===== Largest Feature Values =====")

    for feature in [
        "length",
        "n_branch",
        "n_intersect_200m",
        "intersection_nodes_200m"
    ]:

        print(f"\nTop 5: {feature}")

        print(
            frame.nlargest(
                5, feature
            )[
                ["link_id", feature, "target"]
            ].to_string(index=False)
        )

    return statistics


# ============================================================
# 12. 도로 등급별 라벨 분포
# ============================================================

def analyze_road_rank(frame):

    print("\n===== Road Rank Distribution =====")

    summary = pd.crosstab(
        frame["road_rank"],
        frame["target"]
    )

    summary = summary.reindex(
        columns=[0, 1],
        fill_value=0
    )

    summary.columns = [
        "class_0",
        "class_1"
    ]

    summary["total"] = (
        summary["class_0"]
        + summary["class_1"]
    )

    summary["positive_ratio"] = (
        summary["class_1"]
        / summary["total"]
    )

    print(summary.to_string())

    check(
        "Road rank total count",
        int(summary["total"].sum())
        == EXPECTED_ROWS
    )

    check(
        "Road rank positive count",
        int(summary["class_1"].sum())
        == EXPECTED_TARGETS["target"]
    )

    return summary


# ============================================================
# 13. 피처 상관관계 분석
# ============================================================

def analyze_correlation(frame):

    print("\n===== Pearson Correlation =====")

    # road_rank는 범주형 변수이므로 제외
    numeric_features = [
        feature
        for feature in FEATURES
        if feature != "road_rank"
    ]

    correlation = frame[
        numeric_features + ["target"]
    ].corr(method="pearson")

    print(
        correlation.round(6).to_string()
    )

    print("\n===== Correlation with Target =====")

    print(
        correlation["target"]
        .drop("target")
        .sort_values(
            key=lambda values: values.abs(),
            ascending=False
        )
        .to_string()
    )

    print("\n===== Strong Feature Correlations =====")

    found = False

    for i, first in enumerate(numeric_features):

        for second in numeric_features[i + 1:]:

            value = correlation.loc[
                first, second
            ]

            if abs(value) >= 0.8:

                found = True

                print(
                    f"{first} / {second}: "
                    f"{value:.6f}"
                )

    if not found:
        print("None at absolute correlation >= 0.8")

    return correlation


# ============================================================
# 14. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Dataset Validation =====")
    print("Dataset:", DATASET_PATH)

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for path in [
        FEATURE_STATS_PATH,
        CORRELATION_PATH,
        RANK_SUMMARY_PATH,
        VALIDATION_PATH
    ]:

        if path.exists():
            raise FileExistsError(path)

    dataset_file, road_file = validate_metadata()

    validate_original_features(
        dataset_file,
        road_file
    )

    del dataset_file
    del road_file

    validate_feature_sets()

    frame = load_dataset()

    print_memory("After Dataset Load")

    validate_data_quality(frame)
    validate_targets(frame)
    validate_match_source(frame)

    statistics = analyze_features(frame)

    rank_summary = analyze_road_rank(frame)

    correlation = analyze_correlation(frame)

    # 검증 결과 저장
    statistics.to_csv(
        FEATURE_STATS_PATH,
        encoding="utf-8-sig"
    )

    rank_summary.to_csv(
        RANK_SUMMARY_PATH,
        encoding="utf-8-sig"
    )

    correlation.to_csv(
        CORRELATION_PATH,
        encoding="utf-8-sig"
    )

    validation = pd.DataFrame(
        validation_results
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== Final Validation Summary =====")

    print("Total Checks:", len(validation))

    print(
        "Passed:",
        int(validation["passed"].sum())
    )

    print(
        "Failed:",
        int((~validation["passed"]).sum())
    )

    print("\n===== Saved Results =====")

    print(FEATURE_STATS_PATH)
    print(RANK_SUMMARY_PATH)
    print(CORRELATION_PATH)
    print(VALIDATION_PATH)

    print_memory("Final Memory")

    print("\n===== Validation Completed =====")

    print(
        f"Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    print("Records Removed: 0")
    print("Dataset Modified: No")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()