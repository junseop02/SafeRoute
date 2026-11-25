from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import psutil


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

DATASET_PATH = (
    PROCESSED_DIR
    / "national_accident_risk_dataset.parquet"
)

SPLIT_PATH = (
    PROCESSED_DIR
    / "national_buffered_split_2000m.parquet"
)

VALIDATION_PATH = (
    RESULTS_DIR
    / "national_spatial_validation_summary.csv"
)

FINAL_PATH = (
    PROCESSED_DIR
    / "national_accident_risk_final_dataset.parquet"
)

EXCLUDED_PATH = (
    PROCESSED_DIR
    / "national_accident_risk_buffer_excluded.parquet"
)

CONFIG_PATH = (
    RESULTS_DIR
    / "national_training_config.json"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "national_final_dataset_summary.csv"
)

GRID_SIZE = 2000
BUFFER_DISTANCE = 400.0
RANDOM_STATE = 42

EXPECTED_ROWS = 1562356
EXPECTED_POSITIVE = 207391

EXPECTED_DISTRIBUTION = {
    "train": (911037, 141754),
    "validation": (212767, 30615),
    "test": (234353, 31108),
    "excluded": (204199, 3914)
}


# ============================================================
# 2. 피처 구성
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

ALL_FEATURES = FEATURE_SETS["C"]

TARGET = "target"

SENSITIVITY_TARGETS = [
    "target_10",
    "target_25",
    "target_50",
    "target_75",
    "target_90"
]

CATEGORICAL_FEATURES = [
    "road_rank"
]

ROAD_RANK_CATEGORIES = [
    101, 102, 103, 104, 105, 106, 107
]

BASELINES = [
    "n_intersect_200m",
    "intersection_nodes_200m"
]

METADATA_COLUMNS = [
    "link_id",
    "center_x",
    "center_y",
    "grid_x",
    "grid_y",
    "spatial_group",
    "split"
]


# ============================================================
# 3. 메모리 및 해시 확인
# ============================================================

def print_memory(title):

    memory = psutil.virtual_memory()

    print(f"\n===== {title} =====")

    print(
        f"Available RAM: "
        f"{memory.available / 1024**3:.2f} GiB"
    )

    print(f"RAM Usage: {memory.percent}%")


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            digest.update(chunk)

    return digest.hexdigest()


def hash_test_data(frame):

    # Test 데이터의 ID, 피처, 라벨을 고정된 순서로 정렬
    columns = [
        "link_id",
        *ALL_FEATURES,
        TARGET,
        *SENSITIVITY_TARGETS,
        "spatial_group",
        "grid_x",
        "grid_y"
    ]

    test = (
        frame.loc[
            frame["split"] == "test",
            columns
        ]
        .sort_values("link_id")
    )

    digest = hashlib.sha256()

    digest.update(
        test.to_csv(
            index=False,
            lineterminator="\n",
            float_format="%.17g"
        ).encode("utf-8")
    )

    return digest.hexdigest()


# ============================================================
# 4. 공간 검증 결과 확인
# ============================================================

def validate_spatial_summary():

    summary = pd.read_csv(
        VALIDATION_PATH
    )

    selected = summary.loc[
        summary["grid_size_m"] == GRID_SIZE
    ]

    assert len(selected) == 1

    row = selected.iloc[0]

    assert (
        str(row["validation_passed"]).lower()
        == "true"
    )

    assert (
        row["buffer_distance_m"]
        == BUFFER_DISTANCE
    )

    assert (
        row["total_spatial_violations"]
        == 0
    )

    assert row["group_violations"] == 0
    assert row["grid_violations"] == 0
    assert row["hotspot_violations"] == 0

    assert row["test_removed"] == 0
    assert row["test_added"] == 0
    assert row["label_violations"] == 0

    assert row["retained_links"] == 1358157
    assert row["excluded_links"] == 204199

    print("\n===== Spatial Validation =====")

    print("Selected Grid:", GRID_SIZE)
    print("Buffer Distance:", BUFFER_DISTANCE)
    print("Spatial Violations: 0")
    print("Validation Status: PASS")


# ============================================================
# 5. 원본 데이터 및 공간 분할 결합
# ============================================================

def load_dataset():

    dataset_columns = [
        "link_id",
        *ALL_FEATURES,
        "center_x",
        "center_y",
        TARGET,
        *SENSITIVITY_TARGETS
    ]

    split_columns = [
        "link_id",
        "target",
        "grid_x",
        "grid_y",
        "spatial_group",
        "split",
        "original_split",
        "buffer_excluded",
        "exclusion_reason"
    ]

    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=dataset_columns
    )

    split = pd.read_parquet(
        SPLIT_PATH,
        columns=split_columns
    )

    assert len(dataset) == EXPECTED_ROWS
    assert len(split) == EXPECTED_ROWS

    assert dataset["link_id"].is_unique
    assert split["link_id"].is_unique

    split = split.rename(
        columns={
            "target": "split_target"
        }
    )

    frame = dataset.merge(

        split,

        on="link_id",

        how="left",

        validate="one_to_one",

        indicator=True

    )

    assert len(frame) == EXPECTED_ROWS

    assert (
        frame["_merge"] == "both"
    ).all()

    assert (
        frame[TARGET]
        == frame["split_target"]
    ).all()

    frame = frame.drop(
        columns=[
            "_merge",
            "split_target"
        ]
    )

    print("\n===== Dataset Merge =====")

    print("Original Rows:", len(dataset))
    print("Split Rows:", len(split))
    print("Merged Rows:", len(frame))
    print("Missing Link IDs: 0")
    print("Label Mismatches: 0")

    return frame


# ============================================================
# 6. 피처 및 라벨 검증
# ============================================================

def validate_dataset(frame):

    assert frame["link_id"].is_unique

    assert len(frame) == EXPECTED_ROWS

    assert (
        int(frame[TARGET].sum())
        == EXPECTED_POSITIVE
    )

    assert frame[
        ALL_FEATURES
    ].notna().all().all()

    assert np.isfinite(
        frame[ALL_FEATURES].to_numpy(
            dtype=np.float64
        )
    ).all()

    assert frame["road_rank"].isin(
        ROAD_RANK_CATEGORIES
    ).all()

    assert (frame["lanes"] >= 1).all()
    assert (frame["max_spd"] > 0).all()
    assert (frame["length"] > 0).all()
    assert (frame["n_branch"] >= 0).all()

    assert (
        frame["n_intersect_200m"] >= 0
    ).all()

    assert (
        frame["intersection_nodes_200m"] >= 0
    ).all()

    for target in [
        TARGET,
        *SENSITIVITY_TARGETS
    ]:

        assert frame[target].isin(
            [0, 1]
        ).all()

    # 중첩 기준이 높아질수록 양성 집합은 감소
    previous_target = TARGET

    for target in SENSITIVITY_TARGETS:

        assert (
            frame[target]
            <= frame[previous_target]
        ).all()

        previous_target = target

    assert frame[
        METADATA_COLUMNS
    ].notna().all().all()

    assert frame["split"].isin(
        EXPECTED_DISTRIBUTION
    ).all()

    assert (
        frame["buffer_excluded"]
        == (frame["split"] == "excluded")
    ).all()

    print("\n===== Feature Validation =====")

    print("Missing Features: 0")
    print("Non-finite Features: 0")
    print("Invalid Labels: 0")
    print("Duplicate Link IDs: 0")
    print("Target Hierarchy: PASS")

    for name, features in FEATURE_SETS.items():

        print(f"\nFeature Set {name}:")

        for feature in features:

            print(" -", feature)


# ============================================================
# 7. 최종 분할 통계
# ============================================================

def build_summary(frame):

    results = []

    for split_name, expected in (
        EXPECTED_DISTRIBUTION.items()
    ):

        subset = frame.loc[
            frame["split"] == split_name
        ]

        total = len(subset)

        positive = int(
            subset[TARGET].sum()
        )

        negative = total - positive

        assert (total, positive) == expected

        groups = subset.groupby(
            "spatial_group",
            observed=True
        )[TARGET].sum()

        results.append({

            "split": split_name,

            "total_links": total,

            "positive_links": positive,

            "negative_links": negative,

            "positive_ratio":
                positive / total,

            "total_ratio":
                total / EXPECTED_ROWS,

            "spatial_groups":
                len(groups),

            "positive_groups":
                int((groups > 0).sum()),

            "max_positive_group_share":
                float(groups.max() / positive),

            "scale_pos_weight":
                negative / positive

        })

    summary = pd.DataFrame(results)

    assert (
        summary["total_links"].sum()
        == EXPECTED_ROWS
    )

    assert (
        summary["positive_links"].sum()
        == EXPECTED_POSITIVE
    )

    print("\n===== Final Dataset Distribution =====")

    print(
        summary.to_string(index=False)
    )

    return summary


# ============================================================
# 8. 최종 학습 데이터 구성
# ============================================================

def prepare_final_dataset(frame):

    final_columns = [
        *METADATA_COLUMNS,
        *ALL_FEATURES,
        TARGET,
        *SENSITIVITY_TARGETS
    ]

    retained = frame.loc[

        frame["split"] != "excluded",

        final_columns

    ].copy()

    excluded = frame.loc[

        frame["split"] == "excluded",

        final_columns + [
            "original_split",
            "buffer_excluded",
            "exclusion_reason"
        ]

    ].copy()

    assert len(retained) == 1358157
    assert len(excluded) == 204199

    assert (
        len(retained) + len(excluded)
        == EXPECTED_ROWS
    )

    assert not retained["link_id"].isin(
        excluded["link_id"]
    ).any()

    # 보존 데이터의 공간 그룹 및 격자 중복 재확인
    group_violations = (

        retained.groupby(
            "spatial_group",
            observed=True
        )["split"].nunique() > 1

    ).sum()

    grid_violations = (

        retained.groupby(
            ["grid_x", "grid_y"],
            observed=True
        )["split"].nunique() > 1

    ).sum()

    assert group_violations == 0
    assert grid_violations == 0

    # 도로 등급은 명목형 범주로 저장
    for data in [retained, excluded]:

        data["road_rank"] = pd.Categorical(

            data["road_rank"].astype("int16"),

            categories=ROAD_RANK_CATEGORIES,

            ordered=False

        )

        assert data["road_rank"].notna().all()

        for target in [
            TARGET,
            *SENSITIVITY_TARGETS
        ]:

            data[target] = data[target].astype(
                "int8"
            )

    print("\n===== Final Dataset =====")

    print("Retained Rows:", len(retained))
    print("Excluded Rows:", len(excluded))

    print(
        "Retained Columns:",
        len(retained.columns)
    )

    print("Group Violations:", group_violations)
    print("Grid Violations:", grid_violations)

    return retained, excluded


# ============================================================
# 9. 실험 설정 저장
# ============================================================

def save_config(
    retained,
    summary,
    source_hashes
):

    train = summary.loc[
        summary["split"] == "train"
    ].iloc[0]

    test_hash = hash_test_data(
        retained
    )

    config = {

        "experiment_name":
            "national_accident_hotspot_classification",

        "random_state":
            RANDOM_STATE,

        "grid_size_m":
            GRID_SIZE,

        "buffer_distance_m":
            BUFFER_DISTANCE,

        "coordinate_crs":
            "EPSG:5179",

        "objective":
            "binary",

        "target":
            TARGET,

        "target_definition":
            "positive_polygon_overlap_length",

        "target_meaning":
            "historical_hotspot_membership",

        "feature_sets":
            FEATURE_SETS,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "road_rank_categories":
            ROAD_RANK_CATEGORIES,

        "baseline_features":
            BASELINES,

        "sensitivity_targets":
            SENSITIVITY_TARGETS,

        "primary_metric":
            "average_precision",

        "secondary_metrics": [
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy"
        ],

        "class_weight_candidates": [

            1.0,

            float(train["scale_pos_weight"])

        ],

        "threshold_selection":
            "validation_max_f1",

        "model_selection":
            "validation_average_precision",

        "test_policy":
            "final_evaluation_only",

        "test_data_sha256":
            test_hash,

        "test_frozen":
            True,

        "source_file_sha256":
            source_hashes,

        "final_dataset_sha256":
            sha256_file(FINAL_PATH),

        "excluded_dataset_sha256":
            sha256_file(EXCLUDED_PATH),

        "split_distribution":
            summary.to_dict(
                orient="records"
            ),

        "limitations": [

            "Target zero does not mean accident-free.",

            "Hotspot membership is not actual "
            "accident probability.",

            "Hotspots cover selected vulnerable "
            "road-user accident categories.",

            "Historical accident records and road "
            "network dates are not time-aligned.",

            "Spatial buffering does not eliminate "
            "all spatial dependence.",

            "Grid selection was based on data "
            "retention and group concentration "
            "before model training."

        ],

        "deployment_status":
            "not_deployed"

    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            config,

            file,

            ensure_ascii=False,

            indent=4,

            allow_nan=False

        )

    print("\n===== Frozen Experiment Configuration =====")

    print("Grid Size:", GRID_SIZE)
    print("Buffer Distance:", BUFFER_DISTANCE)
    print("Feature Sets:", list(FEATURE_SETS))
    print("Primary Metric: Average Precision")

    print(
        "Scale Pos Weight:",
        f"{train['scale_pos_weight']:.8f}"
    )

    print("Test Data SHA-256:", test_hash)

    print("Final Test Set Frozen: Yes")

    return config


# ============================================================
# 10. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== Finalize National Training Dataset =====")

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_paths = [
        FINAL_PATH,
        EXCLUDED_PATH,
        CONFIG_PATH,
        SUMMARY_PATH
    ]

    # 기존 결과 파일 덮어쓰기 방지
    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    validate_spatial_summary()

    source_paths = [
        DATASET_PATH,
        SPLIT_PATH,
        VALIDATION_PATH
    ]

    source_hashes = {

        path.name: sha256_file(path)

        for path in source_paths

    }

    frame = load_dataset()

    print_memory("After Dataset Load")

    validate_dataset(frame)

    summary = build_summary(frame)

    retained, excluded = prepare_final_dataset(
        frame
    )

    del frame

    # 새로운 파일로 저장
    retained.to_parquet(
        FINAL_PATH,
        index=False
    )

    excluded.to_parquet(
        EXCLUDED_PATH,
        index=False
    )

    summary.to_csv(

        SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    config = save_config(
        retained,
        summary,
        source_hashes
    )

    # 저장된 데이터 재검증
    reloaded = pd.read_parquet(
        FINAL_PATH
    )

    # Parquet 재로딩 시 정수형으로 반환된
    # road_rank를 기존 범주형 자료형으로 복원
    assert reloaded["road_rank"].isin(
        ROAD_RANK_CATEGORIES
    ).all()

    reloaded["road_rank"] = pd.Categorical(
        reloaded["road_rank"],
        categories=ROAD_RANK_CATEGORIES,
        ordered=False
    )

    assert reloaded["road_rank"].notna().all()

    assert len(reloaded) == 1358157

    assert reloaded["link_id"].is_unique

    assert (
        hash_test_data(reloaded)
        == config["test_data_sha256"]
    )

    # 기존 코드
    # pd.testing.assert_frame_equal(
    #     retained,
    #     reloaded
    # )

    # Parquet 저장 시 index=False를 사용하므로
    # 인덱스를 초기화한 후 실제 데이터의 일치 여부 검증
    pd.testing.assert_frame_equal(
        retained.reset_index(drop=True),
        reloaded.reset_index(drop=True),
        check_exact=True
    )

    for path in source_paths:

        assert (
            sha256_file(path)
            == source_hashes[path.name]
        )

    print("\n===== Saved Files =====")

    for path in output_paths:

        print("Path:", path)

        print(

            f"File Size: "

            f"{path.stat().st_size / 1024**2:.4f} MiB"

        )

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(

        f"Elapsed Time: "

        f"{time.perf_counter() - start_time:.2f} seconds"

    )

    print("Final Grid Size: 2000 m")
    print("Spatial Buffer: 400 m")
    print("Final Dataset Validation: PASS")
    print("Saved Dataset Reload: PASS")
    print("Test Hash Verification: PASS")
    print("Final Test Set Frozen: Yes")
    print("Original Dataset Modified: No")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()