from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import numpy as np
import joblib


# 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"

FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_intersect_200m",
    "n_branch"
]


# 1. 데이터 불러오기
roads = gpd.read_file(
    ROOT / "source/links.geojson"
)

dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_dataset.parquet"
)

with open(
    ROOT / "data/link_scores.json",
    "r",
    encoding="utf-8"
) as file:

    scores = pd.DataFrame(json.load(file))


# 2. 신규 모델 불러오기
bundle = joblib.load(
    ROOT / "models/accident_risk_lightgbm_final.joblib"
)


# 3. 데이터 기본 정보
print("===== Data Information =====")

for name, frame in [
    ("Service Roads", roads),
    ("Service Link Scores", scores),
    ("New Model Dataset", dataset)
]:

    print(f"\n[{name}]")
    print("Shape:", frame.shape)
    print("Columns:", frame.columns.tolist())
    print("Missing IDs:", frame["link_id"].isna().sum())

    frame["link_id"] = frame["link_id"].astype(str)

    print("Unique IDs:", frame["link_id"].nunique())
    print("Duplicate IDs:", frame["link_id"].duplicated().sum())


# 4. 도로 링크 ID 비교
road_ids = set(roads["link_id"])
score_ids = set(scores["link_id"])
dataset_ids = set(dataset["link_id"])

print("\n===== Link ID Compatibility =====")

print(
    "Roads vs Scores:",
    len(road_ids & score_ids)
)

print(
    "Roads vs New Dataset:",
    len(road_ids & dataset_ids)
)

print(
    "Scores vs New Dataset:",
    len(score_ids & dataset_ids)
)

print(
    "Roads Missing in New Dataset:",
    len(road_ids - dataset_ids)
)

print(
    "New Dataset Missing in Roads:",
    len(dataset_ids - road_ids)
)

print(
    "Scores Missing in New Dataset:",
    len(score_ids - dataset_ids)
)

print(
    "New Dataset Missing in Scores:",
    len(dataset_ids - score_ids)
)

print(
    "All ID Sets Identical:",
    road_ids == score_ids == dataset_ids
)


# 5. 기존 점수 데이터 확인
print("\n===== Existing Link Score Sample =====")

print(
    scores.head(3).to_string(index=False)
)


# 6. 공통 피처 비교
print("\n===== Shared Feature Comparison =====")

shared_features = [
    feature for feature in FEATURES
    if feature in scores.columns
    and feature in dataset.columns
]

print("Shared Features:", shared_features)

if (
    not scores["link_id"].duplicated().any()
    and not dataset["link_id"].duplicated().any()
):

    merged = scores[
        ["link_id"] + shared_features
    ].merge(
        dataset[["link_id"] + shared_features],
        on="link_id",
        suffixes=("_service", "_new"),
        validate="one_to_one"
    )

    for feature in shared_features:

        old = pd.to_numeric(
            merged[f"{feature}_service"],
            errors="coerce"
        ).to_numpy(dtype=float)

        new = pd.to_numeric(
            merged[f"{feature}_new"],
            errors="coerce"
        ).to_numpy(dtype=float)

        equal = np.isclose(
            old,
            new,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=False
        )

        print(
            f"{feature}: "
            f"Matched={equal.sum()}, "
            f"Different={(~equal).sum()}"
        )

else:
    print("Comparison skipped: duplicate IDs.")


# 7. 모델 호환성 확인
print("\n===== Model Compatibility =====")

print("Candidate:", bundle["candidate"])
print("Model Features:", bundle["features"])
print("Expected Features:", FEATURES)

print(
    "Feature Order Match:",
    bundle["features"] == FEATURES
)

print(
    "Road Rank Categories:",
    bundle["road_rank_categories"]
)

print(
    "Missing Model Features:",
    [
        feature for feature in FEATURES
        if feature not in dataset.columns
    ]
)

print(
    "Model Threshold:",
    bundle["threshold"]
)

print(
    "Deployment Status:",
    bundle["deployment_status"]
)

print(
    "Score Meaning:",
    bundle["score_meaning"]
)


# 8. 기존 도로 좌표계 및 형상 검사
print("\n===== Service Geometry =====")

print("CRS:", roads.crs)
print("Geometry Types:")
print(roads.geometry.geom_type.value_counts())

print(
    "Missing Geometry:",
    roads.geometry.isna().sum()
)

print(
    "Empty Geometry:",
    roads.geometry.is_empty.sum()
)

print(
    "Invalid Geometry:",
    (~roads.geometry.is_valid).sum()
)


# 9. 검사 종료
print("\n===== Compatibility Inspection Completed =====")

print("Existing Service Files: Not Modified")
print("Existing Model: Not Modified")
print("New Model: Not Modified")