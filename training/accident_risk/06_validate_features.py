from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/legacy/processed"
OUTPUT_DIR = ROOT / "data/accident_risk/processed"

# 학습 데이터 불러오기
dataset = pd.read_parquet(
    OUTPUT_DIR / "accident_risk_dataset.parquet"
)

# 기존 도로 특성 데이터 불러오기
original = pd.read_parquet(
    DATA_DIR / "intersection_features_cheonan_asan.parquet"
)

# 교통량 데이터 불러오기
traffic = pd.read_parquet(
    DATA_DIR / "traffic_features_cheonan_asan.parquet"
)

# 사고다발지역 매칭 결과 불러오기
matches = gpd.read_parquet(
    OUTPUT_DIR / "accident_link_matches.parquet"
)

# 모델 입력 피처
FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch"
]

TARGET = "target"

# 모델 입력 금지 컬럼
FORBIDDEN = [
    "link_id",
    "region_code",
    "hotspot_count",
    "max_overlap_length_m",
    "max_overlap_ratio",
    "first_accident_year",
    "last_accident_year",
    "center_x",
    "center_y",
    "target",
    "target_10",
    "target_25",
    "target_50",
    "target_75",
    "target_90"
]

print("===== Dataset Information =====")
print("Rows:", len(dataset))
print("Input Features:", len(FEATURES))
print("Target:", TARGET)

# 필수 컬럼 검사
missing = [
    col for col in FEATURES + [TARGET]
    if col not in dataset.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 모델 입력 금지 컬럼 검사
leakage = set(FEATURES) & set(FORBIDDEN)

if leakage:
    raise ValueError(f"Forbidden features: {leakage}")

print("\n===== Direct Leakage Check =====")
print("Forbidden Input Features:", sorted(leakage))
print("Input Features:", FEATURES)

# 링크 ID 검사
if dataset["link_id"].duplicated().any():
    raise ValueError("Duplicate link IDs.")

# 입력 데이터 검사
X = dataset[FEATURES]
y = dataset[TARGET]

print("\n===== Missing Values =====")
print(X.isna().sum())

print("\n===== Infinite Values =====")
print(np.isinf(X).sum())

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite feature values.")

if y.isna().any() or set(y.unique()) != {0, 1}:
    raise ValueError("Invalid target values.")

# 원본 피처와 비교
dataset["link_id"] = dataset["link_id"].astype(str)
original["link_id"] = original["link_id"].astype(str)

if original["link_id"].duplicated().any():
    raise ValueError("Duplicate original link IDs.")

if set(dataset["link_id"]) != set(original["link_id"]):
    raise ValueError("Link ID mismatch.")

current = dataset.set_index("link_id").sort_index()
reference = original.set_index("link_id").sort_index()

print("\n===== Original Feature Comparison =====")

for col in FEATURES:

    equal = np.array_equal(
        current[col].to_numpy(),
        reference[col].to_numpy()
    )

    print(f"{col}: {equal}")

    if not equal:
        raise ValueError(f"Feature mismatch: {col}")

# 정답 라벨 재검증
positive = matches[
    matches["overlap_length_m"] > 0
].copy()

positive_ids = set(
    positive["link_id"].astype(str)
)

expected_target = (
    dataset["link_id"]
    .isin(positive_ids)
    .astype(int)
)

if not np.array_equal(
    expected_target.to_numpy(),
    y.to_numpy()
):
    raise ValueError("Target mismatch.")

print("\n===== Target Validation =====")
print("Target Consistency: True")

# 물리적으로 부적절한 값 검사
print("\n===== Feature Range Check =====")

print("Lanes <= 0:", (X["lanes"] <= 0).sum())
print("Speed <= 0:", (X["max_spd"] <= 0).sum())
print("Length <= 0:", (X["length"] <= 0).sum())
print("Curvature < 1:", (X["curvature"] < 1).sum())

print(
    "Negative Intersection Count:",
    (X["n_intersect_200m"] < 0).sum()
)

print(
    "Negative Branch Count:",
    (X["n_branch"] < 0).sum()
)

# 극단값 확인
print("\n===== Feature Quantiles =====")

print(
    X.quantile([
        0.00,
        0.01,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
        1.00
    ]).T.to_string()
)

# 굴곡도 상위 10개
print("\n===== Top 10 Curvature =====")

print(
    dataset.nlargest(10, "curvature")[
        [
            "link_id",
            "region_code",
            "length",
            "curvature",
            "target"
        ]
    ].to_string(index=False)
)

# 교차로 후보 수 상위 10개
print("\n===== Top 10 Intersection Counts =====")

print(
    dataset.nlargest(10, "n_intersect_200m")[
        [
            "link_id",
            "region_code",
            "length",
            "n_intersect_200m",
            "target"
        ]
    ].to_string(index=False)
)

# 도로 등급별 라벨 분포
print("\n===== Road Rank Target Distribution =====")

rank_table = pd.crosstab(
    dataset["road_rank"],
    dataset[TARGET]
).reindex(columns=[0, 1], fill_value=0)

rank_table["total"] = (
    rank_table[0] + rank_table[1]
)

rank_table["positive_ratio_pct"] = (
    rank_table[1] / rank_table["total"] * 100
).round(2)

print(rank_table.to_string())

# 입력 피처 간 상관관계
# 범주형 코드인 road_rank는 제외
numeric_features = [
    col for col in FEATURES
    if col != "road_rank"
]

print("\n===== Feature Correlation =====")

print(
    dataset[numeric_features]
    .corr()
    .round(4)
    .to_string()
)

# 교통량 데이터 검토
print("\n===== Traffic Proxy Distribution =====")
print(traffic["traffic_is_proxy"].value_counts())

print("\n===== Traffic Proxy Ratio (%) =====")

print(
    traffic["traffic_is_proxy"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n===== Traffic Statistics =====")
print(traffic["traffic"].describe())

# 사고다발지역별 매칭 도로 개수
print("\n===== Links Per Hotspot =====")

hotspot_sizes = positive.groupby(
    "hotspot_id"
)["link_id"].nunique()

print(hotspot_sizes.describe())

# 여러 사고다발지역에 속하는 링크 확인
print("\n===== Hotspots Per Link =====")

link_hotspots = positive.groupby(
    "link_id"
)["hotspot_id"].nunique()

print(link_hotspots.value_counts().sort_index())

print("\n===== Validation Completed =====")