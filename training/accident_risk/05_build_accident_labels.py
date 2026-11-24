from pathlib import Path

import pandas as pd
import geopandas as gpd

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/legacy/processed"
OUTPUT_DIR = ROOT / "data/accident_risk/processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 기존 도로 및 교차로 특성 불러오기
features = pd.read_parquet(
    DATA_DIR / "intersection_features_cheonan_asan.parquet"
)

# 도로 공간 데이터 불러오기
links = gpd.read_parquet(
    DATA_DIR / "links_cheonan_asan.parquet"
)

# 사고다발지역 공간 매칭 결과 불러오기
matches = gpd.read_parquet(
    OUTPUT_DIR / "accident_link_matches.parquet"
)

# 링크 ID 자료형 통일
features["link_id"] = features["link_id"].astype(str)
matches["link_id"] = matches["link_id"].astype(str)

links = links.rename(
    columns={"LINK_ID": "link_id"}
)

links["link_id"] = links["link_id"].astype(str)

# 기본 데이터 확인
print("===== Input Data =====")
print("Road Features:", len(features))
print("Road Links:", len(links))
print("Matched Pairs:", len(matches))

# 링크 ID 중복 확인
if features["link_id"].duplicated().any():
    raise ValueError("Duplicate feature link IDs.")

if links["link_id"].duplicated().any():
    raise ValueError("Duplicate road link IDs.")

# 도로와 특성 데이터의 ID 일치 확인
if set(features["link_id"]) != set(links["link_id"]):
    raise ValueError("Road and feature IDs do not match.")

# 매칭 데이터의 링크 ID 검사
if not set(matches["link_id"]).issubset(
    set(features["link_id"])
):
    raise ValueError("Unknown matched link IDs.")

# 실제 중첩 길이가 존재하는 매칭 선택
positive = matches[
    matches["overlap_length_m"] > 0
].copy()

# 링크별 사고다발지역 매칭 정보 집계
summary = positive.groupby("link_id").agg(
    hotspot_count=("hotspot_id", "nunique"),
    max_overlap_length_m=("overlap_length_m", "max"),
    max_overlap_ratio=("overlap_ratio", "max"),
    first_accident_year=("사고연도", "min"),
    last_accident_year=("사고연도", "max")
).reset_index()

# 기존 도로 특성과 매칭 결과 결합
dataset = features.merge(
    summary,
    on="link_id",
    how="left",
    validate="one_to_one"
)

# 미매칭 도로의 매칭 정보 처리
fill_columns = [
    "hotspot_count",
    "max_overlap_length_m",
    "max_overlap_ratio"
]

dataset[fill_columns] = (
    dataset[fill_columns].fillna(0)
)

dataset["hotspot_count"] = (
    dataset["hotspot_count"].astype(int)
)

# 기본 정답 라벨 생성
dataset["target"] = (
    dataset["max_overlap_length_m"] > 0
).astype(int)

# 중첩 비율별 비교용 라벨 생성
thresholds = {
    "target_10": 0.10,
    "target_25": 0.25,
    "target_50": 0.50,
    "target_75": 0.75,
    "target_90": 0.90
}

for column, threshold in thresholds.items():

    dataset[column] = (
        dataset["max_overlap_ratio"] >= threshold
    ).astype(int)

# 신규 모델 입력 피처 지정
FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch"
]

# 필수 컬럼 확인
missing_columns = [
    col for col in FEATURES
    if col not in dataset.columns
]

if missing_columns:
    raise ValueError(
        f"Missing features: {missing_columns}"
    )

# 데이터 크기 검증
if len(dataset) != len(features):
    raise ValueError("Dataset row count changed.")

# 입력 피처 및 기본 라벨 결측치 검증
if dataset[FEATURES + ["target"]].isna().any().any():
    raise ValueError("Missing feature or target values.")

# 기본 라벨 분포
print("\n===== Target Distribution =====")
print(dataset["target"].value_counts().sort_index())

negative_count = int(
    (dataset["target"] == 0).sum()
)

positive_count = int(
    (dataset["target"] == 1).sum()
)

print("Negative:", negative_count)
print("Positive:", positive_count)

print(
    "Positive Ratio (%):",
    round(positive_count / len(dataset) * 100, 2)
)

# 클래스 불균형 비율
if positive_count == 0:
    raise ValueError("No positive labels.")

print(
    "Negative / Positive Ratio:",
    round(negative_count / positive_count, 4)
)

# 중첩 기준별 라벨 비교
print("\n===== Threshold Comparison =====")

for column in ["target"] + list(thresholds):

    count = int(dataset[column].sum())

    print(
        f"{column}: {count:,} positive links "
        f"({count / len(dataset) * 100:.2f}%)"
    )

# 지역별 라벨 분포
print("\n===== Regional Target Distribution =====")

print(
    pd.crosstab(
        dataset["region_code"],
        dataset["target"]
    )
)

# 라벨별 입력 피처 평균 비교
print("\n===== Feature Means by Target =====")

print(
    dataset.groupby("target")[FEATURES]
    .mean()
    .T
    .to_string()
)

# 공간 분할에 사용할 도로 중심점 생성
if links.crs is None:
    raise ValueError("Road CRS is missing.")

links = links.to_crs("EPSG:5179")

midpoints = links.geometry.interpolate(
    0.5,
    normalized=True
)

links["center_x"] = midpoints.x
links["center_y"] = midpoints.y

# 공간 검증용 좌표 결합
dataset = dataset.merge(
    links[["link_id", "center_x", "center_y"]],
    on="link_id",
    how="left",
    validate="one_to_one"
)

if dataset[["center_x", "center_y"]].isna().any().any():
    raise ValueError("Missing spatial coordinates.")

# 최종 데이터 구조 확인
print("\n===== Final Dataset =====")
print("Shape:", dataset.shape)
print("Columns:", dataset.columns.tolist())

# 학습 입력 변수 확인
print("\n===== Model Input Features =====")

for column in FEATURES:
    print(column)

# 데이터 표본 확인
print("\n===== Positive Sample =====")

print(
    dataset.loc[
        dataset["target"] == 1,
        ["link_id"] + FEATURES + ["target"]
    ].head().to_string(index=False)
)

print("\n===== Negative Sample =====")

print(
    dataset.loc[
        dataset["target"] == 0,
        ["link_id"] + FEATURES + ["target"]
    ].head().to_string(index=False)
)

# 학습용 데이터 저장
output_path = (
    OUTPUT_DIR / "accident_risk_dataset.parquet"
)

dataset.to_parquet(
    output_path,
    index=False
)

print("\n===== Output =====")
print("Saved:", output_path)

print("\n===== Label Generation Completed =====")