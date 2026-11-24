from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/legacy/processed"

# 기존 도로 및 노드 데이터 불러오기
links = gpd.read_parquet(
    DATA_DIR / "links_cheonan_asan.parquet"
)

nodes = gpd.read_parquet(
    DATA_DIR / "nodes_cheonan_asan.parquet"
)

# 기존 전처리 특성 데이터 불러오기
road = pd.read_parquet(
    DATA_DIR / "road_features_cheonan_asan.parquet"
)

intersection = pd.read_parquet(
    DATA_DIR / "intersection_features_cheonan_asan.parquet"
)

traffic = pd.read_parquet(
    DATA_DIR / "traffic_features_cheonan_asan.parquet"
)

# 데이터 크기 확인
print("===== Dataset Shape =====")
print("Links:", links.shape)
print("Nodes:", nodes.shape)
print("Road Features:", road.shape)
print("Intersection Features:", intersection.shape)
print("Traffic Features:", traffic.shape)

# 좌표계 확인
print("\n===== CRS =====")
print("Link CRS:", links.crs)
print("Node CRS:", nodes.crs)
print("Link EPSG:", links.crs.to_epsg() if links.crs else None)
print("Node EPSG:", nodes.crs.to_epsg() if nodes.crs else None)

# 공간 데이터 검사
print("\n===== Geometry Check =====")
print("Link Geometry Types:")
print(links.geometry.geom_type.value_counts())

print("Missing Geometry:", links.geometry.isna().sum())
print("Empty Geometry:", links.geometry.is_empty.sum())
print("Invalid Geometry:", (~links.geometry.is_valid).sum())
print("Link Bounds:", links.total_bounds)

# 컬럼 확인
print("\n===== Link Columns =====")
print(links.columns.tolist())

print("\n===== Road Feature Columns =====")
print(road.columns.tolist())

print("\n===== Intersection Feature Columns =====")
print(intersection.columns.tolist())

print("\n===== Traffic Feature Columns =====")
print(traffic.columns.tolist())

# 링크 ID 검사
print("\n===== Link ID Check =====")

link_ids = links["LINK_ID"].astype(str)
road_ids = road["link_id"].astype(str)
intersection_ids = intersection["link_id"].astype(str)

print("Link ID Duplicates:", link_ids.duplicated().sum())
print("Road ID Duplicates:", road_ids.duplicated().sum())
print(
    "Intersection ID Duplicates:",
    intersection_ids.duplicated().sum()
)

print(
    "Links Missing Road Features:",
    len(set(link_ids) - set(road_ids))
)

print(
    "Links Missing Intersection Features:",
    len(set(link_ids) - set(intersection_ids))
)

# 새로운 모델의 입력 후보
feature_cols = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch"
]

# 필수 특성 존재 확인
missing_cols = [
    col for col in feature_cols
    if col not in intersection.columns
]

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

features = intersection[feature_cols]

# 결측치 확인
print("\n===== Missing Values =====")
print(features.isna().sum())

# 무한대 값 확인
print("\n===== Infinite Values =====")
print(np.isinf(features).sum())

# 기초 통계
print("\n===== Feature Statistics =====")
print(features.describe().T.to_string())

# 도로 등급 분포
print("\n===== Road Rank Distribution =====")
print(features["road_rank"].value_counts().sort_index())

# 지역별 도로 링크 수
print("\n===== Regional Distribution =====")
print(intersection["region_code"].value_counts())

# 교통량 특성 결측치 확인
print("\n===== Traffic Missing Values =====")
print(traffic.isna().sum())

# 데이터 표본
print("\n===== Feature Sample =====")
print(
    intersection[
        ["link_id", "region_code"] + feature_cols
    ].head(5).to_string(index=False)
)

print("\n===== Inspection Completed =====")