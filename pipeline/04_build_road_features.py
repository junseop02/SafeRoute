from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LINK_PATH = PROCESSED_DIR / "links_cheonan_asan.parquet"
NODE_PATH = PROCESSED_DIR / "nodes_cheonan_asan.parquet"

OUTPUT_PATH = PROCESSED_DIR / "road_features_cheonan_asan.parquet"


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print("=== SafeRoute 기본 도로 Feature 생성 ===")

print("\n[1] 데이터 불러오는 중...")

links = gpd.read_parquet(LINK_PATH)
nodes = gpd.read_parquet(NODE_PATH)

print("LINK:", len(links))
print("NODE:", len(nodes))
print("LINK CRS:", links.crs)
print("NODE CRS:", nodes.crs)


# ============================================================
# 3. 기본 도로 Feature 생성
# ============================================================

print("\n[2] 기본 도로 Feature 생성...")

features = pd.DataFrame({
    "link_id": links["LINK_ID"].astype(str),
    "region_code": links["region_code"].astype(int),
    "lanes": pd.to_numeric(
        links["LANES"],
        errors="coerce"
    ),
    "max_spd": pd.to_numeric(
        links["MAX_SPD"],
        errors="coerce"
    ),
    "road_rank": pd.to_numeric(
        links["ROAD_RANK"],
        errors="coerce"
    ),
    "length": pd.to_numeric(
        links["LENGTH"],
        errors="coerce"
    ),
})

print(features.head())


# ============================================================
# 4. NODE 좌표 준비
# ============================================================

print("\n[3] NODE 좌표 준비...")

node_xy = pd.DataFrame({
    "NODE_ID": nodes["NODE_ID"].astype(str),
    "x": nodes.geometry.x,
    "y": nodes.geometry.y,
})

f_node_xy = node_xy.rename(
    columns={
        "NODE_ID": "F_NODE",
        "x": "f_x",
        "y": "f_y",
    }
)

t_node_xy = node_xy.rename(
    columns={
        "NODE_ID": "T_NODE",
        "x": "t_x",
        "y": "t_y",
    }
)


# ============================================================
# 5. LINK에 시작/끝 NODE 좌표 연결
# ============================================================

print("\n[4] LINK와 NODE 좌표 연결...")

geometry_data = pd.DataFrame({
    "link_id": links["LINK_ID"].astype(str),
    "F_NODE": links["F_NODE"].astype(str),
    "T_NODE": links["T_NODE"].astype(str),
    "length": pd.to_numeric(
        links["LENGTH"],
        errors="coerce"
    ),
})

geometry_data = geometry_data.merge(
    f_node_xy,
    on="F_NODE",
    how="left"
)

geometry_data = geometry_data.merge(
    t_node_xy,
    on="T_NODE",
    how="left"
)

print(
    "시작 NODE 좌표 누락:",
    geometry_data[["f_x", "f_y"]]
    .isna()
    .any(axis=1)
    .sum()
)

print(
    "끝 NODE 좌표 누락:",
    geometry_data[["t_x", "t_y"]]
    .isna()
    .any(axis=1)
    .sum()
)


# ============================================================
# 6. curvature 계산
#
# curvature =
# 실제 도로 길이 / 시작-끝 NODE 직선거리
# ============================================================

print("\n[5] curvature 계산...")

geometry_data["straight_distance"] = np.sqrt(
    (geometry_data["t_x"] - geometry_data["f_x"]) ** 2
    +
    (geometry_data["t_y"] - geometry_data["f_y"]) ** 2
)

geometry_data["curvature"] = (
    geometry_data["length"]
    / geometry_data["straight_distance"]
)

# 시작점과 끝점의 직선거리가 0인 경우
geometry_data.loc[
    geometry_data["straight_distance"] <= 0,
    "curvature"
] = np.nan


# 수치 오차로 인해 1보다 아주 조금 작아지는 경우만 보정
small_error = (
    geometry_data["curvature"].notna()
    & (geometry_data["curvature"] < 1)
    & (geometry_data["curvature"] >= 0.99)
)

geometry_data.loc[
    small_error,
    "curvature"
] = 1.0


# ============================================================
# 7. curvature를 Feature에 결합
# ============================================================

features = features.merge(
    geometry_data[
        [
            "link_id",
            "curvature",
        ]
    ],
    on="link_id",
    how="left"
)


# ============================================================
# 8. 결과 검사
# ============================================================

print("\n[6] 결과 검사...")

print("전체 행:", len(features))
print(
    "link_id 중복:",
    features["link_id"].duplicated().sum()
)

print("\n[컬럼]")
print(features.columns.tolist())

print("\n[결측치]")
print(features.isna().sum())

print("\n[curvature 통계]")
print(
    features["curvature"].describe()
)

print(
    "\ncurvature < 1:",
    (features["curvature"] < 1).sum()
)

print(
    "curvature 결측:",
    features["curvature"].isna().sum()
)


# ============================================================
# 9. 저장
# ============================================================

print("\n[7] 저장...")

features.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("저장:", OUTPUT_PATH)
print(
    "파일 존재:",
    OUTPUT_PATH.exists()
)

print("\n=== 기본 도로 Feature 생성 완료 ===")