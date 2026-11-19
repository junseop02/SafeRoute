from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LINK_PATH = PROCESSED_DIR / "links_cheonan_asan.parquet"
NODE_PATH = PROCESSED_DIR / "nodes_cheonan_asan.parquet"

ROAD_FEATURE_PATH = (
    PROCESSED_DIR / "road_features_cheonan_asan.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "intersection_features_cheonan_asan.parquet"
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print("=== SafeRoute 교차로 Feature 생성 ===")

print("\n[1] 데이터 불러오기...")

links = gpd.read_parquet(LINK_PATH)
nodes = gpd.read_parquet(NODE_PATH)
features = pd.read_parquet(ROAD_FEATURE_PATH)

print("LINK:", len(links))
print("NODE:", len(nodes))
print("기존 Feature:", len(features))

print("LINK CRS:", links.crs)
print("NODE CRS:", nodes.crs)


# ============================================================
# 3. ID 자료형 통일
# ============================================================

print("\n[2] ID 자료형 통일...")

links["LINK_ID"] = links["LINK_ID"].astype(str)
links["F_NODE"] = links["F_NODE"].astype(str)
links["T_NODE"] = links["T_NODE"].astype(str)

nodes["NODE_ID"] = nodes["NODE_ID"].astype(str)

features["link_id"] = features["link_id"].astype(str)


# ============================================================
# 4. NODE degree 계산
#
# degree =
# 해당 NODE에 연결된 전체 LINK 수
#
# F_NODE와 T_NODE에 등장한 횟수를 모두 합산
# ============================================================

print("\n[3] NODE degree 계산...")

all_node_ids = pd.concat(
    [
        links["F_NODE"],
        links["T_NODE"],
    ],
    ignore_index=True
)

degree = (
    all_node_ids
    .value_counts()
    .rename_axis("NODE_ID")
    .reset_index(name="degree")
)

nodes_with_degree = nodes.merge(
    degree,
    on="NODE_ID",
    how="left"
)

nodes_with_degree["degree"] = (
    nodes_with_degree["degree"]
    .fillna(0)
    .astype(int)
)

print("degree 계산 완료")

print("\n[degree 분포]")
print(
    nodes_with_degree["degree"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. 교차로 후보 NODE 추출
#
# 확정 정의:
# degree >= 3인 NODE를 교차로 후보로 사용
# ============================================================

print("\n[4] 교차로 후보 NODE 추출...")

intersection_nodes = nodes_with_degree[
    nodes_with_degree["degree"] >= 3
].copy()

print(
    "degree >= 3 교차로 후보:",
    len(intersection_nodes)
)


# ============================================================
# 6. LINK 중심점 생성
#
# n_intersect_200m 계산 기준점
# ============================================================

print("\n[5] LINK 중심점 생성...")

link_centers = gpd.GeoDataFrame(
    {
        "link_id": links["LINK_ID"],
    },
    geometry=links.geometry.interpolate(
        0.5,
        normalized=True
    ),
    crs=links.crs
)

print(
    "LINK 중심점:",
    len(link_centers)
)


# ============================================================
# 7. LINK 중심점 기준 200m Buffer 생성
# ============================================================

print("\n[6] LINK 중심점 200m Buffer 생성...")

link_buffers = link_centers.copy()

link_buffers["geometry"] = (
    link_buffers.geometry.buffer(200)
)


# ============================================================
# 8. 200m 이내 교차로 NODE 공간조인
#
# n_intersect_200m =
# LINK 중심점 200m 이내의
# degree >= 3 NODE 개수
# ============================================================

print("\n[7] n_intersect_200m 계산...")

intersection_gdf = gpd.GeoDataFrame(
    intersection_nodes[
        [
            "NODE_ID",
            "degree",
        ]
    ].copy(),
    geometry=intersection_nodes.geometry,
    crs=nodes.crs
)

joined = gpd.sjoin(
    link_buffers,
    intersection_gdf,
    how="left",
    predicate="contains"
)

intersection_count = (
    joined[
        joined["NODE_ID"].notna()
    ]
    .groupby("link_id")
    .size()
    .rename("n_intersect_200m")
    .reset_index()
)

link_intersection_features = (
    link_centers[
        ["link_id"]
    ]
    .merge(
        intersection_count,
        on="link_id",
        how="left"
    )
)

link_intersection_features[
    "n_intersect_200m"
] = (
    link_intersection_features[
        "n_intersect_200m"
    ]
    .fillna(0)
    .astype(int)
)

print(
    "n_intersect_200m 계산 완료"
)

print("\n[n_intersect_200m 통계]")

print(
    link_intersection_features[
        "n_intersect_200m"
    ].describe()
)


# ============================================================
# 9. n_branch 계산
#
# n_branch =
# 현재 LINK의 T_NODE에서
# 출발하는 LINK 개수
# ============================================================

print("\n[8] n_branch 계산...")

outgoing_count = (
    links["F_NODE"]
    .value_counts()
    .rename_axis("T_NODE")
    .reset_index(name="n_branch")
)

branch_features = pd.DataFrame({
    "link_id": links["LINK_ID"],
    "T_NODE": links["T_NODE"],
})

branch_features = branch_features.merge(
    outgoing_count,
    on="T_NODE",
    how="left"
)

branch_features["n_branch"] = (
    branch_features["n_branch"]
    .fillna(0)
    .astype(int)
)

print("n_branch 계산 완료")

print("\n[n_branch 분포]")

print(
    branch_features["n_branch"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. 기존 Feature에 결합
# ============================================================

print("\n[9] 기존 Feature와 결합...")

features = features.merge(
    link_intersection_features[
        [
            "link_id",
            "n_intersect_200m",
        ]
    ],
    on="link_id",
    how="left"
)

features = features.merge(
    branch_features[
        [
            "link_id",
            "n_branch",
        ]
    ],
    on="link_id",
    how="left"
)


# ============================================================
# 11. 결과 검사
# ============================================================

print("\n[10] 결과 검사...")

print(
    "전체 행:",
    len(features)
)

print(
    "link_id 중복:",
    features["link_id"]
    .duplicated()
    .sum()
)

print("\n[컬럼]")

print(
    features.columns.tolist()
)

print("\n[결측치]")

print(
    features.isna().sum()
)

print("\n[n_intersect_200m 통계]")

print(
    features["n_intersect_200m"]
    .describe()
)

print("\n[n_branch 통계]")

print(
    features["n_branch"]
    .describe()
)

print("\n[샘플]")

print(
    features.head(10)
    .to_string(index=False)
)


# ============================================================
# 12. 저장
# ============================================================

print("\n[11] 저장...")

features.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    "저장:",
    OUTPUT_PATH
)

print(
    "파일 존재:",
    OUTPUT_PATH.exists()
)

print(
    "\n=== 교차로 Feature 생성 완료 ==="
)