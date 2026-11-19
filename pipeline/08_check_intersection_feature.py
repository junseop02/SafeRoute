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

FEATURE_PATH = (
    PROCESSED_DIR
    / "intersection_features_cheonan_asan.parquet"
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print("=== SafeRoute n_intersect_200m 검증 ===")

print("\n[1] 데이터 불러오기...")

links = gpd.read_parquet(LINK_PATH)
nodes = gpd.read_parquet(NODE_PATH)
features = pd.read_parquet(FEATURE_PATH)

print("LINK:", len(links))
print("NODE:", len(nodes))
print("Feature:", len(features))

print("LINK CRS:", links.crs)
print("NODE CRS:", nodes.crs)


# ============================================================
# 3. ID 자료형 통일
# ============================================================

links["LINK_ID"] = links["LINK_ID"].astype(str)
links["F_NODE"] = links["F_NODE"].astype(str)
links["T_NODE"] = links["T_NODE"].astype(str)

nodes["NODE_ID"] = nodes["NODE_ID"].astype(str)
features["link_id"] = features["link_id"].astype(str)


# ============================================================
# 4. NODE degree 다시 계산
# ============================================================

print("\n[2] NODE degree 계산...")

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

nodes_check = nodes.merge(
    degree,
    on="NODE_ID",
    how="left"
)

nodes_check["degree"] = (
    nodes_check["degree"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 5. n_intersect_200m 상위 LINK 확인
# ============================================================

print("\n[3] n_intersect_200m 상위 20개 LINK")

top_links = (
    features
    .sort_values(
        "n_intersect_200m",
        ascending=False
    )
    .head(20)
    .copy()
)

# 원본 LINK 정보 결합
link_info = links[
    [
        "LINK_ID",
        "F_NODE",
        "T_NODE",
        "ROAD_NAME",
        "ROAD_RANK",
        "LANES",
        "MAX_SPD",
        "LENGTH",
        "geometry",
    ]
].copy()

link_info = link_info.rename(
    columns={
        "LINK_ID": "link_id"
    }
)

top_links = top_links.merge(
    link_info,
    on="link_id",
    how="left"
)

print(
    top_links[
        [
            "link_id",
            "region_code",
            "ROAD_NAME",
            "ROAD_RANK",
            "LANES",
            "MAX_SPD",
            "LENGTH",
            "n_intersect_200m",
            "n_branch",
        ]
    ].to_string(index=False)
)


# ============================================================
# 6. 상위 5개 LINK 주변 NODE 상세 검사
# ============================================================

print("\n[4] 상위 5개 LINK 주변 NODE 검사")

intersection_nodes = nodes_check[
    nodes_check["degree"] >= 3
].copy()

intersection_nodes = gpd.GeoDataFrame(
    intersection_nodes,
    geometry="geometry",
    crs=nodes.crs
)


for _, row in top_links.head(5).iterrows():

    link_id = row["link_id"]

    # 해당 LINK geometry
    link_row = links[
        links["LINK_ID"] == link_id
    ].iloc[0]

    # LINK 중심점
    center = link_row.geometry.interpolate(
        0.5,
        normalized=True
    )

    # 중심점에서 200m Buffer
    buffer_200m = center.buffer(200)

    # Buffer 내부의 degree >= 3 NODE
    nearby = intersection_nodes[
        intersection_nodes.geometry.within(
            buffer_200m
        )
    ].copy()

    # 중심점과 각 NODE 거리
    nearby["distance_m"] = (
        nearby.geometry.distance(center)
    )

    nearby = nearby.sort_values(
        "distance_m"
    )

    print("\n" + "=" * 70)

    print("LINK_ID:", link_id)
    print("지역코드:", row["region_code"])
    print("도로명:", row["ROAD_NAME"])
    print("도로등급:", row["ROAD_RANK"])
    print("차로수:", row["LANES"])
    print("제한속도:", row["MAX_SPD"])
    print("LINK 길이:", row["LENGTH"])

    print(
        "저장된 n_intersect_200m:",
        row["n_intersect_200m"]
    )

    print(
        "재계산 주변 NODE 수:",
        len(nearby)
    )

    print("\n[NODE_TYPE 분포]")

    print(
        nearby["NODE_TYPE"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\n[degree 분포]")

    print(
        nearby["degree"]
        .value_counts()
        .sort_index()
    )

    print("\n[가까운 NODE 30개]")

    print(
        nearby[
            [
                "NODE_ID",
                "NODE_TYPE",
                "NODE_NAME",
                "TURN_P",
                "degree",
                "distance_m",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# 7. n_intersect_200m 구간별 분포
# ============================================================

print("\n[5] n_intersect_200m 구간별 LINK 수")

bins = [
    -1,
    0,
    5,
    10,
    20,
    50,
    100,
    float("inf"),
]

labels = [
    "0",
    "1~5",
    "6~10",
    "11~20",
    "21~50",
    "51~100",
    "101 이상",
]

groups = pd.cut(
    features["n_intersect_200m"],
    bins=bins,
    labels=labels
)

print(
    groups
    .value_counts()
    .sort_index()
)


# ============================================================
# 8. NODE_TYPE별 degree >= 3 개수
# ============================================================

print("\n[6] degree >= 3 NODE의 NODE_TYPE 분포")

print(
    intersection_nodes["NODE_TYPE"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 9. NODE_TYPE=101만 사용했을 때 비교
#
# 공식 NODE_TYPE=101(교차로)을 이용했을 경우
# 얼마나 차이가 발생하는지 참고용으로만 비교
#
# 현재 Feature 정의를 변경하는 코드는 아님
# ============================================================

print("\n[7] 참고: NODE_TYPE=101 + degree>=3")

type101_intersections = intersection_nodes[
    intersection_nodes["NODE_TYPE"].astype(str)
    == "101"
]

print(
    "기존 degree >= 3 NODE:",
    len(intersection_nodes)
)

print(
    "NODE_TYPE=101 + degree>=3 NODE:",
    len(type101_intersections)
)

difference = (
    len(intersection_nodes)
    - len(type101_intersections)
)

print(
    "차이:",
    difference
)


# ============================================================
# 10. 완료
# ============================================================

print("\n=== n_intersect_200m 검증 완료 ===")