from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NODE_PATH = PROCESSED_DIR / "nodes_cheonan_asan.parquet"
LINK_PATH = PROCESSED_DIR / "links_cheonan_asan.parquet"


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print("=== SafeRoute NODE 구조 검사 ===")

print("\n[1] 데이터 불러오기...")

nodes = gpd.read_parquet(NODE_PATH)
links = gpd.read_parquet(LINK_PATH)

print("NODE:", len(nodes))
print("LINK:", len(links))

print("\n[NODE 컬럼]")
print(nodes.columns.tolist())


# ============================================================
# 3. NODE_TYPE 분포 확인
# ============================================================

print("\n[2] NODE_TYPE 값 분포")

print(
    nodes["NODE_TYPE"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 4. TURN_P 분포 확인
# ============================================================

print("\n[3] TURN_P 값 분포")

print(
    nodes["TURN_P"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 5. NODE_NAME 결측 및 주요 값 확인
# ============================================================

print("\n[4] NODE_NAME 검사")

print(
    "NODE_NAME 결측:",
    nodes["NODE_NAME"].isna().sum()
)

print("\nNODE_NAME 상위 30개")

print(
    nodes["NODE_NAME"]
    .value_counts(dropna=False)
    .head(30)
)


# ============================================================
# 6. NODE_TYPE별 샘플 확인
# ============================================================

print("\n[5] NODE_TYPE별 샘플")

node_types = (
    nodes["NODE_TYPE"]
    .dropna()
    .unique()
)

for node_type in sorted(node_types):

    print(
        f"\n--- NODE_TYPE = {node_type} ---"
    )

    sample = nodes[
        nodes["NODE_TYPE"] == node_type
    ][
        [
            "NODE_ID",
            "NODE_TYPE",
            "NODE_NAME",
            "TURN_P",
        ]
    ].head(10)

    print(
        sample.to_string(index=False)
    )


# ============================================================
# 7. LINK 연결 개수 기반 NODE degree 계산
# ============================================================

print("\n[6] NODE 연결 차수 계산")

links["F_NODE"] = links["F_NODE"].astype(str)
links["T_NODE"] = links["T_NODE"].astype(str)
nodes["NODE_ID"] = nodes["NODE_ID"].astype(str)


# 각 NODE에 연결된 LINK 수
all_nodes = pd.concat(
    [
        links["F_NODE"],
        links["T_NODE"],
    ],
    ignore_index=True
)

degree = (
    all_nodes
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


print("\n[degree 통계]")

print(
    nodes_check["degree"]
    .describe()
)


print("\n[degree 값 분포]")

print(
    nodes_check["degree"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 8. NODE_TYPE과 degree 관계 확인
# ============================================================

print("\n[7] NODE_TYPE × degree")

cross_table = pd.crosstab(
    nodes_check["NODE_TYPE"],
    nodes_check["degree"]
)

print(cross_table.to_string())


# ============================================================
# 9. 교차로 후보 확인
#    합의서 기준: degree >= 3
# ============================================================

intersection_nodes = nodes_check[
    nodes_check["degree"] >= 3
]

print("\n[8] degree >= 3 교차로 후보")

print(
    "교차로 후보 NODE:",
    len(intersection_nodes)
)

print("\nNODE_TYPE 분포")

print(
    intersection_nodes["NODE_TYPE"]
    .value_counts(dropna=False)
    .sort_index()
)


print("\n교차로 후보 샘플")

print(
    intersection_nodes[
        [
            "NODE_ID",
            "NODE_TYPE",
            "NODE_NAME",
            "TURN_P",
            "degree",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 10. 요약
# ============================================================

print("\n=== 요약 ===")

print(
    "전체 NODE:",
    len(nodes_check)
)

print(
    "degree >= 3 NODE:",
    len(intersection_nodes)
)

print(
    "NODE_TYPE 종류:",
    nodes["NODE_TYPE"].nunique(dropna=True)
)

print(
    "TURN_P 종류:",
    nodes["TURN_P"].nunique(dropna=True)
)

print("\n=== NODE 구조 검사 완료 ===")