from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LINK_PATH = PROCESSED_DIR / "links_cheonan_asan.parquet"


# ============================================================
# 2. LINK 불러오기
# ============================================================

print("=== SafeRoute is_oneway 구조 검사 ===")

links = gpd.read_parquet(LINK_PATH)

print("\n전체 LINK:", len(links))

# ID는 문자열로 통일
links["LINK_ID"] = links["LINK_ID"].astype(str)
links["F_NODE"] = links["F_NODE"].astype(str)
links["T_NODE"] = links["T_NODE"].astype(str)


# ============================================================
# 3. 관련 원본 컬럼 분포 확인
# ============================================================

check_columns = [
    "ROAD_USE",
    "MULTI_LINK",
    "CONNECT",
    "ROAD_TYPE",
    "ROAD_RANK",
]

for col in check_columns:
    print(f"\n[{col} 값 분포]")
    print(
        links[col]
        .value_counts(dropna=False)
        .sort_index()
    )


# ============================================================
# 4. 현재 방식으로 reverse link 확인
# ============================================================

directed_pairs = set(
    zip(
        links["F_NODE"],
        links["T_NODE"]
    )
)

links["has_exact_reverse"] = [
    (t_node, f_node) in directed_pairs
    for f_node, t_node in zip(
        links["F_NODE"],
        links["T_NODE"]
    )
]

print("\n[정확히 반대 NODE 쌍 존재 여부]")
print(
    links["has_exact_reverse"]
    .value_counts(dropna=False)
)

print("\n정확한 reverse link가 있는 LINK 샘플")

reverse_sample = links[
    links["has_exact_reverse"]
][
    [
        "LINK_ID",
        "F_NODE",
        "T_NODE",
        "LANES",
        "ROAD_RANK",
        "ROAD_TYPE",
        "ROAD_USE",
        "MULTI_LINK",
        "CONNECT",
        "ROAD_NAME",
    ]
]

print(reverse_sample.head(30).to_string(index=False))


# ============================================================
# 5. 동일 도로명에서 NODE 구조 확인
# ============================================================

print("\n[도로명별 LINK 개수 상위 20개]")

print(
    links["ROAD_NAME"]
    .value_counts(dropna=False)
    .head(20)
)


# ============================================================
# 6. LINK_ID 끝자리 분포
# ============================================================

print("\n[LINK_ID 마지막 1자리 분포]")

print(
    links["LINK_ID"]
    .str[-1]
    .value_counts()
    .sort_index()
)


print("\n[LINK_ID 마지막 2자리 분포 상위 30개]")

print(
    links["LINK_ID"]
    .str[-2:]
    .value_counts()
    .head(30)
)


# ============================================================
# 7. NODE 연결 구조
# ============================================================

print("\n[NODE 출발 차수 통계]")

out_degree = (
    links.groupby("F_NODE")
    .size()
)

print(out_degree.describe())


print("\n[NODE 도착 차수 통계]")

in_degree = (
    links.groupby("T_NODE")
    .size()
)

print(in_degree.describe())


# ============================================================
# 8. 요약
# ============================================================

reverse_count = int(
    links["has_exact_reverse"].sum()
)

print("\n=== 요약 ===")

print("전체 LINK:", len(links))
print("정확한 reverse LINK 있음:", reverse_count)
print(
    "정확한 reverse LINK 없음:",
    len(links) - reverse_count
)

print(
    "reverse 존재 비율:",
    f"{reverse_count / len(links) * 100:.4f}%"
)

print("\n=== 검사 완료 ===")