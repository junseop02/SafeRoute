from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAFFIC_PATH = (
    PROCESSED_DIR
    / "traffic_points_cheonan_asan.parquet"
)

LINK_PATH = (
    PROCESSED_DIR
    / "links_cheonan_asan.parquet"
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print(
    "=== SafeRoute AADT ↔ LINK 매칭 검증 ==="
)

print("\n[1] 데이터 불러오기...")

traffic = gpd.read_parquet(
    TRAFFIC_PATH
)

links = gpd.read_parquet(
    LINK_PATH
)

print(
    "AADT 측정지점:",
    len(traffic)
)

print(
    "LINK:",
    len(links)
)

print(
    "AADT CRS:",
    traffic.crs
)

print(
    "LINK CRS:",
    links.crs
)


# ============================================================
# 3. CRS 통일
# ============================================================

print("\n[2] CRS 확인...")

if traffic.crs != links.crs:

    print(
        "CRS가 달라 LINK CRS로 변환합니다."
    )

    traffic = traffic.to_crs(
        links.crs
    )

print(
    "최종 AADT CRS:",
    traffic.crs
)

print(
    "최종 LINK CRS:",
    links.crs
)


# ============================================================
# 4. LINK 컬럼 확인
# ============================================================

print("\n[3] LINK 컬럼 확인")

print(
    links.columns.tolist()
)


# ============================================================
# 5. 필요한 LINK 정보 준비
# ============================================================

link_columns = [
    "LINK_ID",
    "ROAD_NAME",
    "ROAD_RANK",
    "ROAD_NO",
    "LANES",
    "MAX_SPD",
    "LENGTH",
    "geometry",
]

missing_columns = [
    column
    for column in link_columns
    if column not in links.columns
]

if missing_columns:
    raise ValueError(
        "LINK에 필요한 컬럼이 없습니다: "
        + str(missing_columns)
    )

links_for_match = links[
    link_columns
].copy()


# ============================================================
# 6. 가장 가까운 LINK 찾기
#
# EPSG:5179이므로 거리 단위는 meter
# ============================================================

print(
    "\n[4] 각 AADT 지점의 가장 가까운 LINK 탐색..."
)

matched = gpd.sjoin_nearest(
    traffic,
    links_for_match,
    how="left",
    distance_col="distance_m"
)

print(
    "매칭 결과 행:",
    len(matched)
)


# ============================================================
# 7. 하나의 측정점에 여러 LINK가 동거리로 잡혔는지 확인
# ============================================================

print(
    "\n[5] 최근접 LINK 중복 매칭 검사"
)

# 원래 traffic index를 측정점 식별용으로 사용
matched = matched.reset_index().rename(
    columns={
        "index": "traffic_index"
    }
)

match_count = (
    matched
    .groupby("traffic_index")
    .size()
)

print(
    "원본 AADT 지점:",
    len(traffic)
)

print(
    "최근접 매칭 결과:",
    len(matched)
)

print(
    "1개 LINK만 매칭된 측정점:",
    (match_count == 1).sum()
)

print(
    "2개 이상 LINK가 동거리로 매칭된 측정점:",
    (match_count > 1).sum()
)

if (match_count > 1).any():

    print(
        "\n[동거리 중복 매칭 개수 분포]"
    )

    print(
        match_count[
            match_count > 1
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# 8. 거리 통계
# ============================================================

print(
    "\n[6] 최근접 LINK 거리 통계"
)

print(
    matched[
        "distance_m"
    ].describe()
)


# ============================================================
# 9. 거리 구간별 분포
# ============================================================

print(
    "\n[7] 거리 구간별 매칭 수"
)

distance_bins = pd.cut(
    matched["distance_m"],
    bins=[
        -0.001,
        5,
        10,
        20,
        30,
        50,
        100,
        float("inf"),
    ],
    labels=[
        "0~5m",
        "5~10m",
        "10~20m",
        "20~30m",
        "30~50m",
        "50~100m",
        "100m 초과",
    ]
)

print(
    distance_bins
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 10. 매칭 결과 전체 출력
# ============================================================

print(
    "\n[8] AADT ↔ 최근접 LINK 매칭 결과"
)

display_columns = [
    "traffic_index",
    "region_code",
    "도로종류",
    "지점번호",
    "호선",
    "구간",
    "차로",
    "AADT",
    "LINK_ID",
    "ROAD_NAME",
    "ROAD_RANK",
    "ROAD_NO",
    "LANES",
    "MAX_SPD",
    "LENGTH",
    "distance_m",
]

print(
    matched[
        display_columns
    ]
    .sort_values(
        "distance_m"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 11. 거리가 먼 매칭 별도 확인
# ============================================================

print(
    "\n[9] 최근접 LINK까지 30m 초과인 측정점"
)

far_30 = matched[
    matched[
        "distance_m"
    ] > 30
].copy()

print(
    "30m 초과:",
    len(far_30)
)

if len(far_30) > 0:

    print(
        far_30[
            display_columns
        ]
        .sort_values(
            "distance_m",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


print(
    "\n[10] 최근접 LINK까지 50m 초과인 측정점"
)

far_50 = matched[
    matched[
        "distance_m"
    ] > 50
].copy()

print(
    "50m 초과:",
    len(far_50)
)

if len(far_50) > 0:

    print(
        far_50[
            display_columns
        ]
        .sort_values(
            "distance_m",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 12. 도로 번호 일치 여부 참고 검사
#
# AADT의 '호선'과 국가표준노드링크 ROAD_NO를
# 바로 같은 값이라고 단정하지 않고 참고용으로만 확인
# ============================================================

print(
    "\n[11] AADT 호선 ↔ LINK ROAD_NO 참고 비교"
)


def normalize_road_number(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    # Excel에서 1.0처럼 읽힌 경우
    if text.endswith(".0"):
        text = text[:-2]

    # ROAD_NO 등에 앞자리 0이 있을 가능성을 고려
    text = text.lstrip("0")

    if text == "":
        return "0"

    return text


matched[
    "aadt_road_no_norm"
] = matched[
    "호선"
].apply(
    normalize_road_number
)

matched[
    "link_road_no_norm"
] = matched[
    "ROAD_NO"
].apply(
    normalize_road_number
)

matched[
    "road_no_match"
] = (
    matched[
        "aadt_road_no_norm"
    ]
    == matched[
        "link_road_no_norm"
    ]
)

valid_compare = matched[
    matched[
        "aadt_road_no_norm"
    ].notna()
    &
    matched[
        "link_road_no_norm"
    ].notna()
].copy()

print(
    "비교 가능 행:",
    len(valid_compare)
)

print(
    "호선/ROAD_NO 일치:",
    valid_compare[
        "road_no_match"
    ].sum()
)

print(
    "호선/ROAD_NO 불일치:",
    (
        ~valid_compare[
            "road_no_match"
        ]
    ).sum()
)


# ============================================================
# 13. 불일치 샘플
# ============================================================

print(
    "\n[12] 호선/ROAD_NO 불일치 샘플"
)

road_no_mismatch = valid_compare[
    ~valid_compare[
        "road_no_match"
    ]
].copy()

mismatch_columns = [
    "도로종류",
    "지점번호",
    "호선",
    "구간",
    "AADT",
    "LINK_ID",
    "ROAD_NAME",
    "ROAD_RANK",
    "ROAD_NO",
    "LANES",
    "distance_m",
]

if len(road_no_mismatch) == 0:

    print(
        "불일치 없음"
    )

else:

    print(
        road_no_mismatch[
            mismatch_columns
        ]
        .sort_values(
            "distance_m"
        )
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# 14. 차로수 비교
#
# 이것도 매칭 판정 기준으로 확정하지 않고
# 참고용으로만 확인
# ============================================================

print(
    "\n[13] AADT 차로 ↔ LINK LANES 참고 비교"
)

lane_compare = matched[
    matched["차로"].notna()
    &
    matched["LANES"].notna()
].copy()

lane_compare[
    "lane_difference"
] = (
    lane_compare["차로"].astype(float)
    - lane_compare["LANES"].astype(float)
).abs()

print(
    lane_compare[
        "lane_difference"
    ].describe()
)

print(
    "\n차로수 완전 일치:",
    (
        lane_compare[
            "lane_difference"
        ] == 0
    ).sum()
)

print(
    "차로수 차이 1:",
    (
        lane_compare[
            "lane_difference"
        ] == 1
    ).sum()
)

print(
    "차로수 차이 2 이상:",
    (
        lane_compare[
            "lane_difference"
        ] >= 2
    ).sum()
)


# ============================================================
# 15. 최종 요약
# ============================================================

print(
    "\n=== 매칭 검증 요약 ==="
)

print(
    "AADT 원본 지점:",
    len(traffic)
)

print(
    "최근접 매칭 결과 행:",
    len(matched)
)

print(
    "최근접 거리 중앙값:",
    matched[
        "distance_m"
    ].median()
)

print(
    "최근접 거리 최대값:",
    matched[
        "distance_m"
    ].max()
)

print(
    "30m 초과:",
    len(far_30)
)

print(
    "50m 초과:",
    len(far_50)
)

print(
    "동거리 다중 매칭 측정점:",
    (match_count > 1).sum()
)

print(
    "\n=== AADT ↔ LINK 매칭 검증 완료 ==="
)