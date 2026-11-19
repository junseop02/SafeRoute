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
    "=== SafeRoute AADT 주변 동일 ROAD_NO LINK 검사 ==="
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

    traffic = traffic.to_crs(
        links.crs
    )

print(
    "최종 CRS:",
    links.crs
)


# ============================================================
# 4. 도로번호 정규화 함수
#
# AADT 예:
# 1.0  -> 1
# 21.0 -> 21
# 25.2 -> 25
# 17.4 -> 17
#
# 고속국도에서 25.2, 17.4처럼 들어온 값은
# LINK ROAD_NO가 25, 17 형태이므로 정수부를 사용합니다.
#
# 이번 단계는 매칭 규칙을 확정하는 단계가 아니라
# 후보 LINK 분포를 확인하기 위한 검증입니다.
# ============================================================

def normalize_road_number(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        number = float(text)

        return str(
            int(number)
        )

    except ValueError:

        text = text.lstrip("0")

        if text == "":
            return "0"

        return text


# ============================================================
# 5. AADT / LINK 도로번호 정규화
# ============================================================

print(
    "\n[3] ROAD_NO 정규화..."
)

traffic[
    "aadt_road_no"
] = traffic[
    "호선"
].apply(
    normalize_road_number
)

links[
    "link_road_no"
] = links[
    "ROAD_NO"
].apply(
    normalize_road_number
)

print(
    "AADT 도로번호 결측:",
    traffic[
        "aadt_road_no"
    ].isna().sum()
)

print(
    "LINK 도로번호 결측:",
    links[
        "link_road_no"
    ].isna().sum()
)


# ============================================================
# 6. LINK 공간 인덱스 생성
# ============================================================

print(
    "\n[4] LINK 공간 인덱스 준비..."
)

_ = links.sindex

print(
    "공간 인덱스 준비 완료"
)


# ============================================================
# 7. 측정점별 거리 범위 검사
#
# 각 AADT 측정점 주변에서
# 동일 ROAD_NO를 가진 LINK 개수를 검사
#
# 100m / 300m / 500m / 1000m
# ============================================================

print(
    "\n[5] 측정점별 동일 ROAD_NO LINK 검사..."
)

distance_ranges = [
    100,
    300,
    500,
    1000,
]

results = []


for traffic_index, point in traffic.iterrows():

    road_no = point[
        "aadt_road_no"
    ]

    if road_no is None:
        continue

    same_road_links = links[
        links[
            "link_road_no"
        ] == road_no
    ]

    row = {
        "traffic_index": traffic_index,
        "region_code": point[
            "region_code"
        ],
        "도로종류": point[
            "도로종류"
        ],
        "지점번호": point[
            "지점번호"
        ],
        "호선": point[
            "호선"
        ],
        "road_no_norm": road_no,
        "구간": point[
            "구간"
        ],
        "AADT": point[
            "AADT"
        ],
        "same_road_total": len(
            same_road_links
        ),
    }

    for distance in distance_ranges:

        # 측정점 중심 buffer
        search_area = (
            point.geometry.buffer(
                distance
            )
        )

        candidate_index = list(
            same_road_links.sindex.query(
                search_area,
                predicate="intersects"
            )
        )

        candidate_links = (
            same_road_links.iloc[
                candidate_index
            ].copy()
        )

        if len(candidate_links) > 0:

            candidate_links[
                "distance_m"
            ] = candidate_links.geometry.distance(
                point.geometry
            )

            candidate_links = candidate_links[
                candidate_links[
                    "distance_m"
                ] <= distance
            ]

        row[
            f"links_{distance}m"
        ] = len(
            candidate_links
        )

    results.append(
        row
    )


result_df = pd.DataFrame(
    results
)

print(
    "검사 완료:",
    len(result_df)
)


# ============================================================
# 8. 전체 통계
# ============================================================

print(
    "\n[6] 거리별 동일 ROAD_NO LINK 개수 통계"
)

for distance in distance_ranges:

    column = (
        f"links_{distance}m"
    )

    print(
        f"\n--- {distance}m ---"
    )

    print(
        result_df[
            column
        ].describe()
    )


# ============================================================
# 9. 거리별 0개인 측정점 검사
# ============================================================

print(
    "\n[7] 동일 ROAD_NO LINK를 찾지 못한 측정점"
)

for distance in distance_ranges:

    column = (
        f"links_{distance}m"
    )

    zero_count = (
        result_df[
            column
        ] == 0
    ).sum()

    print(
        f"{distance}m: {zero_count}"
    )


# ============================================================
# 10. 전체 결과
# ============================================================

print(
    "\n[8] 측정점별 결과"
)

display_columns = [
    "traffic_index",
    "region_code",
    "도로종류",
    "지점번호",
    "호선",
    "road_no_norm",
    "구간",
    "AADT",
    "same_road_total",
    "links_100m",
    "links_300m",
    "links_500m",
    "links_1000m",
]

print(
    result_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 11. 100m 안에 후보가 없는 지점
# ============================================================

print(
    "\n[9] 100m 안에 동일 ROAD_NO LINK가 없는 지점"
)

no_100m = result_df[
    result_df[
        "links_100m"
    ] == 0
]

print(
    "개수:",
    len(no_100m)
)

if len(no_100m) > 0:

    print(
        no_100m[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# 12. 300m 안에 후보가 너무 많은 지점
#
# 후보가 많으면 단순 원형 buffer 전파가
# 과도하게 많은 LINK를 포함할 수 있음
# ============================================================

print(
    "\n[10] 300m 안에 동일 ROAD_NO LINK가 많은 지점"
)

many_300m = result_df[
    result_df[
        "links_300m"
    ] >= 10
].copy()

print(
    "10개 이상:",
    len(many_300m)
)

if len(many_300m) > 0:

    print(
        many_300m[
            display_columns
        ]
        .sort_values(
            "links_300m",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 13. 500m 안에 후보가 너무 많은 지점
# ============================================================

print(
    "\n[11] 500m 안에 동일 ROAD_NO LINK가 많은 지점"
)

many_500m = result_df[
    result_df[
        "links_500m"
    ] >= 20
].copy()

print(
    "20개 이상:",
    len(many_500m)
)

if len(many_500m) > 0:

    print(
        many_500m[
            display_columns
        ]
        .sort_values(
            "links_500m",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 14. 대표 측정점 상세 검사
#
# AADT가 큰 지점 / 작은 지점 / 도로종류별 사례를
# 일부 골라 실제 LINK 목록을 확인
# ============================================================

print(
    "\n[12] 대표 측정점 상세 검사"
)


# AADT가 큰 순서 상위 3개
high_aadt_indices = (
    traffic[
        "AADT"
    ]
    .nlargest(3)
    .index
    .tolist()
)

# AADT가 작은 순서 하위 3개
low_aadt_indices = (
    traffic[
        "AADT"
    ]
    .nsmallest(3)
    .index
    .tolist()
)

representative_indices = []

for idx in (
    high_aadt_indices
    + low_aadt_indices
):

    if idx not in representative_indices:

        representative_indices.append(
            idx
        )


for idx in representative_indices:

    point = traffic.loc[
        idx
    ]

    road_no = point[
        "aadt_road_no"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "traffic_index:",
        idx
    )

    print(
        "도로종류:",
        point[
            "도로종류"
        ]
    )

    print(
        "지점번호:",
        point[
            "지점번호"
        ]
    )

    print(
        "호선:",
        point[
            "호선"
        ]
    )

    print(
        "정규화 ROAD_NO:",
        road_no
    )

    print(
        "구간:",
        point[
            "구간"
        ]
    )

    print(
        "AADT:",
        point[
            "AADT"
        ]
    )

    same_road_links = links[
        links[
            "link_road_no"
        ] == road_no
    ].copy()

    # 500m 안의 동일 도로번호 LINK
    search_area = (
        point.geometry.buffer(
            500
        )
    )

    candidate_index = list(
        same_road_links.sindex.query(
            search_area,
            predicate="intersects"
        )
    )

    nearby = (
        same_road_links.iloc[
            candidate_index
        ].copy()
    )

    if len(nearby) == 0:

        print(
            "500m 내 동일 ROAD_NO LINK 없음"
        )

        continue

    nearby[
        "distance_m"
    ] = nearby.geometry.distance(
        point.geometry
    )

    nearby = nearby[
        nearby[
            "distance_m"
        ] <= 500
    ].copy()

    nearby = nearby.sort_values(
        "distance_m"
    )

    detail_columns = [
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
        "\n500m 내 동일 ROAD_NO LINK:"
    )

    print(
        nearby[
            detail_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# 15. 요약
# ============================================================

print(
    "\n=== 전파 범위 검증 요약 ==="
)

print(
    "AADT 측정지점:",
    len(result_df)
)

for distance in distance_ranges:

    column = (
        f"links_{distance}m"
    )

    print(
        f"{distance}m 평균 LINK 수:",
        result_df[
            column
        ].mean()
    )

    print(
        f"{distance}m 중앙값 LINK 수:",
        result_df[
            column
        ].median()
    )

    print(
        f"{distance}m 최대 LINK 수:",
        result_df[
            column
        ].max()
    )


print(
    "\n=== AADT 주변 동일 ROAD_NO LINK 검사 완료 ==="
)