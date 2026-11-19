from pathlib import Path
from collections import deque

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

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
    "=== SafeRoute AADT 도로망 연결성 검사 ==="
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
# 4. ROAD_NO 정규화
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


traffic[
    "road_no_norm"
] = traffic[
    "호선"
].apply(
    normalize_road_number
)

links[
    "road_no_norm"
] = links[
    "ROAD_NO"
].apply(
    normalize_road_number
)


print(
    "\n[3] ROAD_NO 정규화 완료"
)

print(
    "AADT ROAD_NO 결측:",
    traffic[
        "road_no_norm"
    ].isna().sum()
)

print(
    "LINK ROAD_NO 결측:",
    links[
        "road_no_norm"
    ].isna().sum()
)


# ============================================================
# 5. 시작 LINK 찾기
#
# 12단계와 달리 이번에는 각 AADT 측정점에서
# 같은 ROAD_NO를 가진 LINK 중 최근접 LINK를 찾습니다.
#
# 단, 같은 ROAD_NO LINK가 너무 멀리 있는 경우를 막기 위해
# 100m 이내인 경우만 정상 시작 LINK로 인정합니다.
# ============================================================

print(
    "\n[4] AADT별 시작 LINK 탐색..."
)

start_results = []


for traffic_index, point in traffic.iterrows():

    road_no = point[
        "road_no_norm"
    ]

    same_road_links = links[
        links[
            "road_no_norm"
        ] == road_no
    ].copy()

    if len(same_road_links) == 0:

        start_results.append(
            {
                "traffic_index": traffic_index,
                "start_link_id": None,
                "distance_m": None,
                "valid_start": False,
            }
        )

        continue

    distances = (
        same_road_links.geometry.distance(
            point.geometry
        )
    )

    nearest_idx = (
        distances.idxmin()
    )

    nearest_distance = float(
        distances.loc[
            nearest_idx
        ]
    )

    nearest_link_id = (
        same_road_links.loc[
            nearest_idx,
            "LINK_ID"
        ]
    )

    start_results.append(
        {
            "traffic_index": traffic_index,
            "start_link_id": nearest_link_id,
            "distance_m": nearest_distance,
            "valid_start": (
                nearest_distance <= 100
            ),
        }
    )


start_df = pd.DataFrame(
    start_results
)


print(
    "시작 LINK 탐색 완료:",
    len(start_df)
)

print(
    "100m 이내 정상 시작 LINK:",
    start_df[
        "valid_start"
    ].sum()
)

print(
    "정상 시작 LINK 없음:",
    (
        ~start_df[
            "valid_start"
        ]
    ).sum()
)


# ============================================================
# 6. 시작 LINK 실패 지점 확인
# ============================================================

print(
    "\n[5] 정상 시작 LINK가 없는 AADT"
)

invalid_start = start_df[
    ~start_df[
        "valid_start"
    ]
].copy()

print(
    "개수:",
    len(invalid_start)
)

if len(invalid_start) > 0:

    invalid_detail = (
        invalid_start
        .merge(
            traffic[
                [
                    "도로종류",
                    "지점번호",
                    "호선",
                    "구간",
                    "AADT",
                ]
            ],
            left_on="traffic_index",
            right_index=True,
            how="left",
        )
    )

    print(
        invalid_detail.to_string(
            index=False
        )
    )


# ============================================================
# 7. NODE → LINK 인덱스 준비
#
# 도로망을 따라가기 위해
# 각 NODE에 연결된 LINK 목록을 만듭니다.
#
# 여기서는 방향성보다 "도로 연결성" 자체를 검사하기 위해
# F_NODE와 T_NODE 모두 연결로 봅니다.
# ============================================================

print(
    "\n[6] NODE ↔ LINK 연결 인덱스 생성..."
)

node_to_links = {}


for idx, row in links.iterrows():

    f_node = row[
        "F_NODE"
    ]

    t_node = row[
        "T_NODE"
    ]

    node_to_links.setdefault(
        f_node,
        []
    ).append(
        idx
    )

    node_to_links.setdefault(
        t_node,
        []
    ).append(
        idx
    )


print(
    "연결 인덱스 NODE 수:",
    len(node_to_links)
)


# ============================================================
# 8. 연결 LINK 탐색 함수
#
# 시작 LINK에서 F_NODE/T_NODE를 통해
# 같은 ROAD_NO를 가진 LINK만 BFS 방식으로 따라갑니다.
#
# max_depth:
#   몇 번의 LINK 연결까지 따라갈지
#
# max_total_length:
#   너무 멀리 퍼지는 것을 방지하기 위한
#   누적 탐색 길이 제한
#
# 현재는 검증용으로 10단계 / 3000m 사용
# ============================================================

def trace_connected_links(
    start_link_index,
    road_no,
    max_depth=10,
    max_total_length=3000,
):

    visited = set()

    queue = deque()

    queue.append(
        (
            start_link_index,
            0,
            0.0,
        )
    )

    result = []

    while queue:

        current_idx, depth, cumulative_length = (
            queue.popleft()
        )

        if current_idx in visited:
            continue

        current_link = links.loc[
            current_idx
        ]

        current_road_no = current_link[
            "road_no_norm"
        ]

        if current_road_no != road_no:
            continue

        current_length = float(
            current_link[
                "LENGTH"
            ]
        )

        new_cumulative_length = (
            cumulative_length
            + current_length
        )

        if (
            new_cumulative_length
            > max_total_length
            and depth > 0
        ):
            continue

        visited.add(
            current_idx
        )

        result.append(
            {
                "link_index": current_idx,
                "LINK_ID": current_link[
                    "LINK_ID"
                ],
                "ROAD_NAME": current_link[
                    "ROAD_NAME"
                ],
                "ROAD_RANK": current_link[
                    "ROAD_RANK"
                ],
                "ROAD_NO": current_link[
                    "ROAD_NO"
                ],
                "LANES": current_link[
                    "LANES"
                ],
                "MAX_SPD": current_link[
                    "MAX_SPD"
                ],
                "LENGTH": current_length,
                "depth": depth,
                "cumulative_length": (
                    new_cumulative_length
                ),
            }
        )

        if depth >= max_depth:
            continue

        connected_nodes = [
            current_link[
                "F_NODE"
            ],
            current_link[
                "T_NODE"
            ],
        ]

        for node_id in connected_nodes:

            candidate_indices = (
                node_to_links.get(
                    node_id,
                    []
                )
            )

            for candidate_idx in candidate_indices:

                if candidate_idx in visited:
                    continue

                candidate_link = links.loc[
                    candidate_idx
                ]

                if (
                    candidate_link[
                        "road_no_norm"
                    ]
                    != road_no
                ):
                    continue

                queue.append(
                    (
                        candidate_idx,
                        depth + 1,
                        new_cumulative_length,
                    )
                )

    return pd.DataFrame(
        result
    )


# ============================================================
# 9. 대표 AADT 측정점 선택
#
# 고속도로 / 일반국도 / 지방도 /
# 국가지원지방도를 골고루 확인합니다.
# ============================================================

print(
    "\n[7] 대표 AADT 측정점 선택..."
)

representative_indices = []


for road_type in [
    "고속국도",
    "일반국도",
    "지방도",
    "국가지원지방도",
]:

    subset = traffic[
        traffic[
            "도로종류"
        ] == road_type
    ]

    if len(subset) == 0:
        continue

    # 해당 도로종류에서
    # AADT가 가장 큰 지점
    high_idx = (
        subset[
            "AADT"
        ].idxmax()
    )

    # 해당 도로종류에서
    # AADT가 가장 작은 지점
    low_idx = (
        subset[
            "AADT"
        ].idxmin()
    )

    for idx in [
        high_idx,
        low_idx,
    ]:

        if idx not in representative_indices:

            representative_indices.append(
                idx
            )


# 12단계에서 문제가 확인된
# traffic_index 36도 별도로 포함
if 36 in traffic.index:

    if 36 not in representative_indices:

        representative_indices.append(
            36
        )


print(
    "대표 측정점:",
    representative_indices
)


# ============================================================
# 10. 대표 측정점 연결성 상세 검사
# ============================================================

print(
    "\n[8] 대표 측정점 도로망 연결성 검사"
)


for traffic_index in representative_indices:

    point = traffic.loc[
        traffic_index
    ]

    start_info = start_df[
        start_df[
            "traffic_index"
        ] == traffic_index
    ].iloc[0]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "traffic_index:",
        traffic_index
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
        point[
            "road_no_norm"
        ]
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

    print(
        "시작 LINK:",
        start_info[
            "start_link_id"
        ]
    )

    print(
        "시작 LINK 거리:",
        start_info[
            "distance_m"
        ]
    )

    if not start_info[
        "valid_start"
    ]:

        print(
            "→ 100m 이내 동일 ROAD_NO 시작 LINK가 없어 "
            "연결성 검사를 건너뜁니다."
        )

        continue

    start_link_id = start_info[
        "start_link_id"
    ]

    start_candidates = links[
        links[
            "LINK_ID"
        ].astype(str)
        == str(
            start_link_id
        )
    ]

    if len(start_candidates) == 0:

        print(
            "시작 LINK를 찾을 수 없습니다."
        )

        continue

    start_link_index = (
        start_candidates.index[0]
    )

    traced = trace_connected_links(
        start_link_index=start_link_index,
        road_no=point[
            "road_no_norm"
        ],
        max_depth=10,
        max_total_length=3000,
    )

    print(
        "연결 탐색 LINK 수:",
        len(traced)
    )

    if len(traced) == 0:
        continue

    print(
        "ROAD_NAME 분포:"
    )

    print(
        traced[
            "ROAD_NAME"
        ]
        .fillna(
            "결측"
        )
        .value_counts()
        .to_string()
    )

    print(
        "\n연결 LINK 상세:"
    )

    detail_columns = [
        "depth",
        "LINK_ID",
        "ROAD_NAME",
        "ROAD_RANK",
        "ROAD_NO",
        "LANES",
        "MAX_SPD",
        "LENGTH",
        "cumulative_length",
    ]

    print(
        traced[
            detail_columns
        ]
        .sort_values(
            [
                "depth",
                "cumulative_length",
            ]
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 11. 모든 정상 AADT 측정점에 대해 연결 LINK 수 계산
# ============================================================

print(
    "\n[9] 전체 AADT 연결성 통계 계산..."
)

connectivity_results = []


for _, start_info in start_df.iterrows():

    traffic_index = int(
        start_info[
            "traffic_index"
        ]
    )

    point = traffic.loc[
        traffic_index
    ]

    if not start_info[
        "valid_start"
    ]:

        connectivity_results.append(
            {
                "traffic_index": traffic_index,
                "도로종류": point[
                    "도로종류"
                ],
                "호선": point[
                    "호선"
                ],
                "AADT": point[
                    "AADT"
                ],
                "connected_links": 0,
            }
        )

        continue

    start_link_id = start_info[
        "start_link_id"
    ]

    start_candidates = links[
        links[
            "LINK_ID"
        ].astype(str)
        == str(
            start_link_id
        )
    ]

    if len(start_candidates) == 0:

        connected_count = 0

    else:

        start_link_index = (
            start_candidates.index[0]
        )

        traced = trace_connected_links(
            start_link_index=start_link_index,
            road_no=point[
                "road_no_norm"
            ],
            max_depth=10,
            max_total_length=3000,
        )

        connected_count = len(
            traced
        )

    connectivity_results.append(
        {
            "traffic_index": traffic_index,
            "도로종류": point[
                "도로종류"
            ],
            "호선": point[
                "호선"
            ],
            "AADT": point[
                "AADT"
            ],
            "connected_links": (
                connected_count
            ),
        }
    )


connectivity_df = pd.DataFrame(
    connectivity_results
)


# ============================================================
# 12. 전체 통계 출력
# ============================================================

print(
    "\n[10] 연결 LINK 수 통계"
)

print(
    connectivity_df[
        "connected_links"
    ].describe()
)


print(
    "\n도로종류별 연결 LINK 수:"
)

print(
    connectivity_df.groupby(
        "도로종류"
    )[
        "connected_links"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
    .to_string()
)


# ============================================================
# 13. 연결 LINK가 적은 측정점
# ============================================================

print(
    "\n[11] 연결 LINK가 2개 이하인 측정점"
)

few_links = connectivity_df[
    connectivity_df[
        "connected_links"
    ] <= 2
]

print(
    "개수:",
    len(few_links)
)

if len(few_links) > 0:

    print(
        few_links.to_string(
            index=False
        )
    )


# ============================================================
# 14. 연결 LINK가 많은 측정점
# ============================================================

print(
    "\n[12] 연결 LINK가 30개 이상인 측정점"
)

many_links = connectivity_df[
    connectivity_df[
        "connected_links"
    ] >= 30
]

print(
    "개수:",
    len(many_links)
)

if len(many_links) > 0:

    print(
        many_links
        .sort_values(
            "connected_links",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 15. 최종 요약
# ============================================================

print(
    "\n=== 도로망 연결성 검증 요약 ==="
)

print(
    "AADT 측정지점:",
    len(traffic)
)

print(
    "정상 시작 LINK:",
    start_df[
        "valid_start"
    ].sum()
)

print(
    "시작 LINK 실패:",
    (
        ~start_df[
            "valid_start"
        ]
    ).sum()
)

print(
    "연결 LINK 평균:",
    connectivity_df[
        "connected_links"
    ].mean()
)

print(
    "연결 LINK 중앙값:",
    connectivity_df[
        "connected_links"
    ].median()
)

print(
    "연결 LINK 최대:",
    connectivity_df[
        "connected_links"
    ].max()
)

print(
    "\n=== AADT 도로망 연결성 검사 완료 ==="
)