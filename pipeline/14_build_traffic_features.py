from pathlib import Path
from collections import defaultdict
import heapq

import geopandas as gpd
import pandas as pd
import numpy as np


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

FEATURE_PATH = (
    PROCESSED_DIR
    / "intersection_features_cheonan_asan.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "traffic_features_cheonan_asan.parquet"
)


# ============================================================
# 2. 설정
#
# 실측 AADT를 인정하는 최대 네트워크 거리
# ============================================================

MAX_NETWORK_DISTANCE = 3000.0

# 시작 LINK는 같은 ROAD_NO 기준 100m 이내만 인정
MAX_START_DISTANCE = 100.0


# ============================================================
# 3. ROAD_NO 정규화
# ============================================================

def normalize_road_number(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        number = float(text)
        return str(int(number))

    except ValueError:

        text = text.lstrip("0")

        if text == "":
            return "0"

        return text


# ============================================================
# 4. 데이터 불러오기
# ============================================================

print(
    "=== SafeRoute Traffic Feature 생성 ==="
)

print("\n[1] 데이터 불러오기...")

traffic = gpd.read_parquet(
    TRAFFIC_PATH
)

links = gpd.read_parquet(
    LINK_PATH
)

features = pd.read_parquet(
    FEATURE_PATH
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
    "기존 Feature:",
    len(features)
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
# 5. CRS 확인
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
# 6. 기본 검증
# ============================================================

print(
    "\n[3] 기본 데이터 검증..."
)

required_link_columns = [
    "LINK_ID",
    "F_NODE",
    "T_NODE",
    "ROAD_RANK",
    "ROAD_NO",
    "LENGTH",
    "geometry",
]

required_traffic_columns = [
    "호선",
    "AADT",
    "geometry",
]

required_feature_columns = [
    "link_id",
    "road_rank",
]

for column in required_link_columns:

    if column not in links.columns:

        raise ValueError(
            f"LINK 필수 컬럼 없음: {column}"
        )


for column in required_traffic_columns:

    if column not in traffic.columns:

        raise ValueError(
            f"AADT 필수 컬럼 없음: {column}"
        )


for column in required_feature_columns:

    if column not in features.columns:

        raise ValueError(
            f"Feature 필수 컬럼 없음: {column}"
        )


print(
    "필수 컬럼 확인 완료"
)


# ============================================================
# 7. ROAD_NO 정규화
# ============================================================

print(
    "\n[4] ROAD_NO 정규화..."
)

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
# 8. LINK_ID 문자열 통일
# ============================================================

links[
    "link_id_norm"
] = links[
    "LINK_ID"
].astype(str)

features[
    "link_id"
] = features[
    "link_id"
].astype(str)


# ============================================================
# 9. AADT별 시작 LINK 찾기
#
# 같은 ROAD_NO를 가진 LINK 중
# 가장 가까운 LINK를 찾음
#
# 100m 이내인 경우만 유효한 실측지점으로 사용
# ============================================================

print(
    "\n[5] AADT 시작 LINK 탐색..."
)

valid_stations = []
invalid_stations = []


for traffic_index, point in traffic.iterrows():

    road_no = point[
        "road_no_norm"
    ]

    same_road_links = links[
        links[
            "road_no_norm"
        ] == road_no
    ]

    if len(same_road_links) == 0:

        invalid_stations.append(
            {
                "traffic_index": traffic_index,
                "reason": "동일 ROAD_NO LINK 없음",
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

    nearest_link = links.loc[
        nearest_idx
    ]

    if nearest_distance > MAX_START_DISTANCE:

        invalid_stations.append(
            {
                "traffic_index": traffic_index,
                "지점번호": point.get(
                    "지점번호"
                ),
                "호선": point[
                    "호선"
                ],
                "AADT": point[
                    "AADT"
                ],
                "nearest_distance": (
                    nearest_distance
                ),
                "reason": "100m 초과",
            }
        )

        continue

    valid_stations.append(
        {
            "traffic_index": traffic_index,
            "road_no": road_no,
            "aadt": float(
                point[
                    "AADT"
                ]
            ),
            "start_link_index": (
                nearest_idx
            ),
            "start_link_id": (
                nearest_link[
                    "link_id_norm"
                ]
            ),
            "start_distance": (
                nearest_distance
            ),
        }
    )


print(
    "유효 AADT 측정지점:",
    len(valid_stations)
)

print(
    "제외 AADT 측정지점:",
    len(invalid_stations)
)


if len(invalid_stations) > 0:

    print(
        "\n[제외된 AADT 측정지점]"
    )

    print(
        pd.DataFrame(
            invalid_stations
        ).to_string(
            index=False
        )
    )


# ============================================================
# 10. NODE → LINK 연결 인덱스 생성
# ============================================================

print(
    "\n[6] 도로망 연결 인덱스 생성..."
)

node_to_links = defaultdict(
    list
)


for idx, row in links.iterrows():

    node_to_links[
        row[
            "F_NODE"
        ]
    ].append(
        idx
    )

    node_to_links[
        row[
            "T_NODE"
        ]
    ].append(
        idx
    )


print(
    "NODE 수:",
    len(node_to_links)
)


# ============================================================
# 11. LINK별 인접 LINK 생성 함수
#
# 같은 ROAD_NO + NODE 연결 조건
# ============================================================

def get_neighbors(
    link_index,
    road_no,
):

    row = links.loc[
        link_index
    ]

    candidate_indices = set()

    for node_id in [
        row[
            "F_NODE"
        ],
        row[
            "T_NODE"
        ],
    ]:

        for candidate_idx in node_to_links.get(
            node_id,
            []
        ):

            if candidate_idx == link_index:
                continue

            candidate_indices.add(
                candidate_idx
            )

    neighbors = []

    for candidate_idx in candidate_indices:

        candidate = links.loc[
            candidate_idx
        ]

        if (
            candidate[
                "road_no_norm"
            ]
            != road_no
        ):
            continue

        neighbors.append(
            candidate_idx
        )

    return neighbors


# ============================================================
# 12. 모든 AADT 측정점에서 네트워크 탐색
#
# 각 LINK에는
# 가장 가까운 네트워크 거리의 AADT만 남김
# ============================================================

print(
    "\n[7] 실측 AADT 네트워크 전파 시작..."
)

# link index →
# {
#   network_distance,
#   aadt,
#   traffic_index
# }

best_assignment = {}


for station_number, station in enumerate(
    valid_stations,
    start=1,
):

    traffic_index = station[
        "traffic_index"
    ]

    road_no = station[
        "road_no"
    ]

    aadt = station[
        "aadt"
    ]

    start_idx = station[
        "start_link_index"
    ]

    print(
        f"[{station_number}/{len(valid_stations)}] "
        f"traffic_index={traffic_index}, "
        f"ROAD_NO={road_no}, "
        f"AADT={aadt}"
    )

    # priority queue
    # (network_distance, link_index)

    queue = []

    heapq.heappush(
        queue,
        (
            0.0,
            start_idx,
        )
    )

    local_best = {
        start_idx: 0.0
    }

    while queue:

        network_distance, current_idx = (
            heapq.heappop(
                queue
            )
        )

        if (
            network_distance
            > MAX_NETWORK_DISTANCE
        ):
            continue

        if (
            network_distance
            > local_best.get(
                current_idx,
                float("inf"),
            )
        ):
            continue

        # --------------------------------------------
        # 현재 LINK에 AADT 후보 등록
        # --------------------------------------------

        previous = best_assignment.get(
            current_idx
        )

        if (
            previous is None
            or network_distance
            < previous[
                "network_distance"
            ]
        ):

            best_assignment[
                current_idx
            ] = {
                "network_distance": (
                    network_distance
                ),
                "aadt": aadt,
                "traffic_index": (
                    traffic_index
                ),
            }

        # --------------------------------------------
        # 인접 LINK 탐색
        # --------------------------------------------

        neighbors = get_neighbors(
            current_idx,
            road_no,
        )

        for neighbor_idx in neighbors:

            neighbor_length = float(
                links.loc[
                    neighbor_idx,
                    "LENGTH",
                ]
            )

            new_distance = (
                network_distance
                + neighbor_length
            )

            if (
                new_distance
                > MAX_NETWORK_DISTANCE
            ):
                continue

            if (
                new_distance
                < local_best.get(
                    neighbor_idx,
                    float("inf"),
                )
            ):

                local_best[
                    neighbor_idx
                ] = new_distance

                heapq.heappush(
                    queue,
                    (
                        new_distance,
                        neighbor_idx,
                    )
                )


print(
    "\n실측 AADT 할당 LINK:",
    len(best_assignment)
)


# ============================================================
# 13. LINK 단위 실측 traffic 테이블 생성
# ============================================================

print(
    "\n[8] LINK별 실측 traffic 생성..."
)

link_traffic = pd.DataFrame(
    {
        "link_index": (
            links.index
        ),
        "link_id": (
            links[
                "link_id_norm"
            ].values
        ),
        "road_rank": (
            links[
                "ROAD_RANK"
            ].values
        ),
    }
)

link_traffic[
    "traffic"
] = np.nan

link_traffic[
    "traffic_is_proxy"
] = True

link_traffic[
    "aadt_network_distance"
] = np.nan

link_traffic[
    "aadt_source_index"
] = pd.NA


for link_index, assignment in (
    best_assignment.items()
):

    mask = (
        link_traffic[
            "link_index"
        ] == link_index
    )

    link_traffic.loc[
        mask,
        "traffic",
    ] = assignment[
        "aadt"
    ]

    link_traffic.loc[
        mask,
        "traffic_is_proxy",
    ] = False

    link_traffic.loc[
        mask,
        "aadt_network_distance",
    ] = assignment[
        "network_distance"
    ]

    link_traffic.loc[
        mask,
        "aadt_source_index",
    ] = assignment[
        "traffic_index"
    ]


actual_count = (
    ~link_traffic[
        "traffic_is_proxy"
    ]
).sum()

print(
    "실측 AADT 적용 LINK:",
    actual_count
)

print(
    "Proxy 필요 LINK:",
    link_traffic[
        "traffic_is_proxy"
    ].sum()
)


# ============================================================
# 14. road_rank별 실측 AADT 평균 계산
#
# 확정 사양:
# unmatched urban road →
# road-rank mean proxy
# ============================================================

print(
    "\n[9] road_rank별 실측 AADT 평균 계산..."
)

actual_rows = link_traffic[
    ~link_traffic[
        "traffic_is_proxy"
    ]
].copy()


road_rank_mean = (
    actual_rows.groupby(
        "road_rank"
    )[
        "traffic"
    ]
    .mean()
)


print(
    road_rank_mean.to_string()
)


# ============================================================
# 15. road_rank에 실측값이 하나도 없는 경우 확인
# ============================================================

all_road_ranks = sorted(
    link_traffic[
        "road_rank"
    ]
    .dropna()
    .unique()
)

missing_proxy_ranks = [
    rank
    for rank in all_road_ranks
    if rank not in road_rank_mean.index
]


print(
    "\n실측 AADT 평균이 없는 road_rank:",
    missing_proxy_ranks
)


# ============================================================
# 16. Proxy 생성
#
# 1) 해당 ROAD_RANK에 실측 AADT가 있으면
#    ROAD_RANK 평균 사용
#
# 2) ROAD_RANK에 실측이 없으면
#    105 + 106 실측 LINK를 기준으로
#    LANES + MAX_SPD 조건 평균 사용
#
# 3) 같은 조합이 없으면 LANES 평균
#
# 4) 그것도 없으면 105 + 106 전체 평균
# ============================================================

print(
    "\n[10] Proxy 생성..."
)


# ------------------------------------------------------------
# 16-1. 105 / 106 실측 LINK 추출
# ------------------------------------------------------------

similar_actual = actual_rows[
    actual_rows[
        "road_rank"
    ].astype(str).isin(
        [
            "105",
            "106",
        ]
    )
].copy()


# 원본 LINK에서 LANES / MAX_SPD 가져오기
link_condition = links[
    [
        "link_id_norm",
        "LANES",
        "MAX_SPD",
    ]
].copy()

link_condition = link_condition.rename(
    columns={
        "link_id_norm": "link_id"
    }
)


similar_actual = similar_actual.merge(
    link_condition,
    on="link_id",
    how="left",
    validate="one_to_one",
)


print(
    "105/106 실측 LINK:",
    len(similar_actual)
)


# ------------------------------------------------------------
# 16-2. LANES + MAX_SPD 평균
# ------------------------------------------------------------

lane_speed_mean = (
    similar_actual.groupby(
        [
            "LANES",
            "MAX_SPD",
        ]
    )[
        "traffic"
    ]
    .mean()
)


# ------------------------------------------------------------
# 16-3. LANES 평균
# ------------------------------------------------------------

lane_mean = (
    similar_actual.groupby(
        "LANES"
    )[
        "traffic"
    ]
    .mean()
)


# ------------------------------------------------------------
# 16-4. 105/106 전체 평균
# ------------------------------------------------------------

similar_overall_mean = (
    similar_actual[
        "traffic"
    ].mean()
)


print(
    "105/106 전체 실측 평균:",
    similar_overall_mean
)


# ------------------------------------------------------------
# 16-5. link_traffic에도 LANES / MAX_SPD 추가
# ------------------------------------------------------------

link_traffic = link_traffic.merge(
    link_condition,
    on="link_id",
    how="left",
    validate="one_to_one",
)


# ------------------------------------------------------------
# 16-6. Proxy 값 결정
# ------------------------------------------------------------

proxy_method_counts = {
    "road_rank_mean": 0,
    "lanes_speed_mean": 0,
    "lanes_mean": 0,
    "similar_overall_mean": 0,
}


proxy_indices = link_traffic[
    link_traffic[
        "traffic_is_proxy"
    ]
].index


for idx in proxy_indices:

    road_rank = str(
        link_traffic.loc[
            idx,
            "road_rank",
        ]
    )

    lanes = link_traffic.loc[
        idx,
        "LANES",
    ]

    max_spd = link_traffic.loc[
        idx,
        "MAX_SPD",
    ]


    # ----------------------------------------
    # 1순위: 해당 ROAD_RANK 실측 평균
    # ----------------------------------------

    if road_rank in (
        road_rank_mean.index
        .astype(str)
    ):

        matching_rank = [
            rank
            for rank in road_rank_mean.index
            if str(rank) == road_rank
        ][0]

        proxy_value = (
            road_rank_mean.loc[
                matching_rank
            ]
        )

        proxy_method_counts[
            "road_rank_mean"
        ] += 1


    # ----------------------------------------
    # 2순위:
    # 105/106의 동일 LANES + MAX_SPD
    # ----------------------------------------

    elif (
        lanes,
        max_spd,
    ) in lane_speed_mean.index:

        proxy_value = (
            lane_speed_mean.loc[
                (
                    lanes,
                    max_spd,
                )
            ]
        )

        proxy_method_counts[
            "lanes_speed_mean"
        ] += 1


    # ----------------------------------------
    # 3순위:
    # 105/106의 동일 LANES 평균
    # ----------------------------------------

    elif lanes in lane_mean.index:

        proxy_value = (
            lane_mean.loc[
                lanes
            ]
        )

        proxy_method_counts[
            "lanes_mean"
        ] += 1


    # ----------------------------------------
    # 4순위:
    # 105/106 전체 실측 평균
    # ----------------------------------------

    else:

        proxy_value = (
            similar_overall_mean
        )

        proxy_method_counts[
            "similar_overall_mean"
        ] += 1


    link_traffic.loc[
        idx,
        "traffic",
    ] = float(
        proxy_value
    )


print(
    "\nProxy 생성 방법별 LINK 수:"
)

for method, count in (
    proxy_method_counts.items()
):

    print(
        method,
        ":",
        count
    )


# ============================================================
# 17. traffic 최종 검증
# ============================================================

print(
    "\n[10] Traffic Feature 검증..."
)

print(
    "전체 LINK:",
    len(link_traffic)
)

print(
    "traffic 결측:",
    link_traffic[
        "traffic"
    ].isna().sum()
)

print(
    "traffic_is_proxy 결측:",
    link_traffic[
        "traffic_is_proxy"
    ].isna().sum()
)

print(
    "\ntraffic_is_proxy 분포:"
)

print(
    link_traffic[
        "traffic_is_proxy"
    ].value_counts()
)


print(
    "\ntraffic 통계:"
)

print(
    link_traffic[
        "traffic"
    ].describe()
)


print(
    "\n실측 traffic 통계:"
)

print(
    link_traffic.loc[
        ~link_traffic[
            "traffic_is_proxy"
        ],
        "traffic",
    ].describe()
)


print(
    "\nProxy traffic 통계:"
)

print(
    link_traffic.loc[
        link_traffic[
            "traffic_is_proxy"
        ],
        "traffic",
    ].describe()
)


# ============================================================
# 18. road_rank별 Proxy 값 확인
# ============================================================

print(
    "\n[11] road_rank별 traffic 요약"
)

rank_summary = (
    link_traffic.groupby(
        "road_rank"
    )
    .agg(
        link_count=(
            "link_id",
            "count",
        ),
        traffic_mean=(
            "traffic",
            "mean",
        ),
        traffic_min=(
            "traffic",
            "min",
        ),
        traffic_max=(
            "traffic",
            "max",
        ),
        proxy_count=(
            "traffic_is_proxy",
            "sum",
        ),
    )
)

rank_summary[
    "actual_count"
] = (
    rank_summary[
        "link_count"
    ]
    - rank_summary[
        "proxy_count"
    ]
)

print(
    rank_summary.to_string()
)


# ============================================================
# 19. 기존 Feature와 병합
# ============================================================

print(
    "\n[12] 기존 Feature와 병합..."
)

traffic_for_merge = link_traffic[
    [
        "link_id",
        "traffic",
        "traffic_is_proxy",
    ]
].copy()


final_features = features.merge(
    traffic_for_merge,
    on="link_id",
    how="left",
    validate="one_to_one",
)


print(
    "병합 후 행:",
    len(final_features)
)

print(
    "link_id 중복:",
    final_features[
        "link_id"
    ].duplicated().sum()
)

print(
    "traffic 결측:",
    final_features[
        "traffic"
    ].isna().sum()
)

print(
    "traffic_is_proxy 결측:",
    final_features[
        "traffic_is_proxy"
    ].isna().sum()
)


# ============================================================
# 20. 최종 컬럼 확인
# ============================================================

print(
    "\n[13] 최종 컬럼"
)

print(
    final_features.columns.tolist()
)


# ============================================================
# 21. 저장
# ============================================================

print(
    "\n[14] 저장..."
)

final_features.to_parquet(
    OUTPUT_PATH,
    index=False,
)


print(
    "저장 경로:",
    OUTPUT_PATH
)

print(
    "파일 존재:",
    OUTPUT_PATH.exists()
)


# ============================================================
# 22. 최종 요약
# ============================================================

print(
    "\n=== Traffic Feature 생성 요약 ==="
)

print(
    "전체 LINK:",
    len(final_features)
)

print(
    "유효 AADT 측정지점:",
    len(valid_stations)
)

print(
    "제외 AADT 측정지점:",
    len(invalid_stations)
)

print(
    "실측 AADT 적용 LINK:",
    (
        ~final_features[
            "traffic_is_proxy"
        ]
    ).sum()
)

print(
    "Proxy 적용 LINK:",
    final_features[
        "traffic_is_proxy"
    ].sum()
)

print(
    "traffic 결측:",
    final_features[
        "traffic"
    ].isna().sum()
)

print(
    "traffic_is_proxy 결측:",
    final_features[
        "traffic_is_proxy"
    ].isna().sum()
)

print(
    "최종 Feature 컬럼:"
)

print(
    final_features.columns.tolist()
)

print(
    "\n=== SafeRoute Traffic Feature 생성 완료 ==="
)