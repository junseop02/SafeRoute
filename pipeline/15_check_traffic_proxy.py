from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LINK_PATH = (
    PROCESSED_DIR
    / "links_cheonan_asan.parquet"
)

TRAFFIC_FEATURE_PATH = (
    PROCESSED_DIR
    / "traffic_features_cheonan_asan.parquet"
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print("=== SafeRoute Traffic Proxy 검증 ===")

print("\n[1] 데이터 불러오기...")

links = gpd.read_parquet(
    LINK_PATH
)

features = pd.read_parquet(
    TRAFFIC_FEATURE_PATH
)

print("LINK:", len(links))
print("Feature:", len(features))
print("LINK CRS:", links.crs)


# ============================================================
# 3. LINK_ID 문자열 통일
# ============================================================

links["link_id"] = (
    links["LINK_ID"]
    .astype(str)
)

features["link_id"] = (
    features["link_id"]
    .astype(str)
)


# ============================================================
# 4. 필요한 원본 LINK 정보 병합
# ============================================================

print("\n[2] LINK 상세정보 병합...")

link_detail = links[
    [
        "link_id",
        "ROAD_NAME",
        "ROAD_RANK",
        "ROAD_NO",
        "LANES",
        "MAX_SPD",
        "LENGTH",
        "geometry",
    ]
].copy()

check = link_detail.merge(
    features[
        [
            "link_id",
            "traffic",
            "traffic_is_proxy",
        ]
    ],
    on="link_id",
    how="left",
    validate="one_to_one",
)

print("병합 행:", len(check))

print(
    "traffic 결측:",
    check["traffic"].isna().sum()
)

print(
    "traffic_is_proxy 결측:",
    check["traffic_is_proxy"].isna().sum()
)


# ============================================================
# 5. ROAD_RANK 분포
# ============================================================

print("\n[3] ROAD_RANK 분포")

print(
    check["ROAD_RANK"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 6. ROAD_RANK별 차로 수
# ============================================================

print("\n[4] ROAD_RANK별 LANES 분포")

lane_table = pd.crosstab(
    check["ROAD_RANK"],
    check["LANES"],
)

print(
    lane_table.to_string()
)


# ============================================================
# 7. ROAD_RANK별 제한속도 통계
# ============================================================

print("\n[5] ROAD_RANK별 MAX_SPD 통계")

speed_stats = (
    check.groupby(
        "ROAD_RANK"
    )["MAX_SPD"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    speed_stats.to_string()
)


# ============================================================
# 8. ROAD_RANK별 길이 통계
# ============================================================

print("\n[6] ROAD_RANK별 LENGTH 통계")

length_stats = (
    check.groupby(
        "ROAD_RANK"
    )["LENGTH"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    length_stats.to_string()
)


# ============================================================
# 9. ROAD_RANK 104 / 107 집중 검사
# ============================================================

print(
    "\n[7] ROAD_RANK 104 / 107 상세 검사"
)

for rank in ["104", "107"]:

    subset = check[
        check["ROAD_RANK"].astype(str)
        == rank
    ].copy()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ROAD_RANK:",
        rank
    )

    print(
        "LINK 수:",
        len(subset)
    )

    if len(subset) == 0:
        continue

    print(
        "\nLANES 분포:"
    )

    print(
        subset["LANES"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nMAX_SPD 분포:"
    )

    print(
        subset["MAX_SPD"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nROAD_NAME 상위 30개:"
    )

    print(
        subset["ROAD_NAME"]
        .fillna("결측")
        .value_counts()
        .head(30)
        .to_string()
    )

    print(
        "\nROAD_NO 상위 30개:"
    )

    print(
        subset["ROAD_NO"]
        .fillna("결측")
        .value_counts()
        .head(30)
        .to_string()
    )

    print(
        "\n샘플 LINK:"
    )

    sample_columns = [
        "link_id",
        "ROAD_NAME",
        "ROAD_NO",
        "LANES",
        "MAX_SPD",
        "LENGTH",
        "traffic",
        "traffic_is_proxy",
    ]

    print(
        subset[
            sample_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# 10. 실측 AADT가 있는 LINK만 분리
# ============================================================

print(
    "\n[8] 실측 AADT LINK 분석"
)

actual = check[
    check["traffic_is_proxy"]
    == False
].copy()

print(
    "실측 LINK:",
    len(actual)
)

print(
    "\n실측 LINK ROAD_RANK 분포:"
)

print(
    actual["ROAD_RANK"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 11. 실측 LINK의 LANES별 AADT
# ============================================================

print(
    "\n[9] 실측 LINK LANES별 AADT 통계"
)

lane_aadt = (
    actual.groupby(
        "LANES"
    )["traffic"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    lane_aadt.to_string()
)


# ============================================================
# 12. 실측 LINK의 MAX_SPD별 AADT
# ============================================================

print(
    "\n[10] 실측 LINK MAX_SPD별 AADT 통계"
)

speed_aadt = (
    actual.groupby(
        "MAX_SPD"
    )["traffic"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    speed_aadt.to_string()
)


# ============================================================
# 13. ROAD_RANK + LANES별 실측 AADT
#
# 107에 직접 AADT가 없으므로
# 다른 등급의 비슷한 차로 조건에서
# 어느 정도의 AADT가 관측되는지 확인
# ============================================================

print(
    "\n[11] ROAD_RANK + LANES별 실측 AADT"
)

rank_lane_aadt = (
    actual.groupby(
        [
            "ROAD_RANK",
            "LANES",
        ]
    )["traffic"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    rank_lane_aadt.to_string()
)


# ============================================================
# 14. ROAD_RANK + MAX_SPD별 실측 AADT
# ============================================================

print(
    "\n[12] ROAD_RANK + MAX_SPD별 실측 AADT"
)

rank_speed_aadt = (
    actual.groupby(
        [
            "ROAD_RANK",
            "MAX_SPD",
        ]
    )["traffic"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    rank_speed_aadt.to_string()
)


# ============================================================
# 15. 107과 다른 ROAD_RANK의 특성 비교
#
# 차로/속도 조합 분포를 확인
# ============================================================

print(
    "\n[13] ROAD_RANK별 LANES + MAX_SPD 조합"
)

combination = (
    check.groupby(
        [
            "ROAD_RANK",
            "LANES",
            "MAX_SPD",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

for rank in sorted(
    check["ROAD_RANK"]
    .astype(str)
    .unique()
):

    print(
        "\n"
        + "-" * 60
    )

    print(
        "ROAD_RANK:",
        rank
    )

    rank_combination = combination[
        combination[
            "ROAD_RANK"
        ].astype(str)
        == rank
    ].sort_values(
        "count",
        ascending=False,
    )

    print(
        rank_combination
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# 16. 107 LINK 주변 실측 AADT 거리 검사
#
# 모든 107 LINK를 전부 최근접 계산하면 비용이 크므로
# 대표 샘플 최대 200개만 검사
#
# LINK의 중간점을 기준으로
# 가장 가까운 실측 AADT LINK까지 거리를 확인
# ============================================================

print(
    "\n[14] ROAD_RANK 107 주변 실측 LINK 거리 검사"
)

rank107 = check[
    check["ROAD_RANK"].astype(str)
    == "107"
].copy()

actual_geo = actual.copy()


if (
    len(rank107) > 0
    and len(actual_geo) > 0
):

    # 최대 200개를 전체 구간에서 골고루 추출
    sample_size = min(
        200,
        len(rank107),
    )

    rank107_sample = rank107.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    # LINK 중간점
    rank107_sample[
        "center"
    ] = rank107_sample.geometry.interpolate(
        0.5,
        normalized=True,
    )

    actual_geo[
        "center"
    ] = actual_geo.geometry.interpolate(
        0.5,
        normalized=True,
    )

    actual_centers = gpd.GeoDataFrame(
        actual_geo[
            [
                "link_id",
                "traffic",
            ]
        ].copy(),
        geometry=actual_geo[
            "center"
        ],
        crs=links.crs,
    )

    distance_results = []

    for _, row in rank107_sample.iterrows():

        distances = (
            actual_centers.geometry.distance(
                row["center"]
            )
        )

        nearest_idx = (
            distances.idxmin()
        )

        distance_results.append(
            {
                "link_id": row[
                    "link_id"
                ],
                "ROAD_NAME": row[
                    "ROAD_NAME"
                ],
                "LANES": row[
                    "LANES"
                ],
                "MAX_SPD": row[
                    "MAX_SPD"
                ],
                "nearest_actual_distance": float(
                    distances.loc[
                        nearest_idx
                    ]
                ),
                "nearest_actual_aadt": float(
                    actual_centers.loc[
                        nearest_idx,
                        "traffic",
                    ]
                ),
            }
        )

    distance_df = pd.DataFrame(
        distance_results
    )

    print(
        "107 검사 샘플:",
        len(distance_df)
    )

    print(
        "\n최근접 실측 LINK 거리 통계:"
    )

    print(
        distance_df[
            "nearest_actual_distance"
        ].describe()
    )

    print(
        "\n최근접 실측 AADT 통계:"
    )

    print(
        distance_df[
            "nearest_actual_aadt"
        ].describe()
    )

    print(
        "\n거리 구간별 개수:"
    )

    distance_bins = pd.cut(
        distance_df[
            "nearest_actual_distance"
        ],
        bins=[
            -1,
            100,
            300,
            500,
            1000,
            3000,
            float("inf"),
        ],
        labels=[
            "0~100m",
            "100~300m",
            "300~500m",
            "500m~1km",
            "1~3km",
            "3km 초과",
        ],
    )

    print(
        distance_bins
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# 17. 최종 요약
# ============================================================

print(
    "\n=== Traffic Proxy 검증 요약 ==="
)

print(
    "전체 LINK:",
    len(check)
)

print(
    "실측 AADT LINK:",
    len(actual)
)

print(
    "Proxy LINK:",
    (
        check["traffic_is_proxy"]
        == True
    ).sum()
)

print(
    "ROAD_RANK 104 LINK:",
    (
        check["ROAD_RANK"]
        .astype(str)
        == "104"
    ).sum()
)

print(
    "ROAD_RANK 107 LINK:",
    (
        check["ROAD_RANK"]
        .astype(str)
        == "107"
    ).sum()
)

print(
    "\n=== Traffic Proxy 검증 완료 ==="
)