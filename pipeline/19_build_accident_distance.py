from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "links_cheonan_asan.parquet"
)

ACCIDENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "accident_points_cheonan_asan.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "accident_distance_cheonan_asan.parquet"
)


# ============================================================
# 2. 시작
# ============================================================

print(
    "=== SafeRoute LINK-사고지점 최소거리 계산 ==="
)


# ============================================================
# 3. 데이터 불러오기
# ============================================================

print(
    "\n[1] 데이터 불러오기..."
)

links = gpd.read_parquet(
    LINK_PATH
)

accidents = gpd.read_parquet(
    ACCIDENT_PATH
)


print(
    "LINK:",
    len(links)
)

print(
    "사고지점:",
    len(accidents)
)

print(
    "LINK CRS:",
    links.crs
)

print(
    "사고 CRS:",
    accidents.crs
)


# ============================================================
# 4. 기본 검증
# ============================================================

print(
    "\n[2] 기본 검증..."
)


if links.crs is None:
    raise RuntimeError(
        "LINK CRS가 없습니다."
    )


if accidents.crs is None:
    raise RuntimeError(
        "사고지점 CRS가 없습니다."
    )


if links.crs != accidents.crs:

    print(
        "CRS가 달라 사고지점을 LINK CRS로 변환합니다."
    )

    accidents = accidents.to_crs(
        links.crs
    )


if "LINK_ID" not in links.columns:

    raise RuntimeError(
        "LINK 데이터에 LINK_ID 컬럼이 없습니다."
    )


if "accident_type" not in accidents.columns:

    raise RuntimeError(
        "사고 데이터에 accident_type 컬럼이 없습니다."
    )


if links["LINK_ID"].duplicated().any():

    raise RuntimeError(
        "LINK_ID 중복이 존재합니다."
    )


print(
    "LINK geometry 결측:",
    links.geometry.isna().sum()
)

print(
    "사고 geometry 결측:",
    accidents.geometry.isna().sum()
)

print(
    "LINK_ID 중복:",
    links["LINK_ID"].duplicated().sum()
)

print(
    "기본 검증 완료"
)


# ============================================================
# 5. 필요한 컬럼만 준비
# ============================================================

print(
    "\n[3] 거리 계산용 데이터 준비..."
)


link_work = links[
    [
        "LINK_ID",
        "geometry",
    ]
].copy()


link_work = link_work.rename(
    columns={
        "LINK_ID": "link_id"
    }
)


link_work["link_id"] = (
    link_work["link_id"]
    .astype(str)
)


accident_work = accidents[
    [
        "사고다발지fid",
        "사고다발지id",
        "accident_type",
        "지점명",
        "사고건수",
        "geometry",
    ]
].copy()


# 내부적으로 각 사고 행을 구분하기 위한 번호
accident_work[
    "accident_row_id"
] = range(
    len(accident_work)
)


print(
    "거리 계산 대상 LINK:",
    len(link_work)
)

print(
    "거리 계산 대상 사고지점:",
    len(accident_work)
)


# ============================================================
# 6. 최근접 사고지점 계산
#
# 중요:
# LINK의 중심점이 아니라
# LineString 전체와 Point 사이의 최소거리를 계산
#
# EPSG:5179이므로 거리 단위는 metre
# ============================================================

print(
    "\n[4] LINK별 최근접 사고지점 계산..."
)


nearest = gpd.sjoin_nearest(
    link_work,
    accident_work,
    how="left",
    distance_col="accident_distance_m",
)


print(
    "최근접 조인 결과 행:",
    len(nearest)
)


# ============================================================
# 7. 동일 최소거리 tie 확인
#
# 하나의 LINK에서 서로 다른 사고지점이
# 정확히 같은 최소거리에 있을 경우
# sjoin_nearest가 여러 행을 반환할 수 있음
# ============================================================

print(
    "\n[5] 최근접 거리 tie 검사..."
)


tie_mask = (
    nearest["link_id"]
    .duplicated(
        keep=False
    )
)


tie_rows = nearest[
    tie_mask
].copy()


print(
    "tie에 포함된 행:",
    len(tie_rows)
)

print(
    "tie가 발생한 LINK:",
    tie_rows[
        "link_id"
    ].nunique()
)


# ============================================================
# 8. tie가 있는 경우 대표 1행 선택
#
# label에는 최소거리만 필요하므로
# 같은 최소거리 중 accident_row_id가 작은 행을 선택
# ============================================================

nearest = (
    nearest
    .sort_values(
        [
            "link_id",
            "accident_distance_m",
            "accident_row_id",
        ]
    )
    .drop_duplicates(
        subset=[
            "link_id",
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)


print(
    "tie 정리 후 LINK:",
    len(nearest)
)


# ============================================================
# 9. 거리 결과 검증
# ============================================================

print(
    "\n[6] 최소거리 결과 검증..."
)


distance = pd.to_numeric(
    nearest[
        "accident_distance_m"
    ],
    errors="coerce",
)


print(
    "거리 결측:",
    distance.isna().sum()
)

print(
    "거리 음수:",
    int(
        (
            distance < 0
        ).sum()
    )
)

print(
    "\n최소거리 통계:"
)

print(
    distance.describe()
)


# ============================================================
# 10. 거리 구간별 LINK 수
#
# 아직 label을 생성하지 않음.
# 80m / 300m 기준의 분포만 검사.
# ============================================================

print(
    "\n[7] 거리 구간별 LINK 수..."
)


distance_group = pd.cut(
    distance,
    bins=[
        -0.000001,
        80,
        300,
        500,
        1000,
        3000,
        float("inf"),
    ],
    labels=[
        "<=80m",
        "80~300m",
        "300~500m",
        "500m~1km",
        "1~3km",
        ">3km",
    ],
    include_lowest=True,
    right=True,
)


distance_counts = (
    distance_group
    .value_counts()
    .sort_index()
)


print(
    distance_counts.to_string()
)


# ============================================================
# 11. 80m / 300m 기준 사전 확인
# ============================================================

print(
    "\n[8] label 기준 사전 확인..."
)


positive_count = int(
    (
        distance <= 80
    ).sum()
)


gray_count = int(
    (
        (distance > 80)
        & (distance <= 300)
    ).sum()
)


negative_count = int(
    (
        distance > 300
    ).sum()
)


print(
    "80m 이하:",
    positive_count
)

print(
    "80m 초과 ~ 300m 이하:",
    gray_count
)

print(
    "300m 초과:",
    negative_count
)


print(
    "합계:",
    (
        positive_count
        + gray_count
        + negative_count
    )
)


# ============================================================
# 12. 최근접 사고유형 분포
# ============================================================

print(
    "\n[9] 최근접 사고유형 분포..."
)


print(
    nearest[
        "accident_type"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# 13. 80m 이하 LINK의 최근접 사고유형
# ============================================================

print(
    "\n[10] 80m 이하 LINK의 최근접 사고유형..."
)


positive_nearest = nearest[
    nearest[
        "accident_distance_m"
    ] <= 80
]


print(
    positive_nearest[
        "accident_type"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# 14. 저장용 컬럼 정리
#
# 아직 label은 넣지 않음.
# 거리와 최근접 사고정보만 저장.
# ============================================================

print(
    "\n[11] 저장..."
)


output = nearest[
    [
        "link_id",
        "accident_distance_m",
        "사고다발지fid",
        "사고다발지id",
        "accident_type",
        "지점명",
        "사고건수",
    ]
].copy()


# 숫자형 보장
output[
    "accident_distance_m"
] = pd.to_numeric(
    output[
        "accident_distance_m"
    ],
    errors="coerce",
)


output.to_parquet(
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
# 15. 저장 결과 재검증
# ============================================================

print(
    "\n[12] 저장 결과 검증..."
)


print(
    "전체 행:",
    len(output)
)

print(
    "link_id 중복:",
    output[
        "link_id"
    ].duplicated().sum()
)

print(
    "link_id 결측:",
    output[
        "link_id"
    ].isna().sum()
)

print(
    "거리 결측:",
    output[
        "accident_distance_m"
    ].isna().sum()
)


# ============================================================
# 16. 최종 요약
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== LINK-사고지점 거리 계산 요약 ==="
)

print(
    "전체 LINK:",
    len(link_work)
)

print(
    "사고지점:",
    len(accident_work)
)

print(
    "최종 거리 결과:",
    len(output)
)

print(
    "80m 이하:",
    positive_count
)

print(
    "80~300m:",
    gray_count
)

print(
    "300m 초과:",
    negative_count
)

print(
    "거리 결측:",
    output[
        "accident_distance_m"
    ].isna().sum()
)

print(
    "link_id 중복:",
    output[
        "link_id"
    ].duplicated().sum()
)


print(
    "\n※ 아직 label은 생성하지 않았습니다."
)

print(
    "=== SafeRoute LINK-사고지점 최소거리 계산 완료 ==="
)