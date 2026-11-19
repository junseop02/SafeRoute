from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "accident_points_cheonan_asan.parquet"
)


# ============================================================
# 2. 시작
# ============================================================

print(
    "=== SafeRoute 사고지점 중복 구조 검사 ==="
)


# ============================================================
# 3. 데이터 불러오기
# ============================================================

print(
    "\n[1] 데이터 불러오기..."
)

accidents = gpd.read_parquet(
    INPUT_PATH
)


print(
    "전체 사고지점:",
    len(accidents)
)

print(
    "CRS:",
    accidents.crs
)


# ============================================================
# 4. 기본 검증
# ============================================================

print(
    "\n[2] 기본 검증..."
)

required_columns = [
    "사고다발지fid",
    "사고다발지id",
    "accident_type",
    "source_file",
    "region_code",
    "지점명",
    "사고건수",
    "경도",
    "위도",
    "geometry",
]


missing_columns = [
    col
    for col in required_columns
    if col not in accidents.columns
]


if missing_columns:

    raise RuntimeError(
        f"필수 컬럼 누락: {missing_columns}"
    )


print(
    "필수 컬럼 확인 완료"
)

print(
    "geometry 결측:",
    accidents.geometry.isna().sum()
)


# ============================================================
# 5. 완전히 동일한 행 검사
#
# geometry/source_file 등을 제외하고
# 핵심 원본 정보가 완전히 같은 행이 있는지 확인
# ============================================================

print(
    "\n[3] 핵심 정보 완전 중복 검사..."
)


exact_columns = [
    "사고다발지fid",
    "사고다발지id",
    "accident_type",
    "지점명",
    "사고건수",
    "경도",
    "위도",
]


exact_duplicate_mask = (
    accidents.duplicated(
        subset=exact_columns,
        keep=False,
    )
)


exact_duplicate_rows = accidents[
    exact_duplicate_mask
].copy()


print(
    "완전 중복에 포함된 행:",
    len(exact_duplicate_rows)
)


if len(exact_duplicate_rows) > 0:

    print(
        "완전 중복 제거 시 삭제되는 행:",
        int(
            accidents.duplicated(
                subset=exact_columns,
                keep="first",
            ).sum()
        )
    )


# ============================================================
# 6. 사고다발지fid 중복 검사
# ============================================================

print(
    "\n[4] 사고다발지fid 중복 검사..."
)


fid_text = (
    accidents["사고다발지fid"]
    .astype(str)
)


fid_duplicate_mask = (
    fid_text.duplicated(
        keep=False
    )
)


print(
    "중복 FID에 포함된 행:",
    int(
        fid_duplicate_mask.sum()
    )
)

print(
    "중복된 고유 FID:",
    int(
        fid_text[
            fid_duplicate_mask
        ].nunique()
    )
)


# ============================================================
# 7. 사고다발지id 중복 검사
# ============================================================

print(
    "\n[5] 사고다발지id 중복 검사..."
)


id_text = (
    accidents["사고다발지id"]
    .astype(str)
)


id_duplicate_mask = (
    id_text.duplicated(
        keep=False
    )
)


duplicate_ids = (
    accidents.loc[
        id_duplicate_mask
    ]
    .copy()
)


print(
    "중복 ID에 포함된 행:",
    len(duplicate_ids)
)

print(
    "중복된 고유 ID:",
    int(
        id_text[
            id_duplicate_mask
        ].nunique()
    )
)


# ============================================================
# 8. 같은 ID가 몇 개 사고유형에 등장하는지
# ============================================================

print(
    "\n[6] 같은 ID의 사고유형 중복 검사..."
)


id_type_count = (
    accidents
    .assign(
        accident_id_text=id_text
    )
    .groupby(
        "accident_id_text"
    )[
        "accident_type"
    ]
    .nunique()
)


multi_type_ids = (
    id_type_count[
        id_type_count > 1
    ]
)


print(
    "2개 이상 사고유형에 등장하는 ID:",
    len(multi_type_ids)
)


# ============================================================
# 9. 같은 ID의 좌표가 동일한지 검사
# ============================================================

print(
    "\n[7] 같은 ID의 좌표 구조 검사..."
)


id_coord_count = (
    accidents
    .assign(
        accident_id_text=id_text
    )
    .groupby(
        "accident_id_text"
    )[
        ["경도", "위도"]
    ]
    .apply(
        lambda x: len(
            x.drop_duplicates()
        )
    )
)


duplicate_id_coord = (
    id_coord_count[
        id_coord_count > 1
    ]
)


print(
    "여러 좌표에 등장하는 ID:",
    len(duplicate_id_coord)
)


# ============================================================
# 10. 동일 좌표 검사
# ============================================================

print(
    "\n[8] 동일 좌표 검사..."
)


coord_group = (
    accidents.groupby(
        [
            "경도",
            "위도",
        ],
        dropna=False,
    )
)


coord_size = (
    coord_group
    .size()
)


duplicate_coord_groups = (
    coord_size[
        coord_size > 1
    ]
)


print(
    "중복된 고유 좌표:",
    len(
        duplicate_coord_groups
    )
)

print(
    "중복 좌표에 포함된 총 행:",
    int(
        duplicate_coord_groups.sum()
    )
)


# ============================================================
# 11. 동일 좌표의 사고유형 수
# ============================================================

print(
    "\n[9] 동일 좌표의 사고유형 검사..."
)


coord_type_count = (
    coord_group[
        "accident_type"
    ]
    .nunique()
)


multi_type_coords = (
    coord_type_count[
        coord_type_count > 1
    ]
)


print(
    "2개 이상 사고유형에 등장한 좌표:",
    len(
        multi_type_coords
    )
)


# ============================================================
# 12. 동일 좌표의 ID 수
# ============================================================

coord_id_count = (
    accidents
    .assign(
        accident_id_text=id_text
    )
    .groupby(
        [
            "경도",
            "위도",
        ]
    )[
        "accident_id_text"
    ]
    .nunique()
)


multi_id_coords = (
    coord_id_count[
        coord_id_count > 1
    ]
)


print(
    "2개 이상 ID가 존재하는 좌표:",
    len(
        multi_id_coords
    )
)


# ============================================================
# 13. 중복 ID 유형 요약
# ============================================================

print(
    "\n[10] 중복 ID 구조 요약..."
)


id_summary_rows = []


for accident_id, group in (
    accidents
    .assign(
        accident_id_text=id_text
    )
    .groupby(
        "accident_id_text"
    )
):

    if len(group) <= 1:
        continue


    coord_count = len(
        group[
            [
                "경도",
                "위도",
            ]
        ].drop_duplicates()
    )


    type_count = (
        group[
            "accident_type"
        ].nunique()
    )


    region_count = (
        group[
            "region_code"
        ].nunique()
    )


    id_summary_rows.append(
        {
            "accident_id": accident_id,
            "rows": len(group),
            "coord_count": coord_count,
            "type_count": type_count,
            "region_count": region_count,
        }
    )


id_summary = pd.DataFrame(
    id_summary_rows
)


if len(id_summary) > 0:

    print(
        "중복 ID 그룹 수:",
        len(id_summary)
    )

    print(
        "\n중복 ID 그룹의 행 개수 분포:"
    )

    print(
        id_summary[
            "rows"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print(
        "\n중복 ID 그룹의 좌표 개수 분포:"
    )

    print(
        id_summary[
            "coord_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print(
        "\n중복 ID 그룹의 사고유형 개수 분포:"
    )

    print(
        id_summary[
            "type_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# 14. 동일 좌표 중복 구조 요약
# ============================================================

print(
    "\n[11] 동일 좌표 중복 구조 요약..."
)


coord_summary_rows = []


for (
    lon,
    lat,
), group in coord_group:

    if len(group) <= 1:
        continue


    coord_summary_rows.append(
        {
            "경도": lon,
            "위도": lat,
            "rows": len(group),
            "id_count": (
                group[
                    "사고다발지id"
                ]
                .astype(str)
                .nunique()
            ),
            "type_count": (
                group[
                    "accident_type"
                ]
                .nunique()
            ),
            "region_count": (
                group[
                    "region_code"
                ]
                .nunique()
            ),
        }
    )


coord_summary = pd.DataFrame(
    coord_summary_rows
)


if len(coord_summary) > 0:

    print(
        "중복 좌표 그룹 수:",
        len(coord_summary)
    )


    print(
        "\n중복 좌표 그룹의 행 개수 분포:"
    )

    print(
        coord_summary[
            "rows"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print(
        "\n중복 좌표 그룹의 ID 개수 분포:"
    )

    print(
        coord_summary[
            "id_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print(
        "\n중복 좌표 그룹의 사고유형 개수 분포:"
    )

    print(
        coord_summary[
            "type_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# 15. 다중 사고유형 좌표 상세 출력
#
# 현재 3개뿐이므로 전체 출력
# ============================================================

print(
    "\n[12] 다중 사고유형 좌표 상세..."
)


if len(multi_type_coords) == 0:

    print(
        "해당 좌표 없음"
    )

else:

    multi_type_index = set(
        multi_type_coords.index
    )


    mask = [
        (lon, lat) in multi_type_index
        for lon, lat in zip(
            accidents["경도"],
            accidents["위도"],
        )
    ]


    detail = accidents.loc[
        mask,
        [
            "사고다발지fid",
            "사고다발지id",
            "accident_type",
            "region_code",
            "지점명",
            "사고건수",
            "경도",
            "위도",
        ],
    ].copy()


    print(
        detail
        .sort_values(
            [
                "경도",
                "위도",
                "accident_type",
            ]
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 16. ID 앞 4자리 패턴 검사
#
# 중요:
# 연도라고 해석하지 않음.
# 단순 문자열 패턴만 검사.
# ============================================================

print(
    "\n[13] 사고다발지id 앞 4자리 패턴..."
)

print(
    "※ 아래 값은 연도로 해석하지 않고 "
    "ID 문자열 패턴만 확인합니다."
)


id_prefix = (
    id_text
    .str.strip()
    .str[:4]
)


print(
    id_prefix
    .value_counts(
        dropna=False
    )
    .sort_index()
    .to_string()
)


print(
    "\n고유 앞 4자리 수:",
    id_prefix.nunique()
)


# ============================================================
# 17. ID 앞 4자리 × 사고유형
# ============================================================

print(
    "\n[14] ID 앞 4자리 × 사고유형..."
)


prefix_type_table = pd.crosstab(
    id_prefix,
    accidents[
        "accident_type"
    ],
)


print(
    prefix_type_table.to_string()
)


# ============================================================
# 18. 최종 요약
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== 사고지점 중복 검사 요약 ==="
)

print(
    "전체 사고지점:",
    len(accidents)
)

print(
    "완전 중복 포함 행:",
    len(
        exact_duplicate_rows
    )
)

print(
    "중복 ID 고유 개수:",
    int(
        id_text[
            id_duplicate_mask
        ].nunique()
    )
)

print(
    "여러 좌표에 등장하는 ID:",
    len(
        duplicate_id_coord
    )
)

print(
    "중복 좌표 고유 개수:",
    len(
        duplicate_coord_groups
    )
)

print(
    "다중 사고유형 좌표:",
    len(
        multi_type_coords
    )
)

print(
    "다중 ID 좌표:",
    len(
        multi_id_coords
    )
)


print(
    "\n※ 이 스크립트는 데이터를 수정하거나 삭제하지 않습니다."
)

print(
    "=== SafeRoute 사고지점 중복 구조 검사 완료 ==="
)