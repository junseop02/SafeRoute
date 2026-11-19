from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ACCIDENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "accident"
)

BOUNDARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundary"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "accident_points_cheonan_asan.parquet"
)


# SafeRoute 지역 코드
REGION_MAP = {
    "34011": 44131,  # 천안시 동남구
    "34012": 44133,  # 천안시 서북구
    "34040": 44200,  # 아산시
}


# ============================================================
# 2. CSV 읽기 함수
# ============================================================

def read_csv_auto(path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]

    for encoding in encodings:

        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )

        except Exception:
            pass

    raise RuntimeError(
        f"CSV 읽기 실패: {path}"
    )


# ============================================================
# 3. 시작
# ============================================================

print(
    "=== SafeRoute 천안·아산 사고다발지점 추출 ==="
)


# ============================================================
# 4. SGIS 경계 파일 찾기
# ============================================================

print(
    "\n[1] SGIS 경계 파일 탐색..."
)

boundary_files = list(
    BOUNDARY_DIR.rglob(
        "bnd_sigungu_00_2025_2Q.shp"
    )
)


if len(boundary_files) == 0:

    raise FileNotFoundError(
        "bnd_sigungu_00_2025_2Q.shp를 찾지 못했습니다."
    )


boundary_path = boundary_files[0]

print(
    "경계 파일:",
    boundary_path
)


# ============================================================
# 5. 천안·아산 경계 추출
# ============================================================

print(
    "\n[2] 천안·아산 경계 추출..."
)

boundary = gpd.read_file(
    boundary_path
)

boundary["SIGUNGU_CD"] = (
    boundary["SIGUNGU_CD"]
    .astype(str)
)


target_boundary = boundary[
    boundary["SIGUNGU_CD"].isin(
        REGION_MAP.keys()
    )
].copy()


target_boundary[
    "region_code"
] = (
    target_boundary[
        "SIGUNGU_CD"
    ]
    .map(
        REGION_MAP
    )
)


print(
    "경계 CRS:",
    target_boundary.crs
)

print(
    "대상 경계 수:",
    len(target_boundary)
)


print(
    target_boundary[
        [
            "SIGUNGU_CD",
            "SIGUNGU_NM",
            "region_code",
        ]
    ].to_string(
        index=False
    )
)


if len(target_boundary) != 3:

    raise RuntimeError(
        "천안 동남구/서북구/아산시 경계 3개를 "
        "정상적으로 찾지 못했습니다."
    )


# ============================================================
# 6. 사고 CSV 탐색
# ============================================================

print(
    "\n[3] 사고 CSV 탐색..."
)

csv_files = sorted(
    ACCIDENT_DIR.rglob(
        "*.csv"
    )
)


print(
    "CSV 파일 수:",
    len(csv_files)
)


if len(csv_files) == 0:

    raise FileNotFoundError(
        "사고 CSV를 찾지 못했습니다."
    )


# ============================================================
# 7. 사고 데이터 통합
# ============================================================

print(
    "\n[4] 전국 사고다발지점 통합..."
)

accident_frames = []


for path in csv_files:

    df = read_csv_auto(
        path
    )


    required_columns = [
        "사고다발지fid",
        "사고다발지id",
        "시도시군구명",
        "지점명",
        "사고건수",
        "경도",
        "위도",
    ]


    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]


    if missing:

        raise RuntimeError(
            f"{path.name} 필수 컬럼 누락: {missing}"
        )


    # 파일명으로 사고 유형 보존
    df["source_file"] = path.name


    # 파일명 앞부분 제거해서 보기 편한 유형 생성
    accident_type = (
        path.stem
        .replace(
            "12_25_",
            "",
        )
        .replace(
            "13_25_",
            "",
        )
        .replace(
            "15_25_",
            "",
        )
        .replace(
            "17_25_",
            "",
        )
        .replace(
            "19_25_",
            "",
        )
    )


    df["accident_type"] = (
        accident_type
    )


    accident_frames.append(
        df
    )


accidents = pd.concat(
    accident_frames,
    ignore_index=True,
    sort=False,
)


print(
    "전국 통합 사고지점:",
    len(accidents)
)

print(
    "사고 유형 수:",
    accidents[
        "accident_type"
    ].nunique()
)


# ============================================================
# 8. 좌표 숫자 변환
# ============================================================

print(
    "\n[5] 사고 좌표 검증..."
)

accidents["경도"] = pd.to_numeric(
    accidents["경도"],
    errors="coerce",
)

accidents["위도"] = pd.to_numeric(
    accidents["위도"],
    errors="coerce",
)


valid_coord = (
    accidents["경도"].notna()
    & accidents["위도"].notna()
)


print(
    "유효 좌표:",
    int(
        valid_coord.sum()
    )
)

print(
    "좌표 결측:",
    int(
        (~valid_coord).sum()
    )
)


accidents = accidents[
    valid_coord
].copy()


# ============================================================
# 9. GeoDataFrame 생성
#
# 원본 사고 데이터:
# 경도/위도 → EPSG:4326
# ============================================================

print(
    "\n[6] 사고지점 GeoDataFrame 생성..."
)

accident_gdf = gpd.GeoDataFrame(
    accidents,
    geometry=gpd.points_from_xy(
        accidents["경도"],
        accidents["위도"],
    ),
    crs="EPSG:4326",
)


print(
    "원본 사고 CRS:",
    accident_gdf.crs
)


# ============================================================
# 10. SGIS 경계 CRS로 변환
# ============================================================

accident_gdf = (
    accident_gdf.to_crs(
        target_boundary.crs
    )
)


print(
    "변환 사고 CRS:",
    accident_gdf.crs
)


# ============================================================
# 11. 천안·아산 공간 추출
# ============================================================

print(
    "\n[7] 천안·아산 공간 추출..."
)

boundary_for_join = (
    target_boundary[
        [
            "SIGUNGU_CD",
            "SIGUNGU_NM",
            "region_code",
            "geometry",
        ]
    ].copy()
)


target_accidents = gpd.sjoin(
    accident_gdf,
    boundary_for_join,
    how="inner",
    predicate="within",
)


target_accidents = (
    target_accidents
    .drop(
        columns=[
            "index_right",
        ],
        errors="ignore",
    )
    .reset_index(
        drop=True
    )
)


print(
    "천안·아산 사고지점:",
    len(target_accidents)
)


# ============================================================
# 12. 지역별 개수
# ============================================================

print(
    "\n[8] 지역별 사고지점 수"
)

region_summary = (
    target_accidents.groupby(
        [
            "region_code",
            "SIGUNGU_NM",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)


print(
    region_summary.to_string(
        index=False
    )
)


# ============================================================
# 13. 사고 유형별 개수
# ============================================================

print(
    "\n[9] 사고 유형별 사고지점 수"
)

type_summary = (
    target_accidents[
        "accident_type"
    ]
    .value_counts()
)


print(
    type_summary.to_string()
)


# ============================================================
# 14. 지역 × 사고유형
# ============================================================

print(
    "\n[10] 지역 × 사고유형"
)

region_type_summary = pd.crosstab(
    target_accidents[
        "SIGUNGU_NM"
    ],
    target_accidents[
        "accident_type"
    ],
)


print(
    region_type_summary.to_string()
)


# ============================================================
# 15. 사고다발지 ID 중복 검사
# ============================================================

print(
    "\n[11] 사고다발지id 중복 검사"
)

id_text = (
    target_accidents[
        "사고다발지id"
    ]
    .astype(str)
)


id_duplicate = (
    id_text.duplicated(
        keep=False
    )
)


print(
    "중복 ID에 포함된 행:",
    int(
        id_duplicate.sum()
    )
)

print(
    "중복된 고유 ID 수:",
    int(
        id_text[
            id_duplicate
        ].nunique()
    )
)


# ============================================================
# 16. 정확히 동일한 좌표 중복 검사
# ============================================================

print(
    "\n[12] 동일 좌표 중복 검사"
)

coordinate_duplicate = (
    target_accidents.duplicated(
        subset=[
            "경도",
            "위도",
        ],
        keep=False,
    )
)


print(
    "동일 좌표 중복에 포함된 행:",
    int(
        coordinate_duplicate.sum()
    )
)


duplicate_coordinates = (
    target_accidents.loc[
        coordinate_duplicate,
        [
            "경도",
            "위도",
        ],
    ]
    .drop_duplicates()
)


print(
    "중복된 고유 좌표 수:",
    len(
        duplicate_coordinates
    )
)


# ============================================================
# 17. 중복 좌표가 여러 사고 유형에 존재하는지 검사
# ============================================================

print(
    "\n[13] 동일 좌표의 사고유형 중복 검사"
)


coord_type_count = (
    target_accidents.groupby(
        [
            "경도",
            "위도",
        ]
    )[
        "accident_type"
    ]
    .nunique()
)


multi_type_coordinates = (
    coord_type_count[
        coord_type_count > 1
    ]
)


print(
    "2개 이상 사고유형에 등장한 좌표:",
    len(
        multi_type_coordinates
    )
)


# ============================================================
# 18. 중복 사례 일부 출력
# ============================================================

if len(
    multi_type_coordinates
) > 0:

    print(
        "\n[동일 위치 다중 사고유형 샘플]"
    )


    sample_coords = (
        multi_type_coordinates
        .head(10)
        .index
    )


    sample_mask = pd.Series(
        False,
        index=target_accidents.index,
    )


    for lon, lat in sample_coords:

        sample_mask |= (
            (
                target_accidents[
                    "경도"
                ]
                == lon
            )
            &
            (
                target_accidents[
                    "위도"
                ]
                == lat
            )
        )


    print(
        target_accidents.loc[
            sample_mask,
            [
                "사고다발지id",
                "accident_type",
                "SIGUNGU_NM",
                "지점명",
                "사고건수",
                "경도",
                "위도",
            ],
        ]
        .sort_values(
            [
                "경도",
                "위도",
            ]
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 19. 사고건수 기본 통계
# ============================================================

print(
    "\n[14] 사고건수 통계"
)

accident_count = pd.to_numeric(
    target_accidents[
        "사고건수"
    ],
    errors="coerce",
)


print(
    accident_count.describe()
)


# ============================================================
# 20. 저장용 컬럼 정리
#
# 원본 정보는 최대한 보존하되,
# label 생성에 필요한 핵심 컬럼을 앞쪽에 배치
# ============================================================

print(
    "\n[15] 저장..."
)


keep_columns = [
    "사고다발지fid",
    "사고다발지id",
    "accident_type",
    "source_file",
    "법정동코드",
    "지점코드",
    "시도시군구명",
    "SIGUNGU_NM",
    "region_code",
    "지점명",
    "사고건수",
    "사상자수",
    "사망자수",
    "중상자수",
    "경상자수",
    "부상신고자수",
    "경도",
    "위도",
]


# 법규위반 CSV에만 존재
if "법규위반" in target_accidents.columns:

    keep_columns.append(
        "법규위반"
    )


keep_columns.append(
    "geometry"
)


keep_columns = [
    col
    for col in keep_columns
    if col in target_accidents.columns
]


output = target_accidents[
    keep_columns
].copy()


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
# 21. 최종 검증
# ============================================================

print(
    "\n=== 사고지점 추출 요약 ==="
)

print(
    "전국 사고지점:",
    len(accidents)
)

print(
    "천안·아산 사고지점:",
    len(output)
)

print(
    "지역코드 결측:",
    output[
        "region_code"
    ].isna().sum()
)

print(
    "geometry 결측:",
    output[
        "geometry"
    ].isna().sum()
)

print(
    "사고 유형 수:",
    output[
        "accident_type"
    ].nunique()
)

print(
    "동일 좌표 중복 행:",
    int(
        coordinate_duplicate.sum()
    )
)

print(
    "다중 사고유형 좌표:",
    len(
        multi_type_coordinates
    )
)

print(
    "\n=== SafeRoute 천안·아산 사고지점 추출 완료 ==="
)