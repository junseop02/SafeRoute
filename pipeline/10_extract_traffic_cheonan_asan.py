from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAFFIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "traffic"
)

BOUNDARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundary"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "traffic_points_cheonan_asan.parquet"
)


# ============================================================
# 2. AADT Excel 파일 찾기
# ============================================================

print("=== SafeRoute 천안·아산 AADT 지점 추출 ===")

print("\n[1] AADT Excel 파일 찾기...")

excel_files = list(
    TRAFFIC_DIR.rglob("*.xlsx")
)

if len(excel_files) == 0:
    raise FileNotFoundError(
        "data/raw/traffic에서 xlsx 파일을 찾지 못했습니다."
    )

if len(excel_files) > 1:
    print(
        "주의: Excel 파일이 여러 개 있습니다."
    )

    for path in excel_files:
        print(" -", path)


traffic_path = excel_files[0]

print("사용 파일:")
print(traffic_path)


# ============================================================
# 3. AADT 데이터 불러오기
# ============================================================

print("\n[2] X,Y좌표 Sheet 불러오기...")

traffic = pd.read_excel(
    traffic_path,
    sheet_name="X,Y좌표"
)

print(
    "전국 AADT 지점:",
    len(traffic)
)


# ============================================================
# 4. 좌표 유효성 검사
#
# 확인된 원본 데이터:
# XCODE = 위도
# YCODE = 경도
# ============================================================

print("\n[3] 좌표 검사...")

traffic["XCODE"] = pd.to_numeric(
    traffic["XCODE"],
    errors="coerce"
)

traffic["YCODE"] = pd.to_numeric(
    traffic["YCODE"],
    errors="coerce"
)

print(
    "XCODE 결측:",
    traffic["XCODE"].isna().sum()
)

print(
    "YCODE 결측:",
    traffic["YCODE"].isna().sum()
)

if (
    traffic["XCODE"].isna().any()
    or traffic["YCODE"].isna().any()
):
    raise ValueError(
        "좌표에 숫자로 변환할 수 없는 값이 있습니다."
    )


# ============================================================
# 5. GeoDataFrame 생성
#
# EPSG:4326에서는
# x = longitude = YCODE
# y = latitude  = XCODE
# ============================================================

print("\n[4] AADT Point 생성...")

traffic_gdf = gpd.GeoDataFrame(
    traffic.copy(),
    geometry=gpd.points_from_xy(
        traffic["YCODE"],
        traffic["XCODE"]
    ),
    crs="EPSG:4326"
)

print(
    "생성된 Point:",
    len(traffic_gdf)
)

print(
    "원본 CRS:",
    traffic_gdf.crs
)


# ============================================================
# 6. EPSG:5179 변환
# ============================================================

print("\n[5] EPSG:5179 변환...")

traffic_gdf = traffic_gdf.to_crs(
    epsg=5179
)

print(
    "변환 CRS:",
    traffic_gdf.crs
)


# ============================================================
# 7. SGIS 시군구 경계 찾기
# ============================================================

print("\n[6] 시군구 경계 파일 찾기...")

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
    "사용 경계:",
    boundary_path
)


# ============================================================
# 8. 경계 불러오기
# ============================================================

print("\n[7] 시군구 경계 불러오기...")

boundary = gpd.read_file(
    boundary_path
)

print(
    "전체 경계:",
    len(boundary)
)

print(
    "경계 CRS:",
    boundary.crs
)


# ============================================================
# 9. 천안·아산 경계 추출
#
# SGIS 코드
# 34011 = 천안시 동남구
# 34012 = 천안시 서북구
# 34040 = 아산시
# ============================================================

print("\n[8] 천안·아산 경계 추출...")

target_sgis_codes = [
    "34011",
    "34012",
    "34040",
]

boundary["SIGUNGU_CD"] = (
    boundary["SIGUNGU_CD"]
    .astype(str)
)

target_boundary = boundary[
    boundary["SIGUNGU_CD"].isin(
        target_sgis_codes
    )
].copy()

print(
    target_boundary[
        [
            "SIGUNGU_CD",
            "SIGUNGU_NM",
        ]
    ].to_string(index=False)
)

print(
    "추출 경계 수:",
    len(target_boundary)
)

if len(target_boundary) != 3:
    raise ValueError(
        "천안·아산 경계 3개를 정확히 찾지 못했습니다."
    )


# ============================================================
# 10. CRS 통일
# ============================================================

print("\n[9] CRS 통일...")

if target_boundary.crs != traffic_gdf.crs:
    target_boundary = (
        target_boundary.to_crs(
            traffic_gdf.crs
        )
    )

print(
    "AADT CRS:",
    traffic_gdf.crs
)

print(
    "경계 CRS:",
    target_boundary.crs
)


# ============================================================
# 11. 공간조인
#
# 실제 좌표가 천안·아산 경계 내부에 있는
# AADT 지점만 추출
# ============================================================

print("\n[10] 천안·아산 AADT 공간조인...")

target_boundary_join = target_boundary[
    [
        "SIGUNGU_CD",
        "SIGUNGU_NM",
        "geometry",
    ]
].copy()

traffic_joined = gpd.sjoin(
    traffic_gdf,
    target_boundary_join,
    how="inner",
    predicate="within"
)

print(
    "천안·아산 AADT 지점:",
    len(traffic_joined)
)


# ============================================================
# 12. SafeRoute region_code 부여
# ============================================================

print("\n[11] region_code 부여...")

region_map = {
    "34011": 44131,
    "34012": 44133,
    "34040": 44200,
}

traffic_joined["region_code"] = (
    traffic_joined[
        "SIGUNGU_CD"
    ].map(region_map)
)

print(
    "region_code 결측:",
    traffic_joined[
        "region_code"
    ].isna().sum()
)


# ============================================================
# 13. 지역별 지점 수
# ============================================================

print("\n[12] 지역별 AADT 지점 수")

region_counts = (
    traffic_joined
    .groupby(
        [
            "SIGUNGU_CD",
            "SIGUNGU_NM",
            "region_code",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

print(
    region_counts.to_string(
        index=False
    )
)


# ============================================================
# 14. 도로종류별 분포
# ============================================================

print("\n[13] 도로종류별 분포")

print(
    traffic_joined[
        "도로종류"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 15. AADT 통계
# ============================================================

print("\n[14] AADT 통계")

print(
    traffic_joined[
        "AADT"
    ].describe()
)


# ============================================================
# 16. 추출 결과 샘플
# ============================================================

print("\n[15] 추출 결과 샘플")

sample_columns = [
    "연도",
    "도로종류",
    "지점번호",
    "호선",
    "구간",
    "차로",
    "도/시",
    "시/군",
    "읍/면",
    "동/리",
    "구분",
    "AADT",
    "XCODE",
    "YCODE",
    "SIGUNGU_CD",
    "SIGUNGU_NM",
    "region_code",
]

print(
    traffic_joined[
        sample_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 17. 좌표 중복 검사
# ============================================================

print("\n[16] 중복 검사")

duplicate_points = (
    traffic_joined[
        [
            "XCODE",
            "YCODE",
        ]
    ]
    .duplicated()
    .sum()
)

print(
    "중복 좌표:",
    duplicate_points
)

print(
    "지점번호 중복:",
    traffic_joined[
        "지점번호"
    ]
    .astype(str)
    .duplicated()
    .sum()
)


# ============================================================
# 18. 저장 전 필요한 컬럼만 정리
# ============================================================

print("\n[17] 저장용 데이터 정리...")

# 지점번호는 숫자와 문자열이 혼합되어 있으므로
# Parquet 저장을 위해 문자열로 통일
traffic_joined["지점번호"] = (
    traffic_joined["지점번호"]
    .astype(str)
)

output_columns = [
    "연도",
    "도로종류",
    "지점번호",
    "호선",
    "구간",
    "차로",
    "도/시",
    "시/군",
    "읍/면",
    "동/리",
    "구분",
    "AADT",
    "XCODE",
    "YCODE",
    "SIGUNGU_CD",
    "SIGUNGU_NM",
    "region_code",
    "geometry",
]

traffic_output = traffic_joined[
    output_columns
].copy()

traffic_output = gpd.GeoDataFrame(
    traffic_output,
    geometry="geometry",
    crs=traffic_gdf.crs
)

print(
    "최종 행:",
    len(traffic_output)
)

print(
    "최종 컬럼:"
)

print(
    traffic_output.columns.tolist()
)


# ============================================================
# 19. 저장
# ============================================================

print("\n[18] 저장...")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

traffic_output.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    "저장:",
    OUTPUT_PATH
)

print(
    "파일 존재:",
    OUTPUT_PATH.exists()
)


# ============================================================
# 20. 완료
# ============================================================

print(
    "\n=== 천안·아산 AADT 지점 추출 완료 ==="
)