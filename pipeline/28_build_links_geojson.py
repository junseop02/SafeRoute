from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 03번에서 생성한 천안·아산 LINK
LINK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "links_cheonan_asan.parquet"
)

# 최종 모델 출력
RISK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "risk.parquet"
)

# 최종 GeoJSON
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "links.geojson"
)


EXPECTED_LINK_COUNT = 36521


print(
    "=== SafeRoute links.geojson 생성 ==="
)


# ============================================================
# 2. 파일 존재 확인
# ============================================================

print(
    "\n[1] 파일 존재 확인..."
)


if not LINK_PATH.exists():

    raise FileNotFoundError(
        f"LINK 파일이 없습니다: {LINK_PATH}"
    )


if not RISK_PATH.exists():

    raise FileNotFoundError(
        f"risk.parquet 파일이 없습니다: {RISK_PATH}"
    )


print(
    "LINK:",
    LINK_PATH
)

print(
    "Risk:",
    RISK_PATH
)


# ============================================================
# 3. 데이터 불러오기
# ============================================================

print(
    "\n[2] 데이터 불러오기..."
)


links = gpd.read_parquet(
    LINK_PATH
)

risk = pd.read_parquet(
    RISK_PATH
)


print(
    "LINK:",
    len(links)
)

print(
    "Risk:",
    len(risk)
)

print(
    "LINK CRS:",
    links.crs
)


# ============================================================
# 4. LINK_ID 컬럼 확인
#
# 원본 국가표준노드링크는 LINK_ID 사용
# 최종 인터페이스에서는 link_id 사용
# ============================================================

print(
    "\n[3] LINK ID 컬럼 확인..."
)


if "link_id" in links.columns:

    link_id_column = "link_id"

elif "LINK_ID" in links.columns:

    link_id_column = "LINK_ID"

else:

    raise RuntimeError(
        "LINK 데이터에서 link_id 또는 LINK_ID를 찾을 수 없습니다."
    )


print(
    "사용 LINK ID 컬럼:",
    link_id_column
)


# ============================================================
# 5. 최종 GeoDataFrame 생성
#
# 백엔드 geometry 파일에는
# link_id + geometry만 포함
# ============================================================

print(
    "\n[4] GeoJSON 데이터 생성..."
)


geo_df = links[
    [
        link_id_column,
        "geometry",
    ]
].copy()


geo_df = geo_df.rename(
    columns={
        link_id_column: "link_id"
    }
)


geo_df[
    "link_id"
] = (
    geo_df[
        "link_id"
    ]
    .astype(str)
)


risk[
    "link_id"
] = (
    risk[
        "link_id"
    ]
    .astype(str)
)


print(
    "GeoJSON 대상 LINK:",
    len(geo_df)
)


# ============================================================
# 6. 행 수 검사
# ============================================================

print(
    "\n[5] 행 수 검사..."
)


if len(geo_df) != EXPECTED_LINK_COUNT:

    raise RuntimeError(
        f"LINK 행 수 오류: {len(geo_df)} "
        f"(기대값: {EXPECTED_LINK_COUNT})"
    )


if len(geo_df) != len(risk):

    raise RuntimeError(
        "GeoJSON LINK 수와 risk.parquet LINK 수가 다릅니다."
    )


print(
    "행 수:",
    len(geo_df)
)

print(
    "행 수 검사: PASS"
)


# ============================================================
# 7. link_id 검사
# ============================================================

print(
    "\n[6] link_id 검사..."
)


link_missing = int(
    geo_df[
        "link_id"
    ]
    .isna()
    .sum()
)


link_duplicate = int(
    geo_df[
        "link_id"
    ]
    .duplicated()
    .sum()
)


print(
    "link_id 결측:",
    link_missing
)

print(
    "link_id 중복:",
    link_duplicate
)


if link_missing != 0:

    raise RuntimeError(
        "GeoJSON에 link_id 결측이 존재합니다."
    )


if link_duplicate != 0:

    raise RuntimeError(
        "GeoJSON에 link_id 중복이 존재합니다."
    )


print(
    "link_id 검사: PASS"
)


# ============================================================
# 8. geometry 검사
# ============================================================

print(
    "\n[7] geometry 검사..."
)


geometry_missing = int(
    geo_df[
        "geometry"
    ]
    .isna()
    .sum()
)


geometry_empty = int(
    geo_df[
        "geometry"
    ]
    .is_empty
    .sum()
)


geometry_valid = int(
    geo_df[
        "geometry"
    ]
    .is_valid
    .sum()
)


geometry_invalid = (
    len(geo_df)
    - geometry_valid
)


print(
    "geometry 결측:",
    geometry_missing
)

print(
    "geometry empty:",
    geometry_empty
)

print(
    "geometry invalid:",
    geometry_invalid
)


if geometry_missing != 0:

    raise RuntimeError(
        "geometry 결측이 존재합니다."
    )


if geometry_empty != 0:

    raise RuntimeError(
        "빈 geometry가 존재합니다."
    )


if geometry_invalid != 0:

    raise RuntimeError(
        "유효하지 않은 geometry가 존재합니다."
    )


print(
    "geometry 검사: PASS"
)


# ============================================================
# 9. geometry 타입 확인
# ============================================================

print(
    "\n[8] geometry 타입 확인..."
)


geometry_types = (
    geo_df[
        "geometry"
    ]
    .geom_type
    .value_counts()
)


print(
    geometry_types.to_string()
)


# ============================================================
# 10. risk.parquet과 link_id 집합 비교
# ============================================================

print(
    "\n[9] risk.parquet과 LINK 일치 검사..."
)


geo_links = set(
    geo_df[
        "link_id"
    ]
)


risk_links = set(
    risk[
        "link_id"
    ]
)


missing_in_geo = (
    risk_links
    - geo_links
)


extra_in_geo = (
    geo_links
    - risk_links
)


print(
    "GeoJSON에 없는 Risk LINK:",
    len(
        missing_in_geo
    )
)

print(
    "GeoJSON에만 있는 LINK:",
    len(
        extra_in_geo
    )
)


if missing_in_geo:

    raise RuntimeError(
        "risk.parquet에는 있지만 GeoJSON에는 없는 LINK가 있습니다."
    )


if extra_in_geo:

    raise RuntimeError(
        "GeoJSON에는 있지만 risk.parquet에는 없는 LINK가 있습니다."
    )


print(
    "LINK 일치 검사: PASS"
)


# ============================================================
# 11. CRS 확인 및 EPSG:4326 변환
#
# 현재 도로 데이터는 EPSG:5179 기반.
#
# GeoJSON / 웹 지도 / 백엔드 전달용 좌표는
# WGS84 EPSG:4326으로 변환.
# ============================================================

print(
    "\n[10] CRS 변환..."
)


if geo_df.crs is None:

    raise RuntimeError(
        "LINK geometry에 CRS 정보가 없습니다."
    )


print(
    "변환 전 CRS:",
    geo_df.crs
)


geo_df = geo_df.to_crs(
    epsg=4326
)


print(
    "변환 후 CRS:",
    geo_df.crs
)


# ============================================================
# 12. 좌표 범위 확인
#
# 천안·아산이면 대한민국 영역의
# 정상적인 경위도 범위에 있어야 함.
# ============================================================

print(
    "\n[11] 좌표 범위 확인..."
)


minx, miny, maxx, maxy = (
    geo_df.total_bounds
)


print(
    f"Longitude: {minx:.6f} ~ {maxx:.6f}"
)

print(
    f"Latitude : {miny:.6f} ~ {maxy:.6f}"
)


# 대한민국 밖으로 명백히 벗어난 경우만 오류 처리
if not (
    124.0 <= minx <= 132.0
    and
    124.0 <= maxx <= 132.0
    and
    33.0 <= miny <= 39.5
    and
    33.0 <= maxy <= 39.5
):

    raise RuntimeError(
        "EPSG:4326 변환 후 좌표 범위가 비정상적입니다."
    )


print(
    "좌표 범위 검사: PASS"
)


# ============================================================
# 13. 정렬
# ============================================================

print(
    "\n[12] link_id 정렬..."
)


geo_df = (
    geo_df
    .sort_values(
        "link_id"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 14. GeoJSON 저장
# ============================================================

print(
    "\n[13] links.geojson 저장..."
)


geo_df.to_file(
    OUTPUT_PATH,
    driver="GeoJSON",
)


print(
    "저장:",
    OUTPUT_PATH
)


# ============================================================
# 15. 저장 후 다시 불러오기
# ============================================================

print(
    "\n[14] 저장 파일 재검증..."
)


check = gpd.read_file(
    OUTPUT_PATH
)


check[
    "link_id"
] = (
    check[
        "link_id"
    ]
    .astype(str)
)


print(
    "저장 LINK:",
    len(check)
)

print(
    "저장 컬럼:",
    check.columns.tolist()
)

print(
    "저장 CRS:",
    check.crs
)


saved_duplicate = int(
    check[
        "link_id"
    ]
    .duplicated()
    .sum()
)


saved_geometry_missing = int(
    check[
        "geometry"
    ]
    .isna()
    .sum()
)


saved_geometry_empty = int(
    check[
        "geometry"
    ]
    .is_empty
    .sum()
)


print(
    "link_id 중복:",
    saved_duplicate
)

print(
    "geometry 결측:",
    saved_geometry_missing
)

print(
    "geometry empty:",
    saved_geometry_empty
)


if len(check) != EXPECTED_LINK_COUNT:

    raise RuntimeError(
        "저장 후 LINK 수가 달라졌습니다."
    )


if saved_duplicate != 0:

    raise RuntimeError(
        "저장 후 link_id 중복이 발생했습니다."
    )


if saved_geometry_missing != 0:

    raise RuntimeError(
        "저장 후 geometry 결측이 발생했습니다."
    )


if saved_geometry_empty != 0:

    raise RuntimeError(
        "저장 후 빈 geometry가 발생했습니다."
    )


# ============================================================
# 16. 최종 risk.parquet과 다시 비교
# ============================================================

print(
    "\n[15] 저장 후 risk.parquet과 최종 비교..."
)


saved_links = set(
    check[
        "link_id"
    ]
)


risk_links = set(
    risk[
        "link_id"
    ]
)


final_missing = (
    risk_links
    - saved_links
)


final_extra = (
    saved_links
    - risk_links
)


print(
    "누락:",
    len(
        final_missing
    )
)

print(
    "추가:",
    len(
        final_extra
    )
)


if final_missing:

    raise RuntimeError(
        "저장된 GeoJSON에 누락된 LINK가 있습니다."
    )


if final_extra:

    raise RuntimeError(
        "저장된 GeoJSON에 추가 LINK가 있습니다."
    )


# ============================================================
# 17. 샘플 확인
# ============================================================

print(
    "\n[16] GeoJSON 샘플..."
)


print(
    check[
        [
            "link_id",
            "geometry",
        ]
    ]
    .head(5)
    .to_string(
        index=False
    )
)


# ============================================================
# 18. 최종 결과
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute links.geojson 생성 완료 ==="
)

print(
    "=" * 70
)


print(
    "\n최종 파일:"
)

print(
    OUTPUT_PATH
)


print(
    "\n인터페이스:"
)

print(
    "link_id  : str"
)

print(
    "geometry : LineString"
)

print(
    "CRS      : EPSG:4326"
)


print(
    "\n검증 결과:"
)

print(
    f"LINK 수          : {len(check)}"
)

print(
    f"link_id 중복     : {saved_duplicate}"
)

print(
    f"geometry 결측    : {saved_geometry_missing}"
)

print(
    f"geometry empty   : {saved_geometry_empty}"
)

print(
    f"risk 누락 LINK   : {len(final_missing)}"
)

print(
    f"risk 추가 LINK   : {len(final_extra)}"
)


print(
    "\n모든 검사: PASS"
)


print(
    "\n백엔드 전달 핵심 파일:"
)

print(
    "- outputs/risk.parquet"
)

print(
    "- data/processed/links.geojson"
)


print(
    "\n=== 완료 ==="
)