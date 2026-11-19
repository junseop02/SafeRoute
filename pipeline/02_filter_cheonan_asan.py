from pathlib import Path
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOUNDARY_ROOT = PROJECT_ROOT / "data" / "raw" / "boundary"

# 하위 폴더 전체에서 시군구 SHP 자동 검색
matches = list(
    BOUNDARY_ROOT.rglob("bnd_sigungu_00_2025_2Q.shp")
)

if len(matches) != 1:
    raise RuntimeError(
        f"시군구 SHP를 정확히 1개 찾아야 합니다. 현재: {len(matches)}개"
    )

BOUNDARY_PATH = matches[0]

print("=== SafeRoute 시군구 경계 검사 ===")
print("경계 파일:", BOUNDARY_PATH)

sigungu = gpd.read_file(BOUNDARY_PATH)

print("\n[기본 정보]")
print("행 개수:", len(sigungu))
print("CRS:", sigungu.crs)

print("\n[컬럼]")
print(sigungu.columns.tolist())

print("\n[앞 10개 데이터]")
print(sigungu.drop(columns="geometry").head(10))

print("\n[geometry 종류]")
print(sigungu.geometry.geom_type.value_counts())

# 천안·아산 후보 확인
print("\n[천안·아산 시군구 확인]")

target = sigungu[
    sigungu["SIGUNGU_NM"].str.contains(
        "동남구|서북구|아산시",
        na=False
    )
]

print(
    target[
        ["BASE_DATE", "SIGUNGU_CD", "SIGUNGU_NM"]
    ].to_string(index=False)
)

print("\n=== 검사 완료 ===")