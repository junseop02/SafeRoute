from pathlib import Path
import json
import re

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.validation import make_valid

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data/legacy/raw"
    / "전국교통사고다발지역표준데이터.csv"
)

OUTPUT_DIR = ROOT / "data/accident_risk/processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 원본 사고 데이터 불러오기
df = pd.read_csv(DATA_PATH, encoding="cp949")

# 천안·아산 지역 추출
local = df[
    df["사고다발지역시도시군구"].str.contains(
        "천안시|아산시",
        na=False
    )
].copy().reset_index(drop=True)

print("===== Regional Data =====")
print("Total Records:", len(local))


# 비표준 JSON 폴리곤 변환
def parse_polygon(value):

    text = value.strip()

    # JSON 키에 따옴표 추가
    text = re.sub(
        r'([{,]\s*)(type|coordinates)\s*:',
        r'\1"\2":',
        text
    )

    # Geometry 유형에 따옴표 추가
    text = re.sub(
        r'("type"\s*:\s*)(Polygon|MultiPolygon)\b',
        r'\1"\2"',
        text
    )

    # Shapely Geometry 변환
    geometry = shape(json.loads(text))

    return geometry


# 폴리곤 변환
geometries = []
errors = []

for index, value in enumerate(
    local["사고다발지역폴리곤정보"]
):

    try:
        geometries.append(parse_polygon(value))

    except Exception as error:
        geometries.append(None)
        errors.append((index, str(error)))


print("\n===== Polygon Parsing =====")
print("Successful:", len(local) - len(errors))
print("Failed:", len(errors))

for index, error in errors[:5]:
    print(index, error)

# 변환 실패 시 중단
if errors:
    raise ValueError("Polygon parsing failed.")


# 좌표계 후보 EPSG:5179 적용
polygons = gpd.GeoDataFrame(
    local.drop(columns=["사고다발지역폴리곤정보"]),
    geometry=geometries,
    crs="EPSG:5179"
)


# Geometry 유효성 검사
print("\n===== Geometry Check =====")
print(polygons.geometry.geom_type.value_counts())

print("Missing:", polygons.geometry.isna().sum())
print("Empty:", polygons.geometry.is_empty.sum())
print("Invalid:", (~polygons.geometry.is_valid).sum())

print("Polygon Bounds:", polygons.total_bounds)


# 잘못된 Geometry 보정
invalid = ~polygons.geometry.is_valid

if invalid.any():

    polygons.loc[invalid, "geometry"] = (
        polygons.loc[invalid, "geometry"]
        .apply(make_valid)
    )

print(
    "Invalid After Repair:",
    (~polygons.geometry.is_valid).sum()
)

# 면 형태가 아닌 Geometry 확인
polygon_types = polygons.geometry.geom_type.isin(
    ["Polygon", "MultiPolygon"]
)

if (
    polygons.geometry.is_empty.any()
    or not polygons.geometry.is_valid.all()
    or not polygon_types.all()
):
    raise ValueError("Geometry requires additional inspection.")


# 위도·경도 기반 중심점 생성
points = gpd.GeoDataFrame(
    local[["위도", "경도"]].copy(),
    geometry=gpd.points_from_xy(
        local["경도"],
        local["위도"]
    ),
    crs="EPSG:4326"
)

# EPSG:5179로 좌표 변환
points = points.to_crs("EPSG:5179")


# 폴리곤 중심점과 제공 좌표 비교
centroids = polygons.geometry.centroid

distance = centroids.distance(points.geometry)

print("\n===== CRS Validation =====")
print("Centroid Distance Statistics (m):")
print(distance.describe())

# 제공 좌표가 폴리곤 내부 또는 경계에 있는지 확인
inside = polygons.geometry.covers(
    points.geometry
)

print("Points Inside Polygon:", inside.sum())
print("Points Outside Polygon:", (~inside).sum())

print(
    "Inside Ratio (%):",
    round(inside.mean() * 100, 2)
)


# 폴리곤 면적 확인
print("\n===== Polygon Area (m2) =====")
print(polygons.geometry.area.describe())


# 도로 링크 좌표 범위 비교
links = gpd.read_parquet(
    ROOT / "data/legacy/processed"
    / "links_cheonan_asan.parquet"
)

print("\n===== Bounds Comparison =====")
print("Road Bounds:", links.total_bounds)
print("Polygon Bounds:", polygons.total_bounds)


# 좌표계 검증용 결과 저장
polygons["centroid_distance_m"] = distance
polygons["point_inside_polygon"] = inside

output_path = (
    OUTPUT_DIR
    / "accident_polygons_crs_candidate.parquet"
)

polygons.to_parquet(
    output_path,
    index=False
)

print("\n===== Output =====")
print("Saved:", output_path)

print("\n===== Preparation Completed =====")