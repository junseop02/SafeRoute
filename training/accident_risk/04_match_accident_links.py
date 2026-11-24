from pathlib import Path

import geopandas as gpd
import pandas as pd

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/legacy/processed"
OUTPUT_DIR = ROOT / "data/accident_risk/processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 도로 링크 불러오기
links = gpd.read_parquet(
    DATA_DIR / "links_cheonan_asan.parquet"
)

# 사고다발지역 폴리곤 불러오기
polygons = gpd.read_parquet(
    OUTPUT_DIR / "accident_polygons_crs_candidate.parquet"
)

# 필요한 컬럼만 선택
links = links[
    ["LINK_ID", "region_code", "geometry"]
].copy().reset_index(drop=True)

links = links.rename(
    columns={"LINK_ID": "link_id"}
)

links["link_id"] = links["link_id"].astype(str)

polygons = polygons.reset_index(drop=True)

# 사고다발지역 기록별 고유 ID 생성
polygons["hotspot_id"] = polygons.index

# 좌표계 확인
if links.crs is None or polygons.crs is None:
    raise ValueError("CRS is missing.")

if links.crs != polygons.crs:
    polygons = polygons.to_crs(links.crs)

if links.crs.to_epsg() != 5179:
    raise ValueError("Expected EPSG:5179.")

print("===== Input Data =====")
print("Road Links:", len(links))
print("Accident Records:", len(polygons))
print("EPSG:", links.crs.to_epsg())

# 도로의 실제 Geometry 길이
links["geometry_length_m"] = links.geometry.length

if (links["geometry_length_m"] <= 0).any():
    raise ValueError("Zero-length road geometry detected.")

# 사고다발지역과 도로 링크 공간 매칭
matched = gpd.sjoin(
    links,
    polygons[
        [
            "hotspot_id",
            "사고연도",
            "사고유형구분",
            "사고건수",
            "geometry"
        ]
    ],
    how="inner",
    predicate="intersects"
)

matched = matched.reset_index(drop=True)

if matched.empty:
    raise ValueError("No spatial matches found.")

# 매칭된 사고다발지역 Geometry 가져오기
polygon_geometry = gpd.GeoSeries(
    polygons.geometry.iloc[
        matched["index_right"].to_numpy()
    ].to_numpy(),
    index=matched.index,
    crs=links.crs
)

# 도로와 폴리곤의 실제 중첩 구간 계산
overlap_geometry = matched.geometry.intersection(
    polygon_geometry
)

matched["overlap_length_m"] = (
    overlap_geometry.length
)

# 전체 도로 길이 대비 중첩 비율
matched["overlap_ratio"] = (
    matched["overlap_length_m"]
    / matched["geometry_length_m"]
).clip(0, 1)

# 실제 길이가 중첩되는 매칭만 구분
positive_overlap = matched[
    matched["overlap_length_m"] > 0
].copy()

# 기본 공간 매칭 결과
print("\n===== Spatial Matching =====")
print("Total Matched Pairs:", len(matched))

print(
    "Unique Matched Links:",
    matched["link_id"].nunique()
)

print(
    "Matched Accident Records:",
    matched["hotspot_id"].nunique()
)

print(
    "Unmatched Accident Records:",
    len(polygons) - matched["hotspot_id"].nunique()
)

print(
    "Boundary-only Pairs:",
    len(matched) - len(positive_overlap)
)

print(
    "Links With Positive Overlap:",
    positive_overlap["link_id"].nunique()
)

# 중첩 길이 통계
print("\n===== Overlap Length Statistics (m) =====")
print(
    positive_overlap["overlap_length_m"].describe()
)

# 중첩 비율 통계
print("\n===== Overlap Ratio Statistics =====")
print(
    positive_overlap["overlap_ratio"].describe()
)

# 링크별 최대 중첩 비율
max_ratio = positive_overlap.groupby(
    "link_id"
)["overlap_ratio"].max()

# 여러 중첩 기준 비교
print("\n===== Threshold Comparison =====")

for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:

    count = int(
        (max_ratio >= threshold).sum()
    )

    ratio = count / len(links) * 100

    print(
        f"Overlap >= {threshold:.0%}: "
        f"{count:,} links ({ratio:.2f}%)"
    )

# 지역별 매칭 링크 수
print("\n===== Regional Matched Links =====")

print(
    positive_overlap.groupby(
        "region_code"
    )["link_id"].nunique()
)

# 연도별 매칭 링크 수
print("\n===== Yearly Matched Links =====")

print(
    positive_overlap.groupby(
        "사고연도"
    )["link_id"].nunique()
)

# 사고 유형별 매칭 링크 수
print("\n===== Accident Type Matched Links =====")

print(
    positive_overlap.groupby(
        "사고유형구분"
    )["link_id"].nunique()
)

# 여러 사고다발지역에 매칭된 도로 확인
match_counts = positive_overlap.groupby(
    "link_id"
)["hotspot_id"].nunique()

print("\n===== Multiple Hotspot Matches =====")
print(match_counts.describe())

print(
    "Links Matched Multiple Records:",
    (match_counts > 1).sum()
)

# 매칭되지 않은 사고다발지역 확인
unmatched = polygons[
    ~polygons["hotspot_id"].isin(
        positive_overlap["hotspot_id"]
    )
]

print("\n===== Unmatched Accident Records =====")
print("Count:", len(unmatched))

if not unmatched.empty:

    print(
        unmatched[
            [
                "hotspot_id",
                "사고연도",
                "사고유형구분",
                "사고지역위치명"
            ]
        ].to_string(index=False)
    )

# 공간 매칭 결과 저장
output_path = (
    OUTPUT_DIR / "accident_link_matches.parquet"
)

matched.to_parquet(
    output_path,
    index=False
)

print("\n===== Output =====")
print("Saved:", output_path)

print("\n===== Spatial Matching Completed =====")