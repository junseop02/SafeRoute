from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import psutil

from shapely.geometry import shape


# 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    ROOT
    / "data"
    / "legacy"
    / "raw"
    / "전국교통사고다발지역표준데이터.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "national_accident_polygons_crs_candidate.parquet"
)

ERROR_PATH = (
    OUTPUT_DIR
    / "national_accident_polygon_errors.csv"
)

POLYGON_COLUMN = "사고다발지역폴리곤정보"

# 기존 천안·아산 연구와 동일한 후보 좌표계
CANDIDATE_CRS = "EPSG:5179"


# 메모리 확인
def print_memory():
    memory = psutil.virtual_memory()

    print(
        f"Available RAM: "
        f"{memory.available / 1024**3:.2f} GiB"
    )

    print(f"RAM Usage: {memory.percent}%")


# 비표준 JSON 문자열을 폴리곤으로 변환
def parse_polygon(text):

    if not isinstance(text, str):
        raise ValueError("Polygon text is not a string.")

    text = text.strip()

    if not text:
        raise ValueError("Empty polygon text.")

    # type, coordinates 키에 따옴표 추가
    text = re.sub(
        r'([{,]\s*)(type|coordinates)\s*:',
        r'\1"\2":',
        text
    )

    # Polygon, MultiPolygon 값에 따옴표 추가
    text = re.sub(
        r'("type"\s*:\s*)(Polygon|MultiPolygon)\b',
        r'\1"\2"',
        text
    )

    polygon_data = json.loads(text)

    geometry = shape(polygon_data)

    if geometry.geom_type not in [
        "Polygon",
        "MultiPolygon"
    ]:
        raise ValueError(
            f"Unexpected geometry: "
            f"{geometry.geom_type}"
        )

    return geometry


def main():

    print("===== National Accident Polygon Preparation =====")
    print("Source:", RAW_PATH)
    print("Candidate CRS:", CANDIDATE_CRS)

    print("\n===== Initial Memory =====")
    print_memory()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 원본 데이터 읽기
    df = pd.read_csv(
        RAW_PATH,
        encoding="cp949",
        dtype={
            "사고지역관리번호": str,
            "위치코드": str,
            "제공기관코드": str
        }
    )

    df = df.reset_index(drop=True)

    print("\n===== Source Information =====")
    print("Source Records:", len(df))

    # 레코드별 고유 ID 생성
    df["hotspot_id"] = np.arange(
        len(df),
        dtype=np.int64
    )

    print(
        "Unique Hotspot IDs:",
        df["hotspot_id"].nunique()
    )

    # 시도 명칭 정규화
    df["province_raw"] = (
        df["사고다발지역시도시군구"]
        .astype("string")
        .str.strip()
        .str.split()
        .str[0]
    )

    df["province"] = (
        df["province_raw"]
        .str.replace(
            r"^세종특별자치시\d+$",
            "세종특별자치시",
            regex=True
        )
    )

    print("\n===== Province Distribution =====")

    print(
        df["province"]
        .value_counts()
        .to_string()
    )

    print(
        "Unique Provinces:",
        df["province"].nunique()
    )

    # 폴리곤 파싱
    geometries = []
    errors = []

    print("\n===== Polygon Parsing =====")

    for index, row in df.iterrows():

        try:
            geometry = parse_polygon(
                row[POLYGON_COLUMN]
            )

            geometries.append(geometry)

        except Exception as error:

            geometries.append(None)

            errors.append({
                "hotspot_id": row["hotspot_id"],
                "accident_year": row["사고연도"],
                "accident_type": row["사고유형구분"],
                "location_code": row["위치코드"],
                "error": str(error)
            })

        if (index + 1) % 2000 == 0:
            print(
                f"Processed: {index + 1:,}"
                f" / {len(df):,}"
            )

    print("Total Records:", len(df))
    print("Parsed Successfully:", len(df) - len(errors))
    print("Parsing Errors:", len(errors))

    # 파싱 실패 시 원인 저장 후 중단
    if errors:

        pd.DataFrame(errors).to_csv(
            ERROR_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        print("Error File:", ERROR_PATH)

        raise RuntimeError(
            "Polygon parsing failed. "
            "Inspect errors before continuing."
        )

    # 후보 좌표계 지정
    gdf = gpd.GeoDataFrame(
        df.drop(columns=[POLYGON_COLUMN]),
        geometry=geometries,
        crs=CANDIDATE_CRS
    )

    print("\n===== Geometry Validation =====")

    print("CRS:", gdf.crs)

    print("\nGeometry Types:")

    print(
        gdf.geometry.geom_type
        .value_counts()
        .to_string()
    )

    missing_count = int(
        gdf.geometry.isna().sum()
    )

    empty_count = int(
        gdf.geometry.is_empty.sum()
    )

    invalid_count = int(
        (~gdf.geometry.is_valid).sum()
    )

    print("Missing Geometry:", missing_count)
    print("Empty Geometry:", empty_count)
    print("Invalid Geometry:", invalid_count)

    print("Bounds:", gdf.total_bounds)

    # 잘못된 폴리곤을 임의로 수정하지 않음
    if missing_count or empty_count or invalid_count:

        invalid_mask = (
            gdf.geometry.isna()
            | gdf.geometry.is_empty
            | ~gdf.geometry.is_valid
        )

        print("\nInvalid Geometry Samples:")

        print(
            gdf.loc[
                invalid_mask,
                [
                    "hotspot_id",
                    "사고연도",
                    "사고유형구분",
                    "위치코드"
                ]
            ].head(20).to_string(index=False)
        )

        raise RuntimeError(
            "Invalid geometries detected."
        )

    # 제공된 위도·경도를 EPSG:5179로 변환
    reference_points = gpd.GeoSeries(
        gpd.points_from_xy(
            gdf["경도"],
            gdf["위도"]
        ),
        index=gdf.index,
        crs="EPSG:4326"
    ).to_crs(CANDIDATE_CRS)

    # 폴리곤 중심점 계산
    centroids = gdf.geometry.centroid

    # 중심점과 제공 좌표 사이 거리
    gdf["centroid_distance_m"] = (
        centroids.distance(reference_points)
    )

    # 제공 좌표의 폴리곤 포함 여부
    gdf["point_inside_polygon"] = (
        gdf.geometry.covers(reference_points)
    )

    # 폴리곤 면적
    gdf["polygon_area_m2"] = (
        gdf.geometry.area
    )

    print("\n===== CRS Consistency Check =====")

    print("\nCentroid Distance Statistics (m):")

    print(
        gdf["centroid_distance_m"]
        .describe(
            percentiles=[0.5, 0.95, 0.99]
        )
        .to_string()
    )

    for threshold in [1, 10, 50, 100, 500, 1000]:

        count = int(
            (
                gdf["centroid_distance_m"]
                > threshold
            ).sum()
        )

        print(
            f"Distance > {threshold} m: {count}"
        )

    inside_count = int(
        gdf["point_inside_polygon"].sum()
    )

    outside_count = len(gdf) - inside_count

    print("\nReference Points Inside:", inside_count)
    print("Reference Points Outside:", outside_count)

    print(
        f"Inside Ratio: "
        f"{inside_count / len(gdf) * 100:.4f}%"
    )

    print("\n===== Largest Coordinate Differences =====")

    print(
        gdf.nlargest(
            10,
            "centroid_distance_m"
        )[
            [
                "hotspot_id",
                "사고연도",
                "사고유형구분",
                "province",
                "centroid_distance_m",
                "point_inside_polygon"
            ]
        ].to_string(index=False)
    )

    print("\n===== Polygon Area Statistics (m²) =====")

    print(
        gdf["polygon_area_m2"]
        .describe(
            percentiles=[0.5, 0.95, 0.99]
        )
        .to_string()
    )

    zero_area_count = int(
        (gdf["polygon_area_m2"] <= 0).sum()
    )

    print("Zero or Negative Area:", zero_area_count)

    # 동일한 폴리곤 반복 여부 확인
    geometry_wkb = gdf.geometry.to_wkb()

    print("\n===== Exact Geometry Duplicates =====")

    print(
        "Unique Geometry Records:",
        geometry_wkb.nunique()
    )

    print(
        "Exact Geometry Duplicates:",
        geometry_wkb.duplicated().sum()
    )

    # 기존 천안·아산 자료와 비교
    local_mask = (
        gdf["사고다발지역시도시군구"]
        .str.contains(
            "천안시|아산시",
            na=False
        )
    )

    local = gdf.loc[local_mask]

    print("\n===== Cheonan-Asan Comparison =====")

    print("Local Records:", len(local))

    print(
        local["centroid_distance_m"]
        .describe()
        .to_string()
    )

    print(
        "Local Reference Points Inside:",
        local["point_inside_polygon"].sum()
    )

    # 수치 유효성 확인
    finite_distances = np.isfinite(
        gdf["centroid_distance_m"]
    ).all()

    finite_areas = np.isfinite(
        gdf["polygon_area_m2"]
    ).all()

    print("\n===== Final Validation =====")

    print("Finite Distances:", finite_distances)
    print("Finite Areas:", finite_areas)

    if not finite_distances or not finite_areas:
        raise RuntimeError(
            "Non-finite geometry statistics detected."
        )

    if zero_area_count:
        raise RuntimeError(
            "Zero or negative polygon area detected."
        )

    # 후보 좌표계 데이터 저장
    # 좌표계 최종 판단은 검증 결과 확인 후 진행
    gdf.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("\n===== Saved Output =====")
    print("Path:", OUTPUT_PATH)
    print("Shape:", gdf.shape)
    print("CRS:", gdf.crs)

    print(
        f"File Size: "
        f"{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB"
    )

    print("\n===== Final Memory =====")
    print_memory()

    print("\n===== Preparation Completed =====")
    print("Records Removed: 0")
    print("Geometry Repairs: 0")
    print("Original File: Not Modified")
    print("CRS Status: Candidate, pending review")


if __name__ == "__main__":
    main()