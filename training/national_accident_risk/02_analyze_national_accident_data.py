from pathlib import Path

import pandas as pd
import psutil


# 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    ROOT
    / "data"
    / "legacy"
    / "raw"
    / "전국교통사고다발지역표준데이터.csv"
)


# 메모리 사용량 확인
def print_memory():
    memory = psutil.virtual_memory()

    print(f"Available RAM: {memory.available / 1024**3:.2f} GiB")
    print(f"RAM Usage: {memory.percent}%")


def main():
    print("===== National Accident Data Analysis =====")
    print("Source:", RAW_PATH)

    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    print("\n===== Initial Memory =====")
    print_memory()

    # 전국 사고다발지역 데이터 불러오기
    df = pd.read_csv(
        RAW_PATH,
        encoding="cp949",
        dtype={
            "사고지역관리번호": str,
            "위치코드": str,
            "제공기관코드": str
        }
    )

    print("\n===== Dataset Information =====")
    print("Shape:", df.shape)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\n===== Column Names =====")

    for column in df.columns:
        print(column)

    print("\n===== Data Types =====")
    print(df.dtypes.to_string())

    # 필수 컬럼 확인
    required_columns = [
        "사고지역관리번호",
        "사고연도",
        "사고유형구분",
        "위치코드",
        "사고다발지역시도시군구",
        "사고건수",
        "위도",
        "경도",
        "사고다발지역폴리곤정보"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    print("\n===== Required Columns =====")
    print("Missing Columns:", missing_columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # 결측치 확인
    print("\n===== Missing Values =====")
    print(df.isna().sum().to_string())

    print("\nTotal Missing Values:")
    print(int(df.isna().sum().sum()))

    # 중복 데이터 확인
    print("\n===== Duplicate Records =====")
    print("Exact Duplicates:", df.duplicated().sum())

    duplicate_keys = [
        "사고연도",
        "사고유형구분",
        "위치코드"
    ]

    print(
        "Duplicate Year-Type-Location:",
        df.duplicated(subset=duplicate_keys).sum()
    )

    print(
        "Duplicate Management IDs:",
        df["사고지역관리번호"].duplicated().sum()
    )

    # 연도별 사고다발지역 분포
    print("\n===== Records by Year =====")

    year_counts = (
        df["사고연도"]
        .value_counts()
        .sort_index()
    )

    print(year_counts.to_string())

    # 사고 유형별 분포
    print("\n===== Records by Accident Type =====")

    type_counts = (
        df["사고유형구분"]
        .value_counts()
    )

    print(type_counts.to_string())

    # 지역명에서 첫 번째 행정구역 명칭 추출
    region = (
        df["사고다발지역시도시군구"]
        .astype("string")
        .str.strip()
    )

    df["province_raw"] = (
        region.str.split().str[0]
    )

    print("\n===== Raw Province Distribution =====")

    province_counts = (
        df["province_raw"]
        .value_counts(dropna=False)
    )

    print(province_counts.to_string())

    # 세부 지역별 분포
    print("\n===== Top 30 Regions =====")

    print(
        region.value_counts()
        .head(30)
        .to_string()
    )

    # 연도 및 사고 유형 교차표
    print("\n===== Year x Accident Type =====")

    year_type_table = pd.crosstab(
        df["사고연도"],
        df["사고유형구분"]
    )

    print(year_type_table.to_string())

    # 시도 및 사고 유형 교차표
    print("\n===== Province x Accident Type =====")

    province_type_table = pd.crosstab(
        df["province_raw"],
        df["사고유형구분"]
    )

    print(province_type_table.to_string())

    # 사고 및 사상자 관련 통계
    numeric_columns = [
        "사고건수",
        "사상자수",
        "사망자수",
        "중상자수",
        "경상자수",
        "부상신고자수"
    ]

    print("\n===== Accident Statistics =====")

    for column in numeric_columns:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        print(f"\nColumn: {column}")
        print(values.describe().to_string())
        print("Total:", values.sum())
        print("Invalid:", values.isna().sum())
        print("Negative:", (values < 0).sum())

    # 위도 및 경도 검사
    latitude = pd.to_numeric(
        df["위도"],
        errors="coerce"
    )

    longitude = pd.to_numeric(
        df["경도"],
        errors="coerce"
    )

    print("\n===== Coordinate Statistics =====")

    print("Latitude Min:", latitude.min())
    print("Latitude Max:", latitude.max())
    print("Longitude Min:", longitude.min())
    print("Longitude Max:", longitude.max())

    invalid_coordinates = (
        latitude.isna()
        | longitude.isna()
        | ~latitude.between(-90, 90)
        | ~longitude.between(-180, 180)
    )

    print(
        "Invalid Coordinates:",
        invalid_coordinates.sum()
    )

    # 동일 좌표 반복 기록 확인
    coordinate_counts = (
        df.groupby(
            ["위도", "경도"],
            dropna=False
        )
        .size()
    )

    print("\n===== Repeated Coordinates =====")

    print(
        "Unique Coordinate Pairs:",
        len(coordinate_counts)
    )

    print(
        "Repeated Coordinate Locations:",
        (coordinate_counts > 1).sum()
    )

    print(
        "Maximum Records per Coordinate:",
        coordinate_counts.max()
    )

    # 폴리곤 문자열 확인
    polygon_text = (
        df["사고다발지역폴리곤정보"]
        .astype("string")
        .str.strip()
    )

    missing_polygons = (
        polygon_text.isna()
        | polygon_text.eq("")
    )

    print("\n===== Polygon Text Check =====")
    print("Missing or Empty:", missing_polygons.sum())

    print("\nPolygon Text Length:")
    print(
        polygon_text.str.len()
        .describe()
        .to_string()
    )

    valid_polygon_text = polygon_text.dropna()
    valid_polygon_text = valid_polygon_text[
        valid_polygon_text.ne("")
    ]

    if not valid_polygon_text.empty:
        print("\nFirst Polygon Sample:")
        print(valid_polygon_text.iloc[0][:500])

    # 데이터 제공기관 확인
    if "제공기관명" in df.columns:

        print("\n===== Data Providers =====")

        print(
            df["제공기관명"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "데이터기준일자" in df.columns:

        print("\n===== Reference Dates =====")

        print(
            df["데이터기준일자"]
            .value_counts(dropna=False)
            .to_string()
        )

    # 기존 천안·아산 데이터와 비교
    local_mask = region.str.contains(
        "천안시|아산시",
        na=False
    )

    local_count = int(local_mask.sum())

    print("\n===== Cheonan-Asan Comparison =====")
    print("National Records:", len(df))
    print("Cheonan-Asan Records:", local_count)
    print("Other Regions:", len(df) - local_count)

    print(
        f"Cheonan-Asan Ratio: "
        f"{local_count / len(df) * 100:.4f}%"
    )

    print("\n===== Final Memory =====")
    print_memory()

    print("\n===== Analysis Completed =====")
    print("Regional Filtering: Not Applied")
    print("Records Removed: 0")
    print("Original File: Not Modified")
    print("Output Files: Not Created")


if __name__ == "__main__":
    main()