from pathlib import Path
import pandas as pd

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data/legacy/raw"
    / "전국교통사고다발지역표준데이터.csv"
)

# 데이터 불러오기
df = pd.read_csv(DATA_PATH, encoding="cp949")

# 전체 데이터 크기
print("===== Dataset Shape =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 사고연도별 분포
print("\n===== Accident Year =====")
print(df["사고연도"].value_counts().sort_index())

# 사고 유형별 분포
print("\n===== Accident Type =====")
print(df["사고유형구분"].value_counts())

# 천안·아산 데이터 추출
region = df["사고다발지역시도시군구"]

cheonan = df[region.str.contains("천안시", na=False)]
asan = df[region.str.contains("아산시", na=False)]

local = df[
    region.str.contains("천안시|아산시", na=False)
].copy()

# 지역별 데이터 개수
print("\n===== Regional Distribution =====")
print("Cheonan:", len(cheonan))
print("Asan:", len(asan))
print("Total:", len(local))

# 천안·아산 사고연도별 분포
print("\n===== Regional Accident Year =====")
print(local["사고연도"].value_counts().sort_index())

# 천안·아산 사고 유형별 분포
print("\n===== Regional Accident Type =====")
print(local["사고유형구분"].value_counts())

# 지역별 상세 분포
print("\n===== Regional Details =====")
print(local["사고다발지역시도시군구"].value_counts())

# 연도 및 사고 유형 교차표
print("\n===== Year-Type Distribution =====")
print(pd.crosstab(
    local["사고연도"],
    local["사고유형구분"]
).to_string())

# 사고 건수 기초 통계
print("\n===== Accident Count Statistics =====")
print(local["사고건수"].describe())

# 좌표 범위 확인
print("\n===== Coordinate Range =====")
print(local[["위도", "경도"]].describe())

# 동일 연도·유형·위치코드 중복 확인
print("\n===== Location Duplicates =====")
print(local.duplicated(
    subset=["사고연도", "사고유형구분", "위치코드"]
).sum())

# 폴리곤 데이터 형식 확인
print("\n===== Polygon Sample =====")

if not local.empty:
    print(local["사고다발지역폴리곤정보"].iloc[0][:500])