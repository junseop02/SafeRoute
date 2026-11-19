from pathlib import Path

import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAFFIC_DIR = PROJECT_ROOT / "data" / "raw" / "traffic"


# ============================================================
# 2. traffic 폴더의 Excel 파일 찾기
# ============================================================

print("=== SafeRoute 교통량(AADT) 원본 데이터 검사 ===")

print("\n[1] Excel 파일 찾기...")

excel_files = list(TRAFFIC_DIR.rglob("*.xlsx"))

print("찾은 Excel 파일 수:", len(excel_files))

for i, path in enumerate(excel_files, start=1):
    print(f"{i}. {path}")


if len(excel_files) == 0:
    raise FileNotFoundError(
        "data/raw/traffic 폴더에서 xlsx 파일을 찾지 못했습니다."
    )


# 우선 첫 번째 Excel 파일 사용
traffic_path = excel_files[0]

print("\n사용 파일:")
print(traffic_path)


# ============================================================
# 3. Sheet 목록 확인
# ============================================================

print("\n[2] Sheet 목록 확인...")

xls = pd.ExcelFile(traffic_path)

print(xls.sheet_names)


# ============================================================
# 4. X,Y좌표 Sheet 읽기
# ============================================================

sheet_name = "X,Y좌표"

if sheet_name not in xls.sheet_names:
    raise ValueError(
        f"'{sheet_name}' Sheet를 찾지 못했습니다."
    )


print(
    f"\n[3] '{sheet_name}' Sheet 불러오기..."
)

traffic = pd.read_excel(
    traffic_path,
    sheet_name=sheet_name
)

print("행 수:", len(traffic))
print("열 수:", len(traffic.columns))


# ============================================================
# 5. 컬럼 확인
# ============================================================

print("\n[4] 전체 컬럼")

for i, column in enumerate(
    traffic.columns,
    start=1
):
    print(
        f"{i:02d}. {repr(column)}"
    )


# ============================================================
# 6. 데이터 샘플
# ============================================================

print("\n[5] 앞부분 데이터 샘플")

print(
    traffic
    .head(10)
    .to_string()
)


# ============================================================
# 7. 자료형 확인
# ============================================================

print("\n[6] 자료형")

print(
    traffic.dtypes
)


# ============================================================
# 8. 결측치 확인
# ============================================================

print("\n[7] 결측치")

missing = (
    traffic
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(missing.to_string())


# ============================================================
# 9. 좌표 관련 컬럼 탐색
# ============================================================

print("\n[8] 좌표 관련 컬럼 탐색")

coordinate_columns = []

for column in traffic.columns:

    name = str(column).upper()

    if (
        "X" in name
        or "Y" in name
        or "경도" in name
        or "위도" in name
        or "LON" in name
        or "LAT" in name
    ):
        coordinate_columns.append(column)


print(
    "좌표 후보:",
    coordinate_columns
)


# ============================================================
# 10. XCODE / YCODE 검사
# ============================================================

if (
    "XCODE" in traffic.columns
    and "YCODE" in traffic.columns
):

    print("\n[9] XCODE / YCODE 검사")

    traffic["XCODE_numeric"] = pd.to_numeric(
        traffic["XCODE"],
        errors="coerce"
    )

    traffic["YCODE_numeric"] = pd.to_numeric(
        traffic["YCODE"],
        errors="coerce"
    )

    print("\n[XCODE 통계]")

    print(
        traffic[
            "XCODE_numeric"
        ].describe()
    )

    print("\n[YCODE 통계]")

    print(
        traffic[
            "YCODE_numeric"
        ].describe()
    )

    print(
        "\nXCODE 숫자 변환 실패:",
        traffic[
            "XCODE_numeric"
        ].isna().sum()
    )

    print(
        "YCODE 숫자 변환 실패:",
        traffic[
            "YCODE_numeric"
        ].isna().sum()
    )

    print("\n[좌표 샘플]")

    print(
        traffic[
            [
                "XCODE",
                "YCODE",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 11. 교통량(AADT) 관련 컬럼 탐색
# ============================================================

print("\n[10] 교통량 관련 컬럼 탐색")

traffic_columns = []

keywords = [
    "AADT",
    "교통량",
    "평균",
    "24시간",
]

for column in traffic.columns:

    name = str(column).upper()

    if any(
        keyword.upper() in name
        for keyword in keywords
    ):
        traffic_columns.append(column)


print(
    "교통량 후보:",
    traffic_columns
)


# ============================================================
# 12. 지역 관련 값 확인
# ============================================================

print("\n[11] 천안/아산 관련 행 탐색")

text_data = (
    traffic
    .astype(str)
    .agg(
        " ".join,
        axis=1
    )
)

cheonan_mask = text_data.str.contains(
    "천안",
    na=False
)

asan_mask = text_data.str.contains(
    "아산",
    na=False
)


print(
    "천안 포함 행:",
    cheonan_mask.sum()
)

print(
    "아산 포함 행:",
    asan_mask.sum()
)


print("\n[천안 샘플]")

print(
    traffic[
        cheonan_mask
    ]
    .head(10)
    .to_string()
)


print("\n[아산 샘플]")

print(
    traffic[
        asan_mask
    ]
    .head(10)
    .to_string()
)


# ============================================================
# 13. 중복 좌표 확인
# ============================================================

if (
    "XCODE" in traffic.columns
    and "YCODE" in traffic.columns
):

    print("\n[12] 좌표 중복 검사")

    duplicate_coordinates = (
        traffic[
            ["XCODE", "YCODE"]
        ]
        .duplicated()
        .sum()
    )

    print(
        "중복 좌표 행:",
        duplicate_coordinates
    )


# ============================================================
# 14. 완료
# ============================================================

print(
    "\n=== 교통량(AADT) 원본 데이터 검사 완료 ==="
)