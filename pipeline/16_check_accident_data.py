from pathlib import Path
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACCIDENT_DIR = PROJECT_ROOT / "data" / "raw" / "accident"


print("=== SafeRoute 사고다발지역 데이터 검사 ===")


# ============================================================
# 2. CSV 탐색
# ============================================================

csv_files = sorted(
    ACCIDENT_DIR.rglob("*.csv")
)

print("\nCSV 파일 수:", len(csv_files))


if len(csv_files) == 0:
    raise FileNotFoundError(
        "accident 폴더에서 CSV 파일을 찾지 못했습니다."
    )


# ============================================================
# 3. CSV 읽기 함수
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
            return (
                pd.read_csv(
                    path,
                    encoding=encoding,
                ),
                encoding,
            )

        except Exception:
            pass

    raise RuntimeError(
        f"CSV 읽기 실패: {path}"
    )


# ============================================================
# 4. 파일별 핵심 정보 확인
# ============================================================

summary_rows = []


for i, path in enumerate(
    csv_files,
    start=1,
):

    df, encoding = read_csv_auto(path)

    columns = list(df.columns)


    # --------------------------------------------------------
    # 연도 컬럼 탐색
    # --------------------------------------------------------

    year_columns = [
        col
        for col in columns
        if any(
            keyword in str(col).lower()
            for keyword in [
                "연도",
                "년도",
                "year",
            ]
        )
    ]


    # --------------------------------------------------------
    # 경도 컬럼 탐색
    # --------------------------------------------------------

    lon_candidates = [
        col
        for col in columns
        if (
            "경도" in str(col)
            or "longitude" in str(col).lower()
            or str(col).lower() in [
                "lon",
                "lng",
            ]
        )
    ]


    # --------------------------------------------------------
    # 위도 컬럼 탐색
    # --------------------------------------------------------

    lat_candidates = [
        col
        for col in columns
        if (
            "위도" in str(col)
            or "latitude" in str(col).lower()
            or str(col).lower() == "lat"
        )
    ]


    lon_col = (
        lon_candidates[0]
        if lon_candidates
        else None
    )

    lat_col = (
        lat_candidates[0]
        if lat_candidates
        else None
    )


    # --------------------------------------------------------
    # 좌표 검사
    # --------------------------------------------------------

    valid_coord = None
    lon_min = None
    lon_max = None
    lat_min = None
    lat_max = None


    if (
        lon_col is not None
        and lat_col is not None
    ):

        lon = pd.to_numeric(
            df[lon_col],
            errors="coerce",
        )

        lat = pd.to_numeric(
            df[lat_col],
            errors="coerce",
        )

        valid_coord = int(
            (
                lon.notna()
                & lat.notna()
            ).sum()
        )

        lon_min = lon.min()
        lon_max = lon.max()

        lat_min = lat.min()
        lat_max = lat.max()


    # --------------------------------------------------------
    # 요약
    # --------------------------------------------------------

    summary_rows.append(
        {
            "번호": i,
            "파일명": path.name,
            "행수": len(df),
            "열수": len(df.columns),
            "인코딩": encoding,
            "연도컬럼": (
                ",".join(
                    map(
                        str,
                        year_columns,
                    )
                )
                if year_columns
                else "-"
            ),
            "경도컬럼": (
                str(lon_col)
                if lon_col is not None
                else "-"
            ),
            "위도컬럼": (
                str(lat_col)
                if lat_col is not None
                else "-"
            ),
            "유효좌표": (
                valid_coord
                if valid_coord is not None
                else 0
            ),
        }
    )


    # --------------------------------------------------------
    # 파일별 최소 출력
    # --------------------------------------------------------

    print(
        f"\n[{i:02d}] {path.name}"
    )

    print(
        f"행={len(df)}, 열={len(df.columns)}"
    )

    print(
        "연도 컬럼:",
        year_columns
        if year_columns
        else "없음",
    )

    print(
        "좌표 컬럼:",
        lon_col,
        "/",
        lat_col,
    )


    if valid_coord is not None:

        print(
            f"유효 좌표: "
            f"{valid_coord}/{len(df)}"
        )

        print(
            f"경도 범위: "
            f"{lon_min} ~ {lon_max}"
        )

        print(
            f"위도 범위: "
            f"{lat_min} ~ {lat_max}"
        )


# ============================================================
# 5. 전체 요약
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


print(
    "\n"
    + "=" * 80
)

print(
    "=== 전체 요약 ==="
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\n전체 CSV:",
    len(summary)
)

print(
    "전체 사고다발지점:",
    summary["행수"].sum()
)


# ============================================================
# 6. 컬럼 구조 종류만 확인
# ============================================================

patterns = {}


for path in csv_files:

    df, _ = read_csv_auto(path)

    pattern = tuple(
        df.columns.astype(str)
    )

    patterns.setdefault(
        pattern,
        []
    ).append(
        path.name
    )


print(
    "\n서로 다른 컬럼 구조 수:",
    len(patterns)
)


for i, (
    columns,
    files,
) in enumerate(
    patterns.items(),
    start=1,
):

    print(
        f"\n[컬럼 구조 {i}]"
    )

    print(
        "파일 수:",
        len(files)
    )

    print(
        "파일명:"
    )

    for file in files:
        print(" -", file)

    print(
        "컬럼:",
        list(columns)
    )


# ============================================================
# 7. 연도 컬럼 최종 확인
# ============================================================

files_with_year = summary[
    summary["연도컬럼"] != "-"
]


print(
    "\n"
    + "=" * 80
)

print(
    "=== 연도 정보 확인 ==="
)

print(
    "연도 컬럼이 있는 파일:",
    len(files_with_year),
    "/",
    len(summary),
)


if len(files_with_year) == 0:

    print(
        "주의: 명시적인 연도 컬럼을 찾지 못했습니다."
    )

    print(
        "사고 ID 등의 숫자를 연도로 임의 해석하지 않습니다."
    )


print(
    "\n=== 검사 완료 ==="
)