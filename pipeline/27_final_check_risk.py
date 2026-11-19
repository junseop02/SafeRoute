from pathlib import Path

import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_cheonan_asan.parquet"
)

INPUT_RISK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "risk_all_links.parquet"
)

FINAL_RISK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "risk.parquet"
)


EXPECTED_LINK_COUNT = 36521

EXPECTED_COLUMNS = [
    "link_id",
    "risk",
]


print(
    "=== SafeRoute risk.parquet 최종 검증 ==="
)


# ============================================================
# 2. 파일 존재 확인
# ============================================================

print(
    "\n[1] 파일 존재 확인..."
)


if not FEATURE_PATH.exists():

    raise FileNotFoundError(
        f"Feature 파일이 없습니다: {FEATURE_PATH}"
    )


if not INPUT_RISK_PATH.exists():

    raise FileNotFoundError(
        f"Risk 파일이 없습니다: {INPUT_RISK_PATH}"
    )


print(
    "Feature:",
    FEATURE_PATH
)

print(
    "Risk:",
    INPUT_RISK_PATH
)


# ============================================================
# 3. 데이터 불러오기
# ============================================================

print(
    "\n[2] 데이터 불러오기..."
)


features_df = pd.read_parquet(
    FEATURE_PATH
)

risk_df = pd.read_parquet(
    INPUT_RISK_PATH
)


print(
    "Feature LINK:",
    len(features_df)
)

print(
    "Risk LINK:",
    len(risk_df)
)

print(
    "Risk 컬럼:",
    risk_df.columns.tolist()
)


# ============================================================
# 4. 컬럼 규격 검사
# ============================================================

print(
    "\n[3] 컬럼 규격 검사..."
)


if risk_df.columns.tolist() != EXPECTED_COLUMNS:

    raise RuntimeError(
        "risk 파일 컬럼이 최종 인터페이스와 다릅니다.\n"
        f"현재: {risk_df.columns.tolist()}\n"
        f"기대: {EXPECTED_COLUMNS}"
    )


print(
    "컬럼 규격:",
    "PASS"
)


# ============================================================
# 5. link_id 문자열 타입 통일
# ============================================================

features_df[
    "link_id"
] = (
    features_df[
        "link_id"
    ]
    .astype(str)
)


risk_df[
    "link_id"
] = (
    risk_df[
        "link_id"
    ]
    .astype(str)
)


# ============================================================
# 6. 전체 행 수 검사
# ============================================================

print(
    "\n[4] 전체 행 수 검사..."
)


if len(risk_df) != EXPECTED_LINK_COUNT:

    raise RuntimeError(
        f"Risk 행 수 오류: "
        f"{len(risk_df)} "
        f"(기대값 {EXPECTED_LINK_COUNT})"
    )


if len(risk_df) != len(features_df):

    raise RuntimeError(
        "Feature와 Risk의 LINK 수가 다릅니다."
    )


print(
    "Risk 행 수:",
    len(risk_df)
)

print(
    "행 수 검사:",
    "PASS"
)


# ============================================================
# 7. link_id 결측 / 중복 검사
# ============================================================

print(
    "\n[5] link_id 검사..."
)


link_missing = int(
    risk_df[
        "link_id"
    ]
    .isna()
    .sum()
)


link_duplicate = int(
    risk_df[
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
        "link_id 결측이 존재합니다."
    )


if link_duplicate != 0:

    raise RuntimeError(
        "link_id 중복이 존재합니다."
    )


print(
    "link_id 검사:",
    "PASS"
)


# ============================================================
# 8. Feature와 Risk의 link_id 집합 비교
# ============================================================

print(
    "\n[6] 전체 LINK 일치 검사..."
)


feature_links = set(
    features_df[
        "link_id"
    ]
)


risk_links = set(
    risk_df[
        "link_id"
    ]
)


missing_in_risk = (
    feature_links
    - risk_links
)

extra_in_risk = (
    risk_links
    - feature_links
)


print(
    "Risk에 없는 LINK:",
    len(
        missing_in_risk
    )
)

print(
    "Risk에만 있는 LINK:",
    len(
        extra_in_risk
    )
)


if missing_in_risk:

    raise RuntimeError(
        "Risk 파일에서 누락된 LINK가 있습니다."
    )


if extra_in_risk:

    raise RuntimeError(
        "Risk 파일에 알 수 없는 LINK가 있습니다."
    )


print(
    "전체 LINK 일치:",
    "PASS"
)


# ============================================================
# 9. risk 타입 / 결측 검사
# ============================================================

print(
    "\n[7] risk 값 검사..."
)


risk_df[
    "risk"
] = pd.to_numeric(
    risk_df[
        "risk"
    ],
    errors="coerce",
)


risk_missing = int(
    risk_df[
        "risk"
    ]
    .isna()
    .sum()
)


print(
    "risk 결측:",
    risk_missing
)


if risk_missing != 0:

    raise RuntimeError(
        "risk에 숫자가 아닌 값 또는 결측값이 있습니다."
    )


# ============================================================
# 10. risk 범위 검사
# ============================================================

risk_out_of_range = int(
    (
        (
            risk_df[
                "risk"
            ] < 0
        )
        |
        (
            risk_df[
                "risk"
            ] > 1
        )
    ).sum()
)


print(
    "0~1 범위 밖:",
    risk_out_of_range
)


if risk_out_of_range != 0:

    raise RuntimeError(
        "risk가 0~1 범위를 벗어났습니다."
    )


print(
    "risk 값 검사:",
    "PASS"
)


# ============================================================
# 11. 무한대 검사
# ============================================================

print(
    "\n[8] 무한대 검사..."
)


positive_inf = int(
    (
        risk_df[
            "risk"
        ] == float("inf")
    ).sum()
)


negative_inf = int(
    (
        risk_df[
            "risk"
        ] == float("-inf")
    ).sum()
)


print(
    "+inf:",
    positive_inf
)

print(
    "-inf:",
    negative_inf
)


if (
    positive_inf != 0
    or negative_inf != 0
):

    raise RuntimeError(
        "risk에 무한대 값이 존재합니다."
    )


print(
    "무한대 검사:",
    "PASS"
)


# ============================================================
# 12. risk 분포 최종 확인
# ============================================================

print(
    "\n[9] 최종 Risk 분포..."
)


print(
    risk_df[
        "risk"
    ]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


# ============================================================
# 13. Safety Score 참고 확인
#
# safety_score = 100 * (1 - risk)
#
# 최종 risk.parquet에는 저장하지 않음
# ============================================================

print(
    "\n[10] Safety Score 참고 확인..."
)


safety_score = (
    100.0
    * (
        1.0
        - risk_df[
            "risk"
        ]
    )
)


print(
    safety_score
    .describe()
    .to_string()
)


# ============================================================
# 14. risk 기준 샘플 확인
# ============================================================

print(
    "\n[11] Risk 상위 10개..."
)


print(
    risk_df
    .sort_values(
        "risk",
        ascending=False,
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nRisk 하위 10개..."
)


print(
    risk_df
    .sort_values(
        "risk",
        ascending=True,
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. 최종 컬럼 순서 확정
# ============================================================

print(
    "\n[12] 최종 인터페이스 생성..."
)


final_df = (
    risk_df[
        [
            "link_id",
            "risk",
        ]
    ]
    .copy()
)


# 정렬은 결과 재현성과 확인 편의를 위해 사용
final_df = (
    final_df
    .sort_values(
        "link_id"
    )
    .reset_index(
        drop=True
    )
)


print(
    "최종 컬럼:",
    final_df.columns.tolist()
)

print(
    "최종 행:",
    len(final_df)
)


# ============================================================
# 16. 최종 risk.parquet 저장
# ============================================================

print(
    "\n[13] 최종 risk.parquet 저장..."
)


final_df.to_parquet(
    FINAL_RISK_PATH,
    index=False,
)


print(
    "저장:",
    FINAL_RISK_PATH
)


# ============================================================
# 17. 저장 후 다시 읽어서 검증
#
# 실제 파일 자체가 정상 저장됐는지 마지막 확인
# ============================================================

print(
    "\n[14] 저장 파일 재검증..."
)


check_df = pd.read_parquet(
    FINAL_RISK_PATH
)


saved_row_count = len(
    check_df
)

saved_duplicate_count = int(
    check_df[
        "link_id"
    ]
    .duplicated()
    .sum()
)

saved_missing_count = int(
    check_df[
        "risk"
    ]
    .isna()
    .sum()
)

saved_range_error = int(
    (
        (
            check_df[
                "risk"
            ] < 0
        )
        |
        (
            check_df[
                "risk"
            ] > 1
        )
    ).sum()
)


print(
    "저장 행:",
    saved_row_count
)

print(
    "저장 컬럼:",
    check_df.columns.tolist()
)

print(
    "link_id 중복:",
    saved_duplicate_count
)

print(
    "risk 결측:",
    saved_missing_count
)

print(
    "risk 범위 오류:",
    saved_range_error
)


if saved_row_count != EXPECTED_LINK_COUNT:

    raise RuntimeError(
        "저장 후 행 수가 달라졌습니다."
    )


if (
    check_df.columns.tolist()
    != EXPECTED_COLUMNS
):

    raise RuntimeError(
        "저장 후 컬럼 규격이 달라졌습니다."
    )


if saved_duplicate_count != 0:

    raise RuntimeError(
        "저장 후 link_id 중복 발생"
    )


if saved_missing_count != 0:

    raise RuntimeError(
        "저장 후 risk 결측 발생"
    )


if saved_range_error != 0:

    raise RuntimeError(
        "저장 후 risk 범위 오류 발생"
    )


# ============================================================
# 18. 최종 결과
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute risk.parquet 최종 검증 완료 ==="
)

print(
    "=" * 70
)


print(
    "\n최종 파일:"
)

print(
    FINAL_RISK_PATH
)


print(
    "\n인터페이스:"
)

print(
    "link_id : str"
)

print(
    "risk    : float (0~1)"
)


print(
    "\n검증 결과:"
)

print(
    f"LINK 수        : {saved_row_count}"
)

print(
    f"link_id 중복   : {saved_duplicate_count}"
)

print(
    f"risk 결측      : {saved_missing_count}"
)

print(
    f"risk 범위 오류 : {saved_range_error}"
)


print(
    "\n모든 검사: PASS"
)


print(
    "\n주의:"
)

print(
    "risk는 실제 사고 발생 확률이 아니라"
)

print(
    "사고다발 조건에 대한 모델 위험 점수입니다."
)


print(
    "\n다음 단계:"
)

print(
    "28_build_links_geojson.py"
)

print(
    "→ 백엔드용 도로 geometry 파일 생성"
)

print(
    "=== 완료 ==="
)