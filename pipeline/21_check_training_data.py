from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_cheonan_asan.parquet"
)


# ============================================================
# 2. 모델 관련 컬럼 정의
# ============================================================

ID_COLUMN = "link_id"

FEATURE_COLUMNS = [
    "region_code",
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch",
    "traffic",
    "traffic_is_proxy",
]

LABEL_COLUMN = "label"

# label 생성에 사용했으므로 모델 입력 금지
LEAKAGE_COLUMN = "accident_distance_m"


# ============================================================
# 3. 시작
# ============================================================

print("=== SafeRoute 최종 학습 데이터 검증 ===")


# ============================================================
# 4. 데이터 불러오기
# ============================================================

print("\n[1] 데이터 불러오기...")

df = pd.read_parquet(INPUT_PATH)

print("전체 LINK:", len(df))
print("전체 컬럼:", len(df.columns))

print("\n컬럼:")
print(df.columns.tolist())


# ============================================================
# 5. 필수 컬럼 검사
# ============================================================

print("\n[2] 필수 컬럼 검사...")

required_columns = (
    [ID_COLUMN]
    + FEATURE_COLUMNS
    + [LEAKAGE_COLUMN, LABEL_COLUMN]
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise RuntimeError(
        f"필수 컬럼 누락: {missing_columns}"
    )

print("필수 컬럼 확인 완료")


# ============================================================
# 6. 전체 데이터 기본 검사
# ============================================================

print("\n[3] 전체 데이터 기본 검사...")

print(
    "link_id 중복:",
    df[ID_COLUMN].duplicated().sum()
)

print(
    "link_id 결측:",
    df[ID_COLUMN].isna().sum()
)

if df[ID_COLUMN].duplicated().any():
    raise RuntimeError(
        "link_id 중복이 존재합니다."
    )


# ============================================================
# 7. Gray Zone 제외
# ============================================================

print("\n[4] Gray Zone 제외...")

gray_count = int(
    df[LABEL_COLUMN].isna().sum()
)

train_df = (
    df[
        df[LABEL_COLUMN].notna()
    ]
    .copy()
    .reset_index(drop=True)
)

train_df[LABEL_COLUMN] = (
    train_df[LABEL_COLUMN]
    .astype(int)
)

print("전체 LINK:", len(df))
print("Gray Zone:", gray_count)
print("학습 대상:", len(train_df))


# ============================================================
# 8. Label 검사
# ============================================================

print("\n[5] Label 검사...")

label_counts = (
    train_df[LABEL_COLUMN]
    .value_counts()
    .sort_index()
)

print(label_counts.to_string())

unique_labels = set(
    train_df[LABEL_COLUMN].unique()
)

print(
    "고유 Label:",
    sorted(unique_labels)
)

if unique_labels != {0, 1}:
    raise RuntimeError(
        f"예상하지 못한 label 값: {unique_labels}"
    )

positive_count = int(
    (train_df[LABEL_COLUMN] == 1).sum()
)

negative_count = int(
    (train_df[LABEL_COLUMN] == 0).sum()
)

print(
    f"Positive 비율: "
    f"{positive_count / len(train_df) * 100:.2f}%"
)

print(
    f"Negative 비율: "
    f"{negative_count / len(train_df) * 100:.2f}%"
)


# ============================================================
# 9. 모델 입력 X / 정답 y 분리
#
# 중요:
# accident_distance_m 제외
# link_id 제외
# ============================================================

print("\n[6] 모델 입력 컬럼 구성...")

X = train_df[
    FEATURE_COLUMNS
].copy()

y = train_df[
    LABEL_COLUMN
].copy()

print("X 행:", len(X))
print("X 컬럼 수:", len(X.columns))
print("y 행:", len(y))

print("\n모델 Feature:")
for col in X.columns:
    print("-", col)

print(
    "\naccident_distance_m 포함 여부:",
    LEAKAGE_COLUMN in X.columns
)

print(
    "link_id 포함 여부:",
    ID_COLUMN in X.columns
)

if LEAKAGE_COLUMN in X.columns:
    raise RuntimeError(
        "accident_distance_m이 모델 Feature에 포함되었습니다."
    )

if ID_COLUMN in X.columns:
    raise RuntimeError(
        "link_id가 모델 Feature에 포함되었습니다."
    )


# ============================================================
# 10. Feature 데이터 타입 검사
# ============================================================

print("\n[7] Feature 데이터 타입...")

print(
    X.dtypes.to_string()
)


# ============================================================
# 11. 결측값 검사
# ============================================================

print("\n[8] Feature 결측값 검사...")

missing = X.isna().sum()

print(
    missing.to_string()
)

total_missing = int(
    missing.sum()
)

print(
    "\n전체 Feature 결측:",
    total_missing
)

if total_missing > 0:
    raise RuntimeError(
        "모델 Feature에 결측값이 존재합니다."
    )


# ============================================================
# 12. 무한대 검사
# ============================================================

print("\n[9] 무한대 값 검사...")

numeric_columns = (
    X.select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)

inf_counts = {}

for col in numeric_columns:

    values = pd.to_numeric(
        X[col],
        errors="coerce",
    )

    inf_counts[col] = int(
        np.isinf(values).sum()
    )


inf_series = pd.Series(
    inf_counts,
    dtype="int64",
)

print(
    inf_series.to_string()
)

total_inf = int(
    inf_series.sum()
)

print(
    "\n전체 무한대:",
    total_inf
)

if total_inf > 0:
    raise RuntimeError(
        "Feature에 무한대 값이 존재합니다."
    )


# ============================================================
# 13. 고유값 수 검사
#
# 상수 Feature가 있는지 확인
# ============================================================

print("\n[10] Feature 고유값 수...")

unique_counts = X.nunique(
    dropna=False
)

print(
    unique_counts.to_string()
)

constant_columns = (
    unique_counts[
        unique_counts <= 1
    ]
    .index
    .tolist()
)

print(
    "\n상수 Feature:",
    constant_columns
)


# ============================================================
# 14. 범주형 성격 Feature 분포
# ============================================================

print("\n[11] 범주형 성격 Feature 분포...")


categorical_like = [
    "region_code",
    "lanes",
    "max_spd",
    "road_rank",
    "n_branch",
    "traffic_is_proxy",
]


for col in categorical_like:

    print(
        f"\n--- {col} ---"
    )

    counts = (
        train_df[col]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        counts.to_string()
    )


# ============================================================
# 15. 연속형 Feature 기술통계
# ============================================================

print("\n[12] 연속형 Feature 기술통계...")


continuous_columns = [
    "length",
    "curvature",
    "n_intersect_200m",
    "traffic",
]


print(
    train_df[
        continuous_columns
    ]
    .describe()
    .T
    .to_string()
)


# ============================================================
# 16. Label별 Feature 평균
#
# 모델 학습 전 단순 구조 확인용
# 인과관계를 의미하지 않음
# ============================================================

print("\n[13] Label별 Feature 평균...")


mean_columns = [
    "lanes",
    "max_spd",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch",
    "traffic",
]


label_means = (
    train_df
    .groupby(LABEL_COLUMN)[
        mean_columns
    ]
    .mean()
    .T
)

label_means.columns = [
    "label_0_mean",
    "label_1_mean",
]


print(
    label_means.to_string()
)


# ============================================================
# 17. 지역별 Label 분포
# ============================================================

print("\n[14] 지역별 학습 Label 분포...")


region_label = pd.crosstab(
    train_df["region_code"],
    train_df[LABEL_COLUMN],
)


region_label.columns = [
    f"label_{col}"
    for col in region_label.columns
]


region_label[
    "total"
] = region_label.sum(axis=1)


if "label_1" in region_label.columns:

    region_label[
        "positive_ratio"
    ] = (
        region_label["label_1"]
        / region_label["total"]
        * 100
    )


print(
    region_label.to_string()
)


# ============================================================
# 18. traffic proxy와 Label 관계 확인
# ============================================================

print(
    "\n[15] traffic_is_proxy × Label..."
)


proxy_label = pd.crosstab(
    train_df[
        "traffic_is_proxy"
    ],
    train_df[
        LABEL_COLUMN
    ],
)


print(
    proxy_label.to_string()
)


# ============================================================
# 19. Curvature 이상치 확인
#
# 기존 전체 데이터에서 최대값이 컸으므로
# 학습 데이터에서도 상위값 확인
# ============================================================

print(
    "\n[16] Curvature 상위 10개..."
)


curvature_top = (
    train_df[
        [
            "link_id",
            "region_code",
            "length",
            "curvature",
            LABEL_COLUMN,
        ]
    ]
    .sort_values(
        "curvature",
        ascending=False,
    )
    .head(10)
)


print(
    curvature_top.to_string(
        index=False
    )
)


# ============================================================
# 20. n_intersect_200m 상위 10개
# ============================================================

print(
    "\n[17] n_intersect_200m 상위 10개..."
)


intersection_top = (
    train_df[
        [
            "link_id",
            "region_code",
            "n_intersect_200m",
            LABEL_COLUMN,
        ]
    ]
    .sort_values(
        "n_intersect_200m",
        ascending=False,
    )
    .head(10)
)


print(
    intersection_top.to_string(
        index=False
    )
)


# ============================================================
# 21. Traffic 상위 10개
# ============================================================

print(
    "\n[18] Traffic 상위 10개..."
)


traffic_top = (
    train_df[
        [
            "link_id",
            "region_code",
            "road_rank",
            "traffic",
            "traffic_is_proxy",
            LABEL_COLUMN,
        ]
    ]
    .sort_values(
        "traffic",
        ascending=False,
    )
    .head(10)
)


print(
    traffic_top.to_string(
        index=False
    )
)


# ============================================================
# 22. Feature별 Label 상관계수
#
# 단순 참고용.
# 상관관계가 인과관계를 의미하지 않음.
# ============================================================

print(
    "\n[19] 수치 Feature와 Label 상관계수..."
)


correlation_df = (
    train_df[
        mean_columns
        + [LABEL_COLUMN]
    ]
    .corr(
        numeric_only=True
    )
)


label_correlation = (
    correlation_df[
        LABEL_COLUMN
    ]
    .drop(
        LABEL_COLUMN
    )
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False,
    )
)


print(
    label_correlation.to_string()
)


# ============================================================
# 23. 최종 검증
# ============================================================

print(
    "\n[20] 최종 검증..."
)


checks = {
    "학습 데이터 존재": (
        len(train_df) > 0
    ),

    "Label 0/1만 존재": (
        unique_labels == {0, 1}
    ),

    "Feature 결측 없음": (
        total_missing == 0
    ),

    "Feature 무한대 없음": (
        total_inf == 0
    ),

    "상수 Feature 없음": (
        len(constant_columns) == 0
    ),

    "accident_distance_m 제외": (
        LEAKAGE_COLUMN
        not in X.columns
    ),

    "link_id 제외": (
        ID_COLUMN
        not in X.columns
    ),
}


for name, result in checks.items():

    print(
        f"{name}: "
        f"{'PASS' if result else 'FAIL'}"
    )


failed_checks = [
    name
    for name, result in checks.items()
    if not result
]


# ============================================================
# 24. 최종 요약
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute 최종 학습 데이터 검증 요약 ==="
)

print(
    "전체 LINK:",
    len(df)
)

print(
    "Gray Zone 제외:",
    gray_count
)

print(
    "학습 대상:",
    len(train_df)
)

print(
    "Feature 수:",
    len(FEATURE_COLUMNS)
)

print(
    "Positive:",
    positive_count
)

print(
    "Negative:",
    negative_count
)

print(
    f"Positive 비율: "
    f"{positive_count / len(train_df) * 100:.2f}%"
)

print(
    "Feature 결측:",
    total_missing
)

print(
    "Feature 무한대:",
    total_inf
)

print(
    "상수 Feature:",
    len(constant_columns)
)

print(
    "실패한 검증:",
    failed_checks
)


if failed_checks:

    print(
        "\n⚠ 모델 학습 전에 확인이 필요합니다."
    )

else:

    print(
        "\n모든 기본 검증 PASS"
    )

    print(
        "모델 학습 단계로 진행 가능합니다."
    )


print(
    "\n※ accident_distance_m은 모델 Feature에서 제외되었습니다."
)

print(
    "=== SafeRoute 최종 학습 데이터 검증 완료 ==="
)