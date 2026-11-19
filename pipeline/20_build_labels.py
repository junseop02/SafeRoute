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
    / "traffic_features_cheonan_asan.parquet"
)

DISTANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "accident_distance_cheonan_asan.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_cheonan_asan.parquet"
)


# ============================================================
# 2. Label 기준
# ============================================================

POSITIVE_DISTANCE = 80.0
NEGATIVE_DISTANCE = 300.0


# ============================================================
# 3. 시작
# ============================================================

print(
    "=== SafeRoute Label 생성 ==="
)


# ============================================================
# 4. 데이터 불러오기
# ============================================================

print(
    "\n[1] 데이터 불러오기..."
)

features = pd.read_parquet(
    FEATURE_PATH
)

distance = pd.read_parquet(
    DISTANCE_PATH
)


print(
    "Feature:",
    len(features)
)

print(
    "Distance:",
    len(distance)
)


# ============================================================
# 5. 기본 검증
# ============================================================

print(
    "\n[2] 기본 검증..."
)


required_feature_columns = [
    "link_id",
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


missing_feature_columns = [
    col
    for col in required_feature_columns
    if col not in features.columns
]


if missing_feature_columns:

    raise RuntimeError(
        "Feature 필수 컬럼 누락: "
        f"{missing_feature_columns}"
    )


required_distance_columns = [
    "link_id",
    "accident_distance_m",
]


missing_distance_columns = [
    col
    for col in required_distance_columns
    if col not in distance.columns
]


if missing_distance_columns:

    raise RuntimeError(
        "Distance 필수 컬럼 누락: "
        f"{missing_distance_columns}"
    )


features["link_id"] = (
    features["link_id"]
    .astype(str)
)

distance["link_id"] = (
    distance["link_id"]
    .astype(str)
)


print(
    "Feature link_id 중복:",
    features[
        "link_id"
    ].duplicated().sum()
)

print(
    "Distance link_id 중복:",
    distance[
        "link_id"
    ].duplicated().sum()
)


if features["link_id"].duplicated().any():

    raise RuntimeError(
        "Feature에 link_id 중복이 있습니다."
    )


if distance["link_id"].duplicated().any():

    raise RuntimeError(
        "Distance에 link_id 중복이 있습니다."
    )


# ============================================================
# 6. Feature ↔ Distance LINK 일치 검사
# ============================================================

print(
    "\n[3] LINK 일치 검사..."
)


feature_ids = set(
    features["link_id"]
)

distance_ids = set(
    distance["link_id"]
)


only_feature = (
    feature_ids
    - distance_ids
)

only_distance = (
    distance_ids
    - feature_ids
)


print(
    "Feature에만 있는 LINK:",
    len(only_feature)
)

print(
    "Distance에만 있는 LINK:",
    len(only_distance)
)


if (
    len(only_feature) > 0
    or len(only_distance) > 0
):

    raise RuntimeError(
        "Feature와 Distance의 LINK 구성이 다릅니다."
    )


# ============================================================
# 7. 거리 병합
# ============================================================

print(
    "\n[4] 사고 최소거리 병합..."
)


merged = features.merge(
    distance[
        [
            "link_id",
            "accident_distance_m",
        ]
    ],
    on="link_id",
    how="left",
    validate="one_to_one",
)


print(
    "병합 후 행:",
    len(merged)
)

print(
    "거리 결측:",
    merged[
        "accident_distance_m"
    ].isna().sum()
)


# ============================================================
# 8. Label 생성
#
# <= 80m       -> 1
# 80m ~ 300m   -> NaN (Gray Zone)
# > 300m       -> 0
#
# 주의:
# 80m 정확히 포함 -> 1
# 300m 정확히 포함 -> Gray Zone
# ============================================================

print(
    "\n[5] Label 생성..."
)


merged["label"] = pd.NA


positive_mask = (
    merged[
        "accident_distance_m"
    ] <= POSITIVE_DISTANCE
)


gray_mask = (
    (
        merged[
            "accident_distance_m"
        ] > POSITIVE_DISTANCE
    )
    &
    (
        merged[
            "accident_distance_m"
        ] <= NEGATIVE_DISTANCE
    )
)


negative_mask = (
    merged[
        "accident_distance_m"
    ] > NEGATIVE_DISTANCE
)


merged.loc[
    positive_mask,
    "label",
] = 1


merged.loc[
    negative_mask,
    "label",
] = 0


# nullable integer
merged["label"] = (
    merged["label"]
    .astype("Int64")
)


# ============================================================
# 9. Label 분포 확인
# ============================================================

print(
    "\n[6] Label 분포..."
)


positive_count = int(
    (
        merged["label"] == 1
    ).sum()
)

negative_count = int(
    (
        merged["label"] == 0
    ).sum()
)

gray_count = int(
    merged["label"]
    .isna()
    .sum()
)


print(
    "label = 1:",
    positive_count
)

print(
    "label = 0:",
    negative_count
)

print(
    "Gray Zone:",
    gray_count
)

print(
    "전체:",
    len(merged)
)


# ============================================================
# 10. 거리 기준과 Label 일치 검증
# ============================================================

print(
    "\n[7] 거리 ↔ Label 검증..."
)


positive_error = (
    (
        merged["label"] == 1
    )
    &
    (
        merged[
            "accident_distance_m"
        ] > POSITIVE_DISTANCE
    )
).sum()


negative_error = (
    (
        merged["label"] == 0
    )
    &
    (
        merged[
            "accident_distance_m"
        ] <= NEGATIVE_DISTANCE
    )
).sum()


gray_error = (
    merged["label"]
    .isna()
    &
    ~gray_mask
).sum()


print(
    "Positive 기준 오류:",
    int(positive_error)
)

print(
    "Negative 기준 오류:",
    int(negative_error)
)

print(
    "Gray Zone 기준 오류:",
    int(gray_error)
)


if (
    positive_error > 0
    or negative_error > 0
    or gray_error > 0
):

    raise RuntimeError(
        "Label 생성 기준에 맞지 않는 행이 있습니다."
    )


# ============================================================
# 11. 학습 가능 데이터 수
# ============================================================

print(
    "\n[8] 학습 가능 데이터..."
)


trainable_count = (
    positive_count
    + negative_count
)


print(
    "학습 가능 LINK:",
    trainable_count
)

print(
    "학습 제외 Gray Zone:",
    gray_count
)


positive_ratio = (
    positive_count
    / trainable_count
    * 100
)


negative_ratio = (
    negative_count
    / trainable_count
    * 100
)


print(
    f"label=1 비율: "
    f"{positive_ratio:.2f}%"
)

print(
    f"label=0 비율: "
    f"{negative_ratio:.2f}%"
)


# ============================================================
# 12. 지역별 Label 분포
# ============================================================

print(
    "\n[9] 지역별 Label 분포..."
)


region_label = pd.crosstab(
    merged[
        "region_code"
    ],
    merged[
        "label"
    ],
    dropna=False,
)


print(
    region_label.to_string()
)


# ============================================================
# 13. Gray Zone 거리 검증
# ============================================================

print(
    "\n[10] Gray Zone 거리 통계..."
)


gray_distance = merged.loc[
    merged["label"].isna(),
    "accident_distance_m",
]


print(
    gray_distance.describe()
)


# ============================================================
# 14. 최종 컬럼 정리
#
# accident_distance_m은 검증/분석용으로 보존
# 모델 X에는 사용하지 않음.
#
# 사고지점과의 거리로 label을 만들었기 때문에
# 이 컬럼을 모델 입력으로 넣으면 label leakage 발생.
# ============================================================

print(
    "\n[11] 최종 컬럼 정리..."
)


final_columns = [
    "link_id",
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
    "accident_distance_m",
    "label",
]


output = merged[
    final_columns
].copy()


print(
    output.columns.tolist()
)


# ============================================================
# 15. 최종 결측 검사
# ============================================================

print(
    "\n[12] 최종 결측 검사..."
)


print(
    output.isna()
    .sum()
    .to_string()
)


non_label_columns = [
    col
    for col in output.columns
    if col != "label"
]


non_label_missing = (
    output[
        non_label_columns
    ]
    .isna()
    .sum()
    .sum()
)


print(
    "\nlabel 제외 전체 결측:",
    int(
        non_label_missing
    )
)


if non_label_missing > 0:

    raise RuntimeError(
        "label 이외의 컬럼에 결측값이 있습니다."
    )


# ============================================================
# 16. 저장
# ============================================================

print(
    "\n[13] 저장..."
)


output.to_parquet(
    OUTPUT_PATH,
    index=False,
)


print(
    "저장 경로:",
    OUTPUT_PATH
)

print(
    "파일 존재:",
    OUTPUT_PATH.exists()
)


# ============================================================
# 17. 저장 결과 재검증
# ============================================================

print(
    "\n[14] 저장 결과 재검증..."
)


check = pd.read_parquet(
    OUTPUT_PATH
)


print(
    "저장 행:",
    len(check)
)

print(
    "link_id 중복:",
    check[
        "link_id"
    ].duplicated().sum()
)

print(
    "label=1:",
    int(
        (
            check["label"] == 1
        ).sum()
    )
)

print(
    "label=0:",
    int(
        (
            check["label"] == 0
        ).sum()
    )
)

print(
    "label 결측:",
    check[
        "label"
    ].isna().sum()
)


# ============================================================
# 18. 최종 요약
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute Label 생성 요약 ==="
)

print(
    "전체 LINK:",
    len(check)
)

print(
    "Positive (<=80m):",
    positive_count
)

print(
    "Gray Zone (80~300m):",
    gray_count
)

print(
    "Negative (>300m):",
    negative_count
)

print(
    "학습 가능 LINK:",
    trainable_count
)

print(
    f"학습 데이터 Positive 비율: "
    f"{positive_ratio:.2f}%"
)

print(
    f"학습 데이터 Negative 비율: "
    f"{negative_ratio:.2f}%"
)

print(
    "link_id 중복:",
    check[
        "link_id"
    ].duplicated().sum()
)

print(
    "\n주의:"
)

print(
    "- Gray Zone의 label은 NaN으로 유지합니다."
)

print(
    "- 모델 학습 시 Gray Zone은 제외합니다."
)

print(
    "- accident_distance_m은 label 생성/검증용입니다."
)

print(
    "- accident_distance_m은 모델 입력 Feature로 사용하면 안 됩니다."
)

print(
    "\n=== SafeRoute Label 생성 완료 ==="
)