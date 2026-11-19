from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd


# ============================================================
# 1. 경로 / 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_cheonan_asan.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_MODEL_PATH = (
    MODEL_DIR
    / "lightgbm_final.joblib"
)

FINAL_NATIVE_MODEL_PATH = (
    MODEL_DIR
    / "lightgbm_final.txt"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "risk_all_links.parquet"
)


LABEL_COLUMN = "label"

RANDOM_STATE = 42

# 23번 평가 모델의 Best iteration
FINAL_N_ESTIMATORS = 498


FEATURES = [
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


print(
    "=== SafeRoute 최종 LightGBM 학습 + 전체 LINK 예측 ==="
)


# ============================================================
# 2. 전체 데이터 불러오기
# ============================================================

print(
    "\n[1] 데이터 불러오기..."
)

df = pd.read_parquet(
    DATA_PATH
)


print(
    "전체 LINK:",
    len(df)
)

print(
    "전체 컬럼:",
    len(df.columns)
)


# ============================================================
# 3. 기본 데이터 검사
# ============================================================

print(
    "\n[2] 기본 데이터 검사..."
)


if df["link_id"].duplicated().any():

    raise RuntimeError(
        "link_id 중복이 존재합니다."
    )


print(
    "link_id 중복:",
    int(
        df["link_id"]
        .duplicated()
        .sum()
    )
)


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:

    raise RuntimeError(
        f"필요 Feature가 없습니다: {missing_features}"
    )


feature_missing_count = (
    df[
        FEATURES
    ]
    .isna()
    .sum()
    .sum()
)


print(
    "Feature 결측:",
    int(
        feature_missing_count
    )
)


if feature_missing_count > 0:

    raise RuntimeError(
        "Feature에 결측값이 존재합니다."
    )


# ============================================================
# 4. 학습 데이터 생성
#
# label이 존재하는 LINK만 사용
# Gray Zone(label=NaN)은 학습에서 제외
# ============================================================

print(
    "\n[3] 최종 학습 데이터 생성..."
)


train_df = (
    df[
        df[LABEL_COLUMN].notna()
    ]
    .copy()
    .reset_index(drop=True)
)


train_df[
    LABEL_COLUMN
] = (
    train_df[
        LABEL_COLUMN
    ]
    .astype(int)
)


print(
    "최종 학습 LINK:",
    len(train_df)
)

print(
    "Gray Zone 제외:",
    len(df) - len(train_df)
)


label_counts = (
    train_df[
        LABEL_COLUMN
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nLabel 분포:"
)

print(
    label_counts.to_string()
)


# ============================================================
# 5. X / y 준비
# ============================================================

print(
    "\n[4] X / y 준비..."
)


X_train = (
    train_df[
        FEATURES
    ]
    .copy()
)

y_train = (
    train_df[
        LABEL_COLUMN
    ]
    .copy()
)


# bool → int
X_train[
    "traffic_is_proxy"
] = (
    X_train[
        "traffic_is_proxy"
    ]
    .astype(int)
)


# Leakage 방지
forbidden_columns = [
    "link_id",
    "accident_distance_m",
    "label",
]


for column in forbidden_columns:

    if column in X_train.columns:

        raise RuntimeError(
            f"학습 금지 컬럼 발견: {column}"
        )


print(
    "Feature 수:",
    len(FEATURES)
)

print(
    "Leakage 컬럼 없음"
)


# ============================================================
# 6. 클래스 불균형 계산
# ============================================================

print(
    "\n[5] 클래스 불균형 계산..."
)


negative_count = int(
    (
        y_train == 0
    ).sum()
)

positive_count = int(
    (
        y_train == 1
    ).sum()
)


scale_pos_weight = (
    negative_count
    / positive_count
)


print(
    "Negative:",
    negative_count
)

print(
    "Positive:",
    positive_count
)

print(
    "scale_pos_weight:",
    round(
        scale_pos_weight,
        4,
    )
)


# ============================================================
# 7. 최종 LightGBM 생성
#
# 평가 단계에서 Best iteration = 498
# 전체 학습 데이터에서는 validation/test를 사용하지 않음
# ============================================================

print(
    "\n[6] 최종 LightGBM 생성..."
)


final_model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=FINAL_N_ESTIMATORS,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=20,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.0,

    reg_lambda=0.0,

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1,
)


print(
    "n_estimators:",
    FINAL_N_ESTIMATORS
)


# ============================================================
# 8. 전체 학습 가능 데이터로 최종 학습
# ============================================================

print(
    "\n[7] 최종 모델 학습 시작..."
)


final_model.fit(
    X_train,
    y_train,
)


print(
    "최종 모델 학습 완료"
)


# ============================================================
# 9. 최종 모델 저장
# ============================================================

print(
    "\n[8] 최종 모델 저장..."
)


joblib.dump(
    final_model,
    FINAL_MODEL_PATH,
)


final_model.booster_.save_model(
    str(
        FINAL_NATIVE_MODEL_PATH
    )
)


print(
    "Joblib:",
    FINAL_MODEL_PATH
)

print(
    "Native:",
    FINAL_NATIVE_MODEL_PATH
)


# ============================================================
# 10. 전체 36,521개 LINK Feature 준비
#
# 여기서는 Gray Zone도 포함
# 모든 도로에 risk가 필요함
# ============================================================

print(
    "\n[9] 전체 LINK 예측 데이터 준비..."
)


X_all = (
    df[
        FEATURES
    ]
    .copy()
)


X_all[
    "traffic_is_proxy"
] = (
    X_all[
        "traffic_is_proxy"
    ]
    .astype(int)
)


print(
    "예측 대상 LINK:",
    len(X_all)
)


# ============================================================
# 11. 전체 LINK 위험도 예측
#
# risk는 실제 사고 발생 확률이 아니라
# 모델의 사고다발 조건 위험 점수
# ============================================================

print(
    "\n[10] 전체 LINK risk 예측..."
)


risk = (
    final_model
    .predict_proba(
        X_all
    )[:, 1]
)


print(
    "risk 생성:",
    len(risk)
)


# ============================================================
# 12. 결과 DataFrame 생성
# ============================================================

print(
    "\n[11] 예측 결과 생성..."
)


risk_df = pd.DataFrame(
    {
        "link_id": (
            df[
                "link_id"
            ]
            .astype(str)
            .values
        ),

        "risk": risk,
    }
)


print(
    "결과 행:",
    len(risk_df)
)


# ============================================================
# 13. Risk 기본 검증
# ============================================================

print(
    "\n[12] Risk 기본 검증..."
)


duplicate_count = int(
    risk_df[
        "link_id"
    ]
    .duplicated()
    .sum()
)


missing_count = int(
    risk_df[
        "risk"
    ]
    .isna()
    .sum()
)


out_of_range_count = int(
    (
        (
            risk_df["risk"] < 0
        )
        |
        (
            risk_df["risk"] > 1
        )
    ).sum()
)


print(
    "link_id 중복:",
    duplicate_count
)

print(
    "risk 결측:",
    missing_count
)

print(
    "risk 범위 밖:",
    out_of_range_count
)


if duplicate_count != 0:

    raise RuntimeError(
        "risk 결과에 link_id 중복이 있습니다."
    )


if missing_count != 0:

    raise RuntimeError(
        "risk 결과에 결측값이 있습니다."
    )


if out_of_range_count != 0:

    raise RuntimeError(
        "risk가 0~1 범위를 벗어났습니다."
    )


# ============================================================
# 14. Risk 분포 확인
# ============================================================

print(
    "\n[13] Risk 분포..."
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
# 15. Safety Score 참고 계산
#
# 현재 확인용.
# 백엔드 최종 인터페이스 risk.parquet에는
# 우선 link_id / risk만 사용.
#
# safety_score = 100 * (1 - risk)
# ============================================================

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
    "\nSafety Score 분포:"
)


print(
    safety_score
    .describe()
    .to_string()
)


# ============================================================
# 16. 위험도 상위 20개 LINK
# ============================================================

print(
    "\n[14] Risk 상위 20개 LINK..."
)


top_risk = (
    risk_df
    .sort_values(
        "risk",
        ascending=False,
    )
    .head(20)
)


print(
    top_risk.to_string(
        index=False
    )
)


# ============================================================
# 17. Label별 Risk 분포 확인
#
# Label은 모델 출력에는 포함하지 않고
# sanity check 용도로만 사용
# ============================================================

print(
    "\n[15] 기존 Label별 Risk 확인..."
)


check_df = pd.DataFrame(
    {
        "label": df[
            LABEL_COLUMN
        ].values,

        "risk": risk,
    }
)


label_risk_summary = (
    check_df[
        check_df[
            "label"
        ].notna()
    ]
    .groupby(
        "label"
    )[
        "risk"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)


print(
    label_risk_summary.to_string()
)


# Gray Zone도 따로 확인
gray_mask = (
    df[
        LABEL_COLUMN
    ]
    .isna()
)


gray_risk = (
    risk_df.loc[
        gray_mask,
        "risk"
    ]
)


print(
    "\nGray Zone Risk:"
)

print(
    gray_risk
    .describe()
    .to_string()
)


# ============================================================
# 18. 임시 전체 예측 결과 저장
#
# 27번에서 최종 인터페이스 검증 후
# outputs/risk.parquet으로 확정
# ============================================================

print(
    "\n[16] 전체 예측 결과 저장..."
)


risk_df.to_parquet(
    PREDICTION_PATH,
    index=False,
)


print(
    "저장:",
    PREDICTION_PATH
)


# ============================================================
# 19. 최종 요약
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute 최종 모델 학습 / 예측 완료 ==="
)

print(
    "=" * 70
)


print(
    "\n최종 학습:"
)

print(
    "Train LINK:",
    len(train_df)
)

print(
    "Feature:",
    len(FEATURES)
)

print(
    "Trees:",
    FINAL_N_ESTIMATORS
)

print(
    "scale_pos_weight:",
    round(
        scale_pos_weight,
        4,
    )
)


print(
    "\n전체 예측:"
)

print(
    "전체 LINK:",
    len(risk_df)
)

print(
    "중복:",
    duplicate_count
)

print(
    "결측:",
    missing_count
)

print(
    "범위 오류:",
    out_of_range_count
)


print(
    "\n저장 완료:"
)

print(
    "- models/lightgbm_final.joblib"
)

print(
    "- models/lightgbm_final.txt"
)

print(
    "- outputs/risk_all_links.parquet"
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
    "27_final_check_risk.py"
)

print(
    "→ 최종 risk.parquet 생성 및 검증"
)

print(
    "=== 완료 ==="
)