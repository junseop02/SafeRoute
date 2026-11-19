from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
LABEL_COLUMN = "label"


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
    "=== SafeRoute SHAP 분석 ==="
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print(
    "\n[1] 데이터 불러오기..."
)


df = pd.read_parquet(
    DATA_PATH
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


train_df["traffic_is_proxy"] = (
    train_df["traffic_is_proxy"]
    .astype(int)
)


print(
    "전체 LINK:",
    len(df)
)

print(
    "학습 가능 LINK:",
    len(train_df)
)


# ============================================================
# 3. 모델 불러오기
# ============================================================

print(
    "\n[2] LightGBM 모델 불러오기..."
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"모델 파일이 없습니다: {MODEL_PATH}"
    )


model = joblib.load(
    MODEL_PATH
)


print(
    "모델:",
    MODEL_PATH
)


# ============================================================
# 4. 23번과 동일한 Test 데이터 생성
# ============================================================

print(
    "\n[3] 동일 Test 데이터 생성..."
)


X = train_df[
    FEATURES
].copy()

y = train_df[
    LABEL_COLUMN
].copy()


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)


print(
    "Train:",
    len(X_train)
)

print(
    "Test:",
    len(X_test)
)


# ============================================================
# 5. SHAP TreeExplainer 생성
# ============================================================

print(
    "\n[4] SHAP TreeExplainer 생성..."
)


explainer = shap.TreeExplainer(
    model
)


# ============================================================
# 6. Test 데이터 SHAP 계산
# ============================================================

print(
    "\n[5] SHAP 값 계산..."
)


shap_values = explainer.shap_values(
    X_test
)


# SHAP 버전에 따라 binary classifier 반환 형태가
# 다를 수 있으므로 형태를 정리
if isinstance(
    shap_values,
    list,
):

    # binary classification에서
    # class 1 SHAP 사용
    if len(shap_values) == 2:

        shap_values = shap_values[1]

    else:

        shap_values = shap_values[0]


shap_values = np.asarray(
    shap_values
)


# 일부 버전에서
# (rows, features, classes) 형태일 경우
if shap_values.ndim == 3:

    shap_values = shap_values[
        :,
        :,
        1,
    ]


print(
    "SHAP shape:",
    shap_values.shape
)

print(
    "X_test shape:",
    X_test.shape
)


if shap_values.shape != X_test.shape:

    raise RuntimeError(
        "SHAP 값과 X_test의 shape가 일치하지 않습니다."
    )


# ============================================================
# 7. Global SHAP Importance
#
# mean(|SHAP|)
# 값이 클수록 모델 예측에 미치는 평균 영향이 큼
# ============================================================

print(
    "\n[6] Global SHAP Importance 계산..."
)


mean_abs_shap = np.mean(
    np.abs(
        shap_values
    ),
    axis=0,
)


importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_abs_shap": mean_abs_shap,
    }
)


importance_df = (
    importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    "\n=== SHAP Feature Importance ==="
)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 8. SHAP Importance 저장
# ============================================================

importance_path = (
    OUTPUT_DIR
    / "shap_feature_importance.csv"
)


importance_df.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\nSHAP Importance 저장:",
    importance_path
)


# ============================================================
# 9. SHAP Summary Plot
#
# Feature 값과 SHAP 방향을 동시에 확인
# ============================================================

print(
    "\n[7] SHAP Summary Plot 생성..."
)


plt.figure()


shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURES,
    show=False,
)


plt.tight_layout()


summary_path = (
    OUTPUT_DIR
    / "shap_summary_plot.png"
)


plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


print(
    "저장:",
    summary_path
)


# ============================================================
# 10. SHAP Bar Plot
# ============================================================

print(
    "\n[8] SHAP Bar Plot 생성..."
)


plt.figure()


shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURES,
    plot_type="bar",
    show=False,
)


plt.tight_layout()


bar_path = (
    OUTPUT_DIR
    / "shap_bar_plot.png"
)


plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


print(
    "저장:",
    bar_path
)


# ============================================================
# 11. Feature별 SHAP 방향성 요약
#
# 단순 correlation은 인과관계를 의미하지 않음.
# Feature 값이 증가할 때 SHAP 값이 어느 방향으로
# 움직이는지 참고하기 위한 통계.
# ============================================================

print(
    "\n[9] Feature별 SHAP 방향성 확인..."
)


direction_results = []


for i, feature in enumerate(
    FEATURES
):

    feature_values = (
        X_test[
            feature
        ]
        .to_numpy()
    )

    feature_shap = (
        shap_values[
            :,
            i,
        ]
    )


    # 값이 상수일 경우 correlation 계산 불가
    if (
        np.std(
            feature_values
        )
        == 0
        or
        np.std(
            feature_shap
        )
        == 0
    ):

        correlation = np.nan

    else:

        correlation = np.corrcoef(
            feature_values,
            feature_shap,
        )[0, 1]


    direction_results.append(
        {
            "feature": feature,

            "mean_abs_shap": (
                np.mean(
                    np.abs(
                        feature_shap
                    )
                )
            ),

            "shap_value_correlation": (
                correlation
            ),
        }
    )


direction_df = pd.DataFrame(
    direction_results
)


direction_df = (
    direction_df
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    direction_df.to_string(
        index=False
    )
)


direction_path = (
    OUTPUT_DIR
    / "shap_direction_summary.csv"
)


direction_df.to_csv(
    direction_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. 개별 Test LINK SHAP 저장
#
# 이후 특정 도로의 risk가 왜 높게 나왔는지
# 분석할 때 사용할 수 있음
# ============================================================

print(
    "\n[10] 개별 LINK SHAP 값 저장..."
)


test_indices = X_test.index


shap_detail_df = pd.DataFrame(
    shap_values,
    columns=[
        f"shap_{feature}"
        for feature in FEATURES
    ],
    index=test_indices,
)


feature_detail_df = (
    X_test
    .copy()
)


feature_detail_df.columns = [
    f"value_{feature}"
    for feature in FEATURES
]


detail_df = pd.concat(
    [
        train_df.loc[
            test_indices,
            [
                "link_id",
                LABEL_COLUMN,
            ],
        ],

        feature_detail_df,

        shap_detail_df,
    ],
    axis=1,
)


# 모델 위험 점수도 함께 저장
detail_df[
    "risk_score"
] = model.predict_proba(
    X_test
)[:, 1]


detail_df = (
    detail_df
    .sort_values(
        "risk_score",
        ascending=False,
    )
)


detail_path = (
    OUTPUT_DIR
    / "shap_test_details.parquet"
)


detail_df.to_parquet(
    detail_path,
    index=False,
)


print(
    "저장:",
    detail_path
)

print(
    "행:",
    len(detail_df)
)


# ============================================================
# 13. 위험 점수 상위 LINK 확인
# ============================================================

print(
    "\n[11] 위험 점수 상위 10개 Test LINK..."
)


top_columns = [
    "link_id",
    "label",
    "risk_score",
]


print(
    detail_df[
        top_columns
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 14. 최종 요약
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute SHAP 분석 완료 ==="
)

print(
    "=" * 70
)


print(
    "\nSHAP 중요도 TOP 10"
)


for rank, row in (
    importance_df
    .head(10)
    .iterrows()
):

    print(
        f"{rank + 1}. "
        f"{row['feature']} "
        f"(mean|SHAP|="
        f"{row['mean_abs_shap']:.6f})"
    )


print(
    "\n저장 완료:"
)

print(
    "- outputs/shap_feature_importance.csv"
)

print(
    "- outputs/shap_direction_summary.csv"
)

print(
    "- outputs/shap_summary_plot.png"
)

print(
    "- outputs/shap_bar_plot.png"
)

print(
    "- outputs/shap_test_details.parquet"
)


print(
    "\n주의:"
)

print(
    "SHAP은 모델 예측에 대한 설명이며"
)

print(
    "Feature와 실제 사고 사이의 인과관계를 의미하지 않습니다."
)


print(
    "\n다음 단계:"
)

print(
    "전체 36,521개 LINK risk 산출"
)

print(
    "=== 완료 ==="
)