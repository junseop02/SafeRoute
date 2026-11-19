from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_cheonan_asan.parquet"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. 학습 설정
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

LABEL_COLUMN = "label"

TRAFFIC_FEATURES = [
    "traffic",
]

ALL_FEATURES = [
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


# ============================================================
# 3. 시작
# ============================================================

print(
    "=== SafeRoute Logistic Regression Baseline 학습 ==="
)


# ============================================================
# 4. 데이터 불러오기
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


# ============================================================
# 5. Gray Zone 제외
# ============================================================

print(
    "\n[2] Gray Zone 제외..."
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


print(
    "학습 가능 데이터:",
    len(train_df)
)

print(
    "label=0:",
    int(
        (
            train_df[LABEL_COLUMN] == 0
        ).sum()
    )
)

print(
    "label=1:",
    int(
        (
            train_df[LABEL_COLUMN] == 1
        ).sum()
    )
)


# ============================================================
# 6. Feature / Label 준비
# ============================================================

print(
    "\n[3] Feature / Label 준비..."
)

X = train_df[
    ALL_FEATURES
].copy()

y = train_df[
    LABEL_COLUMN
].copy()


print(
    "Feature 수:",
    len(ALL_FEATURES)
)

print(
    "accident_distance_m 사용 여부:",
    "accident_distance_m" in X.columns
)

print(
    "link_id 사용 여부:",
    "link_id" in X.columns
)


if "accident_distance_m" in X.columns:

    raise RuntimeError(
        "Label Leakage: accident_distance_m이 포함되었습니다."
    )


if "link_id" in X.columns:

    raise RuntimeError(
        "link_id가 Feature에 포함되었습니다."
    )


# ============================================================
# 7. Train / Test 분할
#
# stratify=y:
# Train/Test에 label 비율을 비슷하게 유지
#
# random_state=42:
# 이후 LightGBM에서도 동일 분할 사용
# ============================================================

print(
    "\n[4] Train / Test 분할..."
)

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


print(
    "\nTrain Label:"
)

print(
    y_train
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    "\nTest Label:"
)

print(
    y_test
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 8. 모델 생성 함수
#
# Logistic Regression은 Feature 스케일 차이에
# 영향을 받으므로 StandardScaler 적용
#
# class_weight="balanced":
# label 1이 적은 클래스 불균형 대응
# ============================================================

def build_logistic_pipeline(
    feature_columns,
):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )


    model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
    )


    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# 9. 평가 함수
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_eval,
    y_eval,
):

    pred = model.predict(
        X_eval
    )

    prob = model.predict_proba(
        X_eval
    )[:, 1]


    accuracy = accuracy_score(
        y_eval,
        pred,
    )

    precision = precision_score(
        y_eval,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y_eval,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_eval,
        pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_eval,
        prob,
    )

    pr_auc = average_precision_score(
        y_eval,
        prob,
    )


    cm = confusion_matrix(
        y_eval,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{model_name}"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )


    print(
        "\nConfusion Matrix"
    )

    print(
        "                Pred 0   Pred 1"
    )

    print(
        f"Actual 0        {tn:6d}   {fp:6d}"
    )

    print(
        f"Actual 1        {fn:6d}   {tp:6d}"
    )


    result = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


    return result


# ============================================================
# 10. Baseline 1
# Traffic-only Logistic Regression
# ============================================================

print(
    "\n[5] Traffic-only Logistic Regression 학습..."
)


traffic_model = build_logistic_pipeline(
    TRAFFIC_FEATURES
)


traffic_model.fit(
    X_train[
        TRAFFIC_FEATURES
    ],
    y_train,
)


traffic_result = evaluate_model(
    model_name=(
        "Traffic-only Logistic Regression"
    ),
    model=traffic_model,
    X_eval=X_test[
        TRAFFIC_FEATURES
    ],
    y_eval=y_test,
)


# ============================================================
# 11. Baseline 2
# All-feature Logistic Regression
# ============================================================

print(
    "\n[6] All-feature Logistic Regression 학습..."
)


all_model = build_logistic_pipeline(
    ALL_FEATURES
)


all_model.fit(
    X_train[
        ALL_FEATURES
    ],
    y_train,
)


all_result = evaluate_model(
    model_name=(
        "All-feature Logistic Regression"
    ),
    model=all_model,
    X_eval=X_test[
        ALL_FEATURES
    ],
    y_eval=y_test,
)


# ============================================================
# 12. 성능 비교
# ============================================================

print(
    "\n[7] Baseline 성능 비교..."
)


results = pd.DataFrame(
    [
        traffic_result,
        all_result,
    ]
)


metric_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


print(
    results[
        metric_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 13. 모델 저장
# ============================================================

print(
    "\n[8] 모델 저장..."
)


traffic_model_path = (
    MODEL_DIR
    / "logistic_traffic_only.joblib"
)

all_model_path = (
    MODEL_DIR
    / "logistic_all_features.joblib"
)


joblib.dump(
    traffic_model,
    traffic_model_path,
)

joblib.dump(
    all_model,
    all_model_path,
)


print(
    "Traffic-only 모델:",
    traffic_model_path
)

print(
    "All-feature 모델:",
    all_model_path
)


# ============================================================
# 14. 평가 결과 저장
# ============================================================

print(
    "\n[9] 평가 결과 저장..."
)


result_path = (
    OUTPUT_DIR
    / "logistic_baseline_metrics.csv"
)


results.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig",
)


print(
    "평가 결과:",
    result_path
)


# ============================================================
# 15. Test 예측 결과 저장
#
# 이후 LightGBM과 같은 Test 데이터에서 비교할 수 있도록
# link_id도 같이 저장
# ============================================================

print(
    "\n[10] Test 예측 결과 저장..."
)


test_index = X_test.index


prediction_output = pd.DataFrame(
    {
        "link_id": train_df.loc[
            test_index,
            "link_id",
        ].values,

        "label": y_test.values,

        "traffic_only_prob": (
            traffic_model.predict_proba(
                X_test[
                    TRAFFIC_FEATURES
                ]
            )[:, 1]
        ),

        "all_feature_logistic_prob": (
            all_model.predict_proba(
                X_test[
                    ALL_FEATURES
                ]
            )[:, 1]
        ),
    }
)


prediction_path = (
    OUTPUT_DIR
    / "logistic_test_predictions.parquet"
)


prediction_output.to_parquet(
    prediction_path,
    index=False,
)


print(
    "Test 예측 결과:",
    prediction_path
)

print(
    "Test 예측 행:",
    len(prediction_output)
)


# ============================================================
# 16. Logistic 회귀계수 확인
#
# StandardScaler 적용 후의 coefficient
# 절대값이 크다고 곧바로 인과적 중요도라고 해석하면 안 됨
# ============================================================

print(
    "\n[11] All-feature Logistic Regression 계수..."
)


coefficients = (
    all_model
    .named_steps["model"]
    .coef_[0]
)


coefficient_df = pd.DataFrame(
    {
        "feature": ALL_FEATURES,
        "coefficient": coefficients,
    }
)


coefficient_df[
    "abs_coefficient"
] = (
    coefficient_df[
        "coefficient"
    ].abs()
)


coefficient_df = (
    coefficient_df
    .sort_values(
        "abs_coefficient",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print(
    coefficient_df.to_string(
        index=False
    )
)


coefficient_path = (
    OUTPUT_DIR
    / "logistic_coefficients.csv"
)


coefficient_df.to_csv(
    coefficient_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. 최종 요약
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute Logistic Regression Baseline 완료 ==="
)

print(
    "Train:",
    len(X_train)
)

print(
    "Test:",
    len(X_test)
)

print(
    "Random State:",
    RANDOM_STATE
)

print(
    "Test Size:",
    TEST_SIZE
)


print(
    "\nTraffic-only"
)

print(
    f"ROC-AUC: "
    f"{traffic_result['roc_auc']:.4f}"
)

print(
    f"PR-AUC : "
    f"{traffic_result['pr_auc']:.4f}"
)

print(
    f"F1     : "
    f"{traffic_result['f1']:.4f}"
)


print(
    "\nAll-feature"
)

print(
    f"ROC-AUC: "
    f"{all_result['roc_auc']:.4f}"
)

print(
    f"PR-AUC : "
    f"{all_result['pr_auc']:.4f}"
)

print(
    f"F1     : "
    f"{all_result['f1']:.4f}"
)


print(
    "\n저장 완료:"
)

print(
    "- logistic_traffic_only.joblib"
)

print(
    "- logistic_all_features.joblib"
)

print(
    "- logistic_baseline_metrics.csv"
)

print(
    "- logistic_test_predictions.parquet"
)

print(
    "- logistic_coefficients.csv"
)

print(
    "\n다음 단계: LightGBM"
)

print(
    "=== 완료 ==="
)