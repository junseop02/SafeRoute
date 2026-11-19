from pathlib import Path

import joblib
import pandas as pd
import lightgbm as lgb

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


# ============================================================
# 2. 학습 설정
# ============================================================

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
    "=== SafeRoute LightGBM 학습 ==="
)


# ============================================================
# 3. 데이터 불러오기
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
# 4. Gray Zone 제외
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
# 5. Feature / Label 준비
# ============================================================

print(
    "\n[3] Feature / Label 준비..."
)

X = train_df[
    FEATURES
].copy()

y = train_df[
    LABEL_COLUMN
].copy()


# bool -> int
# LightGBM 입력을 명확하게 맞추기 위함
X["traffic_is_proxy"] = (
    X["traffic_is_proxy"]
    .astype(int)
)


print(
    "Feature 수:",
    len(FEATURES)
)

print(
    "Feature 목록:"
)

for feature in FEATURES:
    print(
        "-",
        feature,
    )


# ============================================================
# 6. Leakage 검사
# ============================================================

print(
    "\n[4] Leakage 검사..."
)

forbidden_columns = [
    "link_id",
    "accident_distance_m",
    "label",
]


for column in forbidden_columns:

    if column in X.columns:

        raise RuntimeError(
            f"학습 Feature에 사용 금지 컬럼 발견: {column}"
        )


print(
    "금지 컬럼 없음"
)


# ============================================================
# 7. 결측 / 무한대 검사
# ============================================================

print(
    "\n[5] Feature 상태 검사..."
)

missing_count = (
    X
    .isna()
    .sum()
    .sum()
)

print(
    "전체 결측값:",
    int(missing_count)
)


if missing_count > 0:

    raise RuntimeError(
        "Feature에 결측값이 존재합니다."
    )


# ============================================================
# 8. Train / Test 분할
#
# 22번 Logistic Regression과 동일 조건
# ============================================================

print(
    "\n[6] Train / Test 분할..."
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
# 9. 클래스 불균형 계산
#
# scale_pos_weight =
# negative / positive
# ============================================================

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
    "\n[7] 클래스 불균형..."
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
# 10. LightGBM 모델
#
# 처음부터 과도한 튜닝을 하지 않고
# 비교적 보수적인 기본 설정으로 시작
# ============================================================

print(
    "\n[8] LightGBM 모델 생성..."
)


model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=500,

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
    model
)


# ============================================================
# 11. 학습
#
# validation set을 사용하고
# early stopping 적용
# ============================================================

print(
    "\n[9] LightGBM 학습 시작..."
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (
            X_test,
            y_test,
        )
    ],

    eval_metric="auc",

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50,
            verbose=True,
        )
    ],
)


print(
    "\n학습 완료"
)

print(
    "Best iteration:",
    model.best_iteration_
)


# ============================================================
# 12. 예측
# ============================================================

print(
    "\n[10] Test 예측..."
)


pred = model.predict(
    X_test
)

prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 13. 평가
# ============================================================

print(
    "\n[11] 성능 평가..."
)


accuracy = accuracy_score(
    y_test,
    pred,
)

precision = precision_score(
    y_test,
    pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    prob,
)

pr_auc = average_precision_score(
    y_test,
    prob,
)


cm = confusion_matrix(
    y_test,
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
    "LightGBM"
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


# ============================================================
# 14. Feature Importance
#
# split importance는 참고용.
# 본격적인 설명은 이후 SHAP에서 진행
# ============================================================

print(
    "\n[12] Feature Importance..."
)


importance_df = pd.DataFrame(
    {
        "feature": FEATURES,

        "importance": (
            model.feature_importances_
        ),
    }
)


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 15. 모델 저장
# ============================================================

print(
    "\n[13] 모델 저장..."
)


model_path = (
    MODEL_DIR
    / "lightgbm_model.joblib"
)


joblib.dump(
    model,
    model_path,
)


print(
    "저장:",
    model_path
)


# LightGBM native model도 저장
native_model_path = (
    MODEL_DIR
    / "lightgbm_model.txt"
)


model.booster_.save_model(
    str(
        native_model_path
    )
)


print(
    "Native 모델:",
    native_model_path
)


# ============================================================
# 16. 평가 결과 저장
# ============================================================

print(
    "\n[14] 평가 결과 저장..."
)


metrics_df = pd.DataFrame(
    [
        {
            "model": "LightGBM",
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
            "best_iteration": (
                model.best_iteration_
            ),
            "scale_pos_weight": (
                scale_pos_weight
            ),
        }
    ]
)


metrics_path = (
    OUTPUT_DIR
    / "lightgbm_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig",
)


print(
    "평가 결과:",
    metrics_path
)


# ============================================================
# 17. Feature Importance 저장
# ============================================================

importance_path = (
    OUTPUT_DIR
    / "lightgbm_feature_importance.csv"
)


importance_df.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig",
)


print(
    "Feature Importance:",
    importance_path
)


# ============================================================
# 18. Test 예측 결과 저장
# ============================================================

print(
    "\n[15] Test 예측 결과 저장..."
)


test_index = X_test.index


prediction_df = pd.DataFrame(
    {
        "link_id": train_df.loc[
            test_index,
            "link_id",
        ].values,

        "label": y_test.values,

        "lightgbm_prob": prob,

        "lightgbm_pred": pred,
    }
)


prediction_path = (
    OUTPUT_DIR
    / "lightgbm_test_predictions.parquet"
)


prediction_df.to_parquet(
    prediction_path,
    index=False,
)


print(
    "예측 결과:",
    prediction_path
)

print(
    "예측 행:",
    len(prediction_df)
)


# ============================================================
# 19. Logistic Regression과 비교
# ============================================================

print(
    "\n[16] Logistic Regression Baseline과 비교..."
)


logistic_metrics_path = (
    OUTPUT_DIR
    / "logistic_baseline_metrics.csv"
)


if logistic_metrics_path.exists():

    logistic_df = pd.read_csv(
        logistic_metrics_path
    )


    compare_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]


    lightgbm_compare = metrics_df[
        compare_columns
    ]


    comparison_df = pd.concat(
        [
            logistic_df[
                compare_columns
            ],
            lightgbm_compare,
        ],
        ignore_index=True,
    )


    print(
        comparison_df.to_string(
            index=False
        )
    )


    comparison_path = (
        OUTPUT_DIR
        / "model_comparison.csv"
    )


    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\n비교 결과:",
        comparison_path
    )


else:

    print(
        "Logistic Regression 결과 파일을 찾을 수 없습니다."
    )


# ============================================================
# 20. 최종 결과
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute LightGBM 학습 완료 ==="
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
    "Best iteration:",
    model.best_iteration_
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
    "\n저장 완료:"
)

print(
    "- models/lightgbm_model.joblib"
)

print(
    "- models/lightgbm_model.txt"
)

print(
    "- outputs/lightgbm_metrics.csv"
)

print(
    "- outputs/lightgbm_feature_importance.csv"
)

print(
    "- outputs/lightgbm_test_predictions.parquet"
)

print(
    "- outputs/model_comparison.csv"
)

print(
    "\n다음 단계:"
)

print(
    "LightGBM 결과 검토 및 Baseline 비교"
)

print(
    "=== 완료 ==="
)