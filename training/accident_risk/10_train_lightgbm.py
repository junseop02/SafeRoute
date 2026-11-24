from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = DATA_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 입력 피처 설정
FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch"
]

TARGET = "target"

RANDOM_STATE = 42

# 1. 최종 데이터셋 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_final_dataset.parquet"
)

train = dataset[
    dataset["split"] == "train"
].copy()

val = dataset[
    dataset["split"] == "validation"
].copy()

print("===== Dataset Information =====")
print("Train:", len(train))
print("Validation:", len(val))
print("Input Features:", FEATURES)

# 2. 입력 피처 및 정답 분리
X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int)

X_val = val[FEATURES].copy()
y_val = val[TARGET].astype(int)

# 3. 도로 등급 범주형 처리
# 학습 데이터의 범주를 기준으로 통일
road_rank_categories = sorted(
    X_train["road_rank"].unique()
)

road_rank_dtype = pd.CategoricalDtype(
    categories=road_rank_categories
)

X_train["road_rank"] = (
    X_train["road_rank"].astype(road_rank_dtype)
)

X_val["road_rank"] = (
    X_val["road_rank"].astype(road_rank_dtype)
)

print("\n===== Categorical Feature =====")
print("Feature: road_rank")
print("Categories:", road_rank_categories)

print(
    "Unknown Validation Categories:",
    X_val["road_rank"].isna().sum()
)

# 4. 클래스 가중치 계산
negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

scale_pos_weight = negative / positive

print("\n===== Class Distribution =====")
print("Train Negative:", negative)
print("Train Positive:", positive)
print("Scale Pos Weight:", scale_pos_weight)

print(
    "Validation Positive Ratio:",
    round(float(y_val.mean()), 6)
)

# 5. 공통 하이퍼파라미터
BASE_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 6,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "n_jobs": -1
}

# 비교 모델 설정
MODEL_CONFIGS = {
    "model_a": {
        "scale_pos_weight": 1.0
    },
    "model_b": {
        "scale_pos_weight": scale_pos_weight
    }
}

# 6. 분류 임계값 선정 함수
# Validation F1-score 기준
def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])


# 7. 모델 평가 함수
def evaluate_model(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# 8. 모델 학습 및 비교
results = []
trained_models = {}

for model_name, config in MODEL_CONFIGS.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("Scale Pos Weight:", config["scale_pos_weight"])
    print("=" * 60)

    params = {
        **BASE_PARAMS,
        **config
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        eval_metric="average_precision",
        categorical_feature=["road_rank"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True
            ),
            lgb.log_evaluation(period=50)
        ]
    )

    # Validation 분류 점수
    val_probabilities = model.predict_proba(
        X_val,
        num_iteration=model.best_iteration_
    )[:, 1]

    # 기본 임계값 0.5 평가
    default_metrics = evaluate_model(
        y_val,
        val_probabilities,
        threshold=0.5
    )

    # Validation 기준 최적 임계값 선정
    best_threshold = find_best_threshold(
        y_val,
        val_probabilities
    )

    optimized_metrics = evaluate_model(
        y_val,
        val_probabilities,
        threshold=best_threshold
    )

    print("\n===== Model Information =====")
    print("Model:", model_name)
    print("Best Iteration:", model.best_iteration_)

    print("\n===== Default Threshold =====")

    for key, value in default_metrics.items():
        print(f"{key}: {value:.6f}")

    print("\n===== Optimized Threshold =====")

    for key, value in optimized_metrics.items():
        print(f"{key}: {value:.6f}")

    # 모델별 평가 결과 저장
    results.append({
        "model": model_name,
        "threshold_type": "default",
        "best_iteration": model.best_iteration_,
        "scale_pos_weight": config["scale_pos_weight"],
        **default_metrics
    })

    results.append({
        "model": model_name,
        "threshold_type": "optimized",
        "best_iteration": model.best_iteration_,
        "scale_pos_weight": config["scale_pos_weight"],
        **optimized_metrics
    })

    # 피처 중요도 계산
    importance = pd.DataFrame({
        "feature": FEATURES,
        "gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
        "split": model.booster_.feature_importance(
            importance_type="split"
        )
    })

    total_gain = importance["gain"].sum()

    importance["gain_ratio_pct"] = (
        importance["gain"] / total_gain * 100
        if total_gain > 0
        else 0.0
    )

    importance = importance.sort_values(
        "gain",
        ascending=False
    )

    print("\n===== Feature Importance =====")
    print(importance.to_string(index=False))

    importance.to_csv(
        RESULT_DIR / f"{model_name}_feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 검증 데이터 예측 결과 저장
    val_predictions = val[
        ["link_id", "spatial_group", TARGET]
    ].copy()

    val_predictions["score"] = val_probabilities

    val_predictions["prediction_default"] = (
        val_probabilities >= 0.5
    ).astype(int)

    val_predictions["prediction_optimized"] = (
        val_probabilities >= best_threshold
    ).astype(int)

    val_predictions.to_parquet(
        RESULT_DIR / f"{model_name}_validation_predictions.parquet",
        index=False
    )

    # 모델 및 추론에 필요한 설정 보존
    model_bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "road_rank_categories": road_rank_categories,
        "threshold": best_threshold,
        "best_iteration": model.best_iteration_,
        "scale_pos_weight": config["scale_pos_weight"],
        "validation_metrics": optimized_metrics,
        "buffer_distance_m": 400,
        "random_state": RANDOM_STATE
    }

    trained_models[model_name] = model_bundle

    # 기존 서비스 모델과 분리하여 저장
    model_path = (
        MODEL_DIR
        / f"accident_risk_lightgbm_{model_name}.joblib"
    )

    joblib.dump(
        model_bundle,
        model_path
    )

    print("\nSaved Model:", model_path)


# 9. 모델 비교 결과 정리
results_df = pd.DataFrame(results)

comparison_columns = [
    "model",
    "threshold_type",
    "best_iteration",
    "threshold",
    "average_precision",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1"
]

print("\n===== Validation Model Comparison =====")

print(
    results_df[comparison_columns]
    .round(6)
    .to_string(index=False)
)

results_path = (
    RESULT_DIR / "lightgbm_validation_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)

# 10. Validation AP 기준 우선 모델 선정
optimized_results = results_df[
    results_df["threshold_type"] == "optimized"
].copy()

best_result = optimized_results.sort_values(
    ["average_precision", "roc_auc", "f1"],
    ascending=False
).iloc[0]

best_model_name = best_result["model"]

print("\n===== Recommended Candidate =====")
print("Model:", best_model_name)

print(
    "Average Precision:",
    round(best_result["average_precision"], 6)
)

print(
    "ROC-AUC:",
    round(best_result["roc_auc"], 6)
)

print(
    "F1:",
    round(best_result["f1"], 6)
)

print(
    "Threshold:",
    round(best_result["threshold"], 6)
)

print("\n===== Output =====")
print("Results:", results_path)

print("\n===== Training Completed =====")
print("Test evaluation has not been performed.")