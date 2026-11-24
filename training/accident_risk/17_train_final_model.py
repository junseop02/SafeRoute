from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay
)


# 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"
RESULT_DIR = DATA_DIR / "results"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = RESULT_DIR / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# 실험 설정
RANDOM_STATE = 42
N_SPLITS = 3
BUFFER_DISTANCE = 400

TARGET = "target"

FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_intersect_200m",
    "n_branch"
]


# 1. 최적화 결과 불러오기
config_path = (
    RESULT_DIR / "lightgbm_best_candidate.json"
)

with open(config_path, "r", encoding="utf-8") as file:

    config = json.load(file)


# 선정 모델 및 실험 조건 검사
if config["candidate"] != "C02":
    raise ValueError("Unexpected candidate.")

if config["features"] != FEATURES:
    raise ValueError("Feature mismatch.")

if config["target"] != TARGET:
    raise ValueError("Target mismatch.")

if config["buffer_distance_m"] != BUFFER_DISTANCE:
    raise ValueError("Buffer distance mismatch.")

if config["random_state"] != RANDOM_STATE:
    raise ValueError("Random state mismatch.")

if config["n_splits"] != N_SPLITS:
    raise ValueError("Fold count mismatch.")


# 최종 모델 설정
MODEL_PARAMS = {
    **config["base_params"],
    **config["tuned_params"]
}

# 교차검증 최적 반복 횟수의 중앙값
MODEL_PARAMS["n_estimators"] = int(
    config["best_iteration_median"]
)

if MODEL_PARAMS["n_estimators"] != 30:
    raise ValueError("Unexpected iteration count.")


print("===== Final Model Configuration =====")

print("Candidate:", config["candidate"])

print(
    json.dumps(
        MODEL_PARAMS,
        indent=4
    )
)

print("Features:", FEATURES)


# 2. 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

development = dataset[
    dataset["split"] != "test"
].copy()

test = dataset[
    dataset["split"] == "test"
].copy().reset_index(drop=True)


# 좌표 추출 함수
def get_coordinates(frame):

    return frame[
        ["center_x", "center_y"]
    ].to_numpy()


# 3. Test 주변 개발 데이터 제외
test_tree = cKDTree(
    get_coordinates(test)
)

distances, _ = test_tree.query(
    get_coordinates(development),
    k=1
)

original_development_size = len(development)

development = development.loc[
    distances > BUFFER_DISTANCE
].reset_index(drop=True)


# 데이터 중복 검사
if (
    set(development["link_id"])
    & set(test["link_id"])
):
    raise ValueError("Link leakage.")

if (
    set(development["spatial_group"])
    & set(test["spatial_group"])
):
    raise ValueError("Spatial group leakage.")

if development["link_id"].duplicated().any():
    raise ValueError("Duplicate development links.")

if test["link_id"].duplicated().any():
    raise ValueError("Duplicate test links.")


# 최소 공간 거리 검사
distances, _ = test_tree.query(
    get_coordinates(development),
    k=1
)

minimum_distance = float(
    distances.min()
)

if minimum_distance <= BUFFER_DISTANCE:
    raise ValueError("Spatial buffer violation.")


# 개발 데이터 피처 검사
if not np.isfinite(
    development[FEATURES].to_numpy(dtype=float)
).all():
    raise ValueError("Invalid development features.")

if set(development[TARGET].unique()) != {0, 1}:
    raise ValueError("Invalid development labels.")

if len(development) != 25293:
    raise ValueError("Unexpected development size.")

if len(test) != 5942:
    raise ValueError("Unexpected test size.")


print("\n===== Final Dataset =====")

print(
    "Development Before Buffer:",
    original_development_size
)

print(
    "Excluded Links:",
    original_development_size - len(development)
)

print("Final Development:", len(development))
print("Reserved Test:", len(test))

print(
    "Development Positive:",
    int(development[TARGET].sum())
)

print(
    "Development Negative:",
    int((development[TARGET] == 0).sum())
)

print(
    "Development Spatial Groups:",
    development["spatial_group"].nunique()
)

print(
    "Minimum Development-Test Distance:",
    minimum_distance
)


# 4. 공간 교차검증 구성
splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

folds = list(
    splitter.split(
        development[FEATURES],
        development[TARGET],
        groups=development["spatial_group"]
    )
)


# 도로 등급 범주형 처리 함수
def prepare_features(train_frame, val_frame):

    X_train = train_frame[FEATURES].copy()
    X_val = val_frame[FEATURES].copy()

    categories = sorted(
        X_train["road_rank"].unique()
    )

    rank_dtype = pd.CategoricalDtype(
        categories=categories
    )

    X_train["road_rank"] = (
        X_train["road_rank"].astype(rank_dtype)
    )

    X_val["road_rank"] = (
        X_val["road_rank"].astype(rank_dtype)
    )

    return X_train, X_val, categories


# 5. OOF 예측값 생성
# 모든 Fold에서 학습 반복 횟수를 30으로 고정
oof_scores = np.full(
    len(development),
    np.nan
)

oof_fold = np.zeros(
    len(development),
    dtype=int
)

fold_results = []


print("\n===== OOF Prediction =====")

for fold_number, (train_idx, val_idx) in enumerate(
    folds,
    start=1
):

    fold_train = development.iloc[
        train_idx
    ].copy()

    fold_val = development.iloc[
        val_idx
    ].copy()

    # 공간 그룹 중복 검사
    if (
        set(fold_train["spatial_group"])
        & set(fold_val["spatial_group"])
    ):
        raise ValueError("Fold group leakage.")

    # Validation 주변 학습 데이터 제외
    val_tree = cKDTree(
        get_coordinates(fold_val)
    )

    distances, _ = val_tree.query(
        get_coordinates(fold_train),
        k=1
    )

    fold_train = fold_train.loc[
        distances > BUFFER_DISTANCE
    ].copy()

    # 완충구역 검사
    distances, _ = val_tree.query(
        get_coordinates(fold_train),
        k=1
    )

    fold_minimum_distance = float(
        distances.min()
    )

    if fold_minimum_distance <= BUFFER_DISTANCE:
        raise ValueError("Fold buffer violation.")

    # 입력 피처 준비
    X_train, X_val, _ = prepare_features(
        fold_train,
        fold_val
    )

    y_train = fold_train[TARGET].astype(int)
    y_val = fold_val[TARGET].astype(int)

    if (
        set(y_train.unique()) != {0, 1}
        or set(y_val.unique()) != {0, 1}
    ):
        raise ValueError("Both classes are required.")

    # 모델 생성 및 학습
    model = lgb.LGBMClassifier(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=["road_rank"]
    )

    # Validation 예측
    scores = model.predict_proba(
        X_val
    )[:, 1]

    oof_scores[val_idx] = scores
    oof_fold[val_idx] = fold_number

    ap = float(
        average_precision_score(
            y_val,
            scores
        )
    )

    auc = float(
        roc_auc_score(
            y_val,
            scores
        )
    )

    fold_results.append({
        "fold": fold_number,
        "train_size": len(fold_train),
        "validation_size": len(fold_val),
        "n_estimators": MODEL_PARAMS["n_estimators"],
        "baseline_ap": float(y_val.mean()),
        "average_precision": ap,
        "roc_auc": auc,
        "minimum_distance": fold_minimum_distance
    })

    print(
        f"Fold {fold_number} | "
        f"Train: {len(fold_train)} | "
        f"Validation: {len(fold_val)} | "
        f"AP: {ap:.6f} | "
        f"ROC-AUC: {auc:.6f}"
    )


# OOF 예측 누락 검사
if not np.isfinite(oof_scores).all():
    raise ValueError("Missing OOF predictions.")

if (oof_fold == 0).any():
    raise ValueError("Missing OOF fold assignments.")


# 6. 분류 임계값 결정 함수
def find_best_threshold(y_true, scores):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = int(
        np.argmax(f1_values)
    )

    return float(
        thresholds[best_index]
    )


# 평가 함수
def evaluate_model(
    model_name,
    y_true,
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    ap = float(
        average_precision_score(
            y_true,
            scores
        )
    )

    baseline_ap = float(
        np.mean(y_true)
    )

    return {
        "model": model_name,
        "threshold": float(threshold),
        "baseline_ap": baseline_ap,
        "average_precision": ap,
        "ap_lift": ap / baseline_ap,
        "roc_auc": float(
            roc_auc_score(y_true, scores)
        ),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
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


# 개발 데이터 정답
y_development = development[
    TARGET
].astype(int).to_numpy()


# LightGBM 임계값
final_threshold = find_best_threshold(
    y_development,
    oof_scores
)


# 교차로 수 기준 모델 임계값
baseline_development_scores = development[
    "n_intersect_200m"
].to_numpy(dtype=float)

baseline_threshold = find_best_threshold(
    y_development,
    baseline_development_scores
)


# OOF 평가
oof_metrics = evaluate_model(
    "LightGBM_C02_OOF",
    y_development,
    oof_scores,
    final_threshold
)

baseline_oof_metrics = evaluate_model(
    "Intersection_Baseline_Development",
    y_development,
    baseline_development_scores,
    baseline_threshold
)


print("\n===== Threshold Selection =====")

print(
    "LightGBM Threshold:",
    final_threshold
)

print(
    "Intersection Baseline Threshold:",
    baseline_threshold
)

print("\n===== OOF Metrics =====")

print(
    pd.DataFrame([
        oof_metrics,
        baseline_oof_metrics
    ]).round(6).to_string(index=False)
)


# OOF 결과 저장
oof_output = development[
    ["link_id", "spatial_group", TARGET]
].copy()

oof_output["fold"] = oof_fold
oof_output["score"] = oof_scores

oof_output["prediction"] = (
    oof_scores >= final_threshold
).astype(int)

oof_output.to_parquet(
    RESULT_DIR / "final_oof_predictions.parquet",
    index=False
)

pd.DataFrame(fold_results).to_csv(
    RESULT_DIR / "final_oof_fold_results.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([
    oof_metrics,
    baseline_oof_metrics
]).to_csv(
    RESULT_DIR / "final_oof_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)


# 7. 최종 모델 전체 개발 데이터 학습
X_development = development[
    FEATURES
].copy()

road_rank_categories = sorted(
    X_development["road_rank"].unique()
)

rank_dtype = pd.CategoricalDtype(
    categories=road_rank_categories
)

X_development["road_rank"] = (
    X_development["road_rank"].astype(rank_dtype)
)


print("\n===== Final Model Training =====")

print("Training Links:", len(X_development))

print(
    "Number of Estimators:",
    MODEL_PARAMS["n_estimators"]
)


final_model = lgb.LGBMClassifier(
    **MODEL_PARAMS
)

final_model.fit(
    X_development,
    y_development,
    categorical_feature=["road_rank"]
)

print("Final Model Training Completed.")


# 8. Test 평가 전 모델 설정 고정
frozen_config = {
    "candidate": "C02",
    "target": TARGET,
    "features": FEATURES,
    "model_params": MODEL_PARAMS,
    "threshold": final_threshold,
    "baseline_threshold": baseline_threshold,
    "threshold_method": "development_oof_max_f1",
    "buffer_distance_m": BUFFER_DISTANCE,
    "training_size": len(development),
    "road_rank_categories": [
        int(value)
        for value in road_rank_categories
    ],
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS
}

with open(
    RESULT_DIR / "final_frozen_config.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        frozen_config,
        file,
        indent=4,
        ensure_ascii=False
    )


# 9. 독립 Test 데이터 평가
print("\n===== Reserved Test Evaluation =====")

if not np.isfinite(
    test[FEATURES].to_numpy(dtype=float)
).all():
    raise ValueError("Invalid test features.")

X_test = test[FEATURES].copy()

y_test = test[TARGET].astype(int).to_numpy()

if set(np.unique(y_test)) != {0, 1}:
    raise ValueError("Invalid test labels.")


# 학습 데이터와 동일한 범주형 설정
X_test["road_rank"] = (
    X_test["road_rank"].astype(rank_dtype)
)

unknown_categories = int(
    X_test["road_rank"].isna().sum()
)

print("Test Links:", len(test))

print(
    "Test Positive:",
    int(y_test.sum())
)

print(
    "Test Negative:",
    int((y_test == 0).sum())
)

print(
    "Test Positive Groups:",
    test.loc[
        test[TARGET] == 1,
        "spatial_group"
    ].nunique()
)

print(
    "Unknown Road Rank Categories:",
    unknown_categories
)


# LightGBM 예측
test_scores = final_model.predict_proba(
    X_test
)[:, 1]


# 단순 교차로 수 기준 예측
baseline_test_scores = test[
    "n_intersect_200m"
].to_numpy(dtype=float)


# 10. 최종 성능 비교
test_metrics = evaluate_model(
    "LightGBM_C02",
    y_test,
    test_scores,
    final_threshold
)

baseline_test_metrics = evaluate_model(
    "Intersection_Baseline",
    y_test,
    baseline_test_scores,
    baseline_threshold
)

default_threshold_metrics = evaluate_model(
    "LightGBM_C02_Threshold_0.5",
    y_test,
    test_scores,
    0.5
)

test_results = pd.DataFrame([
    test_metrics,
    baseline_test_metrics,
    default_threshold_metrics
])


print("\n===== Final Test Results =====")

print(
    test_results.round(6).to_string(
        index=False
    )
)


# AP 상대 개선율
ap_improvement = (
    test_metrics["average_precision"]
    - baseline_test_metrics["average_precision"]
)

relative_improvement = (
    ap_improvement
    / baseline_test_metrics["average_precision"]
    * 100
)

print("\n===== Baseline Comparison =====")

print(
    "Absolute AP Improvement:",
    round(ap_improvement, 6)
)

print(
    "Relative AP Improvement (%):",
    round(relative_improvement, 4)
)


# 11. Test 예측 결과 저장
test_output = test[
    [
        "link_id",
        "region_code",
        "spatial_group",
        TARGET
    ]
].copy()

test_output["lightgbm_score"] = test_scores

test_output["lightgbm_prediction"] = (
    test_scores >= final_threshold
).astype(int)

test_output["baseline_score"] = (
    baseline_test_scores
)

test_output["baseline_prediction"] = (
    baseline_test_scores >= baseline_threshold
).astype(int)

test_output.to_parquet(
    RESULT_DIR / "final_test_predictions.parquet",
    index=False
)

test_results.to_csv(
    RESULT_DIR / "final_test_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)


# 12. 피처 중요도 분석
importance = pd.DataFrame({
    "feature": FEATURES,
    "gain": final_model.booster_.feature_importance(
        importance_type="gain"
    ),
    "split": final_model.booster_.feature_importance(
        importance_type="split"
    )
})

total_gain = importance["gain"].sum()

importance["gain_ratio"] = (
    importance["gain"] / total_gain * 100
    if total_gain > 0
    else 0.0
)

importance = importance.sort_values(
    "gain",
    ascending=False
).reset_index(drop=True)


print("\n===== Feature Importance =====")

print(
    importance.round(6).to_string(
        index=False
    )
)

importance.to_csv(
    RESULT_DIR / "final_feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)


# 13. 시각화 설정
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# Confusion Matrix
test_predictions = (
    test_scores >= final_threshold
).astype(int)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Matched",
        "Hotspot Matched"
    ]
)

display.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
    values_format="d"
)

ax.set_title(
    "LightGBM C02 - Test Confusion Matrix"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "final_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# 14. Precision-Recall Curve
fig, ax = plt.subplots(
    figsize=(8, 6)
)

for name, scores in [
    ("LightGBM C02", test_scores),
    ("Intersection Baseline", baseline_test_scores)
]:

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            scores
        )
    )

    ap = average_precision_score(
        y_test,
        scores
    )

    ax.step(
        recall,
        precision,
        where="post",
        label=f"{name} (AP={ap:.4f})"
    )

ax.axhline(
    y=y_test.mean(),
    linestyle="--",
    color="gray",
    label=f"Random Baseline ({y_test.mean():.4f})"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Test Precision-Recall Curve")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "final_pr_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# 15. ROC Curve
fig, ax = plt.subplots(
    figsize=(8, 6)
)

for name, scores in [
    ("LightGBM C02", test_scores),
    ("Intersection Baseline", baseline_test_scores)
]:

    fpr, tpr, _ = roc_curve(
        y_test,
        scores
    )

    auc = roc_auc_score(
        y_test,
        scores
    )

    ax.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.4f})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Baseline"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Test ROC Curve")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "final_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# 16. 피처 중요도 그래프
plot_importance = importance.sort_values(
    "gain_ratio",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    plot_importance["feature"],
    plot_importance["gain_ratio"]
)

ax.set_xlabel("Gain Importance (%)")
ax.set_title("LightGBM C02 - Feature Importance")
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "final_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# 17. 최종 모델 저장
model_path = (
    MODEL_DIR / "accident_risk_lightgbm_final.joblib"
)

model_bundle = {
    "model": final_model,
    "candidate": "C02",
    "features": FEATURES,
    "target": TARGET,
    "threshold": final_threshold,
    "road_rank_categories": [
        int(value)
        for value in road_rank_categories
    ],
    "model_params": MODEL_PARAMS,
    "n_estimators": MODEL_PARAMS["n_estimators"],
    "buffer_distance_m": BUFFER_DISTANCE,
    "random_state": RANDOM_STATE,
    "training_size": len(development),
    "oof_metrics": oof_metrics,
    "test_metrics": test_metrics,
    "baseline_test_metrics": baseline_test_metrics,
    "deployment_status": "candidate_not_deployed",
    "score_meaning": "historical_hotspot_membership_score",
    "limitations": [
        "Not a calibrated future accident probability.",
        "Not a direct novice driving difficulty score.",
        "Historical hotspot labels cover limited accident types.",
        "Reserved test contains limited positive spatial groups."
    ]
}

joblib.dump(
    model_bundle,
    model_path
)


# 18. 저장 모델 재로딩 검증
loaded_bundle = joblib.load(
    model_path
)

loaded_model = loaded_bundle["model"]

reloaded_scores = loaded_model.predict_proba(
    X_test
)[:, 1]

maximum_difference = float(
    np.max(
        np.abs(
            test_scores - reloaded_scores
        )
    )
)

if maximum_difference > 1e-12:
    raise ValueError("Saved model prediction mismatch.")


print("\n===== Model Save Verification =====")

print("Saved Model:", model_path)

print(
    "Model Size (Bytes):",
    model_path.stat().st_size
)

print(
    "Reloaded Prediction Maximum Difference:",
    maximum_difference
)


# 19. 최종 결과 출력
print("\n===== Output Files =====")

print("Results Directory:", RESULT_DIR)
print("Figures Directory:", FIGURE_DIR)
print("Model:", model_path)

print("\n===== Final Training Completed =====")

print("Final Candidate: C02")
print("Reserved Test Evaluation: Completed")
print("Model Deployment: Not Performed")
print("Existing Service Model: Not Modified")