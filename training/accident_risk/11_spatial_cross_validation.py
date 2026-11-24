from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.spatial import cKDTree

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)

# 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"
RESULT_DIR = DATA_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 실험 설정
RANDOM_STATE = 42
N_SPLITS = 3
BUFFER_DISTANCE = 400

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

# 1. 공간 분할 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

development = dataset[
    dataset["split"] != "test"
].copy()

test = dataset[
    dataset["split"] == "test"
].copy()

print("===== Input Dataset =====")
print("Development Links:", len(development))
print("Reserved Test Links:", len(test))


# 2. 도로 중심점 추출 함수
def get_coordinates(frame):

    return frame[
        ["center_x", "center_y"]
    ].to_numpy()


# 3. 기존 Test 주변 개발 데이터 제외
test_tree = cKDTree(
    get_coordinates(test)
)

distances, _ = test_tree.query(
    get_coordinates(development),
    k=1
)

development = development.loc[
    distances > BUFFER_DISTANCE
].reset_index(drop=True)

# 기존 Test 그룹 중복 검사
if (
    set(development["spatial_group"])
    & set(test["spatial_group"])
):

    raise ValueError("Test group leakage.")

print("\n===== Development Dataset =====")
print("Links:", len(development))

print(
    "Positive:",
    int(development[TARGET].sum())
)

print(
    "Negative:",
    int((development[TARGET] == 0).sum())
)

print(
    "Spatial Groups:",
    development["spatial_group"].nunique()
)

print(
    "Positive Groups:",
    development.loc[
        development[TARGET] == 1,
        "spatial_group"
    ].nunique()
)


# 4. 교차검증 분할 설정
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


# 5. 공통 모델 설정
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


# 6. 최적 임계값 계산 함수
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

    best_index = int(np.argmax(f1_values))

    return float(thresholds[best_index])


# 7. 교차검증 실행
results = []

for fold_number, (train_idx, val_idx) in enumerate(
    folds,
    start=1
):

    print("\n" + "=" * 60)
    print("FOLD:", fold_number)
    print("=" * 60)

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

        raise ValueError("Spatial group leakage.")

    # Fold별 공간 완충구역 적용
    val_tree = cKDTree(
        get_coordinates(fold_val)
    )

    distances, _ = val_tree.query(
        get_coordinates(fold_train),
        k=1
    )

    original_train_size = len(fold_train)

    fold_train = fold_train.loc[
        distances > BUFFER_DISTANCE
    ].copy()

    # 양성·음성 클래스 검사
    for name, frame in [
        ("train", fold_train),
        ("validation", fold_val)
    ]:

        if set(frame[TARGET].unique()) != {0, 1}:

            raise ValueError(
                f"Fold {fold_number}: "
                f"Both classes required in {name}."
            )

    # 최종 공간 거리 검사
    distances, _ = val_tree.query(
        get_coordinates(fold_train),
        k=1
    )

    minimum_distance = float(distances.min())

    if minimum_distance <= BUFFER_DISTANCE:

        raise ValueError("Spatial buffer violation.")

    # 입력 및 정답 분리
    X_train = fold_train[FEATURES].copy()
    y_train = fold_train[TARGET].astype(int)

    X_val = fold_val[FEATURES].copy()
    y_val = fold_val[TARGET].astype(int)

    # 도로 등급 범주형 처리
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

    # Fold별 클래스 가중치 계산
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())

    scale_pos_weight = negative / positive

    train_positive_groups = fold_train.loc[
        fold_train[TARGET] == 1,
        "spatial_group"
    ].nunique()

    val_positive_groups = fold_val.loc[
        fold_val[TARGET] == 1,
        "spatial_group"
    ].nunique()

    # Fold별 AP 기준값
    baseline_ap = float(y_val.mean())

    print("\n===== Fold Dataset =====")
    print("Train Before Buffer:", original_train_size)
    print("Train After Buffer:", len(fold_train))
    print("Validation:", len(fold_val))

    print("Train Positive:", positive)
    print("Validation Positive:", int(y_val.sum()))

    print("Train Positive Groups:", train_positive_groups)
    print("Validation Positive Groups:", val_positive_groups)

    print("Minimum Distance:", minimum_distance)
    print("Baseline AP:", baseline_ap)
    print("Scale Pos Weight:", scale_pos_weight)

    print(
        "Unknown Validation Categories:",
        int(X_val["road_rank"].isna().sum())
    )

    # Model A와 Model B 비교
    model_configs = {
        "model_a": 1.0,
        "model_b": scale_pos_weight
    }

    for model_name, weight in model_configs.items():

        print("\n===== Training:", model_name, "=====")

        model = lgb.LGBMClassifier(
            **BASE_PARAMS,
            scale_pos_weight=weight
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            categorical_feature=["road_rank"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=100,
                    first_metric_only=True,
                    verbose=False
                ),
                lgb.log_evaluation(period=0)
            ]
        )

        # 검증 데이터 분류 점수
        scores = model.predict_proba(
            X_val,
            num_iteration=model.best_iteration_
        )[:, 1]

        # Fold별 탐색용 임계값
        threshold = find_best_threshold(
            y_val,
            scores
        )

        predictions = (
            scores >= threshold
        ).astype(int)

        # 평가 지표 계산
        ap = average_precision_score(
            y_val,
            scores
        )

        auc = roc_auc_score(
            y_val,
            scores
        )

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        result = {
            "fold": fold_number,
            "model": model_name,
            "train_size": len(fold_train),
            "validation_size": len(fold_val),
            "train_positive": positive,
            "validation_positive": int(y_val.sum()),
            "train_positive_groups": train_positive_groups,
            "validation_positive_groups": val_positive_groups,
            "minimum_distance": minimum_distance,
            "scale_pos_weight": weight,
            "best_iteration": model.best_iteration_,
            "baseline_ap": baseline_ap,
            "average_precision": ap,
            "ap_lift": ap / baseline_ap,
            "roc_auc": auc,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        results.append(result)

        print("Best Iteration:", model.best_iteration_)
        print("Average Precision:", round(ap, 6))
        print("AP Lift:", round(ap / baseline_ap, 6))
        print("ROC-AUC:", round(auc, 6))
        print("Threshold:", round(threshold, 6))
        print("Precision:", round(precision, 6))
        print("Recall:", round(recall, 6))
        print("F1:", round(f1, 6))


# 8. 교차검증 결과 정리
results_df = pd.DataFrame(results)

comparison_columns = [
    "fold",
    "model",
    "validation_positive_groups",
    "best_iteration",
    "baseline_ap",
    "average_precision",
    "ap_lift",
    "roc_auc",
    "precision",
    "recall",
    "f1"
]

print("\n===== Cross-Validation Results =====")

print(
    results_df[comparison_columns]
    .round(6)
    .to_string(index=False)
)


# 9. 모델별 평균 및 표준편차 계산
summary = results_df.groupby("model").agg(
    ap_mean=("average_precision", "mean"),
    ap_std=("average_precision", "std"),
    ap_min=("average_precision", "min"),
    ap_max=("average_precision", "max"),
    ap_lift_mean=("ap_lift", "mean"),
    auc_mean=("roc_auc", "mean"),
    auc_std=("roc_auc", "std"),
    precision_mean=("precision", "mean"),
    recall_mean=("recall", "mean"),
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
    best_iteration_mean=("best_iteration", "mean")
)

print("\n===== Cross-Validation Summary =====")
print(summary.round(6).to_string())


# 10. 결과 저장
results_path = (
    RESULT_DIR / "spatial_cv_results.csv"
)

summary_path = (
    RESULT_DIR / "spatial_cv_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    summary_path,
    encoding="utf-8-sig"
)

print("\n===== Output =====")
print("Results:", results_path)
print("Summary:", summary_path)

print("\n===== Spatial Cross-Validation Completed =====")
print("Reserved test evaluation has not been performed.")