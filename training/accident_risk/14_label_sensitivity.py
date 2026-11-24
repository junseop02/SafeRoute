from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
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

# E02 입력 피처
FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_intersect_200m",
    "n_branch"
]

# 비교 대상 라벨
TARGETS = [
    "target",
    "target_10",
    "target_25",
    "target_50",
    "target_75",
    "target_90"
]

# 기존 모델 설정 유지
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
    "n_jobs": -1,
    "scale_pos_weight": 1.0
}


# 1. 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

development = dataset[
    dataset["split"] != "test"
].copy()

test = dataset[
    dataset["split"] == "test"
].copy()


# 2. 좌표 추출 함수
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

development = development.loc[
    distances > BUFFER_DISTANCE
].reset_index(drop=True)

if (
    set(development["spatial_group"])
    & set(test["spatial_group"])
):
    raise ValueError("Test group leakage.")

print("===== Development Dataset =====")
print("Links:", len(development))
print("Groups:", development["spatial_group"].nunique())


# 4. 라벨 분포 및 유효성 검사
print("\n===== Label Distribution =====")

for target in TARGETS:

    if set(development[target].unique()) != {0, 1}:
        raise ValueError(f"Invalid label: {target}")

    positive = int(development[target].sum())

    print(
        target,
        "Positive:", positive,
        "Negative:", len(development) - positive,
        "Ratio (%):",
        round(positive / len(development) * 100, 4)
    )

# 중첩 기준이 높아질수록 양성 집합이 감소하는지 검사
for previous, current in zip(TARGETS, TARGETS[1:]):

    if (development[current] > development[previous]).any():
        raise ValueError("Invalid nested labels.")


# 5. 기존 target 기준으로 Fold 고정
splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

folds = list(
    splitter.split(
        development[FEATURES],
        development["target"],
        groups=development["spatial_group"]
    )
)


# 6. 평가 함수
def evaluate_scores(y_true, scores):

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
    threshold = float(thresholds[best_index])

    predictions = (
        scores >= threshold
    ).astype(int)

    baseline_ap = float(y_true.mean())

    ap = average_precision_score(
        y_true,
        scores
    )

    return {
        "baseline_ap": baseline_ap,
        "average_precision": ap,
        "ap_lift": ap / baseline_ap,
        "roc_auc": roc_auc_score(
            y_true,
            scores
        ),
        "threshold": threshold,
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    }


# 7. Fold별 실험
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

    fold_train = fold_train.loc[
        distances > BUFFER_DISTANCE
    ].copy()

    # 공간 거리 재검사
    distances, _ = val_tree.query(
        get_coordinates(fold_train),
        k=1
    )

    minimum_distance = float(distances.min())

    if minimum_distance <= BUFFER_DISTANCE:
        raise ValueError("Spatial buffer violation.")

    print("Train:", len(fold_train))
    print("Validation:", len(fold_val))
    print("Minimum Distance:", minimum_distance)

    # 입력 피처 준비
    X_train = fold_train[FEATURES].copy()
    X_val = fold_val[FEATURES].copy()

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

    # 라벨별 모델 비교
    for target in TARGETS:

        print("\n===== Target:", target, "=====")

        y_train = fold_train[target].astype(int)
        y_val = fold_val[target].astype(int)

        # 양성·음성 클래스 검사
        if (
            set(y_train.unique()) != {0, 1}
            or set(y_val.unique()) != {0, 1}
        ):
            raise ValueError(
                f"Both classes required: "
                f"Fold {fold_number}, {target}"
            )

        train_positive_groups = fold_train.loc[
            fold_train[target] == 1,
            "spatial_group"
        ].nunique()

        val_positive_groups = fold_val.loc[
            fold_val[target] == 1,
            "spatial_group"
        ].nunique()

        common_result = {
            "fold": fold_number,
            "target": target,
            "train_size": len(fold_train),
            "validation_size": len(fold_val),
            "train_positive": int(y_train.sum()),
            "validation_positive": int(y_val.sum()),
            "train_positive_groups": train_positive_groups,
            "validation_positive_groups": val_positive_groups
        }

        print("Train Positive:", int(y_train.sum()))
        print("Validation Positive:", int(y_val.sum()))
        print("Validation Positive Groups:", val_positive_groups)

        # 8. 단일 피처 기준 모델
        baseline_scores = fold_val[
            "n_intersect_200m"
        ].to_numpy(dtype=float)

        baseline_metrics = evaluate_scores(
            y_val,
            baseline_scores
        )

        results.append({
            **common_result,
            "model": "B01_raw_intersection",
            "best_iteration": 0,
            **baseline_metrics
        })

        # 9. E02 LightGBM 학습
        model = lgb.LGBMClassifier(
            **BASE_PARAMS
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

        scores = model.predict_proba(
            X_val,
            num_iteration=model.best_iteration_
        )[:, 1]

        metrics = evaluate_scores(
            y_val,
            scores
        )

        results.append({
            **common_result,
            "model": "E02_without_curvature",
            "best_iteration": model.best_iteration_,
            **metrics
        })

        print(
            "Baseline AP:",
            round(baseline_metrics["average_precision"], 6)
        )

        print(
            "Baseline ROC-AUC:",
            round(baseline_metrics["roc_auc"], 6)
        )

        print("Best Iteration:", model.best_iteration_)

        for key, value in metrics.items():

            print(f"{key}: {value:.6f}")


# 10. 전체 결과 정리
results_df = pd.DataFrame(results)

comparison_columns = [
    "fold",
    "target",
    "model",
    "validation_positive",
    "validation_positive_groups",
    "baseline_ap",
    "average_precision",
    "ap_lift",
    "roc_auc",
    "precision",
    "recall",
    "f1"
]

print("\n===== Label Sensitivity Results =====")

print(
    results_df[comparison_columns]
    .round(6)
    .to_string(index=False)
)


# 11. 라벨별 성능 요약
summary = results_df.groupby(
    ["target", "model"]
).agg(
    baseline_ap_mean=("baseline_ap", "mean"),
    ap_mean=("average_precision", "mean"),
    ap_std=("average_precision", "std"),
    ap_lift_mean=("ap_lift", "mean"),
    auc_mean=("roc_auc", "mean"),
    auc_std=("roc_auc", "std"),
    auc_min=("roc_auc", "min"),
    precision_mean=("precision", "mean"),
    recall_mean=("recall", "mean"),
    f1_mean=("f1", "mean"),
    best_iteration_mean=("best_iteration", "mean")
).reset_index()

summary["target"] = pd.Categorical(
    summary["target"],
    categories=TARGETS,
    ordered=True
)

summary = summary.sort_values(
    ["target", "model"]
)

print("\n===== Label Sensitivity Summary =====")
print(summary.round(6).to_string(index=False))


# 12. LightGBM과 기준 모델 차이
comparison = summary.pivot(
    index="target",
    columns="model",
    values=[
        "ap_mean",
        "auc_mean",
        "f1_mean"
    ]
)

print("\n===== LightGBM vs Baseline =====")
print(comparison.round(6).to_string())


# 13. 기존 E02 재현 여부 확인
reference_path = RESULT_DIR / "ablation_results.csv"

if reference_path.exists():

    reference = pd.read_csv(reference_path)

    reference = reference[
        reference["experiment"] == "E02_without_curvature"
    ]

    current = results_df[
        (results_df["target"] == "target")
        & (results_df["model"] == "E02_without_curvature")
    ]

    check = reference.merge(
        current,
        on="fold",
        suffixes=("_previous", "_current")
    )

    print("\n===== E02 Reproduction Check =====")

    for metric in [
        "average_precision",
        "roc_auc"
    ]:

        difference = (
            check[f"{metric}_previous"]
            - check[f"{metric}_current"]
        ).abs()

        print(
            metric,
            "Maximum Difference:",
            float(difference.max())
        )


# 14. 결과 저장
outputs = {
    "label_sensitivity_results.csv": results_df,
    "label_sensitivity_summary.csv": summary
}

print("\n===== Output =====")

for filename, frame in outputs.items():

    path = RESULT_DIR / filename

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", path)

print("\n===== Label Sensitivity Completed =====")
print("Reserved test evaluation has not been performed.")