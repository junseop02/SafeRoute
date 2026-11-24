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

TARGET = "target"

FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch"
]

# 1. 실험별 피처 구성
EXPERIMENTS = {
    "E01_all_features": FEATURES.copy(),

    "E02_without_curvature": [
        f for f in FEATURES
        if f != "curvature"
    ],

    "E03_without_intersection": [
        f for f in FEATURES
        if f != "n_intersect_200m"
    ],

    "E04_without_both": [
        f for f in FEATURES
        if f not in [
            "curvature",
            "n_intersect_200m"
        ]
    ],

    "E05_intersection_only": [
        "n_intersect_200m"
    ],

    "E06_intersection_branch": [
        "n_intersect_200m",
        "n_branch"
    ]
}

# 2. 기존 Model A 하이퍼파라미터 유지
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

# 3. 기존 공간 분할 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

development = dataset[
    dataset["split"] != "test"
].copy()

test = dataset[
    dataset["split"] == "test"
].copy()


# 4. 도로 중심점 추출
def get_coordinates(frame):

    return frame[
        ["center_x", "center_y"]
    ].to_numpy()


# 5. Test 주변 개발 데이터 제외
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
print("Positive:", int(development[TARGET].sum()))
print("Groups:", development["spatial_group"].nunique())

print("\n===== Experiment Configuration =====")

for name, features in EXPERIMENTS.items():

    print(name)
    print("Features:", features)


# 6. 기존 교차검증 분할 재현
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


# 7. 최적 임계값 계산
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


# 8. 공통 평가 함수
def evaluate_scores(y_true, scores):

    threshold = find_best_threshold(
        y_true,
        scores
    )

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


# 9. Fold별 실험 진행
results = []
importance_results = []

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

    # Fold별 완충구역 적용
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

    # 클래스 검사
    for name, frame in [
        ("train", fold_train),
        ("validation", fold_val)
    ]:

        if set(frame[TARGET].unique()) != {0, 1}:

            raise ValueError(
                f"Fold {fold_number}: "
                f"Both classes required in {name}."
            )

    # 공간 거리 재검사
    distances, _ = val_tree.query(
        get_coordinates(fold_train),
        k=1
    )

    minimum_distance = float(distances.min())

    if minimum_distance <= BUFFER_DISTANCE:

        raise ValueError("Spatial buffer violation.")

    y_train = fold_train[TARGET].astype(int)
    y_val = fold_val[TARGET].astype(int)

    print("Train:", len(fold_train))
    print("Validation:", len(fold_val))
    print("Train Positive:", int(y_train.sum()))
    print("Validation Positive:", int(y_val.sum()))
    print("Minimum Distance:", minimum_distance)

    # 10. 단일 피처 기준 모델
    baseline_scores = fold_val[
        "n_intersect_200m"
    ].to_numpy(dtype=float)

    baseline_metrics = evaluate_scores(
        y_val,
        baseline_scores
    )

    results.append({
        "fold": fold_number,
        "experiment": "B01_raw_intersection",
        "feature_count": 1,
        "best_iteration": 0,
        **baseline_metrics
    })

    print("\n===== B01_raw_intersection =====")

    for key, value in baseline_metrics.items():

        print(f"{key}: {value:.6f}")

    # 11. 피처 조합별 LightGBM 학습
    for experiment_name, features in EXPERIMENTS.items():

        print("\n=====", experiment_name, "=====")

        X_train = fold_train[features].copy()
        X_val = fold_val[features].copy()

        categorical_features = []

        # 도로 등급 범주형 처리
        if "road_rank" in features:

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

            categorical_features.append("road_rank")

        # 모델 생성
        model = lgb.LGBMClassifier(
            **BASE_PARAMS
        )

        # 모델 학습
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=100,
                    first_metric_only=True,
                    verbose=False
                ),
                lgb.log_evaluation(period=0)
            ]
        )

        # 검증 점수 계산
        scores = model.predict_proba(
            X_val,
            num_iteration=model.best_iteration_
        )[:, 1]

        metrics = evaluate_scores(
            y_val,
            scores
        )

        results.append({
            "fold": fold_number,
            "experiment": experiment_name,
            "feature_count": len(features),
            "best_iteration": model.best_iteration_,
            **metrics
        })

        print("Feature Count:", len(features))
        print("Best Iteration:", model.best_iteration_)

        for key, value in metrics.items():

            print(f"{key}: {value:.6f}")

        # 피처 중요도 저장
        gain_values = model.booster_.feature_importance(
            importance_type="gain"
        )

        total_gain = gain_values.sum()

        for feature, gain in zip(
            features,
            gain_values
        ):

            gain_ratio = (
                gain / total_gain * 100
                if total_gain > 0
                else 0.0
            )

            importance_results.append({
                "fold": fold_number,
                "experiment": experiment_name,
                "feature": feature,
                "gain": gain,
                "gain_ratio_pct": gain_ratio
            })


# 12. 전체 실험 결과 정리
results_df = pd.DataFrame(results)

comparison_columns = [
    "fold",
    "experiment",
    "feature_count",
    "best_iteration",
    "baseline_ap",
    "average_precision",
    "ap_lift",
    "roc_auc",
    "precision",
    "recall",
    "f1"
]

print("\n===== Ablation Study Results =====")

print(
    results_df[comparison_columns]
    .round(6)
    .to_string(index=False)
)


# 13. 실험별 평균 및 표준편차
summary = results_df.groupby(
    "experiment"
).agg(
    feature_count=("feature_count", "first"),

    ap_mean=("average_precision", "mean"),
    ap_std=("average_precision", "std"),
    ap_min=("average_precision", "min"),

    ap_lift_mean=("ap_lift", "mean"),

    auc_mean=("roc_auc", "mean"),
    auc_std=("roc_auc", "std"),
    auc_min=("roc_auc", "min"),

    precision_mean=("precision", "mean"),
    recall_mean=("recall", "mean"),

    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),

    best_iteration_mean=("best_iteration", "mean")
)

summary = summary.sort_values(
    "ap_mean",
    ascending=False
)

print("\n===== Ablation Study Summary =====")

print(
    summary.round(6).to_string()
)


# 14. Fold 3 성능 비교
fold3 = results_df[
    results_df["fold"] == 3
].sort_values(
    "average_precision",
    ascending=False
)

print("\n===== Fold 3 Comparison =====")

print(
    fold3[comparison_columns]
    .round(6)
    .to_string(index=False)
)


# 15. 기존 Model A 재현 여부 확인
reference_path = (
    RESULT_DIR / "spatial_cv_results.csv"
)

if reference_path.exists():

    reference = pd.read_csv(reference_path)

    reference = reference[
        reference["model"] == "model_a"
    ]

    current = results_df[
        results_df["experiment"] == "E01_all_features"
    ]

    check = reference.merge(
        current,
        on="fold",
        suffixes=("_previous", "_current")
    )

    print("\n===== Model A Reproduction Check =====")

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


# 16. 실험 결과 저장
importance_df = pd.DataFrame(
    importance_results
)

outputs = {
    "ablation_results.csv": results_df,

    "ablation_summary.csv": summary.reset_index(),

    "ablation_feature_importance.csv": importance_df
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

print("\n===== Ablation Study Completed =====")
print("Reserved test evaluation has not been performed.")