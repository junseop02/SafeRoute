from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.spatial import cKDTree

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_intersect_200m",
    "n_branch"
]

NUMERIC_FEATURES = [
    "lanes",
    "max_spd",
    "length",
    "n_intersect_200m",
    "n_branch"
]

CATEGORICAL_FEATURES = ["road_rank"]

TARGETS = [
    "target",
    "target_50"
]


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


# 2. 좌표 추출
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

for target in TARGETS:

    print(
        target,
        "Positive:", int(development[target].sum())
    )


# 4. 기존 공간 Fold 구성
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


# 5. 평가 함수
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


# 6. Logistic Regression 생성
def build_logistic_regression():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE
            )
        )
    ])


# 7. Random Forest 생성
def build_random_forest():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=30,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])


# 8. 기존 E02 LightGBM 생성
def build_lightgbm():

    return lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=6,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
        scale_pos_weight=1.0
    )


# 9. 공간 교차검증
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

    # 공간 완충구역 적용
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

    # 최소 공간 거리 검사
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

    X_train = fold_train[FEATURES].copy()
    X_val = fold_val[FEATURES].copy()

    # LightGBM 범주형 피처 처리
    X_train_lgbm = X_train.copy()
    X_val_lgbm = X_val.copy()

    categories = sorted(
        X_train["road_rank"].unique()
    )

    rank_dtype = pd.CategoricalDtype(
        categories=categories
    )

    X_train_lgbm["road_rank"] = (
        X_train_lgbm["road_rank"].astype(rank_dtype)
    )

    X_val_lgbm["road_rank"] = (
        X_val_lgbm["road_rank"].astype(rank_dtype)
    )

    # 라벨별 비교
    for target in TARGETS:

        print("\n===== Target:", target, "=====")

        y_train = fold_train[target].astype(int)
        y_val = fold_val[target].astype(int)

        if (
            set(y_train.unique()) != {0, 1}
            or set(y_val.unique()) != {0, 1}
        ):
            raise ValueError(
                f"Both classes required: {target}"
            )

        print("Train Positive:", int(y_train.sum()))
        print("Validation Positive:", int(y_val.sum()))

        common_result = {
            "fold": fold_number,
            "target": target,
            "train_size": len(fold_train),
            "validation_size": len(fold_val),
            "train_positive": int(y_train.sum()),
            "validation_positive": int(y_val.sum()),
            "minimum_distance": minimum_distance
        }

        # B01: 교차로 수 기준
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
            **baseline_metrics
        })

        # M01: Logistic Regression
        logistic_model = build_logistic_regression()

        logistic_model.fit(
            X_train,
            y_train
        )

        logistic_scores = logistic_model.predict_proba(
            X_val
        )[:, 1]

        logistic_metrics = evaluate_scores(
            y_val,
            logistic_scores
        )

        results.append({
            **common_result,
            "model": "M01_logistic_regression",
            **logistic_metrics
        })

        # M02: Random Forest
        forest_model = build_random_forest()

        forest_model.fit(
            X_train,
            y_train
        )

        forest_scores = forest_model.predict_proba(
            X_val
        )[:, 1]

        forest_metrics = evaluate_scores(
            y_val,
            forest_scores
        )

        results.append({
            **common_result,
            "model": "M02_random_forest",
            **forest_metrics
        })

        # M03: LightGBM
        lightgbm_model = build_lightgbm()

        lightgbm_model.fit(
            X_train_lgbm,
            y_train,
            eval_set=[(X_val_lgbm, y_val)],
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

        lightgbm_scores = lightgbm_model.predict_proba(
            X_val_lgbm,
            num_iteration=lightgbm_model.best_iteration_
        )[:, 1]

        lightgbm_metrics = evaluate_scores(
            y_val,
            lightgbm_scores
        )

        results.append({
            **common_result,
            "model": "M03_lightgbm",
            "best_iteration": lightgbm_model.best_iteration_,
            **lightgbm_metrics
        })

        # 모델별 결과 출력
        model_metrics = {
            "B01_raw_intersection": baseline_metrics,
            "M01_logistic_regression": logistic_metrics,
            "M02_random_forest": forest_metrics,
            "M03_lightgbm": lightgbm_metrics
        }

        for model_name, metrics in model_metrics.items():

            print("\n---", model_name, "---")

            for key, value in metrics.items():

                print(f"{key}: {value:.6f}")


# 10. 전체 실험 결과
results_df = pd.DataFrame(results)

comparison_columns = [
    "fold",
    "target",
    "model",
    "baseline_ap",
    "average_precision",
    "ap_lift",
    "roc_auc",
    "precision",
    "recall",
    "f1"
]

print("\n===== Model Comparison Results =====")

print(
    results_df[comparison_columns]
    .round(6)
    .to_string(index=False)
)


# 11. 모델별 평균 성능
summary = results_df.groupby(
    ["target", "model"]
).agg(
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
    f1_std=("f1", "std")
).reset_index()

summary = summary.sort_values(
    ["target", "ap_mean"],
    ascending=[True, False]
)

print("\n===== Model Comparison Summary =====")

print(
    summary.round(6).to_string(index=False)
)


# 12. 기존 LightGBM 재현 검사
reference_path = (
    RESULT_DIR / "label_sensitivity_results.csv"
)

if reference_path.exists():

    reference = pd.read_csv(reference_path)

    reference = reference[
        (
            reference["model"]
            == "E02_without_curvature"
        )
        & reference["target"].isin(TARGETS)
    ]

    current = results_df[
        results_df["model"] == "M03_lightgbm"
    ]

    check = reference.merge(
        current,
        on=["fold", "target"],
        suffixes=("_previous", "_current"),
        validate="one_to_one"
    )

    if len(check) != N_SPLITS * len(TARGETS):
        raise ValueError("Incomplete reproduction check.")

    print("\n===== LightGBM Reproduction Check =====")

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

        if difference.max() > 1e-8:
            raise ValueError(
                f"Reproduction mismatch: {metric}"
            )


# 13. 결과 저장
results_path = (
    RESULT_DIR / "model_comparison_results.csv"
)

summary_path = (
    RESULT_DIR / "model_comparison_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== Output =====")
print("Saved:", results_path)
print("Saved:", summary_path)

print("\n===== Model Comparison Completed =====")
print("Reserved test evaluation has not been performed.")