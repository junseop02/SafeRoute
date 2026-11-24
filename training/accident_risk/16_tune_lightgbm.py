from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.spatial import cKDTree

from sklearn.model_selection import (
    StratifiedGroupKFold,
    ParameterGrid
)

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)


# 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"
RESULT_DIR = DATA_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# 실험 설정
RANDOM_STATE = 42
N_SPLITS = 3
N_CANDIDATES = 30
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


# 1. 공통 모델 설정
BASE_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "n_estimators": 1000,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "n_jobs": -1,
    "scale_pos_weight": 1.0
}


# 기존 E02 모델 설정
REFERENCE_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 6,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "colsample_bytree": 1.0
}


# 2. 하이퍼파라미터 탐색 범위
PARAM_GRID = {
    "learning_rate": [
        0.01, 0.03, 0.05, 0.1
    ],
    "num_leaves": [
        7, 15, 31
    ],
    "max_depth": [
        3, 4, 5, 6
    ],
    "min_child_samples": [
        20, 30, 50, 100
    ],
    "reg_alpha": [
        0.0, 0.1, 0.5, 1.0
    ],
    "reg_lambda": [
        0.1, 1.0, 5.0, 10.0
    ],
    "colsample_bytree": [
        0.8, 1.0
    ]
}


# 3. 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

development = dataset[
    dataset["split"] != "test"
].copy()

test = dataset[
    dataset["split"] == "test"
].copy()


# 좌표 추출 함수
def get_coordinates(frame):

    return frame[
        ["center_x", "center_y"]
    ].to_numpy()


# 4. Test 주변 개발 데이터 제외
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


# Test 공간 그룹 중복 검사
if (
    set(development["spatial_group"])
    & set(test["spatial_group"])
):
    raise ValueError("Test group leakage.")


print("===== Development Dataset =====")

print("Links:", len(development))

print(
    "Spatial Groups:",
    development["spatial_group"].nunique()
)

print(
    "Positive:",
    int(development[TARGET].sum())
)

print(
    "Negative:",
    int((development[TARGET] == 0).sum())
)

print("Features:", FEATURES)


# 5. 기존 공간 Fold 구성
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


# 6. Fold별 데이터 사전 준비
prepared_folds = []

print("\n===== Preparing Spatial Folds =====")

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

    # 최소 거리 검사
    distances, _ = val_tree.query(
        get_coordinates(fold_train),
        k=1
    )

    minimum_distance = float(
        distances.min()
    )

    if minimum_distance <= BUFFER_DISTANCE:

        raise ValueError(
            "Spatial buffer violation."
        )

    # 입력 피처와 정답 라벨
    X_train = fold_train[FEATURES].copy()
    X_val = fold_val[FEATURES].copy()

    y_train = fold_train[TARGET].astype(int)
    y_val = fold_val[TARGET].astype(int)

    # 양성·음성 클래스 검사
    if (
        set(y_train.unique()) != {0, 1}
        or set(y_val.unique()) != {0, 1}
    ):
        raise ValueError(
            "Both classes are required."
        )

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

    # Fold 데이터 저장
    prepared_folds.append({
        "fold": fold_number,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "minimum_distance": minimum_distance,
        "baseline_ap": float(y_val.mean())
    })

    print(f"\n[FOLD {fold_number}]")

    print("Train:", len(X_train))
    print("Validation:", len(X_val))

    print(
        "Train Positive:",
        int(y_train.sum())
    )

    print(
        "Validation Positive:",
        int(y_val.sum())
    )

    print(
        "Minimum Distance:",
        round(minimum_distance, 6)
    )


# 7. 탐색 후보 생성
all_candidates = list(
    ParameterGrid(PARAM_GRID)
)

valid_candidates = [
    params
    for params in all_candidates
    if params["num_leaves"]
    <= 2 ** params["max_depth"]
]

# 기존 설정은 무작위 탐색 대상에서 제외
search_pool = [
    params
    for params in valid_candidates
    if params != REFERENCE_PARAMS
]

rng = np.random.default_rng(
    RANDOM_STATE
)

selected_indices = rng.choice(
    len(search_pool),
    size=N_CANDIDATES - 1,
    replace=False
)

# C00은 기존 E02 모델
candidates = [REFERENCE_PARAMS.copy()]

candidates.extend([
    search_pool[int(index)]
    for index in selected_indices
])


print("\n===== Search Configuration =====")

print(
    "All Parameter Combinations:",
    len(all_candidates)
)

print(
    "Valid Parameter Combinations:",
    len(valid_candidates)
)

print("Selected Candidates:", len(candidates))

print(
    "Total Training Runs:",
    len(candidates) * N_SPLITS
)


# 8. 후보별 공간 교차검증
results = []
candidate_summaries = []

for candidate_index, params in enumerate(
    candidates
):

    candidate_id = f"C{candidate_index:02d}"

    print("\n" + "=" * 60)

    print(
        f"Candidate: {candidate_id} "
        f"({candidate_index + 1}/{len(candidates)})"
    )

    print("=" * 60)

    print(
        json.dumps(
            params,
            indent=4
        )
    )

    candidate_results = []

    for fold_data in prepared_folds:

        fold_number = fold_data["fold"]

        X_train = fold_data["X_train"]
        X_val = fold_data["X_val"]

        y_train = fold_data["y_train"]
        y_val = fold_data["y_val"]

        # 모델 생성
        model = lgb.LGBMClassifier(
            **BASE_PARAMS,
            **params
        )

        # 모델 학습
        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            eval_metric="average_precision",
            categorical_feature=[
                "road_rank"
            ],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=100,
                    first_metric_only=True,
                    verbose=False
                ),
                lgb.log_evaluation(
                    period=0
                )
            ]
        )

        # Validation 점수 예측
        scores = model.predict_proba(
            X_val,
            num_iteration=model.best_iteration_
        )[:, 1]

        # 평가 지표 계산
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

        baseline_ap = fold_data[
            "baseline_ap"
        ]

        ap_lift = ap / baseline_ap

        result = {
            "candidate": candidate_id,
            "fold": fold_number,
            **params,
            "best_iteration": model.best_iteration_,
            "baseline_ap": baseline_ap,
            "average_precision": ap,
            "ap_lift": ap_lift,
            "roc_auc": auc
        }

        results.append(result)
        candidate_results.append(result)

        print(
            f"Fold {fold_number} | "
            f"AP: {ap:.6f} | "
            f"ROC-AUC: {auc:.6f} | "
            f"Best Iteration: "
            f"{model.best_iteration_}"
        )

    # 후보별 평균 성능
    candidate_df = pd.DataFrame(
        candidate_results
    )

    summary = {
        "candidate": candidate_id,
        **params,
        "ap_mean": candidate_df[
            "average_precision"
        ].mean(),
        "ap_std": candidate_df[
            "average_precision"
        ].std(),
        "ap_min": candidate_df[
            "average_precision"
        ].min(),
        "ap_max": candidate_df[
            "average_precision"
        ].max(),
        "ap_lift_mean": candidate_df[
            "ap_lift"
        ].mean(),
        "auc_mean": candidate_df[
            "roc_auc"
        ].mean(),
        "auc_std": candidate_df[
            "roc_auc"
        ].std(),
        "auc_min": candidate_df[
            "roc_auc"
        ].min(),
        "best_iteration_mean": candidate_df[
            "best_iteration"
        ].mean(),
        "best_iteration_median": candidate_df[
            "best_iteration"
        ].median()
    }

    candidate_summaries.append(summary)

    print("\n--- Candidate Summary ---")

    print(
        "Mean AP:",
        round(summary["ap_mean"], 6)
    )

    print(
        "Minimum AP:",
        round(summary["ap_min"], 6)
    )

    print(
        "AP Std:",
        round(summary["ap_std"], 6)
    )

    print(
        "Mean ROC-AUC:",
        round(summary["auc_mean"], 6)
    )

    # 중간 결과 저장
    pd.DataFrame(results).to_csv(
        RESULT_DIR / "tuning_partial_results.csv",
        index=False,
        encoding="utf-8-sig"
    )


# 9. 전체 결과 정리
results_df = pd.DataFrame(results)

summary_df = pd.DataFrame(
    candidate_summaries
)

# 평균 AP 우선 정렬
summary_df = summary_df.sort_values(
    [
        "ap_mean",
        "ap_min",
        "ap_std",
        "auc_mean"
    ],
    ascending=[
        False,
        False,
        True,
        False
    ]
).reset_index(drop=True)


# 10. 기존 모델 재현 검사
reference_path = (
    RESULT_DIR / "model_comparison_results.csv"
)

if not reference_path.exists():

    raise FileNotFoundError(
        reference_path
    )

previous = pd.read_csv(
    reference_path
)

previous = previous[
    (previous["target"] == TARGET)
    & (previous["model"] == "M03_lightgbm")
]

current = results_df[
    results_df["candidate"] == "C00"
]

check = previous.merge(
    current,
    on="fold",
    suffixes=(
        "_previous",
        "_current"
    ),
    validate="one_to_one"
)

if len(check) != N_SPLITS:

    raise ValueError(
        "Incomplete reproduction check."
    )


print("\n===== Reference Reproduction Check =====")

for metric in [
    "average_precision",
    "roc_auc"
]:

    difference = (
        check[f"{metric}_previous"]
        - check[f"{metric}_current"]
    ).abs()

    max_difference = float(
        difference.max()
    )

    print(
        metric,
        "Maximum Difference:",
        max_difference
    )

    if max_difference > 1e-8:

        raise ValueError(
            f"Reproduction mismatch: {metric}"
        )


# 11. 상위 후보 출력
display_columns = [
    "candidate",
    "ap_mean",
    "ap_std",
    "ap_min",
    "ap_lift_mean",
    "auc_mean",
    "auc_std",
    "auc_min",
    "best_iteration_median"
]

print("\n===== Top 10 Candidates =====")

print(
    summary_df[display_columns]
    .head(10)
    .round(6)
    .to_string(index=False)
)


# 12. 기존 모델과 최상위 후보 비교
reference_result = summary_df[
    summary_df["candidate"] == "C00"
].iloc[0]

best_result = summary_df.iloc[0]

ap_improvement = (
    best_result["ap_mean"]
    - reference_result["ap_mean"]
)

relative_improvement = (
    ap_improvement
    / reference_result["ap_mean"]
    * 100
)


print("\n===== Reference vs Best Candidate =====")

print(
    "Reference Candidate:",
    reference_result["candidate"]
)

print(
    "Best Candidate:",
    best_result["candidate"]
)

for metric in [
    "ap_mean",
    "ap_std",
    "ap_min",
    "ap_lift_mean",
    "auc_mean",
    "auc_std",
    "auc_min"
]:

    print(
        f"{metric}: "
        f"{reference_result[metric]:.6f}"
        f" -> "
        f"{best_result[metric]:.6f}"
    )

print(
    "Absolute AP Improvement:",
    round(ap_improvement, 6)
)

print(
    "Relative AP Improvement (%):",
    round(relative_improvement, 4)
)


# 13. 최상위 후보 하이퍼파라미터
best_candidate_index = int(
    best_result["candidate"][1:]
)

best_params = candidates[
    best_candidate_index
]

print("\n===== Best Hyperparameters =====")

print(
    json.dumps(
        best_params,
        indent=4
    )
)

print("\n===== Best Candidate Fold Results =====")

print(
    results_df.loc[
        results_df["candidate"]
        == best_result["candidate"],
        [
            "fold",
            "best_iteration",
            "baseline_ap",
            "average_precision",
            "ap_lift",
            "roc_auc"
        ]
    ]
    .round(6)
    .to_string(index=False)
)


# 14. 결과 저장
results_path = (
    RESULT_DIR / "lightgbm_tuning_results.csv"
)

summary_path = (
    RESULT_DIR / "lightgbm_tuning_summary.csv"
)

candidate_path = (
    RESULT_DIR / "lightgbm_best_candidate.json"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

candidate_config = {
    "candidate": str(
        best_result["candidate"]
    ),
    "selection_status": "provisional",
    "target": TARGET,
    "features": FEATURES,
    "base_params": BASE_PARAMS,
    "tuned_params": best_params,
    "buffer_distance_m": BUFFER_DISTANCE,
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "ap_mean": float(
        best_result["ap_mean"]
    ),
    "ap_std": float(
        best_result["ap_std"]
    ),
    "ap_min": float(
        best_result["ap_min"]
    ),
    "auc_mean": float(
        best_result["auc_mean"]
    ),
    "best_iteration_median": int(
        best_result["best_iteration_median"]
    )
}

with open(
    candidate_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        candidate_config,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n===== Output =====")

print("Saved:", results_path)
print("Saved:", summary_path)
print("Saved:", candidate_path)

print("\n===== Hyperparameter Tuning Completed =====")

print("Total Training Runs:", len(results_df))

print(
    "Reserved test evaluation "
    "has not been performed."
)

print(
    "Existing service model "
    "has not been modified."
)