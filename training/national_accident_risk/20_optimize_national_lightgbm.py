from pathlib import Path
import gc
import json
import importlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score


# ============================================================
# 1. 경로 및 실험 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
    / "results"
)

ASSIGNMENT_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_assignments.parquet"
)

CV_SUMMARY_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_summary.csv"
)

CV_FOLD_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_fold_summary.csv"
)

METRICS_PATH = (
    RESULTS_DIR
    / "national_lightgbm_tuning_metrics.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "national_lightgbm_tuning_summary.csv"
)

BEST_PATH = (
    RESULTS_DIR
    / "national_lightgbm_tuned_candidate.json"
)

RANDOM_STATE = 42
N_CANDIDATES = 10
N_SPLITS = 3
MAX_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)


# ============================================================
# 2. 하이퍼파라미터 탐색 범위
# ============================================================

PARAMETER_SPACE = {

    "learning_rate": [0.03, 0.05, 0.1],

    "num_leaves": [15, 31, 63],

    "max_depth": [4, 6, 8],

    "min_child_samples": [30, 100, 300],

    "reg_alpha": [0.0, 0.5, 1.0],

    "reg_lambda": [1.0, 5.0, 10.0],

    "colsample_bytree": [0.8, 1.0]

}


# ============================================================
# 3. 개발 데이터 및 기존 Fold 불러오기
# ============================================================

def load_development_data(config):

    train, validation = (
        training_utils.load_development_data(config)
    )

    development = pd.concat(
        [train, validation],
        ignore_index=True
    )

    del train, validation
    gc.collect()

    assignments = pd.read_parquet(
        ASSIGNMENT_PATH
    )

    fold_summary = pd.read_csv(
        CV_FOLD_PATH
    )

    assert len(development) == 1123804
    assert len(assignments) == 1123804

    assert development["link_id"].is_unique
    assert assignments["link_id"].is_unique

    assert (
        int(development["target"].sum())
        == 172369
    )

    assert development["split"].isin(
        ["train", "validation"]
    ).all()

    # link_id 기준으로 기존 Fold 순서 정렬
    assignments = assignments.set_index(
        "link_id"
    ).loc[
        development["link_id"]
    ].reset_index()

    assert np.array_equal(
        development["link_id"].to_numpy(),
        assignments["link_id"].to_numpy()
    )

    assert np.array_equal(
        development["target"].to_numpy(),
        assignments["target"].to_numpy()
    )

    assert np.array_equal(
        development["spatial_group"].to_numpy(),
        assignments["spatial_group"].to_numpy()
    )

    validation_count = np.zeros(
        len(development),
        dtype=np.int8
    )

    print("\n===== Existing Spatial Folds =====")

    for fold_id in range(1, N_SPLITS + 1):

        column = f"fold_{fold_id}"

        assert assignments[column].isin(
            ["train", "validation", "excluded"]
        ).all()

        train_mask = (
            assignments[column] == "train"
        ).to_numpy()

        val_mask = (
            assignments[column] == "validation"
        ).to_numpy()

        excluded_mask = (
            assignments[column] == "excluded"
        ).to_numpy()

        validation_count[val_mask] += 1

        expected = fold_summary.loc[
            fold_summary["fold"] == fold_id
        ].iloc[0]

        assert int(train_mask.sum()) == int(
            expected["retained_train"]
        )

        assert int(val_mask.sum()) == int(
            expected["validation_rows"]
        )

        assert int(excluded_mask.sum()) == int(
            expected["excluded_train"]
        )

        assert int(
            development.loc[
                train_mask, "target"
            ].sum()
        ) == int(expected["train_positive"])

        assert int(
            development.loc[
                val_mask, "target"
            ].sum()
        ) == int(expected["validation_positive"])

        train_groups = development.loc[
            train_mask, "spatial_group"
        ].unique()

        val_groups = development.loc[
            val_mask, "spatial_group"
        ].unique()

        assert not np.intersect1d(
            train_groups,
            val_groups
        ).size

        assert int(
            expected["spatial_violations"]
        ) == 0

        print(
            f"Fold {fold_id}",
            "| Train:", int(train_mask.sum()),
            "| Validation:", int(val_mask.sum()),
            "| Excluded:", int(excluded_mask.sum())
        )

    assert (validation_count == 1).all()

    print("Fold Assignment Validation: PASS")
    print("Existing 400m Buffer: Reused")
    print("Test Evaluation: Not Performed")

    return development, assignments


# ============================================================
# 4. 피처 구성 및 후보 생성
# ============================================================

def build_candidates(config):

    full_features = config["feature_sets"]["C"]

    feature_sets = {

        "full": full_features.copy(),

        "without_rank": [
            feature
            for feature in full_features
            if feature != "road_rank"
        ],

        "without_rank_branch": [
            feature
            for feature in full_features
            if feature not in [
                "road_rank",
                "n_branch"
            ]
        ]

    }

    # 모든 피처 구성에 동일한 탐색 조합 사용
    sampled_params = list(

        ParameterSampler(

            PARAMETER_SPACE,

            n_iter=N_CANDIDATES,

            random_state=RANDOM_STATE

        )

    )

    candidates = []
    number = 1

    for name, features in feature_sets.items():

        for sampled in sampled_params:

            params = {

                **training_utils.BASE_PARAMS,

                **sampled,

                "n_estimators": MAX_ESTIMATORS,

                "scale_pos_weight": 1.0,

                "random_state": RANDOM_STATE

            }

            # 최대 깊이에 비해 리프 수가
            # 과도하게 설정되지 않도록 제한
            params["num_leaves"] = min(

                params["num_leaves"],

                2 ** params["max_depth"]

            )

            candidates.append({

                "candidate_id": f"T{number:02d}",

                "feature_set": name,

                "features": features.copy(),

                "params": params

            })

            number += 1

    assert len(candidates) == 30

    print("\n===== Tuning Candidates =====")

    for candidate in candidates:

        selected_params = {

            key: candidate["params"][key]

            for key in PARAMETER_SPACE

        }

        print(
            candidate["candidate_id"],
            "| Set:", candidate["feature_set"],
            "| Params:", selected_params
        )

    print("Total Candidates:", len(candidates))

    print(
        "Total Training Runs:",
        len(candidates) * N_SPLITS
    )

    return candidates


# ============================================================
# 5. 후보별 모델 학습 및 평가
# ============================================================

def train_candidate(
    fold_id,
    candidate,
    train,
    validation
):

    start_time = time.perf_counter()

    features = candidate["features"]

    X_train = train[features]
    y_train = train["target"]

    X_validation = validation[features]
    y_validation = validation["target"]

    categorical_features = [

        feature

        for feature in ["road_rank"]

        if feature in features

    ]

    model = lgb.LGBMClassifier(
        **candidate["params"]
    )

    model.fit(

        X_train,

        y_train,

        eval_X=X_validation,

        eval_y=y_validation,

        eval_names=["validation"],

        eval_metric="average_precision",

        categorical_feature=categorical_features,

        callbacks=[

            lgb.early_stopping(

                stopping_rounds=EARLY_STOPPING_ROUNDS,

                first_metric_only=True,

                verbose=False

            ),

            lgb.log_evaluation(period=0)

        ]

    )

    best_iteration = int(
        model.best_iteration_
    )

    assert best_iteration > 0

    scores = model.predict_proba(

        X_validation,

        num_iteration=best_iteration

    )[:, 1]

    assert np.isfinite(scores).all()

    assert (
        (scores >= 0) & (scores <= 1)
    ).all()

    threshold = baseline_utils.select_threshold(

        y_validation.to_numpy(),

        scores

    )

    metrics = baseline_utils.evaluate_scores(

        candidate["candidate_id"],

        candidate["feature_set"],

        y_validation.to_numpy(),

        scores,

        threshold

    )

    metrics.pop("baseline_id")
    metrics.pop("feature")

    metrics.update({

        "candidate_id":
            candidate["candidate_id"],

        "feature_set":
            candidate["feature_set"],

        "feature_count":
            len(features),

        "fold":
            fold_id,

        "best_iteration":
            best_iteration,

        "train_rows":
            len(train),

        "train_positive":
            int(y_train.sum()),

        "elapsed_seconds":
            time.perf_counter() - start_time

    })

    for key in PARAMETER_SPACE:

        metrics[key] = candidate["params"][key]

    print(

        f"Fold {fold_id}",

        f"| {candidate['candidate_id']}",

        f"| AP: {metrics['average_precision']:.6f}",

        f"| AUC: {metrics['roc_auc']:.6f}",

        f"| F1: {metrics['f1']:.6f}",

        f"| Iter: {best_iteration}",

        f"| Time: {metrics['elapsed_seconds']:.2f}s",

        flush=True

    )

    del model
    del X_train, X_validation
    del y_train, y_validation
    del scores

    gc.collect()

    return metrics


# ============================================================
# 6. 전체 공간 교차검증 실행
# ============================================================

def run_tuning(
    development,
    assignments,
    candidates
):

    results = []

    for fold_id in range(1, N_SPLITS + 1):

        column = f"fold_{fold_id}"

        train_mask = (
            assignments[column] == "train"
        ).to_numpy()

        val_mask = (
            assignments[column] == "validation"
        ).to_numpy()

        train = development.loc[
            train_mask
        ].reset_index(drop=True)

        validation = development.loc[
            val_mask
        ].reset_index(drop=True)

        assert train["target"].nunique() == 2
        assert validation["target"].nunique() == 2

        print(f"\n===== Tuning Fold {fold_id} =====")

        print("Train Rows:", len(train))

        print("Validation Rows:", len(validation))

        for candidate in candidates:

            metrics = train_candidate(

                fold_id,

                candidate,

                train,

                validation

            )

            results.append(metrics)

            print(

                "Progress:",

                f"{len(results)}/90",

                f"({len(results) / 90 * 100:.1f}%)"

            )

        del train, validation

        gc.collect()

        training_utils.print_memory(
            f"After Fold {fold_id}"
        )

    metrics = pd.DataFrame(results)

    assert len(metrics) == 90

    assert (
        metrics.groupby("candidate_id").size() == 3
    ).all()

    return metrics


# ============================================================
# 7. 최적화 결과 요약
# ============================================================

def summarize_results(metrics):

    summary = metrics.groupby(

        ["candidate_id", "feature_set"],

        as_index=False

    ).agg(

        folds=("fold", "nunique"),

        mean_ap=("average_precision", "mean"),

        std_ap=(
            "average_precision",
            lambda x: x.std(ddof=0)
        ),

        min_ap=("average_precision", "min"),

        max_ap=("average_precision", "max"),

        mean_ap_lift=("ap_lift", "mean"),

        mean_auc=("roc_auc", "mean"),

        std_auc=(
            "roc_auc",
            lambda x: x.std(ddof=0)
        ),

        min_auc=("roc_auc", "min"),

        mean_f1=("f1", "mean"),

        mean_precision=("precision", "mean"),

        mean_recall=("recall", "mean"),

        median_best_iteration=(
            "best_iteration", "median"
        ),

        total_training_seconds=(
            "elapsed_seconds", "sum"
        )

    )

    parameter_columns = [

        "candidate_id",

        *PARAMETER_SPACE.keys()

    ]

    parameter_frame = metrics[
        parameter_columns
    ].drop_duplicates()

    assert parameter_frame[
        "candidate_id"
    ].is_unique

    summary = summary.merge(

        parameter_frame,

        on="candidate_id",

        validate="one_to_one"

    )

    previous_cv = pd.read_csv(
        CV_SUMMARY_PATH
    )

    reference_ap = float(

        previous_cv.loc[
            previous_cv["model_id"] == "CV01",
            "mean_ap"
        ].iloc[0]

    )

    baseline_ap = float(

        previous_cv.loc[
            previous_cv["model_id"] == "B02",
            "mean_ap"
        ].iloc[0]

    )

    summary["ap_change_vs_cv01_pct"] = (

        (summary["mean_ap"] - reference_ap)

        / reference_ap

        * 100

    )

    summary["ap_improvement_vs_baseline_pct"] = (

        (summary["mean_ap"] - baseline_ap)

        / baseline_ap

        * 100

    )

    summary = summary.sort_values(

        ["mean_ap", "min_ap", "std_ap", "candidate_id"],

        ascending=[False, False, True, True]

    ).reset_index(drop=True)

    columns = [

        "candidate_id",
        "feature_set",
        "mean_ap",
        "std_ap",
        "min_ap",
        "mean_auc",
        "mean_f1",
        "median_best_iteration",
        "ap_change_vs_cv01_pct"

    ]

    print("\n===== Final Tuning Summary =====")

    print(
        summary[columns].to_string(index=False)
    )

    print("\n===== Best Candidate Per Feature Set =====")

    print(

        summary.drop_duplicates(
            "feature_set"
        )[columns].to_string(index=False)

    )

    print("\n===== Previous CV01 Reference =====")

    print("Mean AP:", reference_ap)

    print("\n===== B02 Reference =====")

    print("Mean AP:", baseline_ap)

    return summary, reference_ap, baseline_ap


# ============================================================
# 8. 기존 CV01과 최적화 후보 비교
# ============================================================

def select_best_candidate(
    summary,
    metrics,
    candidates,
    config,
    reference_ap,
    baseline_ap
):

    best = summary.iloc[0]

    selected = next(

        candidate

        for candidate in candidates

        if candidate["candidate_id"]
        == best["candidate_id"]

    )

    print("\n===== Best Tuned Candidate =====")

    print("Candidate:", best["candidate_id"])
    print("Feature Set:", best["feature_set"])
    print("Mean AP:", best["mean_ap"])
    print("AP Std:", best["std_ap"])
    print("Minimum AP:", best["min_ap"])
    print("Mean AUC:", best["mean_auc"])

    print(
        "Change vs CV01:",
        best["ap_change_vs_cv01_pct"]
    )

    print(
        "Improvement vs B02:",
        best["ap_improvement_vs_baseline_pct"]
    )

    # 최적화가 성능을 높이지 못하면
    # 기존 CV01을 유지
    if float(best["mean_ap"]) > reference_ap:

        selected_id = selected["candidate_id"]

        selected_features = selected["features"]

        selected_params = selected["params"].copy()

        selected_mean_ap = float(best["mean_ap"])

        recommended_iterations = int(
            best["median_best_iteration"]
        )

        fold_results = metrics.loc[

            metrics["candidate_id"] == selected_id

        ].sort_values("fold")

        selection_reason = (
            "tuned_candidate_beats_cv01"
        )

    else:

        selected_id = "CV01"

        selected_features = (
            config["feature_sets"]["C"]
        )

        selected_params = {

            **training_utils.BASE_PARAMS,

            "scale_pos_weight": 1.0

        }

        previous_metrics = pd.read_csv(

            RESULTS_DIR
            / "national_spatial_cv_metrics.csv"

        )

        fold_results = previous_metrics.loc[

            previous_metrics["model_id"] == "CV01"

        ].sort_values("fold")

        selected_mean_ap = reference_ap

        recommended_iterations = int(

            fold_results["best_iteration"].median()

        )

        selection_reason = (
            "original_cv01_retained"
        )

    assert len(fold_results) == 3

    output = {

        "selected_candidate": selected_id,

        "selection_reason": selection_reason,

        "features": selected_features,

        "categorical_features": [

            feature

            for feature in ["road_rank"]

            if feature in selected_features

        ],

        "model_params": selected_params,

        "recommended_n_estimators":
            recommended_iterations,

        "mean_cv_ap": selected_mean_ap,

        "std_cv_ap": float(

            fold_results[
                "average_precision"
            ].std(ddof=0)

        ),

        "min_cv_ap": float(

            fold_results[
                "average_precision"
            ].min()

        ),

        "mean_cv_auc": float(

            fold_results["roc_auc"].mean()

        ),

        "fold_ap": [

            float(value)

            for value in fold_results[
                "average_precision"
            ]

        ],

        "fold_best_iterations": [

            int(value)

            for value in fold_results[
                "best_iteration"
            ]

        ],

        "reference_cv01_ap": reference_ap,

        "baseline_b02_ap": baseline_ap,

        "dataset_sha256":
            config["final_dataset_sha256"],

        "assignment_sha256":
            baseline_utils.sha256_file(
                ASSIGNMENT_PATH
            ),

        "grid_size_m": 2000,

        "buffer_distance_m": 400,

        "random_state": RANDOM_STATE,

        "test_evaluated": False,

        "deployment_status": "not_deployed"

    }

    print("\n===== Selected Candidate =====")

    print("Candidate:", selected_id)

    print("Reason:", selection_reason)

    print("Features:", selected_features)

    print("Parameters:", selected_params)

    print("Mean CV AP:", selected_mean_ap)

    print(
        "Recommended Iterations:",
        recommended_iterations
    )

    return output


# ============================================================
# 9. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National LightGBM Optimization =====")

    training_utils.print_memory(
        "Initial Memory"
    )

    output_paths = [

        METRICS_PATH,

        SUMMARY_PATH,

        BEST_PATH

    ]

    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    config = training_utils.load_config()

    assignment_hash = (
        baseline_utils.sha256_file(
            ASSIGNMENT_PATH
        )
    )

    development, assignments = (
        load_development_data(config)
    )

    training_utils.print_memory(
        "After Dataset Load"
    )

    candidates = build_candidates(
        config
    )

    metrics = run_tuning(

        development,

        assignments,

        candidates

    )

    summary, reference_ap, baseline_ap = (
        summarize_results(metrics)
    )

    best_candidate = select_best_candidate(

        summary,

        metrics,

        candidates,

        config,

        reference_ap,

        baseline_ap

    )

    # 기존 데이터 및 Fold 파일 변경 여부 검증
    assert (

        baseline_utils.sha256_file(
            training_utils.DATASET_PATH
        )

        == config["final_dataset_sha256"]

    )

    assert (

        baseline_utils.sha256_file(
            ASSIGNMENT_PATH
        )

        == assignment_hash

    )

    metrics.to_csv(

        METRICS_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    summary.to_csv(

        SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    with BEST_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            best_candidate,

            file,

            ensure_ascii=False,

            indent=4

        )

    print("\n===== Saved Results =====")

    for path in output_paths:

        print("Path:", path)

        print(

            "File Size:",

            f"{path.stat().st_size / 1024**2:.4f} MiB"

        )

    training_utils.print_memory(
        "Final Memory"
    )

    print("\n===== Processing Completed =====")

    print(

        "Elapsed Time:",

        f"{time.perf_counter() - start_time:.2f}s"

    )

    print("Feature Configurations: 3")
    print("Tuning Candidates: 30")
    print("LightGBM Training Runs: 90")
    print("Dataset Hash Verification: PASS")
    print("Fold Assignment Hash Verification: PASS")
    print("Test Set Frozen: Yes")
    print("Test Evaluation: Not Performed")
    print("Final Model Selected: No")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()