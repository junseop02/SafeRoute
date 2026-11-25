from pathlib import Path
import gc
import importlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


# ============================================================
# 1. 경로 및 기존 함수 불러오기
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
    / "results"
)

METRICS_PATH = (
    RESULTS_DIR
    / "national_lightgbm_metrics.csv"
)

IMPORTANCE_PATH = (
    RESULTS_DIR
    / "national_lightgbm_feature_importance.csv"
)

ABLATION_PATH = (
    RESULTS_DIR
    / "national_feature_ablation_metrics.csv"
)

COMPARISON_PATH = (
    RESULTS_DIR
    / "national_feature_importance_ablation.csv"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "national_feature_ablation_history.csv"
)

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)


# ============================================================
# 2. 제거 실험 구성
# ============================================================

ABLATION_EXPERIMENTS = {

    "E00": [],

    "E01": ["lanes"],

    "E02": ["max_spd"],

    "E03": ["road_rank"],

    "E04": ["length"],

    "E05": ["n_branch"],

    "E06": ["n_intersect_200m"],

    "E07": ["intersection_nodes_200m"],

    "E08": [
        "n_intersect_200m",
        "intersection_nodes_200m"
    ]

}

# 기존 실험과 동일한 조건은 재학습하지 않음
REUSED_MODELS = {

    "E00": "L06",

    "E06": "L04",

    "E07": "L02"

}


# ============================================================
# 3. 기존 실험 결과 불러오기
# ============================================================

def load_previous_results():

    metrics = pd.read_csv(
        METRICS_PATH
    )

    importance = pd.read_csv(
        IMPORTANCE_PATH
    )

    assert set(metrics["model_id"]) == {
        "L01", "L02", "L03",
        "L04", "L05", "L06"
    }

    assert metrics["model_id"].is_unique

    assert (
        metrics["split"] == "validation"
    ).all()

    assert (
        metrics["total_links"] == 212767
    ).all()

    reference = metrics.loc[
        metrics["model_id"] == "L06"
    ].iloc[0]

    reference_importance = importance.loc[
        importance["model_id"] == "L06"
    ].copy()

    assert len(reference_importance) == 7

    assert reference_importance["feature"].is_unique

    assert np.isclose(
        reference_importance["gain_ratio"].sum(),
        1.0
    )

    print("\n===== Reference Model =====")

    print("Model: L06")

    print(
        "Average Precision:",
        f"{reference['average_precision']:.8f}"
    )

    print(
        "Scale Pos Weight:",
        reference["scale_pos_weight"]
    )

    print(
        "Best Iteration:",
        reference["best_iteration"]
    )

    return (
        metrics,
        reference,
        reference_importance
    )


# ============================================================
# 4. 피처 중요도 분석
# ============================================================

def analyze_importance(importance):

    importance = importance.copy()

    importance["gain_pct"] = (
        importance["gain_ratio"] * 100
    )

    importance = importance.sort_values(

        "gain_pct",

        ascending=False

    ).reset_index(drop=True)

    importance["importance_rank"] = (
        np.arange(1, len(importance) + 1)
    )

    print("\n===== L06 Feature Importance =====")

    print(

        importance[
            [
                "importance_rank",
                "feature",
                "gain",
                "gain_pct",
                "split_importance"
            ]
        ].to_string(index=False)

    )

    return importance


# ============================================================
# 5. 기존 모델 결과 재사용
# ============================================================

def reuse_result(
    experiment_id,
    removed_features,
    features,
    previous_metrics,
    class_weight,
    config
):

    model_id = REUSED_MODELS[
        experiment_id
    ]

    row = previous_metrics.loc[
        previous_metrics["model_id"] == model_id
    ].iloc[0]

    original_features = config[
        "feature_sets"
    ][row["feature_set"]]

    assert features == original_features

    assert np.isclose(

        row["scale_pos_weight"],

        class_weight

    )

    result = row.to_dict()

    result.update({

        "experiment_id":
            experiment_id,

        "removed_features":
            ", ".join(removed_features)
            if removed_features else "none",

        "feature_count":
            len(features),

        "result_source":
            f"reused_{model_id}",

        "elapsed_seconds":
            0.0

    })

    print(
        f"\n===== {experiment_id}: Reused {model_id} ====="
    )

    print("Removed:", removed_features)

    print(
        "Average Precision:",
        f"{result['average_precision']:.8f}"
    )

    return result


# ============================================================
# 6. 피처 제거 모델 학습
# ============================================================

def train_ablation(
    experiment_id,
    removed_features,
    features,
    train,
    validation,
    class_weight
):

    start_time = time.perf_counter()

    print(f"\n===== Training {experiment_id} =====")

    print("Removed:", removed_features)

    print("Features:", features)

    print("Feature Count:", len(features))

    X_train = train[features]

    y_train = train["target"]

    X_validation = validation[features]

    y_validation = validation["target"]

    params = {

        **training_utils.BASE_PARAMS,

        "scale_pos_weight": class_weight

    }

    # road_rank 제거 실험에서는 범주형 피처 없음
    categorical_features = [

        feature

        for feature in ["road_rank"]

        if feature in features

    ]

    model = lgb.LGBMClassifier(
        **params
    )

    evaluation_history = {}

    # 기존 eval_set 사용 중단 경고 반영
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

                stopping_rounds=(
                    training_utils.EARLY_STOPPING_ROUNDS
                ),

                first_metric_only=True,

                verbose=True

            ),

            lgb.log_evaluation(
                period=50
            ),

            lgb.record_evaluation(
                evaluation_history
            )

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

    result = baseline_utils.evaluate_scores(

        experiment_id,

        "feature_ablation",

        y_validation.to_numpy(),

        scores,

        threshold

    )

    result.pop("baseline_id")
    result.pop("feature")

    result.update({

        "experiment_id":
            experiment_id,

        "removed_features":
            ", ".join(removed_features),

        "feature_count":
            len(features),

        "scale_pos_weight":
            class_weight,

        "best_iteration":
            best_iteration,

        "result_source":
            "new_training",

        "elapsed_seconds":
            time.perf_counter() - start_time

    })

    history_values = evaluation_history[
        "validation"
    ]["average_precision"]

    history = pd.DataFrame({

        "experiment_id":
            experiment_id,

        "iteration":
            np.arange(1, len(history_values) + 1),

        "validation_average_precision":
            history_values

    })

    print(f"\n===== {experiment_id} Result =====")

    print("Best Iteration:", best_iteration)

    for name in [

        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy"

    ]:

        print(
            f"{name}: {result[name]:.8f}"
        )

    print(

        "Elapsed Time:",

        f"{result['elapsed_seconds']:.2f}s"

    )

    # 제거 실험 모델은 서비스에 사용하지 않으므로
    # 별도 모델 파일을 저장하지 않음
    del model
    del X_train
    del X_validation
    del scores

    gc.collect()

    return result, history


# ============================================================
# 7. 제거 실험 전체 실행
# ============================================================

def run_experiments(
    config,
    train,
    validation,
    previous_metrics,
    reference
):

    all_features = config[
        "feature_sets"
    ]["C"]

    class_weight = float(
        reference["scale_pos_weight"]
    )

    positive = int(train["target"].sum())

    negative = len(train) - positive

    assert np.isclose(

        class_weight,

        negative / positive

    )

    results = []

    histories = []

    for experiment_id, removed in (
        ABLATION_EXPERIMENTS.items()
    ):

        features = [

            feature

            for feature in all_features

            if feature not in removed

        ]

        if experiment_id in REUSED_MODELS:

            result = reuse_result(

                experiment_id,

                removed,

                features,

                previous_metrics,

                class_weight,

                config

            )

        else:

            result, history = train_ablation(

                experiment_id,

                removed,

                features,

                train,

                validation,

                class_weight

            )

            histories.append(history)

        results.append(result)

        gc.collect()

        training_utils.print_memory(
            f"After {experiment_id}"
        )

    results = pd.DataFrame(results)

    history = pd.concat(

        histories,

        ignore_index=True

    )

    return results, history


# ============================================================
# 8. 제거 전후 성능 비교
# ============================================================

def compare_ablation(results, reference):

    reference_ap = float(
        reference["average_precision"]
    )

    reference_auc = float(
        reference["roc_auc"]
    )

    reference_f1 = float(
        reference["f1"]
    )

    # 양수: 피처 제거 후 성능 감소
    # 음수: 피처 제거 후 성능 향상
    results["ap_drop"] = (

        reference_ap

        - results["average_precision"]

    )

    results["ap_drop_pct"] = (

        results["ap_drop"]

        / reference_ap

        * 100

    )

    results["auc_drop"] = (

        reference_auc

        - results["roc_auc"]

    )

    results["f1_drop"] = (

        reference_f1

        - results["f1"]

    )

    results["beats_full_model"] = (

        results["average_precision"]

        > reference_ap

    )

    results = results.sort_values(

        "average_precision",

        ascending=False

    ).reset_index(drop=True)

    columns = [

        "experiment_id",
        "removed_features",
        "feature_count",
        "best_iteration",
        "average_precision",
        "roc_auc",
        "f1",
        "ap_drop",
        "ap_drop_pct",
        "result_source"

    ]

    print("\n===== Final Ablation Comparison =====")

    print(

        results[columns].to_string(
            index=False
        )

    )

    best = results.iloc[0]

    print("\n===== Best Ablation Candidate =====")

    print("Experiment:", best["experiment_id"])

    print("Removed:", best["removed_features"])

    print(

        "Average Precision:",

        f"{best['average_precision']:.8f}"

    )

    print(

        "AP Drop Percent:",

        f"{best['ap_drop_pct']:+.6f}%"

    )

    return results


# ============================================================
# 9. 중요도와 제거 실험 결과 결합
# ============================================================

def combine_importance_ablation(
    importance,
    results
):

    single_feature_results = results.loc[

        results["experiment_id"].isin(
            [
                "E01", "E02", "E03", "E04",
                "E05", "E06", "E07"
            ]
        ),

        [
            "removed_features",
            "experiment_id",
            "average_precision",
            "ap_drop",
            "ap_drop_pct",
            "auc_drop",
            "f1_drop"
        ]

    ].copy()

    comparison = importance.merge(

        single_feature_results,

        left_on="feature",

        right_on="removed_features",

        how="left",

        validate="one_to_one"

    )

    assert len(comparison) == 7

    assert comparison[
        "experiment_id"
    ].notna().all()

    comparison = comparison.sort_values(

        "ap_drop",

        ascending=False

    ).reset_index(drop=True)

    comparison["ablation_rank"] = (
        np.arange(1, len(comparison) + 1)
    )

    print("\n===== Importance vs Ablation =====")

    print(

        comparison[
            [
                "feature",
                "importance_rank",
                "gain_pct",
                "ablation_rank",
                "ap_drop",
                "ap_drop_pct"
            ]
        ].to_string(index=False)

    )

    print("\n===== Largest AP Contribution =====")

    print(

        "Feature:",

        comparison.iloc[0]["feature"]

    )

    print(

        "AP Drop:",

        f"{comparison.iloc[0]['ap_drop']:.8f}"

    )

    print("\n===== Smallest AP Contribution =====")

    print(

        "Feature:",

        comparison.iloc[-1]["feature"]

    )

    print(

        "AP Drop:",

        f"{comparison.iloc[-1]['ap_drop']:.8f}"

    )

    return comparison


# ============================================================
# 10. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Feature Ablation =====")

    training_utils.print_memory(
        "Initial Memory"
    )

    output_paths = [

        ABLATION_PATH,

        COMPARISON_PATH,

        HISTORY_PATH

    ]

    # 기존 결과 덮어쓰기 방지
    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    config = training_utils.load_config()

    (
        previous_metrics,
        reference,
        importance
    ) = load_previous_results()

    importance = analyze_importance(
        importance
    )

    train, validation = (
        training_utils.load_development_data(
            config
        )
    )

    training_utils.print_memory(
        "After Dataset Load"
    )

    results, history = run_experiments(

        config,

        train,

        validation,

        previous_metrics,

        reference

    )

    results = compare_ablation(

        results,

        reference

    )

    comparison = combine_importance_ablation(

        importance,

        results

    )

    assert len(results) == 9

    assert (
        results["result_source"] == "new_training"
    ).sum() == 6

    assert (
        results["result_source"].str.startswith("reused_")
    ).sum() == 3

    # 원본 학습 데이터 파일 변경 여부 검증
    assert (

        baseline_utils.sha256_file(
            training_utils.DATASET_PATH
        )

        == config["final_dataset_sha256"]

    )

    results.to_csv(

        ABLATION_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    comparison.to_csv(

        COMPARISON_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    history.to_csv(

        HISTORY_PATH,

        index=False,

        encoding="utf-8-sig"

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

    print("Total Experiments: 9")

    print("Newly Trained Models: 6")

    print("Reused Models: 3")

    print("Dataset Hash Verification: PASS")

    print("Test Set Frozen: Yes")

    print("Test Evaluation: Not Performed")

    print("Final Features Selected: No")

    print("Final Model Selected: No")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()