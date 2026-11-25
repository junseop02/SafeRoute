from pathlib import Path
import gc
import json
import importlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


# ============================================================
# 1. 경로 및 실험 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

RESULTS_DIR = PROCESSED_DIR / "results"

TUNED_PATH = (
    RESULTS_DIR
    / "national_lightgbm_tuned_candidate.json"
)

ASSIGNMENT_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_assignments.parquet"
)

METRICS_PATH = (
    RESULTS_DIR
    / "national_label_sensitivity_metrics.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "national_label_sensitivity_summary.csv"
)

DISTRIBUTION_PATH = (
    RESULTS_DIR
    / "national_label_sensitivity_distribution.csv"
)

CONFIG_PATH = (
    RESULTS_DIR
    / "national_label_sensitivity_config.json"
)

TARGETS = {
    "target": 0,
    "target_10": 10,
    "target_25": 25,
    "target_50": 50,
    "target_75": 75,
    "target_90": 90
}

N_SPLITS = 3

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)

tuning_utils = importlib.import_module(
    "20_optimize_national_lightgbm"
)


# ============================================================
# 2. 최적화 모델 설정 불러오기
# ============================================================

def load_tuned_config(config):

    with TUNED_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        tuned = json.load(file)

    assert tuned["selected_candidate"] == "T09"

    assert tuned["test_evaluated"] is False

    assert tuned["dataset_sha256"] == (
        config["final_dataset_sha256"]
    )

    assert tuned["assignment_sha256"] == (
        baseline_utils.sha256_file(
            ASSIGNMENT_PATH
        )
    )

    assert tuned["features"] == (
        config["feature_sets"]["C"]
    )

    assert tuned["grid_size_m"] == 2000
    assert tuned["buffer_distance_m"] == 400

    assert tuned["recommended_n_estimators"] == 269

    params = tuned["model_params"].copy()

    # 모든 라벨에서 동일한 반복 횟수 사용
    params["n_estimators"] = (
        tuned["recommended_n_estimators"]
    )

    assert params["scale_pos_weight"] == 1.0

    print("\n===== Selected Model Configuration =====")

    print("Candidate:", tuned["selected_candidate"])
    print("Features:", tuned["features"])
    print("Parameters:", params)
    print("Fixed Iterations:", params["n_estimators"])
    print("Tuned Configuration Validation: PASS")

    return tuned, params


# ============================================================
# 3. 개발 데이터 및 라벨 불러오기
# ============================================================

def load_development_data(config):

    # 20번에서 검증한 공간 Fold 재사용
    development, assignments = (
        tuning_utils.load_development_data(config)
    )

    label_columns = list(TARGETS.keys())

    labels = pd.read_parquet(
        training_utils.DATASET_PATH,
        columns=["link_id", *label_columns],
        filters=[
            ("split", "in", ["train", "validation"])
        ]
    )

    assert len(labels) == len(development)
    assert labels["link_id"].is_unique

    labels = (
        labels.set_index("link_id")
        .loc[development["link_id"]]
        .reset_index()
    )

    assert np.array_equal(
        labels["link_id"].to_numpy(),
        development["link_id"].to_numpy()
    )

    assert np.array_equal(
        labels["target"].to_numpy(),
        development["target"].to_numpy()
    )

    for target in label_columns:

        assert labels[target].notna().all()

        assert labels[target].isin([0, 1]).all()

        development[target] = (
            labels[target].to_numpy(dtype=np.int8)
        )

    # 중첩 기준이 높아질수록 양성 집합은 감소
    for previous, current in zip(
        label_columns[:-1],
        label_columns[1:]
    ):

        assert (
            development[current]
            <= development[previous]
        ).all()

    del labels
    gc.collect()

    print("\n===== Development Label Distribution =====")

    for target, overlap in TARGETS.items():

        positive = int(development[target].sum())

        print(
            target,
            "| Overlap:", overlap,
            "| Positive:", positive,
            "| Negative:", len(development) - positive,
            "| Positive Ratio:",
            f"{positive / len(development):.8f}"
        )

    print("Label Validation: PASS")
    print("Test Evaluation: Not Performed")

    return development, assignments


# ============================================================
# 4. Fold별 라벨 분포 기록
# ============================================================

def summarize_distribution(
    fold_id,
    target,
    train,
    validation
):

    train_positive = int(train[target].sum())

    val_positive = int(validation[target].sum())

    assert train[target].nunique() == 2
    assert validation[target].nunique() == 2

    return {
        "fold": fold_id,
        "target": target,
        "overlap_threshold": TARGETS[target],

        "train_rows": len(train),
        "train_positive": train_positive,
        "train_negative": len(train) - train_positive,

        "train_positive_ratio":
            train_positive / len(train),

        "validation_rows": len(validation),
        "validation_positive": val_positive,

        "validation_negative":
            len(validation) - val_positive,

        "validation_positive_ratio":
            val_positive / len(validation),

        "train_positive_groups":
            train.loc[
                train[target] == 1,
                "spatial_group"
            ].nunique(),

        "validation_positive_groups":
            validation.loc[
                validation[target] == 1,
                "spatial_group"
            ].nunique()
    }


# ============================================================
# 5. 공통 평가 함수
# ============================================================

def evaluate_scores(
    model_id,
    fold_id,
    target,
    y_true,
    scores
):

    assert np.isfinite(scores).all()

    threshold = baseline_utils.select_threshold(
        y_true,
        scores
    )

    metrics = baseline_utils.evaluate_scores(
        model_id,
        target,
        y_true,
        scores,
        threshold
    )

    metrics.pop("baseline_id")
    metrics.pop("feature")

    metrics.update({
        "model_id": model_id,
        "fold": fold_id,
        "target": target,
        "overlap_threshold": TARGETS[target]
    })

    return metrics


# ============================================================
# 6. 교차로 Baseline 평가
# ============================================================

def evaluate_baseline(
    fold_id,
    target,
    validation
):

    scores = validation[
        "intersection_nodes_200m"
    ].to_numpy(dtype=np.float64)

    metrics = evaluate_scores(
        "B02",
        fold_id,
        target,
        validation[target].to_numpy(),
        scores
    )

    metrics["n_estimators"] = 0

    return metrics


# ============================================================
# 7. LightGBM 학습 및 평가
# ============================================================

def train_model(
    fold_id,
    target,
    train,
    validation,
    tuned,
    params
):

    start_time = time.perf_counter()

    features = tuned["features"]

    X_train = train[features]
    y_train = train[target]

    X_validation = validation[features]
    y_validation = validation[target]

    model = lgb.LGBMClassifier(**params)

    # 라벨별 Early Stopping을 사용하지 않고
    # 모든 실험에서 동일한 269회 학습
    model.fit(
        X_train,
        y_train,
        categorical_feature=(
            tuned["categorical_features"]
        )
    )

    scores = model.predict_proba(
        X_validation
    )[:, 1]

    assert (
        (scores >= 0) & (scores <= 1)
    ).all()

    metrics = evaluate_scores(
        "T09",
        fold_id,
        target,
        y_validation.to_numpy(),
        scores
    )

    metrics["n_estimators"] = (
        params["n_estimators"]
    )

    metrics["elapsed_seconds"] = (
        time.perf_counter() - start_time
    )

    del model
    del X_train, X_validation
    del y_train, y_validation
    del scores

    gc.collect()

    return metrics


# ============================================================
# 8. 라벨별 공간 교차검증
# ============================================================

def run_experiments(
    development,
    assignments,
    tuned,
    params
):

    metrics_list = []
    distribution_list = []

    completed = 0
    total_runs = len(TARGETS) * N_SPLITS

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

        assert not np.intersect1d(
            train["spatial_group"].unique(),
            validation["spatial_group"].unique()
        ).size

        print(f"\n===== Fold {fold_id} =====")

        print("Train Rows:", len(train))
        print("Validation Rows:", len(validation))

        for target in TARGETS:

            distribution = summarize_distribution(
                fold_id,
                target,
                train,
                validation
            )

            distribution_list.append(distribution)

            baseline = evaluate_baseline(
                fold_id,
                target,
                validation
            )

            metrics = train_model(
                fold_id,
                target,
                train,
                validation,
                tuned,
                params
            )

            metrics_list.extend([
                baseline,
                metrics
            ])

            completed += 1

            improvement = (
                (
                    metrics["average_precision"]
                    - baseline["average_precision"]
                )
                / baseline["average_precision"]
                * 100
            )

            print(
                f"\nFold {fold_id} | {target}"
            )

            print(
                "Validation Positive Ratio:",
                f"{distribution['validation_positive_ratio']:.8f}"
            )

            print(
                "B02 AP:",
                f"{baseline['average_precision']:.8f}"
            )

            print(
                "T09 AP:",
                f"{metrics['average_precision']:.8f}"
            )

            print(
                "T09 AP Lift:",
                f"{metrics['ap_lift']:.8f}"
            )

            print(
                "T09 ROC-AUC:",
                f"{metrics['roc_auc']:.8f}"
            )

            print(
                "T09 F1:",
                f"{metrics['f1']:.8f}"
            )

            print(
                "AP Improvement vs B02:",
                f"{improvement:.4f}%"
            )

            print(
                "Progress:",
                f"{completed}/{total_runs}",
                f"({completed / total_runs * 100:.1f}%)",
                flush=True
            )

        del train, validation
        gc.collect()

        training_utils.print_memory(
            f"After Fold {fold_id}"
        )

    metrics = pd.DataFrame(metrics_list)

    distribution = pd.DataFrame(
        distribution_list
    )

    assert len(metrics) == 36
    assert len(distribution) == 18

    return metrics, distribution


# ============================================================
# 9. 민감도 분석 결과 요약
# ============================================================

def summarize_results(metrics):

    summary = metrics.groupby(
        [
            "target",
            "overlap_threshold",
            "model_id"
        ],
        as_index=False
    ).agg(
        folds=("fold", "nunique"),

        mean_positive_ratio=(
            "positive_ratio", "mean"
        ),

        mean_ap=(
            "average_precision", "mean"
        ),

        std_ap=(
            "average_precision",
            lambda x: x.std(ddof=0)
        ),

        min_ap=(
            "average_precision", "min"
        ),

        mean_ap_lift=("ap_lift", "mean"),

        mean_auc=("roc_auc", "mean"),

        min_auc=("roc_auc", "min"),

        mean_f1=("f1", "mean"),

        mean_precision=("precision", "mean"),

        mean_recall=("recall", "mean"),

        mean_balanced_accuracy=(
            "balanced_accuracy", "mean"
        )
    )

    assert len(summary) == 12
    assert (summary["folds"] == 3).all()

    baseline = summary.loc[
        summary["model_id"] == "B02",
        ["target", "mean_ap"]
    ].rename(
        columns={"mean_ap": "baseline_mean_ap"}
    )

    summary = summary.merge(
        baseline,
        on="target",
        validate="many_to_one"
    )

    summary["ap_improvement_pct"] = (
        (
            summary["mean_ap"]
            - summary["baseline_mean_ap"]
        )
        / summary["baseline_mean_ap"]
        * 100
    )

    # 각 라벨의 동일 Fold Baseline 비교
    wins = {}

    for target in TARGETS:

        selected = metrics.loc[
            metrics["target"] == target
        ]

        pivot = selected.pivot(
            index="fold",
            columns="model_id",
            values="average_precision"
        )

        wins[(target, "T09")] = int(
            (pivot["T09"] > pivot["B02"]).sum()
        )

        wins[(target, "B02")] = 0

    summary["baseline_wins"] = [
        wins[(target, model_id)]
        for target, model_id in zip(
            summary["target"],
            summary["model_id"]
        )
    ]

    summary = summary.sort_values(
        ["overlap_threshold", "model_id"]
    ).reset_index(drop=True)

    print("\n===== Label Sensitivity Summary =====")

    print(
        summary.to_string(index=False)
    )

    print("\n===== LightGBM Sensitivity Summary =====")

    columns = [
        "target",
        "mean_positive_ratio",
        "mean_ap",
        "std_ap",
        "mean_ap_lift",
        "mean_auc",
        "mean_f1",
        "ap_improvement_pct",
        "baseline_wins"
    ]

    print(
        summary.loc[
            summary["model_id"] == "T09",
            columns
        ].to_string(index=False)
    )

    return summary


# ============================================================
# 10. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Label Sensitivity Analysis =====")

    training_utils.print_memory("Initial Memory")

    output_paths = [
        METRICS_PATH,
        SUMMARY_PATH,
        DISTRIBUTION_PATH,
        CONFIG_PATH
    ]

    # 기존 결과 파일 보호
    for path in output_paths:
        if path.exists():
            raise FileExistsError(path)

    config = training_utils.load_config()

    tuned_hash = baseline_utils.sha256_file(
        TUNED_PATH
    )

    assignment_hash = baseline_utils.sha256_file(
        ASSIGNMENT_PATH
    )

    tuned, params = load_tuned_config(config)

    development, assignments = (
        load_development_data(config)
    )

    training_utils.print_memory(
        "After Dataset Load"
    )

    metrics, distribution = run_experiments(
        development,
        assignments,
        tuned,
        params
    )

    summary = summarize_results(metrics)

    # 원본 데이터 및 실험 설정 변경 여부 확인
    assert baseline_utils.sha256_file(
        training_utils.DATASET_PATH
    ) == config["final_dataset_sha256"]

    assert baseline_utils.sha256_file(
        ASSIGNMENT_PATH
    ) == assignment_hash

    assert baseline_utils.sha256_file(
        TUNED_PATH
    ) == tuned_hash

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

    distribution.to_csv(
        DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    experiment_config = {
        "candidate_id": tuned["selected_candidate"],
        "features": tuned["features"],
        "model_params": params,
        "targets": TARGETS,
        "primary_target": "target",
        "primary_target_definition":
            "positive_polygon_overlap_length",
        "sensitivity_targets": [
            target for target in TARGETS
            if target != "target"
        ],
        "grid_size_m": 2000,
        "buffer_distance_m": 400,
        "cv_folds": N_SPLITS,
        "fixed_iterations": params["n_estimators"],
        "early_stopping": False,
        "dataset_sha256":
            config["final_dataset_sha256"],
        "assignment_sha256": assignment_hash,
        "tuned_config_sha256": tuned_hash,
        "test_evaluated": False,
        "deployment_status": "not_deployed"
    }

    with CONFIG_PATH.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(
            experiment_config,
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

    training_utils.print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(
        "Elapsed Time:",
        f"{time.perf_counter() - start_time:.2f}s"
    )

    print("Label Configurations: 6")
    print("CV Folds: 3")
    print("LightGBM Training Runs: 18")
    print("Baseline Evaluations: 18")
    print("Fixed Iterations: 269")
    print("Dataset Hash Verification: PASS")
    print("Fold Assignment Hash Verification: PASS")
    print("Tuned Config Hash Verification: PASS")
    print("Primary Target: target")
    print("Test Set Frozen: Yes")
    print("Test Evaluation: Not Performed")
    print("Final Model Selected: No")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()