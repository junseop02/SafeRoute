from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import psutil

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. 경로 및 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

RESULTS_DIR = PROCESSED_DIR / "results"

DATASET_PATH = (
    PROCESSED_DIR
    / "national_accident_risk_final_dataset.parquet"
)

CONFIG_PATH = (
    RESULTS_DIR
    / "national_training_config.json"
)

METRICS_PATH = (
    RESULTS_DIR
    / "national_baseline_metrics.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "national_baseline_validation_predictions.parquet"
)

THRESHOLDS_PATH = (
    RESULTS_DIR
    / "national_baseline_thresholds.json"
)

BASELINE_FEATURES = {
    "B01": "n_intersect_200m",
    "B02": "intersection_nodes_200m"
}

EXPECTED_TRAIN = 911037
EXPECTED_VALIDATION = 212767

EXPECTED_TRAIN_POSITIVE = 141754
EXPECTED_VALIDATION_POSITIVE = 30615


# ============================================================
# 2. 메모리 및 파일 해시
# ============================================================

def print_memory(title):

    memory = psutil.virtual_memory()

    print(f"\n===== {title} =====")

    print(
        f"Available RAM: "
        f"{memory.available / 1024**3:.2f} GiB"
    )

    print(f"RAM Usage: {memory.percent}%")


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 3. 고정된 실험 설정 검증
# ============================================================

def load_config():

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    assert config["test_frozen"] is True

    assert config["grid_size_m"] == 2000

    assert config["buffer_distance_m"] == 400.0

    assert config["target"] == "target"

    assert (
        config["primary_metric"]
        == "average_precision"
    )

    assert (
        config["test_policy"]
        == "final_evaluation_only"
    )

    actual_hash = sha256_file(
        DATASET_PATH
    )

    assert (
        actual_hash
        == config["final_dataset_sha256"]
    )

    print("\n===== Experiment Configuration =====")

    print("Grid Size:", config["grid_size_m"])

    print(
        "Buffer Distance:",
        config["buffer_distance_m"]
    )

    print("Dataset SHA-256:", actual_hash)

    print("Dataset Hash Verification: PASS")

    print("Test Set Frozen: Yes")

    return config


# ============================================================
# 4. Train 및 Validation 데이터 읽기
# ============================================================

def load_development_data():

    columns = [
        "link_id",
        "split",
        "spatial_group",
        "target",
        "n_intersect_200m",
        "intersection_nodes_200m"
    ]

    # Test 행은 평가 데이터로 불러오지 않음
    frame = pd.read_parquet(

        DATASET_PATH,

        columns=columns,

        filters=[
            (
                "split",
                "in",
                ["train", "validation"]
            )
        ],

        engine="pyarrow"

    )

    assert frame["link_id"].is_unique

    assert frame["split"].isin(
        ["train", "validation"]
    ).all()

    assert frame["target"].isin(
        [0, 1]
    ).all()

    assert frame[
        list(BASELINE_FEATURES.values())
    ].notna().all().all()

    assert np.isfinite(

        frame[
            list(BASELINE_FEATURES.values())
        ].to_numpy(dtype=np.float64)

    ).all()

    train = frame.loc[
        frame["split"] == "train"
    ].copy()

    validation = frame.loc[
        frame["split"] == "validation"
    ].copy()

    assert len(train) == EXPECTED_TRAIN

    assert len(validation) == EXPECTED_VALIDATION

    assert (
        int(train["target"].sum())
        == EXPECTED_TRAIN_POSITIVE
    )

    assert (
        int(validation["target"].sum())
        == EXPECTED_VALIDATION_POSITIVE
    )

    assert not train["spatial_group"].isin(
        validation["spatial_group"]
    ).any()

    print("\n===== Development Dataset =====")

    print("Train Rows:", len(train))

    print(
        "Train Positive:",
        int(train["target"].sum())
    )

    print("Validation Rows:", len(validation))

    print(
        "Validation Positive:",
        int(validation["target"].sum())
    )

    print("Shared Spatial Groups: 0")

    print("Test Evaluation: Not Performed")

    return train, validation


# ============================================================
# 5. Validation F1 기준 임계값 선택
# ============================================================

def select_threshold(y_true, scores):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    # 마지막 Precision, Recall 값에는 임계값이 없음
    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_values = np.divide(

        2 * precision * recall,

        denominator,

        out=np.zeros_like(denominator),

        where=denominator > 0

    )

    best_index = int(
        np.argmax(f1_values)
    )

    threshold = float(
        thresholds[best_index]
    )

    return threshold


# ============================================================
# 6. 성능 평가
# ============================================================

def evaluate_scores(
    baseline_id,
    feature_name,
    y_true,
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(

        y_true,

        predictions,

        labels=[0, 1]

    ).ravel()

    prevalence = float(
        np.mean(y_true)
    )

    ap = average_precision_score(
        y_true,
        scores
    )

    return {

        "baseline_id":
            baseline_id,

        "feature":
            feature_name,

        "split":
            "validation",

        "total_links":
            len(y_true),

        "positive_links":
            int(y_true.sum()),

        "positive_ratio":
            prevalence,

        "threshold":
            threshold,

        "average_precision":
            float(ap),

        "ap_lift":
            float(ap / prevalence),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    scores
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    predictions
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),

        "f1":
            float(
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


# ============================================================
# 7. Baseline 실험
# ============================================================

def run_baselines(train, validation):

    y_true = validation[
        "target"
    ].to_numpy(dtype=np.int8)

    train_prevalence = float(
        train["target"].mean()
    )

    validation_prevalence = float(
        validation["target"].mean()
    )

    print("\n===== Class Distribution =====")

    print(
        "Train Positive Ratio:",
        f"{train_prevalence:.8f}"
    )

    print(
        "Validation Positive Ratio:",
        f"{validation_prevalence:.8f}"
    )

    print(
        "Validation No-skill AP:",
        f"{validation_prevalence:.8f}"
    )

    predictions_frame = validation[
        [
            "link_id",
            "spatial_group",
            "target"
        ]
    ].copy()

    baseline_scores = {

        "B00": (

            "constant_train_prevalence",

            np.full(
                len(validation),
                train_prevalence,
                dtype=np.float64
            )

        )

    }

    for baseline_id, feature in (
        BASELINE_FEATURES.items()
    ):

        baseline_scores[baseline_id] = (

            feature,

            validation[
                feature
            ].to_numpy(dtype=np.float64)

        )

    results = []
    thresholds = {}

    for baseline_id, values in (
        baseline_scores.items()
    ):

        feature_name, scores = values

        threshold = select_threshold(
            y_true,
            scores
        )

        result = evaluate_scores(

            baseline_id,

            feature_name,

            y_true,

            scores,

            threshold

        )

        results.append(result)

        thresholds[baseline_id] = {

            "feature": feature_name,

            "threshold": threshold,

            "selection_method":
                "validation_max_f1",

            "score_meaning":

                "constant_train_prevalence"

                if baseline_id == "B00"

                else "raw_intersection_count"

        }

        predictions_frame[
            f"{baseline_id}_score"
        ] = scores

        predictions_frame[
            f"{baseline_id}_prediction"
        ] = (
            scores >= threshold
        ).astype(np.int8)

        print(
            f"\n===== {baseline_id} Result ====="
        )

        for name, value in result.items():

            if isinstance(value, float):

                print(
                    f"{name}: {value:.8f}"
                )

            else:

                print(f"{name}: {value}")

    metrics = pd.DataFrame(results)

    return (
        metrics,
        predictions_frame,
        thresholds
    )


# ============================================================
# 8. 결과 저장
# ============================================================

def save_results(
    metrics,
    predictions,
    thresholds
):

    metrics.to_csv(

        METRICS_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    predictions.to_parquet(

        PREDICTIONS_PATH,

        index=False

    )

    with THRESHOLDS_PATH.open(

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            thresholds,

            file,

            ensure_ascii=False,

            indent=4,

            allow_nan=False

        )

    print("\n===== Saved Results =====")

    for path in [
        METRICS_PATH,
        PREDICTIONS_PATH,
        THRESHOLDS_PATH
    ]:

        print("Path:", path)

        print(

            f"File Size: "
            f"{path.stat().st_size / 1024**2:.4f} MiB"

        )


# ============================================================
# 9. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Baseline Evaluation =====")

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 기존 결과 덮어쓰기 방지
    for path in [

        METRICS_PATH,

        PREDICTIONS_PATH,

        THRESHOLDS_PATH

    ]:

        if path.exists():

            raise FileExistsError(path)

    config = load_config()

    train, validation = (
        load_development_data()
    )

    print_memory("After Dataset Load")

    (
        metrics,
        predictions,
        thresholds
    ) = run_baselines(

        train,

        validation

    )

    comparison = metrics.sort_values(

        "average_precision",

        ascending=False

    )

    display_columns = [

        "baseline_id",

        "feature",

        "threshold",

        "average_precision",

        "ap_lift",

        "roc_auc",

        "precision",

        "recall",

        "f1",

        "balanced_accuracy"

    ]

    print("\n===== Final Baseline Comparison =====")

    print(

        comparison[
            display_columns
        ].to_string(index=False)

    )

    best = comparison.iloc[0]

    print("\n===== Best Validation Baseline =====")

    print("Baseline:", best["baseline_id"])

    print("Feature:", best["feature"])

    print(

        "Average Precision:",

        f"{best['average_precision']:.8f}"

    )

    print(

        "ROC-AUC:",

        f"{best['roc_auc']:.8f}"

    )

    print(

        "F1:",

        f"{best['f1']:.8f}"

    )

    # 데이터 파일 변경 여부 재검증
    assert (

        sha256_file(DATASET_PATH)

        == config["final_dataset_sha256"]

    )

    save_results(

        metrics,

        predictions,

        thresholds

    )

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(

        f"Elapsed Time: "

        f"{time.perf_counter() - start_time:.2f} seconds"

    )

    print("Baseline Evaluation: Completed")

    print("Dataset Hash Verification: PASS")

    print("Test Set Frozen: Yes")

    print("Test Evaluation: Not Performed")

    print("LightGBM Training: Not Started")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()