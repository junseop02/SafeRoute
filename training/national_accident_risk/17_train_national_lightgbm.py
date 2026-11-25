from pathlib import Path
import gc
import hashlib
import importlib
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil


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

BASELINE_PATH = (
    RESULTS_DIR
    / "national_baseline_metrics.csv"
)

METRICS_PATH = (
    RESULTS_DIR
    / "national_lightgbm_metrics.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "national_lightgbm_validation_predictions.parquet"
)

IMPORTANCE_PATH = (
    RESULTS_DIR
    / "national_lightgbm_feature_importance.csv"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "national_lightgbm_learning_history.csv"
)

CANDIDATE_PATH = (
    RESULTS_DIR
    / "national_lightgbm_best_candidate.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "national_accident_risk"
    / "experiments"
)

RANDOM_STATE = 42

EARLY_STOPPING_ROUNDS = 30

EXPECTED_TRAIN = 911037
EXPECTED_VALIDATION = 212767

EXPECTED_TRAIN_POSITIVE = 141754
EXPECTED_VALIDATION_POSITIVE = 30615


# 16번 Baseline 평가 함수 재사용
baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)


# ============================================================
# 2. 공통 모델 파라미터
# ============================================================

BASE_PARAMS = {

    "objective": "binary",

    "boosting_type": "gbdt",

    "n_estimators": 500,

    "learning_rate": 0.05,

    "num_leaves": 31,

    "max_depth": 6,

    "min_child_samples": 50,

    "reg_alpha": 0.5,

    "reg_lambda": 1.0,

    "colsample_bytree": 1.0,

    "subsample": 1.0,

    "random_state": RANDOM_STATE,

    "n_jobs": 8,

    "deterministic": True,

    "force_col_wise": True,

    "metric": "None",

    "verbosity": -1

}


# ============================================================
# 3. 메모리 및 실험 설정 검증
# ============================================================

def print_memory(title):

    memory = psutil.virtual_memory()

    print(f"\n===== {title} =====")

    print(
        f"Available RAM: "
        f"{memory.available / 1024**3:.2f} GiB"
    )

    print(f"RAM Usage: {memory.percent}%")


def load_config():

    config = baseline_utils.load_config()

    assert config["test_frozen"] is True

    assert config["categorical_features"] == [
        "road_rank"
    ]

    assert set(config["feature_sets"]) == {
        "A", "B", "C"
    }

    assert len(config["feature_sets"]["A"]) == 6
    assert len(config["feature_sets"]["B"]) == 6
    assert len(config["feature_sets"]["C"]) == 7

    return config


# ============================================================
# 4. 개발 데이터 불러오기
# ============================================================

def load_development_data(config):

    features = config["feature_sets"]["C"]

    columns = [
        "link_id",
        "split",
        "spatial_group",
        "target",
        *features
    ]

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

    assert frame[features].notna().all().all()

    assert np.isfinite(
        frame[features].to_numpy(
            dtype=np.float64
        )
    ).all()

    # 15번에서 발생했던 범주형 복원 문제 반영
    categories = config[
        "road_rank_categories"
    ]

    assert frame["road_rank"].isin(
        categories
    ).all()

    frame["road_rank"] = pd.Categorical(

        frame["road_rank"],

        categories=categories,

        ordered=False

    )

    assert frame["road_rank"].notna().all()

    train = frame.loc[
        frame["split"] == "train"
    ].reset_index(drop=True)

    validation = frame.loc[
        frame["split"] == "validation"
    ].reset_index(drop=True)

    del frame

    gc.collect()

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

    print("Validation Rows:", len(validation))

    print("Road Rank Dtype:", train["road_rank"].dtype)

    print("Shared Spatial Groups: 0")

    print("Test Evaluation: Not Performed")

    return train, validation


# ============================================================
# 5. 실험 후보 구성
# ============================================================

def build_candidates(config, train):

    positive = int(train["target"].sum())

    negative = len(train) - positive

    class_weight = negative / positive

    assert np.isclose(

        class_weight,

        config["class_weight_candidates"][1]

    )

    candidates = []

    model_number = 1

    for set_name in ["A", "B", "C"]:

        features = config[
            "feature_sets"
        ][set_name]

        for weight in [1.0, class_weight]:

            candidates.append({

                "model_id":
                    f"L{model_number:02d}",

                "feature_set":
                    set_name,

                "features":
                    features.copy(),

                "scale_pos_weight":
                    float(weight)

            })

            model_number += 1

    print("\n===== Experiment Candidates =====")

    for candidate in candidates:

        print(

            candidate["model_id"],

            "| Set:", candidate["feature_set"],

            "| Features:", len(candidate["features"]),

            "| Weight:",
            f"{candidate['scale_pos_weight']:.8f}"

        )

    return candidates


# ============================================================
# 6. LightGBM 학습 및 검증
# ============================================================

def train_candidate(
    candidate,
    train,
    validation,
    config
):

    start_time = time.perf_counter()

    model_id = candidate["model_id"]

    features = candidate["features"]

    print(f"\n===== Training {model_id} =====")

    print("Feature Set:", candidate["feature_set"])

    print("Features:", features)

    print(
        "Scale Pos Weight:",
        candidate["scale_pos_weight"]
    )

    X_train = train[features]

    y_train = train["target"]

    X_validation = validation[features]

    y_validation = validation["target"]

    params = {

        **BASE_PARAMS,

        "scale_pos_weight":
            candidate["scale_pos_weight"]

    }

    model = lgb.LGBMClassifier(
        **params
    )

    evaluation_history = {}

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (X_validation, y_validation)
        ],

        eval_names=["validation"],

        eval_metric="average_precision",

        categorical_feature=[
            "road_rank"
        ],

        callbacks=[

            lgb.early_stopping(

                stopping_rounds=EARLY_STOPPING_ROUNDS,

                first_metric_only=True,

                verbose=True

            ),

            lgb.log_evaluation(
                period=25
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
        (scores >= 0)
        & (scores <= 1)
    ).all()

    # Validation F1 최대 임계값
    threshold = baseline_utils.select_threshold(

        y_validation.to_numpy(),

        scores

    )

    # 16번과 동일한 평가 지표 사용
    metrics = baseline_utils.evaluate_scores(

        model_id,

        f"feature_set_{candidate['feature_set']}",

        y_validation.to_numpy(),

        scores,

        threshold

    )

    metrics.pop("baseline_id")
    metrics.pop("feature")

    metrics.update({

        "model_id": model_id,

        "feature_set":
            candidate["feature_set"],

        "feature_count":
            len(features),

        "scale_pos_weight":
            candidate["scale_pos_weight"],

        "best_iteration":
            best_iteration

    })

    # 피처 중요도
    gain = model.booster_.feature_importance(

        importance_type="gain",

        iteration=best_iteration

    )

    split_importance = (
        model.booster_.feature_importance(

            importance_type="split",

            iteration=best_iteration

        )
    )

    total_gain = float(gain.sum())

    importance = pd.DataFrame({

        "model_id": model_id,

        "feature_set":
            candidate["feature_set"],

        "feature": features,

        "gain": gain,

        "gain_ratio": (

            gain / total_gain

            if total_gain > 0

            else np.zeros(len(features))

        ),

        "split_importance":
            split_importance

    })

    # 학습 과정 기록
    history = pd.DataFrame({

        "model_id": model_id,

        "iteration": np.arange(

            1,

            len(
                evaluation_history[
                    "validation"
                ]["average_precision"]
            ) + 1

        ),

        "validation_average_precision":

            evaluation_history[
                "validation"
            ]["average_precision"]

    })

    # 실험용 모델 저장
    model_path = MODEL_DIR / f"{model_id}.joblib"

    bundle = {

        "model": model,

        "model_id": model_id,

        "feature_set":
            candidate["feature_set"],

        "features": features,

        "road_rank_categories":
            config["road_rank_categories"],

        "threshold": threshold,

        "best_iteration":
            best_iteration,

        "model_params": params,

        "validation_metrics": metrics.copy(),

        "dataset_sha256":
            config["final_dataset_sha256"],

        "score_meaning":
            "historical_hotspot_membership_score",

        "deployment_status":
            "experimental_not_deployed"

    }

    joblib.dump(
        bundle,
        model_path,
        compress=3
    )

    # 저장 모델 재로딩 검증
    reloaded = joblib.load(
        model_path
    )

    sample = X_validation.iloc[:1000]

    original_scores = model.predict_proba(

        sample,

        num_iteration=best_iteration

    )[:, 1]

    reloaded_scores = reloaded[
        "model"
    ].predict_proba(

        sample,

        num_iteration=best_iteration

    )[:, 1]

    np.testing.assert_array_equal(

        original_scores,

        reloaded_scores

    )

    metrics["elapsed_seconds"] = (
        time.perf_counter() - start_time
    )

    print(f"\n===== {model_id} Result =====")

    print("Best Iteration:", best_iteration)

    print("Threshold:", f"{threshold:.8f}")

    for metric_name in [

        "average_precision",

        "ap_lift",

        "roc_auc",

        "precision",

        "recall",

        "f1",

        "balanced_accuracy"

    ]:

        print(

            f"{metric_name}: "

            f"{metrics[metric_name]:.8f}"

        )

    print("Model Reload Verification: PASS")

    print("Saved Model:", model_path)

    print(

        "Elapsed Time:",

        f"{metrics['elapsed_seconds']:.2f}s"

    )

    predictions = (
        scores >= threshold
    ).astype(np.int8)

    del model
    del bundle
    del reloaded
    del X_train
    del X_validation

    gc.collect()

    return (

        metrics,

        scores,

        predictions,

        importance,

        history

    )


# ============================================================
# 7. 결과 비교 및 최우수 후보 저장
# ============================================================

def compare_results(metrics, config):

    baselines = pd.read_csv(
        BASELINE_PATH
    )

    assert set(baselines["baseline_id"]) == {
        "B00", "B01", "B02"
    }

    assert (
        baselines["split"] == "validation"
    ).all()

    assert (
        baselines["total_links"]
        == EXPECTED_VALIDATION
    ).all()

    best_baseline = baselines.sort_values(

        "average_precision",

        ascending=False

    ).iloc[0]

    baseline_ap = float(
        best_baseline["average_precision"]
    )

    metrics["baseline_ap"] = baseline_ap

    metrics["ap_difference"] = (

        metrics["average_precision"]

        - baseline_ap

    )

    metrics["ap_improvement_pct"] = (

        metrics["ap_difference"]

        / baseline_ap

        * 100

    )

    metrics = metrics.sort_values(

        ["average_precision", "roc_auc", "model_id"],

        ascending=[False, False, True]

    ).reset_index(drop=True)

    display_columns = [

        "model_id",

        "feature_set",

        "scale_pos_weight",

        "best_iteration",

        "average_precision",

        "roc_auc",

        "precision",

        "recall",

        "f1",

        "ap_improvement_pct"

    ]

    print("\n===== LightGBM Comparison =====")

    print(

        metrics[
            display_columns
        ].to_string(index=False)

    )

    print("\n===== Best Baseline =====")

    print("Baseline:", best_baseline["baseline_id"])

    print("AP:", f"{baseline_ap:.8f}")

    best = metrics.iloc[0]

    print("\n===== Best LightGBM Candidate =====")

    print("Model:", best["model_id"])

    print("Feature Set:", best["feature_set"])

    print("Best Iteration:", best["best_iteration"])

    print(

        "Average Precision:",

        f"{best['average_precision']:.8f}"

    )

    print(

        "AP Improvement:",

        f"{best['ap_improvement_pct']:+.4f}%"

    )

    print(

        "Beats Baseline:",

        "Yes"

        if best["average_precision"] > baseline_ap

        else "No"

    )

    candidate = {

        "model_id":
            str(best["model_id"]),

        "feature_set":
            str(best["feature_set"]),

        "features":
            config["feature_sets"][
                best["feature_set"]
            ],

        "scale_pos_weight":
            float(best["scale_pos_weight"]),

        "best_iteration":
            int(best["best_iteration"]),

        "threshold":
            float(best["threshold"]),

        "validation_average_precision":
            float(best["average_precision"]),

        "validation_roc_auc":
            float(best["roc_auc"]),

        "validation_f1":
            float(best["f1"]),

        "baseline_id":
            str(best_baseline["baseline_id"]),

        "baseline_average_precision":
            baseline_ap,

        "ap_improvement_pct":
            float(best["ap_improvement_pct"]),

        "model_params": {

            **BASE_PARAMS,

            "scale_pos_weight":
                float(best["scale_pos_weight"])

        },

        "selection_metric":
            "validation_average_precision",

        "selection_status":
            "initial_candidate_not_final",

        "test_evaluated": False,

        "deployment_status":
            "not_deployed"

    }

    return metrics, candidate


# ============================================================
# 8. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National LightGBM Experiments =====")

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_paths = [

        METRICS_PATH,

        PREDICTIONS_PATH,

        IMPORTANCE_PATH,

        HISTORY_PATH,

        CANDIDATE_PATH,

        *[

            MODEL_DIR / f"L{i:02d}.joblib"

            for i in range(1, 7)

        ]

    ]

    # 기존 결과 덮어쓰기 방지
    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    config = load_config()

    train, validation = load_development_data(
        config
    )

    print_memory("After Dataset Load")

    candidates = build_candidates(
        config,
        train
    )

    prediction_frame = validation[

        [
            "link_id",
            "spatial_group",
            "target"
        ]

    ].copy()

    metrics_list = []

    importance_list = []

    history_list = []

    for candidate in candidates:

        (

            metrics,

            scores,

            predictions,

            importance,

            history

        ) = train_candidate(

            candidate,

            train,

            validation,

            config

        )

        model_id = candidate["model_id"]

        metrics_list.append(metrics)

        importance_list.append(importance)

        history_list.append(history)

        prediction_frame[
            f"{model_id}_score"
        ] = scores

        prediction_frame[
            f"{model_id}_prediction"
        ] = predictions

        gc.collect()

        print_memory(f"After {model_id}")

    metrics = pd.DataFrame(
        metrics_list
    )

    importance = pd.concat(

        importance_list,

        ignore_index=True

    )

    history = pd.concat(

        history_list,

        ignore_index=True

    )

    metrics, best_candidate = compare_results(

        metrics,

        config

    )

    # 고정 데이터셋 변경 여부 확인
    assert (

        baseline_utils.sha256_file(DATASET_PATH)

        == config["final_dataset_sha256"]

    )

    metrics.to_csv(

        METRICS_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    prediction_frame.to_parquet(

        PREDICTIONS_PATH,

        index=False

    )

    importance.to_csv(

        IMPORTANCE_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    history.to_csv(

        HISTORY_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    with CANDIDATE_PATH.open(

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            best_candidate,

            file,

            ensure_ascii=False,

            indent=4,

            allow_nan=False

        )

    print("\n===== Saved Results =====")

    for path in output_paths:

        print("Path:", path)

        print(

            "File Size:",

            f"{path.stat().st_size / 1024**2:.4f} MiB"

        )

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(

        "Elapsed Time:",

        f"{time.perf_counter() - start_time:.2f}s"

    )

    print("Trained Models: 6")

    print("Dataset Hash Verification: PASS")

    print("Test Set Frozen: Yes")

    print("Test Evaluation: Not Performed")

    print("Final Model Selected: No")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()