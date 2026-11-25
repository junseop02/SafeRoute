from pathlib import Path
import gc
import json
import importlib
import time
import hashlib

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay
)


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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "national_accident_risk"
)

FIGURE_DIR = RESULTS_DIR / "final_figures"

TUNED_PATH = (
    RESULTS_DIR
    / "national_lightgbm_tuned_candidate.json"
)

ASSIGNMENT_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_assignments.parquet"
)

SENSITIVITY_PATH = (
    RESULTS_DIR
    / "national_label_sensitivity_config.json"
)

MODEL_PATH = (
    MODEL_DIR
    / "national_accident_risk_lightgbm_final.joblib"
)

NATIVE_MODEL_PATH = (
    MODEL_DIR
    / "national_accident_risk_lightgbm_final.txt"
)

FROZEN_CONFIG_PATH = (
    RESULTS_DIR
    / "national_final_frozen_config.json"
)

OOF_METRICS_PATH = (
    RESULTS_DIR
    / "national_final_oof_metrics.csv"
)

OOF_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "national_final_oof_predictions.parquet"
)

TEST_METRICS_PATH = (
    RESULTS_DIR
    / "national_final_test_metrics.csv"
)

TEST_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "national_final_test_predictions.parquet"
)

IMPORTANCE_PATH = (
    RESULTS_DIR
    / "national_final_feature_importance.csv"
)

COMPARISON_PATH = (
    RESULTS_DIR
    / "national_final_model_comparison.json"
)

N_SPLITS = 3
TARGET = "target"

SENSITIVITY_TARGETS = [
    "target_10",
    "target_25",
    "target_50",
    "target_75",
    "target_90"
]

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
# 2. 공통 유틸리티
# ============================================================

def load_json(path):

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):

    with path.open("x", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )


def evaluate_scores(
    model_id,
    split,
    y_true,
    scores,
    threshold
):

    assert np.isfinite(scores).all()

    metrics = baseline_utils.evaluate_scores(
        model_id,
        model_id,
        y_true,
        scores,
        threshold
    )

    metrics.pop("baseline_id")
    metrics.pop("feature")

    metrics["model_id"] = model_id
    metrics["split"] = split

    return metrics


# ============================================================
# 3. 최종 학습 설정 검증
# ============================================================

def load_final_config():

    config = training_utils.load_config()

    tuned = load_json(TUNED_PATH)

    sensitivity = load_json(SENSITIVITY_PATH)

    assert tuned["selected_candidate"] == "T09"

    assert tuned["test_evaluated"] is False
    assert sensitivity["test_evaluated"] is False

    assert tuned["dataset_sha256"] == (
        config["final_dataset_sha256"]
    )

    assert sensitivity["dataset_sha256"] == (
        config["final_dataset_sha256"]
    )

    assignment_hash = baseline_utils.sha256_file(
        ASSIGNMENT_PATH
    )

    assert tuned["assignment_sha256"] == (
        assignment_hash
    )

    assert sensitivity["assignment_sha256"] == (
        assignment_hash
    )

    assert sensitivity["tuned_config_sha256"] == (
        baseline_utils.sha256_file(TUNED_PATH)
    )

    assert sensitivity["primary_target"] == TARGET

    assert tuned["features"] == (
        config["feature_sets"]["C"]
    )

    assert sensitivity["features"] == tuned["features"]

    params = tuned["model_params"].copy()

    params["n_estimators"] = int(
        tuned["recommended_n_estimators"]
    )

    assert params["n_estimators"] == 269

    assert params["scale_pos_weight"] == 1.0

    assert params == sensitivity["model_params"]

    print("\n===== Final Model Configuration =====")

    print("Candidate:", tuned["selected_candidate"])
    print("Target:", TARGET)
    print("Features:", tuned["features"])
    print("Parameters:", params)
    print("Configuration Validation: PASS")

    return config, tuned, params


# ============================================================
# 4. 공간 CV 기반 OOF 예측
# ============================================================

def generate_oof_predictions(
    development,
    assignments,
    tuned,
    params
):

    features = tuned["features"]

    categorical_features = (
        tuned["categorical_features"]
    )

    oof_scores = np.full(
        len(development),
        np.nan,
        dtype=np.float64
    )

    oof_fold = np.zeros(
        len(development),
        dtype=np.int8
    )

    for fold_id in range(1, N_SPLITS + 1):

        start_time = time.perf_counter()

        column = f"fold_{fold_id}"

        train_mask = (
            assignments[column] == "train"
        ).to_numpy()

        val_mask = (
            assignments[column] == "validation"
        ).to_numpy()

        train = development.loc[train_mask]

        validation = development.loc[val_mask]

        assert not np.intersect1d(
            train["spatial_group"].unique(),
            validation["spatial_group"].unique()
        ).size

        assert np.isnan(oof_scores[val_mask]).all()

        model = lgb.LGBMClassifier(**params)

        model.fit(
            train[features],
            train[TARGET],
            categorical_feature=categorical_features
        )

        scores = model.predict_proba(
            validation[features]
        )[:, 1]

        assert np.isfinite(scores).all()

        assert (
            (scores >= 0) & (scores <= 1)
        ).all()

        oof_scores[val_mask] = scores

        oof_fold[val_mask] = fold_id

        # Fold별 진단용 임계값
        fold_threshold = baseline_utils.select_threshold(
            validation[TARGET].to_numpy(),
            scores
        )

        fold_metrics = evaluate_scores(
            "T09",
            f"fold_{fold_id}",
            validation[TARGET].to_numpy(),
            scores,
            fold_threshold
        )

        print(f"\n===== OOF Fold {fold_id} =====")

        print("Train Rows:", len(train))
        print("Validation Rows:", len(validation))
        print("AP:", fold_metrics["average_precision"])
        print("AUC:", fold_metrics["roc_auc"])
        print("F1:", fold_metrics["f1"])

        print(
            "Elapsed:",
            f"{time.perf_counter() - start_time:.2f}s"
        )

        del model, train, validation, scores

        gc.collect()

    assert np.isfinite(oof_scores).all()

    assert np.isin(oof_fold, [1, 2, 3]).all()

    print("\nOOF Prediction Coverage: 100%")

    return oof_scores, oof_fold


# ============================================================
# 5. OOF 기반 최종 분류 임계값 결정
# ============================================================

def select_final_thresholds(
    development,
    oof_scores,
    oof_fold
):

    y_true = development[TARGET].to_numpy()

    baseline_scores = development[
        "intersection_nodes_200m"
    ].to_numpy(dtype=np.float64)

    model_threshold = baseline_utils.select_threshold(
        y_true,
        oof_scores
    )

    baseline_threshold = baseline_utils.select_threshold(
        y_true,
        baseline_scores
    )

    model_metrics = evaluate_scores(
        "T09",
        "development_oof",
        y_true,
        oof_scores,
        model_threshold
    )

    baseline_metrics = evaluate_scores(
        "B02",
        "development_oof",
        y_true,
        baseline_scores,
        baseline_threshold
    )

    metrics = pd.DataFrame([
        model_metrics,
        baseline_metrics
    ])

    predictions = development[
        ["link_id", "spatial_group", TARGET]
    ].copy()

    predictions["oof_fold"] = oof_fold

    predictions["hotspot_score"] = oof_scores

    predictions["hotspot_predicted_label"] = (
        oof_scores >= model_threshold
    ).astype(np.int8)

    predictions["intersection_baseline"] = (
        baseline_scores
    )

    predictions["intersection_predicted_label"] = (
        baseline_scores >= baseline_threshold
    ).astype(np.int8)

    print("\n===== OOF Threshold Selection =====")

    print("LightGBM Threshold:", model_threshold)

    print("Baseline Threshold:", baseline_threshold)

    print(metrics.to_string(index=False))

    return (
        model_threshold,
        baseline_threshold,
        metrics,
        predictions
    )


# ============================================================
# 6. Test 평가 전 설정 고정
# ============================================================

def freeze_configuration(
    config,
    tuned,
    params,
    model_threshold,
    baseline_threshold,
    development
):

    frozen = {

        "candidate_id": "T09",

        "objective": "binary",

        "target": TARGET,

        "target_definition":
            "positive_polygon_overlap_length",

        "score_meaning":
            "historical_hotspot_membership_score",

        "features": tuned["features"],

        "categorical_features":
            tuned["categorical_features"],

        "road_rank_categories":
            config["road_rank_categories"],

        "model_params": params,

        "n_estimators": params["n_estimators"],

        "model_threshold": float(model_threshold),

        "baseline_threshold": float(baseline_threshold),

        "threshold_selection":
            "pooled_spatial_oof_max_f1",

        "training_rows": len(development),

        "training_positive":
            int(development[TARGET].sum()),

        "training_negative":
            len(development)
            - int(development[TARGET].sum()),

        "grid_size_m": 2000,

        "buffer_distance_m": 400,

        "random_state": 42,

        "primary_metric": "average_precision",

        "dataset_sha256":
            config["final_dataset_sha256"],

        "test_data_sha256":
            config["test_data_sha256"],

        "assignment_sha256":
            baseline_utils.sha256_file(ASSIGNMENT_PATH),

        "tuned_config_sha256":
            baseline_utils.sha256_file(TUNED_PATH),

        "sensitivity_config_sha256":
            baseline_utils.sha256_file(SENSITIVITY_PATH),

        "test_policy": "final_evaluation_only",

        "test_evaluated_at_freeze": False,

        "deployment_status": "not_deployed"

    }

    save_json(frozen, FROZEN_CONFIG_PATH)

    print("\n===== Final Configuration Frozen =====")

    print("Model Threshold:", model_threshold)
    print("Baseline Threshold:", baseline_threshold)
    print("Training Rows:", len(development))
    print("Test Evaluation: Not Performed")

    return frozen


# ============================================================
# 7. 전체 개발 데이터로 최종 모델 학습
# ============================================================

def train_final_model(
    development,
    frozen
):

    start_time = time.perf_counter()

    features = frozen["features"]

    model = lgb.LGBMClassifier(
        **frozen["model_params"]
    )

    model.fit(
        development[features],
        development[TARGET],
        categorical_feature=(
            frozen["categorical_features"]
        )
    )

    assert model.n_estimators_ == (
        frozen["n_estimators"]
    )

    assert model.feature_name_ == features

    print("\n===== Final Model Training =====")

    print("Training Rows:", len(development))

    print(
        "Positive Rows:",
        int(development[TARGET].sum())
    )

    print(
        "Negative Rows:",
        len(development)
        - int(development[TARGET].sum())
    )

    print("Trained Iterations:", model.n_estimators_)

    print(
        "Elapsed:",
        f"{time.perf_counter() - start_time:.2f}s"
    )

    return model


# ============================================================
# 8. 독립 Test 데이터 무결성 검증
# ============================================================

def load_test_data(
    config,
    development
):

    all_features = config["feature_sets"]["C"]

    hash_columns = [
        "link_id",
        *all_features,
        TARGET,
        *SENSITIVITY_TARGETS,
        "spatial_group",
        "grid_x",
        "grid_y"
    ]

    test = pd.read_parquet(
        training_utils.DATASET_PATH,
        columns=hash_columns + ["split"],
        filters=[("split", "==", "test")]
    ).reset_index(drop=True)

    assert len(test) == 234353

    assert int(test[TARGET].sum()) == 31108

    assert test["link_id"].is_unique

    assert (test["split"] == "test").all()

    assert not np.intersect1d(
        development["link_id"].to_numpy(),
        test["link_id"].to_numpy()
    ).size

    assert not np.intersect1d(
        development["spatial_group"].unique(),
        test["spatial_group"].unique()
    ).size

    # 15번에서 고정한 Test 해시 방식 재사용
    hash_frame = (
        test[hash_columns]
        .sort_values("link_id")
    )

    test_hash = hashlib.sha256(
        hash_frame.to_csv(
            index=False,
            lineterminator="\n",
            float_format="%.17g"
        ).encode("utf-8")
    ).hexdigest()

    assert test_hash == config["test_data_sha256"], (
        "Frozen Test SHA-256 mismatch"
    )

    del hash_frame

    assert test[all_features].notna().all().all()

    assert np.isfinite(
        test[all_features].to_numpy(dtype=np.float64)
    ).all()

    # 학습 데이터와 동일한 범주형 설정
    test["road_rank"] = pd.Categorical(
        test["road_rank"],
        categories=config["road_rank_categories"],
        ordered=False
    )

    assert test["road_rank"].notna().all()

    print("\n===== Independent Test Dataset =====")

    print("Test Rows:", len(test))

    print("Positive Rows:", int(test[TARGET].sum()))

    print(
        "Negative Rows:",
        len(test) - int(test[TARGET].sum())
    )

    print("Test SHA-256:", test_hash)

    print("Shared Link IDs: 0")
    print("Shared Spatial Groups: 0")
    print("Frozen Test Hash Verification: PASS")

    return test


# ============================================================
# 9. 독립 Test 최종 평가
# ============================================================

def evaluate_final_model(
    model,
    test,
    frozen
):

    y_true = test[TARGET].to_numpy()

    model_scores = model.predict_proba(
        test[frozen["features"]]
    )[:, 1]

    assert np.isfinite(model_scores).all()

    assert (
        (model_scores >= 0)
        & (model_scores <= 1)
    ).all()

    baseline_scores = test[
        "intersection_nodes_200m"
    ].to_numpy(dtype=np.float64)

    # Test 데이터에서 임계값을 다시 탐색하지 않음
    model_metrics = evaluate_scores(
        "T09",
        "test",
        y_true,
        model_scores,
        frozen["model_threshold"]
    )

    baseline_metrics = evaluate_scores(
        "B02",
        "test",
        y_true,
        baseline_scores,
        frozen["baseline_threshold"]
    )

    metrics = pd.DataFrame([
        model_metrics,
        baseline_metrics
    ])

    predictions = test[
        [
            "link_id",
            "spatial_group",
            "grid_x",
            "grid_y",
            TARGET
        ]
    ].copy()

    predictions["hotspot_score"] = model_scores

    predictions["hotspot_predicted_label"] = (
        model_scores >= frozen["model_threshold"]
    ).astype(np.int8)

    predictions["intersection_baseline"] = (
        baseline_scores
    )

    predictions["intersection_predicted_label"] = (
        baseline_scores >= frozen["baseline_threshold"]
    ).astype(np.int8)

    print("\n===== Final Independent Test Metrics =====")

    print(metrics.to_string(index=False))

    return metrics, predictions


# ============================================================
# 10. 피처 중요도 분석
# ============================================================

def calculate_feature_importance(
    model,
    frozen
):

    gain = model.booster_.feature_importance(
        importance_type="gain"
    )

    split = model.booster_.feature_importance(
        importance_type="split"
    )

    assert gain.sum() > 0

    importance = pd.DataFrame({

        "feature": frozen["features"],

        "gain": gain,

        "gain_ratio": gain / gain.sum(),

        "gain_percentage":
            gain / gain.sum() * 100,

        "split_importance": split

    }).sort_values(
        "gain",
        ascending=False
    ).reset_index(drop=True)

    print("\n===== Final Feature Importance =====")

    print(importance.to_string(index=False))

    return importance


# ============================================================
# 11. 최종 평가 그래프 저장
# ============================================================

def save_figures(
    test_metrics,
    test_predictions,
    importance
):

    y_true = test_predictions[TARGET].to_numpy()

    model_scores = test_predictions[
        "hotspot_score"
    ].to_numpy()

    baseline_scores = test_predictions[
        "intersection_baseline"
    ].to_numpy()

    model_result = test_metrics.loc[
        test_metrics["model_id"] == "T09"
    ].iloc[0]

    baseline_result = test_metrics.loc[
        test_metrics["model_id"] == "B02"
    ].iloc[0]

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))

    for name, scores, result in [
        ("LightGBM", model_scores, model_result),
        ("Baseline", baseline_scores, baseline_result)
    ]:

        precision, recall, _ = precision_recall_curve(
            y_true,
            scores
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{name} "
                f"(AP={result['average_precision']:.4f})"
            )
        )

    plt.axhline(
        y_true.mean(),
        linestyle="--",
        color="gray",
        label="Positive prevalence"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Independent Test Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "precision_recall_curve.png",
        dpi=200
    )

    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))

    for name, scores, result in [
        ("LightGBM", model_scores, model_result),
        ("Baseline", baseline_scores, baseline_result)
    ]:

        fpr, tpr, _ = roc_curve(
            y_true,
            scores
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{name} "
                f"(AUC={result['roc_auc']:.4f})"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Independent Test ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=200
    )

    plt.close()

    # Confusion Matrix
    for model_id in ["T09", "B02"]:

        result = test_metrics.loc[
            test_metrics["model_id"] == model_id
        ].iloc[0]

        matrix = np.array([
            [result["tn"], result["fp"]],
            [result["fn"], result["tp"]]
        ], dtype=np.int64)

        fig, ax = plt.subplots(figsize=(6, 5))

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[
                "Non-hotspot",
                "Hotspot"
            ]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            colorbar=False
        )

        ax.set_title(
            f"{model_id} Test Confusion Matrix"
        )

        fig.tight_layout()

        fig.savefig(
            FIGURE_DIR
            / f"confusion_matrix_{model_id}.png",
            dpi=200
        )

        plt.close(fig)

    # Feature Importance
    ordered = importance.sort_values(
        "gain_percentage"
    )

    plt.figure(figsize=(9, 5))

    plt.barh(
        ordered["feature"],
        ordered["gain_percentage"]
    )

    plt.xlabel("Gain Importance (%)")
    plt.title("Final LightGBM Feature Importance")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "feature_importance.png",
        dpi=200
    )

    plt.close()


# ============================================================
# 12. Baseline 대비 최종 성능 비교
# ============================================================

def compare_final_results(
    test_metrics,
    oof_metrics
):

    model = test_metrics.loc[
        test_metrics["model_id"] == "T09"
    ].iloc[0]

    baseline = test_metrics.loc[
        test_metrics["model_id"] == "B02"
    ].iloc[0]

    oof = oof_metrics.loc[
        oof_metrics["model_id"] == "T09"
    ].iloc[0]

    ap_difference = float(
        model["average_precision"]
        - baseline["average_precision"]
    )

    ap_improvement = float(
        ap_difference
        / baseline["average_precision"]
        * 100
    )

    comparison = {

        "model_id": "T09",

        "baseline_id": "B02",

        "test_average_precision":
            float(model["average_precision"]),

        "baseline_test_average_precision":
            float(baseline["average_precision"]),

        "test_ap_difference": ap_difference,

        "test_ap_improvement_pct": ap_improvement,

        "test_roc_auc":
            float(model["roc_auc"]),

        "baseline_test_roc_auc":
            float(baseline["roc_auc"]),

        "test_f1": float(model["f1"]),

        "baseline_test_f1": float(baseline["f1"]),

        "oof_average_precision":
            float(oof["average_precision"]),

        "test_beats_baseline_ap":
            bool(ap_difference > 0),

        "test_evaluated": True,

        "deployment_status":
            "candidate_not_deployed"

    }

    print("\n===== Final Model Comparison =====")

    for key, value in comparison.items():

        print(f"{key}: {value}")

    return comparison


# ============================================================
# 13. 최종 모델 저장 및 재로드 검증
# ============================================================

def save_final_model(
    model,
    frozen,
    oof_metrics,
    test_metrics,
    comparison,
    test,
    test_predictions
):

    bundle = {

        "model": model,

        "candidate_id": "T09",

        "features": frozen["features"],

        "target": TARGET,

        "threshold": frozen["model_threshold"],

        "baseline_threshold":
            frozen["baseline_threshold"],

        "categorical_features":
            frozen["categorical_features"],

        "road_rank_categories":
            frozen["road_rank_categories"],

        "model_params": frozen["model_params"],

        "n_estimators": frozen["n_estimators"],

        "training_size": frozen["training_rows"],

        "dataset_sha256": frozen["dataset_sha256"],

        "test_data_sha256":
            frozen["test_data_sha256"],

        "buffer_distance_m": 400,

        "random_state": 42,

        "score_meaning":
            "historical_hotspot_membership_score",

        "oof_metrics":
            oof_metrics.to_dict(orient="records"),

        "test_metrics":
            test_metrics.to_dict(orient="records"),

        "comparison": comparison,

        "deployment_status":
            "candidate_not_deployed",

        "limitations": [

            "Target indicates historical hotspot membership.",

            "Scores are not calibrated accident probabilities.",

            "Negative labels do not guarantee safe roads.",

            "Hotspot data mainly covers vulnerable road users.",

            "Road network and accident data are not time-aligned.",

            "No novice-driver outcome labels are available.",

            "Spatial validation does not guarantee "
            "generalization to every unseen region."

        ]

    }

    joblib.dump(
        bundle,
        MODEL_PATH,
        compress=3
    )

    model.booster_.save_model(
        str(NATIVE_MODEL_PATH)
    )

    reloaded = joblib.load(MODEL_PATH)

    assert reloaded["features"] == frozen["features"]

    assert reloaded["threshold"] == (
        frozen["model_threshold"]
    )

    sample = test[
        frozen["features"]
    ].iloc[:1000]

    original_scores = test_predictions[
        "hotspot_score"
    ].iloc[:1000].to_numpy()

    reloaded_scores = reloaded[
        "model"
    ].predict_proba(sample)[:, 1]

    np.testing.assert_array_equal(
        original_scores,
        reloaded_scores
    )

    native_model = lgb.Booster(
        model_file=str(NATIVE_MODEL_PATH)
    )

    native_scores = native_model.predict(sample)

    np.testing.assert_allclose(
        original_scores,
        native_scores,
        rtol=1e-12,
        atol=1e-12
    )

    print("\n===== Final Model Reload Validation =====")

    print("Joblib Reload: PASS")
    print("Native LightGBM Reload: PASS")
    print("Prediction Consistency: PASS")

    return bundle


# ============================================================
# 14. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== Final National LightGBM Training =====")

    training_utils.print_memory("Initial Memory")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_paths = [

        MODEL_PATH,
        NATIVE_MODEL_PATH,
        FROZEN_CONFIG_PATH,
        OOF_METRICS_PATH,
        OOF_PREDICTIONS_PATH,
        TEST_METRICS_PATH,
        TEST_PREDICTIONS_PATH,
        IMPORTANCE_PATH,
        COMPARISON_PATH,
        FIGURE_DIR

    ]

    # 기존 결과 덮어쓰기 방지
    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    config, tuned, params = load_final_config()

    development, assignments = (
        tuning_utils.load_development_data(config)
    )

    assert len(development) == 1123804

    assert int(development[TARGET].sum()) == 172369

    assert development["link_id"].is_unique

    # 1. OOF 예측
    oof_scores, oof_fold = generate_oof_predictions(
        development,
        assignments,
        tuned,
        params
    )

    # 2. 최종 임계값 결정
    (
        model_threshold,
        baseline_threshold,
        oof_metrics,
        oof_predictions
    ) = select_final_thresholds(
        development,
        oof_scores,
        oof_fold
    )

    # 3. Test 평가 전에 설정 고정
    frozen = freeze_configuration(
        config,
        tuned,
        params,
        model_threshold,
        baseline_threshold,
        development
    )

    frozen_hash = baseline_utils.sha256_file(
        FROZEN_CONFIG_PATH
    )

    oof_metrics.to_csv(
        OOF_METRICS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    oof_predictions.to_parquet(
        OOF_PREDICTIONS_PATH,
        index=False
    )

    del assignments
    del oof_scores, oof_fold, oof_predictions

    gc.collect()

    # 4. 최종 모델 학습
    model = train_final_model(
        development,
        frozen
    )

    # 5. 독립 Test 데이터 확인
    test = load_test_data(
        config,
        development
    )

    del development

    gc.collect()

    # 6. 독립 Test 평가
    test_metrics, test_predictions = (
        evaluate_final_model(
            model,
            test,
            frozen
        )
    )

    # 7. 피처 중요도 및 Baseline 비교
    importance = calculate_feature_importance(
        model,
        frozen
    )

    comparison = compare_final_results(
        test_metrics,
        oof_metrics
    )

    # 8. 최종 데이터 및 설정 변경 여부 확인
    assert baseline_utils.sha256_file(
        training_utils.DATASET_PATH
    ) == frozen["dataset_sha256"]

    assert baseline_utils.sha256_file(
        ASSIGNMENT_PATH
    ) == frozen["assignment_sha256"]

    assert baseline_utils.sha256_file(
        TUNED_PATH
    ) == frozen["tuned_config_sha256"]

    assert baseline_utils.sha256_file(
        SENSITIVITY_PATH
    ) == frozen["sensitivity_config_sha256"]

    assert baseline_utils.sha256_file(
        FROZEN_CONFIG_PATH
    ) == frozen_hash

    # 9. 최종 평가 결과 저장
    test_metrics.to_csv(
        TEST_METRICS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    test_predictions.to_parquet(
        TEST_PREDICTIONS_PATH,
        index=False
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    save_json(
        comparison,
        COMPARISON_PATH
    )

    # 10. 최종 모델 저장
    save_final_model(
        model,
        frozen,
        oof_metrics,
        test_metrics,
        comparison,
        test,
        test_predictions
    )

    # 11. 평가 그래프 생성
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=False
    )

    save_figures(
        test_metrics,
        test_predictions,
        importance
    )

    print("\n===== Saved Results =====")

    for path in output_paths:

        if path.is_file():

            print("Path:", path)

            print(
                "File Size:",
                f"{path.stat().st_size / 1024**2:.4f} MiB"
            )

    print("\n===== Saved Figures =====")

    for path in sorted(FIGURE_DIR.glob("*.png")):

        print("Path:", path)

    training_utils.print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(
        "Elapsed Time:",
        f"{time.perf_counter() - start_time:.2f}s"
    )

    print("OOF Training Runs: 3")
    print("Final Training Runs: 1")
    print("Final Training Rows: 1123804")
    print("Independent Test Rows: 234353")
    print("Fixed Iterations: 269")
    print("Dataset Hash Verification: PASS")
    print("Frozen Test Hash Verification: PASS")
    print("Frozen Configuration Verification: PASS")
    print("Model Reload Verification: PASS")
    print("Test Evaluation: Completed")
    print("Final Research Model: T09")

    print(
        "Test AP Beats Baseline:",
        comparison["test_beats_baseline_ap"]
    )

    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()