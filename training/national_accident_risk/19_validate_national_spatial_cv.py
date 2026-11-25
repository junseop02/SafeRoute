from pathlib import Path
import gc
import importlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

from shapely import STRtree
from sklearn.model_selection import StratifiedGroupKFold


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

ROAD_PATH = (
    PROCESSED_DIR
    / "national_road_features.parquet"
)

METRICS_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_metrics.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_summary.csv"
)

FOLD_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_fold_summary.csv"
)

ASSIGNMENT_PATH = (
    RESULTS_DIR
    / "national_spatial_cv_assignments.parquet"
)

N_SPLITS = 3
RANDOM_STATE = 42
BUFFER_DISTANCE = 400.0
BATCH_SIZE = 1000

EXPECTED_ROWS = 1123804
EXPECTED_POSITIVE = 172369

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)


# ============================================================
# 2. 개발 데이터 통합
# ============================================================

def load_development_data(config):

    train, validation = (
        training_utils.load_development_data(config)
    )

    development = pd.concat(
        [train, validation],
        ignore_index=True
    )

    del train
    del validation

    gc.collect()

    assert len(development) == EXPECTED_ROWS

    assert development["link_id"].is_unique

    assert (
        int(development["target"].sum())
        == EXPECTED_POSITIVE
    )

    assert development["split"].isin(
        ["train", "validation"]
    ).all()

    print("\n===== Combined Development Dataset =====")

    print("Total Rows:", len(development))

    print(
        "Positive:",
        int(development["target"].sum())
    )

    print(
        "Negative:",
        len(development)
        - int(development["target"].sum())
    )

    print(
        "Spatial Groups:",
        development["spatial_group"].nunique()
    )

    print("Test Evaluation: Not Performed")

    return development


# ============================================================
# 3. 개발 데이터 도로 형상 불러오기
# ============================================================

def load_geometries(development):

    print("\n===== Loading Road Geometries =====")

    road_file = pq.ParquetFile(ROAD_PATH)

    metadata = road_file.schema_arrow.metadata

    assert metadata[b"geometry_crs"] == b"EPSG:5179"

    link_index = pd.Index(
        development["link_id"]
    )

    geometries = np.empty(
        len(development),
        dtype=object
    )

    geometries[:] = None

    seen = np.zeros(
        len(development),
        dtype=bool
    )

    processed = 0

    for batch in road_file.iter_batches(
        batch_size=5000,
        columns=["link_id", "geometry_wkb"]
    ):

        frame = batch.to_pandas()

        positions = link_index.get_indexer(
            frame["link_id"]
        )

        selected = positions >= 0

        selected_positions = positions[selected]

        if selected.any():

            assert not seen[
                selected_positions
            ].any()

            decoded = shapely.from_wkb(

                frame.loc[
                    selected,
                    "geometry_wkb"
                ].to_numpy()

            )

            assert not shapely.is_missing(
                decoded
            ).any()

            assert not shapely.is_empty(
                decoded
            ).any()

            assert shapely.is_valid(
                decoded
            ).all()

            geometries[
                selected_positions
            ] = decoded

            seen[selected_positions] = True

        processed += len(frame)

    assert seen.all()

    print("Source Roads:", processed)

    print(
        "Development Geometries:",
        int(seen.sum())
    )

    print("Geometry Validation: PASS")

    return geometries


# ============================================================
# 4. 공간 그룹 교차검증 분할
# ============================================================

def create_folds(development):

    splitter = StratifiedGroupKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE

    )

    targets = development["target"].to_numpy()

    groups = development[
        "spatial_group"
    ].to_numpy()

    folds = list(

        splitter.split(

            np.zeros(
                len(development),
                dtype=np.int8
            ),

            targets,

            groups

        )

    )

    validation_count = np.zeros(
        len(development),
        dtype=np.int8
    )

    for train_idx, val_idx in folds:

        validation_count[val_idx] += 1

        assert not np.intersect1d(

            groups[train_idx],

            groups[val_idx]

        ).size

        assert len(
            np.unique(targets[train_idx])
        ) == 2

        assert len(
            np.unique(targets[val_idx])
        ) == 2

    assert (validation_count == 1).all()

    print("\n===== Spatial CV Folds =====")

    for fold_id, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        print(

            f"Fold {fold_id}",

            "| Train:", len(train_idx),

            "| Validation:", len(val_idx),

            "| Validation Positive:",

            int(targets[val_idx].sum())

        )

    return folds


# ============================================================
# 5. Fold별 400m 공간 버퍼 적용
# ============================================================

def apply_spatial_buffer(
    geometries,
    train_idx,
    val_idx
):

    tree = STRtree(
        geometries[val_idx]
    )

    excluded = np.zeros(
        len(train_idx),
        dtype=bool
    )

    for start in range(
        0,
        len(train_idx),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(train_idx)
        )

        batch_indices = train_idx[start:end]

        pairs = tree.query(

            geometries[batch_indices],

            predicate="dwithin",

            distance=BUFFER_DISTANCE

        )

        if pairs.shape[1] > 0:

            affected = np.unique(
                pairs[0]
            )

            excluded[
                start + affected
            ] = True

    retained_idx = train_idx[~excluded]

    excluded_idx = train_idx[excluded]

    # 버퍼 적용 후 독립적인 거리 재검증
    violations = 0

    for start in range(
        0,
        len(retained_idx),
        BATCH_SIZE
    ):

        batch_indices = retained_idx[
            start:start + BATCH_SIZE
        ]

        pairs = tree.query(

            geometries[batch_indices],

            predicate="dwithin",

            distance=BUFFER_DISTANCE

        )

        violations += pairs.shape[1]

    assert violations == 0

    del tree

    gc.collect()

    return retained_idx, excluded_idx


# ============================================================
# 6. 실험 후보 구성
# ============================================================

def build_candidates(config):

    full_features = config[
        "feature_sets"
    ]["C"]

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

    candidates = []

    number = 1

    for name, features in feature_sets.items():

        for weighted in [False, True]:

            candidates.append({

                "model_id": f"CV{number:02d}",

                "feature_set": name,

                "features": features.copy(),

                "weighted": weighted

            })

            number += 1

    print("\n===== CV Candidates =====")

    for candidate in candidates:

        print(

            candidate["model_id"],

            "| Set:", candidate["feature_set"],

            "| Weighted:", candidate["weighted"],

            "| Features:", candidate["features"]

        )

    return candidates


# ============================================================
# 7. Fold별 Baseline 평가
# ============================================================

def evaluate_baselines(
    fold_id,
    train,
    validation
):

    results = []

    baselines = {

        "B01": "n_intersect_200m",

        "B02": "intersection_nodes_200m"

    }

    for model_id, feature in baselines.items():

        scores = validation[
            feature
        ].to_numpy(dtype=np.float64)

        y_true = validation[
            "target"
        ].to_numpy()

        threshold = baseline_utils.select_threshold(

            y_true,

            scores

        )

        metrics = baseline_utils.evaluate_scores(

            model_id,

            feature,

            y_true,

            scores,

            threshold

        )

        metrics.pop("baseline_id")
        metrics.pop("feature")

        metrics.update({

            "fold": fold_id,

            "model_id": model_id,

            "feature_set": feature,

            "feature_count": 1,

            "weighted": False,

            "scale_pos_weight": 1.0,

            "best_iteration": 0,

            "train_rows": len(train),

            "train_positive":
                int(train["target"].sum())

        })

        results.append(metrics)

        print(

            f"Fold {fold_id} | {model_id}",

            f"| AP: {metrics['average_precision']:.8f}",

            f"| AUC: {metrics['roc_auc']:.8f}"

        )

    return results


# ============================================================
# 8. Fold별 LightGBM 학습
# ============================================================

def train_candidate(
    fold_id,
    candidate,
    train,
    validation
):

    start_time = time.perf_counter()

    features = candidate["features"]

    positive = int(train["target"].sum())

    negative = len(train) - positive

    assert positive > 0
    assert negative > 0

    weight = (

        negative / positive

        if candidate["weighted"]

        else 1.0

    )

    params = {

        **training_utils.BASE_PARAMS,

        "scale_pos_weight": weight

    }

    categorical_features = [

        feature

        for feature in ["road_rank"]

        if feature in features

    ]

    X_train = train[features]

    y_train = train["target"]

    X_validation = validation[features]

    y_validation = validation["target"]

    model = lgb.LGBMClassifier(
        **params
    )

    print(

        f"\n===== Fold {fold_id} "
        f"| {candidate['model_id']} ====="

    )

    print("Features:", features)

    print("Class Weight:", weight)

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

                verbose=False

            ),

            lgb.log_evaluation(period=100)

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

    threshold = baseline_utils.select_threshold(

        y_validation.to_numpy(),

        scores

    )

    metrics = baseline_utils.evaluate_scores(

        candidate["model_id"],

        candidate["feature_set"],

        y_validation.to_numpy(),

        scores,

        threshold

    )

    metrics.pop("baseline_id")
    metrics.pop("feature")

    metrics.update({

        "fold": fold_id,

        "model_id": candidate["model_id"],

        "feature_set": candidate["feature_set"],

        "feature_count": len(features),

        "weighted": candidate["weighted"],

        "scale_pos_weight": weight,

        "best_iteration": best_iteration,

        "train_rows": len(train),

        "train_positive": positive,

        "elapsed_seconds":
            time.perf_counter() - start_time

    })

    print("Best Iteration:", best_iteration)

    for name in [

        "average_precision",

        "ap_lift",

        "roc_auc",

        "precision",

        "recall",

        "f1",

        "balanced_accuracy"

    ]:

        print(
            f"{name}: {metrics[name]:.8f}"
        )

    del model
    del X_train
    del X_validation
    del scores

    gc.collect()

    return metrics


# ============================================================
# 9. Fold별 데이터 분포 검증
# ============================================================

def summarize_fold(
    fold_id,
    development,
    train_idx,
    retained_idx,
    val_idx,
    excluded_idx
):

    targets = development["target"].to_numpy()

    groups = development[
        "spatial_group"
    ].to_numpy()

    assert (
        len(retained_idx) + len(excluded_idx)
        == len(train_idx)
    )

    assert not np.intersect1d(

        groups[retained_idx],

        groups[val_idx]

    ).size

    assert len(
        np.unique(targets[retained_idx])
    ) == 2

    assert len(
        np.unique(targets[val_idx])
    ) == 2

    summary = {

        "fold": fold_id,

        "original_train": len(train_idx),

        "retained_train": len(retained_idx),

        "excluded_train": len(excluded_idx),

        "train_retention_ratio":
            len(retained_idx) / len(train_idx),

        "train_positive":
            int(targets[retained_idx].sum()),

        "train_positive_ratio":
            float(targets[retained_idx].mean()),

        "validation_rows": len(val_idx),

        "validation_positive":
            int(targets[val_idx].sum()),

        "validation_positive_ratio":
            float(targets[val_idx].mean()),

        "train_groups":
            len(np.unique(groups[retained_idx])),

        "validation_groups":
            len(np.unique(groups[val_idx])),

        "train_positive_groups":
            len(np.unique(
                groups[retained_idx][
                    targets[retained_idx] == 1
                ]
            )),

        "validation_positive_groups":
            len(np.unique(
                groups[val_idx][
                    targets[val_idx] == 1
                ]
            )),

        "spatial_violations": 0

    }

    print(f"\n===== Fold {fold_id} Distribution =====")

    for key, value in summary.items():

        print(f"{key}: {value}")

    return summary


# ============================================================
# 10. 전체 교차검증 결과 비교
# ============================================================

def summarize_results(metrics):

    assert len(metrics) == 24

    assert (
        metrics.groupby("model_id").size() == 3
    ).all()

    summary = metrics.groupby(

        ["model_id", "feature_set", "weighted"],

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
            "best_iteration",
            "median"
        )

    )

    baseline = summary.loc[

        summary["model_id"].isin(
            ["B01", "B02"]
        )

    ].sort_values(

        "mean_ap",

        ascending=False

    ).iloc[0]

    baseline_ap = float(
        baseline["mean_ap"]
    )

    summary["ap_improvement_pct"] = (

        (summary["mean_ap"] - baseline_ap)

        / baseline_ap

        * 100

    )

    # 각 Fold에서 동일한 Baseline과 비교
    baseline_fold_ap = metrics.loc[

        metrics["model_id"] == baseline["model_id"]

    ].set_index("fold")["average_precision"]

    wins = {}

    for model_id, frame in metrics.groupby("model_id"):

        model_fold_ap = frame.set_index(
            "fold"
        )["average_precision"]

        wins[model_id] = int(

            (model_fold_ap > baseline_fold_ap).sum()

        )

    summary["baseline_wins"] = (
        summary["model_id"].map(wins)
    )

    summary = summary.sort_values(

        ["mean_ap", "min_ap", "model_id"],

        ascending=[False, False, True]

    ).reset_index(drop=True)

    print("\n===== Fold AP Comparison =====")

    print(

        metrics.pivot(

            index="model_id",

            columns="fold",

            values="average_precision"

        ).to_string()

    )

    print("\n===== Final Spatial CV Summary =====")

    print(summary.to_string(index=False))

    best = summary.loc[

        summary["model_id"].str.startswith("CV")

    ].iloc[0]

    print("\n===== Best Spatial CV Candidate =====")

    print("Model:", best["model_id"])

    print("Feature Set:", best["feature_set"])

    print("Weighted:", best["weighted"])

    print("Mean AP:", best["mean_ap"])

    print("AP Std:", best["std_ap"])

    print("Minimum AP:", best["min_ap"])

    print("Mean AUC:", best["mean_auc"])

    print("Minimum AUC:", best["min_auc"])

    print("Best Baseline:", baseline["model_id"])

    print("Baseline Mean AP:", baseline_ap)

    print(
        "AP Improvement:",
        best["ap_improvement_pct"]
    )

    print(
        "Baseline Wins:",
        f"{best['baseline_wins']}/{N_SPLITS}"
    )

    return summary


# ============================================================
# 11. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Spatial Cross Validation =====")

    training_utils.print_memory(
        "Initial Memory"
    )

    output_paths = [

        METRICS_PATH,

        SUMMARY_PATH,

        FOLD_PATH,

        ASSIGNMENT_PATH

    ]

    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    config = training_utils.load_config()

    development = load_development_data(
        config
    )

    training_utils.print_memory(
        "After Development Load"
    )

    geometries = load_geometries(
        development
    )

    training_utils.print_memory(
        "After Geometry Load"
    )

    folds = create_folds(
        development
    )

    candidates = build_candidates(
        config
    )

    metrics_list = []

    fold_summaries = []

    assignments = development[

        ["link_id", "spatial_group", "target"]

    ].copy()

    for fold_id, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        print(
            f"\n===== Preparing Fold {fold_id} ====="
        )

        retained_idx, excluded_idx = (
            apply_spatial_buffer(

                geometries,

                train_idx,

                val_idx

            )
        )

        fold_summary = summarize_fold(

            fold_id,

            development,

            train_idx,

            retained_idx,

            val_idx,

            excluded_idx

        )

        fold_summaries.append(
            fold_summary
        )

        # Fold별 분할 정보 저장
        split_codes = np.full(

            len(development),

            -1,

            dtype=np.int8

        )

        split_codes[retained_idx] = 0

        split_codes[val_idx] = 1

        split_codes[excluded_idx] = 2

        assert (split_codes >= 0).all()

        assignments[f"fold_{fold_id}"] = (

            pd.Categorical.from_codes(

                split_codes,

                categories=[
                    "train",
                    "validation",
                    "excluded"
                ]

            )

        )

        train = development.iloc[
            retained_idx
        ].reset_index(drop=True)

        validation = development.iloc[
            val_idx
        ].reset_index(drop=True)

        metrics_list.extend(

            evaluate_baselines(

                fold_id,

                train,

                validation

            )

        )

        for candidate in candidates:

            metrics = train_candidate(

                fold_id,

                candidate,

                train,

                validation

            )

            metrics_list.append(metrics)

        del train
        del validation

        gc.collect()

        training_utils.print_memory(
            f"After Fold {fold_id}"
        )

    del geometries

    gc.collect()

    metrics = pd.DataFrame(
        metrics_list
    )

    fold_summary = pd.DataFrame(
        fold_summaries
    )

    summary = summarize_results(
        metrics
    )

    assert (

        baseline_utils.sha256_file(
            training_utils.DATASET_PATH
        )

        == config["final_dataset_sha256"]

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

    fold_summary.to_csv(

        FOLD_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    assignments.to_parquet(

        ASSIGNMENT_PATH,

        index=False

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

    print("CV Folds: 3")

    print("LightGBM Training Runs: 18")

    print("Baseline Evaluations: 6")

    print("Spatial Buffer: 400 m")

    print("Spatial Violations: 0")

    print("Dataset Hash Verification: PASS")

    print("Test Set Frozen: Yes")

    print("Test Evaluation: Not Performed")

    print("Final Model Selected: No")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()