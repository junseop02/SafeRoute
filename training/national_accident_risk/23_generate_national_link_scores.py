from pathlib import Path
import gc
import json
import time
import importlib

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "national_accident_risk"
    / "national_accident_risk_lightgbm_final.joblib"
)

ROAD_PATH = (
    PROCESSED_DIR
    / "national_road_features.parquet"
)

FROZEN_CONFIG_PATH = (
    RESULTS_DIR
    / "national_final_frozen_config.json"
)

TEST_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "national_final_test_predictions.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "national_link_risk_scores.parquet"
)

TEMP_PATH = (
    PROCESSED_DIR
    / "national_link_risk_scores.tmp.parquet"
)

METADATA_PATH = (
    RESULTS_DIR
    / "national_link_risk_scores_metadata.json"
)

BATCH_SIZE = 50000

EXPECTED_ROWS = 1562356

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)


# ============================================================
# 2. 최종 모델 및 설정 불러오기
# ============================================================

def load_model():

    bundle = joblib.load(MODEL_PATH)

    with FROZEN_CONFIG_PATH.open(
        "r", encoding="utf-8"
    ) as file:

        frozen = json.load(file)

    assert bundle["candidate_id"] == "T09"

    assert bundle["features"] == frozen["features"]

    assert bundle["features"] == (
        bundle["model"].feature_name_
    )

    assert bundle["road_rank_categories"] == (
        frozen["road_rank_categories"]
    )

    assert bundle["threshold"] == (
        frozen["model_threshold"]
    )

    assert bundle["n_estimators"] == 269

    assert bundle["model"].n_estimators_ == 269

    assert bundle["dataset_sha256"] == (
        frozen["dataset_sha256"]
    )

    assert bundle["test_data_sha256"] == (
        frozen["test_data_sha256"]
    )

    assert bundle["score_meaning"] == (
        "historical_hotspot_membership_score"
    )

    assert bundle["deployment_status"] == (
        "candidate_not_deployed"
    )

    assert bundle["comparison"]["test_evaluated"]

    assert bundle["comparison"][
        "test_beats_baseline_ap"
    ]

    print("\n===== Final Model Configuration =====")

    print("Candidate:", bundle["candidate_id"])
    print("Features:", bundle["features"])
    print("Iterations:", bundle["n_estimators"])
    print("Threshold:", bundle["threshold"])
    print("Training Rows:", bundle["training_size"])

    print(
        "Test AP:",
        bundle["comparison"]["test_average_precision"]
    )

    print("Model Configuration Validation: PASS")

    return bundle, frozen


# ============================================================
# 3. 전국 도로 데이터 호환성 검증
# ============================================================

def validate_road_file(bundle):

    road = pq.ParquetFile(ROAD_PATH)

    assert road.metadata.num_rows == EXPECTED_ROWS

    required_columns = [
        "link_id",
        *bundle["features"],
        "geometry_length_m"
    ]

    missing = set(required_columns) - set(
        road.schema_arrow.names
    )

    assert not missing, (
        f"Missing Columns: {missing}"
    )

    print("\n===== National Road Dataset =====")

    print("Rows:", road.metadata.num_rows)
    print("Row Groups:", road.num_row_groups)
    print("Required Columns:", required_columns)
    print("Missing Columns:", sorted(missing))
    print("Schema Compatibility: PASS")

    return road, required_columns


# ============================================================
# 4. 출력 Parquet 스키마 설정
# ============================================================

def create_output_schema():

    schema = pa.schema([

        ("link_id", pa.string()),

        ("lanes", pa.float64()),

        ("max_spd", pa.float64()),

        ("road_rank", pa.int32()),

        ("length", pa.float64()),

        ("geometry_length_m", pa.float64()),

        ("n_branch", pa.int32()),

        ("n_intersect_200m", pa.int32()),

        ("intersection_nodes_200m", pa.int32()),

        ("hotspot_score", pa.float64()),

        ("hotspot_predicted_label", pa.int8())

    ])

    return schema.with_metadata({

        b"model_id": b"T09",

        b"score_meaning":
            b"historical_hotspot_membership_score",

        b"score_range": b"0_to_1",

        b"deployment_status": b"not_deployed"

    })


# ============================================================
# 5. 배치 데이터 검증 및 예측
# ============================================================

def predict_batch(batch, bundle):

    df = batch.to_pandas()

    features = bundle["features"]

    assert df["link_id"].notna().all()

    assert df["link_id"].is_unique

    assert df[features].notna().all().all()

    assert np.isfinite(
        df[features].to_numpy(dtype=np.float64)
    ).all()

    assert np.isfinite(
        df["geometry_length_m"].to_numpy()
    ).all()

    assert (df["lanes"] > 0).all()

    assert (df["max_spd"] > 0).all()

    assert (df["length"] > 0).all()

    assert (df["geometry_length_m"] > 0).all()

    assert (df["n_branch"] >= 0).all()

    assert (df["n_intersect_200m"] >= 0).all()

    assert (df["intersection_nodes_200m"] >= 0).all()

    assert df["road_rank"].isin(
        bundle["road_rank_categories"]
    ).all()

    # 학습 당시 피처 순서 유지
    X = df[features].copy()

    # 학습 당시와 동일한 범주형 자료형 적용
    X["road_rank"] = pd.Categorical(
        X["road_rank"],
        categories=bundle["road_rank_categories"],
        ordered=False
    )

    assert X["road_rank"].notna().all()

    # 양성 클래스의 연속 예측 점수
    scores = bundle["model"].predict_proba(
        X
    )[:, 1]

    assert len(scores) == len(df)

    assert np.isfinite(scores).all()

    assert (
        (scores >= 0) & (scores <= 1)
    ).all()

    # OOF에서 확정한 기존 임계값 사용
    predicted_labels = (
        scores >= bundle["threshold"]
    ).astype(np.int8)

    df["road_rank"] = (
        df["road_rank"].astype(np.int32)
    )

    df["hotspot_score"] = scores

    df["hotspot_predicted_label"] = (
        predicted_labels
    )

    output_columns = [

        "link_id",

        "lanes",

        "max_spd",

        "road_rank",

        "length",

        "geometry_length_m",

        "n_branch",

        "n_intersect_200m",

        "intersection_nodes_200m",

        "hotspot_score",

        "hotspot_predicted_label"

    ]

    return df[output_columns]


# ============================================================
# 6. 전국 도로 예측 점수 생성
# ============================================================

def generate_scores(
    road,
    required_columns,
    bundle
):

    schema = create_output_schema()

    processed_rows = 0
    batch_count = 0

    score_sum = 0.0

    score_min = float("inf")
    score_max = float("-inf")

    predicted_positive = 0

    print("\n===== National Link Score Generation =====")

    with pq.ParquetWriter(
        TEMP_PATH,
        schema=schema,
        compression="snappy"
    ) as writer:

        for batch in road.iter_batches(
            batch_size=BATCH_SIZE,
            columns=required_columns
        ):

            result = predict_batch(
                batch,
                bundle
            )

            table = pa.Table.from_pandas(
                result,
                schema=schema,
                preserve_index=False,
                safe=True
            )

            writer.write_table(table)

            scores = result[
                "hotspot_score"
            ].to_numpy()

            processed_rows += len(result)

            batch_count += 1

            score_sum += float(scores.sum())

            score_min = min(
                score_min,
                float(scores.min())
            )

            score_max = max(
                score_max,
                float(scores.max())
            )

            predicted_positive += int(
                result["hotspot_predicted_label"].sum()
            )

            print(
                f"Batch {batch_count:02d} | "
                f"Rows: {len(result):,} | "
                f"Processed: {processed_rows:,} / "
                f"{EXPECTED_ROWS:,} | "
                f"{processed_rows / EXPECTED_ROWS * 100:.2f}%",
                flush=True
            )

            del result, table, scores, batch

            gc.collect()

    assert processed_rows == EXPECTED_ROWS

    summary = {

        "total_links": processed_rows,

        "batch_count": batch_count,

        "score_min": score_min,

        "score_max": score_max,

        "score_mean": score_sum / processed_rows,

        "predicted_positive": predicted_positive,

        "predicted_negative":
            processed_rows - predicted_positive,

        "predicted_positive_ratio":
            predicted_positive / processed_rows

    }

    print("\n===== Generation Summary =====")

    for key, value in summary.items():

        print(f"{key}: {value}")

    return summary


# ============================================================
# 7. 전체 링크 ID 및 예측 결과 검증
# ============================================================

def validate_output(bundle, summary):

    print("\n===== Output Validation =====")

    source_ids = pd.read_parquet(
        ROAD_PATH,
        columns=["link_id"]
    )

    result = pd.read_parquet(
        TEMP_PATH,
        columns=[
            "link_id",
            "hotspot_score",
            "hotspot_predicted_label"
        ]
    )

    assert len(source_ids) == EXPECTED_ROWS

    assert len(result) == EXPECTED_ROWS

    assert source_ids["link_id"].is_unique

    assert result["link_id"].is_unique

    # 원본 링크 순서 및 ID 완전 일치
    np.testing.assert_array_equal(
        source_ids["link_id"].to_numpy(),
        result["link_id"].to_numpy()
    )

    assert result.notna().all().all()

    scores = result["hotspot_score"].to_numpy()

    assert np.isfinite(scores).all()

    assert (
        (scores >= 0) & (scores <= 1)
    ).all()

    expected_labels = (
        scores >= bundle["threshold"]
    ).astype(np.int8)

    np.testing.assert_array_equal(
        expected_labels,
        result["hotspot_predicted_label"].to_numpy()
    )

    assert int(expected_labels.sum()) == (
        summary["predicted_positive"]
    )

    print("Source Link IDs:", len(source_ids))
    print("Output Link IDs:", len(result))
    print("Duplicate Link IDs: 0")
    print("Missing Link IDs: 0")
    print("Invalid Scores: 0")
    print("Link Order Consistency: PASS")
    print("Classification Consistency: PASS")

    del source_ids, expected_labels, scores

    gc.collect()

    return result


# ============================================================
# 8. 기존 Test 예측 결과와 일치 여부 확인
# ============================================================

def validate_test_predictions(result):

    previous = pd.read_parquet(
        TEST_PREDICTIONS_PATH,
        columns=[
            "link_id",
            "hotspot_score",
            "hotspot_predicted_label"
        ]
    )

    assert len(previous) == 234353

    assert previous["link_id"].is_unique

    matched = previous.merge(

        result,

        on="link_id",

        how="left",

        validate="one_to_one",

        suffixes=("_previous", "_national")

    )

    assert len(matched) == 234353

    assert matched.notna().all().all()

    previous_scores = matched[
        "hotspot_score_previous"
    ].to_numpy()

    national_scores = matched[
        "hotspot_score_national"
    ].to_numpy()

    np.testing.assert_allclose(
        previous_scores,
        national_scores,
        rtol=1e-12,
        atol=1e-12
    )

    np.testing.assert_array_equal(
        matched[
            "hotspot_predicted_label_previous"
        ].to_numpy(),

        matched[
            "hotspot_predicted_label_national"
        ].to_numpy()
    )

    max_difference = float(
        np.max(
            np.abs(
                previous_scores - national_scores
            )
        )
    )

    print("\n===== Previous Test Prediction Consistency =====")

    print("Matched Test Links:", len(matched))

    print("Maximum Score Difference:", max_difference)

    print("Score Consistency: PASS")
    print("Label Consistency: PASS")
    print("Test Re-evaluation: Not Performed")

    del previous, matched
    del previous_scores, national_scores

    gc.collect()

    return max_difference


# ============================================================
# 9. 예측 결과 메타데이터 저장
# ============================================================

def save_metadata(
    bundle,
    frozen,
    summary,
    model_hash,
    road_hash,
    frozen_hash,
    max_difference
):

    metadata = {

        "model_id": bundle["candidate_id"],

        "model_type": "LightGBMClassifier",

        "features": bundle["features"],

        "categorical_features":
            bundle["categorical_features"],

        "road_rank_categories":
            bundle["road_rank_categories"],

        "score_meaning":
            "historical_hotspot_membership_score",

        "score_range": [0.0, 1.0],

        "classification_threshold":
            float(bundle["threshold"]),

        "threshold_selection":
            frozen["threshold_selection"],

        "training_rows":
            bundle["training_size"],

        "n_estimators":
            bundle["n_estimators"],

        "batch_size": BATCH_SIZE,

        "prediction_summary": summary,

        "model_sha256": model_hash,

        "road_features_sha256": road_hash,

        "frozen_config_sha256": frozen_hash,

        "training_dataset_sha256":
            bundle["dataset_sha256"],

        "output_sha256":
            baseline_utils.sha256_file(OUTPUT_PATH),

        "test_prediction_consistency": {

            "matched_links": 234353,

            "max_absolute_difference":
                max_difference,

            "status": "PASS"

        },

        "all_national_links_scored": True,

        "test_re_evaluated": False,

        "route_score_formula_selected": False,

        "deployment_status": "not_deployed",

        "limitations": bundle["limitations"]

    }

    with METADATA_PATH.open(
        "x", encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# 10. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Link Risk Score Generation =====")

    training_utils.print_memory(
        "Initial Memory"
    )

    # 기존 결과 파일 보호
    for path in [
        OUTPUT_PATH,
        TEMP_PATH,
        METADATA_PATH
    ]:

        if path.exists():

            raise FileExistsError(path)

    # 원본 파일 해시 기록
    model_hash = baseline_utils.sha256_file(
        MODEL_PATH
    )

    road_hash = baseline_utils.sha256_file(
        ROAD_PATH
    )

    frozen_hash = baseline_utils.sha256_file(
        FROZEN_CONFIG_PATH
    )

    # 모델 및 도로 데이터 불러오기
    bundle, frozen = load_model()

    road, required_columns = validate_road_file(
        bundle
    )

    # 전국 도로 점수 생성
    summary = generate_scores(
        road,
        required_columns,
        bundle
    )

    training_utils.print_memory(
        "After Score Generation"
    )

    # 전체 결과 검증
    result = validate_output(
        bundle,
        summary
    )

    # 기존 Test 예측값과 비교
    max_difference = validate_test_predictions(
        result
    )

    del result

    gc.collect()

    # 원본 모델 및 데이터 변경 여부 검증
    assert baseline_utils.sha256_file(
        MODEL_PATH
    ) == model_hash

    assert baseline_utils.sha256_file(
        ROAD_PATH
    ) == road_hash

    assert baseline_utils.sha256_file(
        FROZEN_CONFIG_PATH
    ) == frozen_hash

    # 모든 검증 통과 후 최종 파일명 적용
    TEMP_PATH.rename(OUTPUT_PATH)

    save_metadata(
        bundle,
        frozen,
        summary,
        model_hash,
        road_hash,
        frozen_hash,
        max_difference
    )

    print("\n===== Saved Results =====")

    for path in [
        OUTPUT_PATH,
        METADATA_PATH
    ]:

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

    print("Model: T09")
    print("Total National Links:", EXPECTED_ROWS)
    print("Scored Links:", summary["total_links"])
    print("Score Coverage: 100%")
    print("Model Hash Verification: PASS")
    print("Road Data Hash Verification: PASS")
    print("Frozen Config Verification: PASS")
    print("Previous Test Prediction Consistency: PASS")
    print("Additional Model Training: Not Performed")
    print("Test Re-evaluation: Not Performed")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()