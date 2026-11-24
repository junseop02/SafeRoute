from pathlib import Path
import json
import hashlib

import geopandas as gpd
import pandas as pd
import numpy as np
import joblib


# 1. 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"

DATASET_PATH = DATA_DIR / "accident_risk_dataset.parquet"

MODEL_PATH = (
    ROOT / "models/accident_risk_lightgbm_final.joblib"
)

SERVICE_ROAD_PATH = ROOT / "source/links.geojson"

LEGACY_ROAD_PATH = (
    ROOT / "data/legacy/processed/links_cheonan_asan.parquet"
)

SERVICE_SCORE_PATH = ROOT / "data/link_scores.json"

CONFIG_PATH = (
    DATA_DIR / "results/final_frozen_config.json"
)

OUTPUT_JSON = DATA_DIR / "link_risk_scores.json"

OUTPUT_PARQUET = DATA_DIR / "link_risk_scores.parquet"

OUTPUT_METADATA = DATA_DIR / "link_risk_metadata.json"


# 모델 입력 피처
FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_intersect_200m",
    "n_branch"
]


# 파일 해시 계산
def calculate_sha256(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# 2. 데이터 및 모델 불러오기
print("===== Load Data and Model =====")

dataset = pd.read_parquet(DATASET_PATH)

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    config = json.load(file)

service_roads = gpd.read_file(
    SERVICE_ROAD_PATH
)

legacy_roads = gpd.read_parquet(
    LEGACY_ROAD_PATH
)

with open(
    SERVICE_SCORE_PATH,
    "r",
    encoding="utf-8"
) as file:

    service_scores = pd.DataFrame(
        json.load(file)
    )


print("Dataset Links:", len(dataset))
print("Service Roads:", len(service_roads))
print("Legacy Roads:", len(legacy_roads))
print("Service Scores:", len(service_scores))
print("Model Candidate:", bundle["candidate"])


# 3. 링크 ID 검증
print("\n===== Link ID Validation =====")

legacy_roads = legacy_roads.rename(
    columns={"LINK_ID": "link_id"}
)

for name, frame in [
    ("Dataset", dataset),
    ("Service Roads", service_roads),
    ("Legacy Roads", legacy_roads),
    ("Service Scores", service_scores)
]:

    if frame["link_id"].isna().any():
        raise ValueError(
            f"{name}: Missing Link IDs"
        )

    frame["link_id"] = (
        frame["link_id"].astype(str)
    )

    if frame["link_id"].duplicated().any():
        raise ValueError(
            f"{name}: Duplicate Link IDs"
        )


dataset_ids = set(dataset["link_id"])

service_road_ids = set(
    service_roads["link_id"]
)

legacy_road_ids = set(
    legacy_roads["link_id"]
)

service_score_ids = set(
    service_scores["link_id"]
)

if not (
    dataset_ids
    == service_road_ids
    == legacy_road_ids
    == service_score_ids
):
    raise ValueError(
        "Link ID sets are different."
    )

print("All ID Sets Identical: True")
print("Unique Links:", len(dataset_ids))


# 4. 도로 형상 비교
print("\n===== Geometry Compatibility =====")

service_geometry = (
    service_roads
    .to_crs(epsg=5179)
    .set_index("link_id")
    .sort_index()
    .geometry
)

legacy_geometry = (
    legacy_roads
    .to_crs(epsg=5179)
    .set_index("link_id")
    .sort_index()
    .geometry
)

# 도로 형상 간 Hausdorff 거리
geometry_difference = (
    service_geometry.hausdorff_distance(
        legacy_geometry
    )
)

if not np.isfinite(
    geometry_difference.to_numpy()
).all():
    raise ValueError(
        "Invalid geometry differences."
    )

print(
    "Geometry Difference Mean (m):",
    geometry_difference.mean()
)

print(
    "Geometry Difference Median (m):",
    geometry_difference.median()
)

print(
    "Geometry Difference P95 (m):",
    geometry_difference.quantile(0.95)
)

print(
    "Geometry Difference Max (m):",
    geometry_difference.max()
)

for tolerance in [0.01, 0.1, 1.0, 5.0]:

    count = int(
        (geometry_difference > tolerance).sum()
    )

    print(
        f"Links Above {tolerance}m:",
        count
    )

print("\nLargest Geometry Differences:")

print(
    geometry_difference
    .nlargest(5)
    .to_string()
)

# 임시 형상 호환성 검사 기준
if (geometry_difference > 1.0).any():

    raise ValueError(
        "Geometry difference exceeds 1m. "
        "Inspect road data before generating scores."
    )


# 5. 모델 입력 피처 검증
print("\n===== Model Feature Validation =====")

if bundle["features"] != FEATURES:

    raise ValueError(
        "Model feature order mismatch."
    )

missing_features = [
    feature for feature in FEATURES
    if feature not in dataset.columns
]

if missing_features:

    raise ValueError(
        f"Missing features: {missing_features}"
    )

X = dataset[FEATURES].copy()

for feature in FEATURES:

    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )

if not np.isfinite(
    X.to_numpy(dtype=float)
).all():

    raise ValueError(
        "Missing or infinite feature values."
    )


# 도로 등급 범주 설정
road_rank_categories = (
    bundle["road_rank_categories"]
)

unknown_categories = set(
    X["road_rank"].unique()
) - set(road_rank_categories)

if unknown_categories:

    raise ValueError(
        f"Unknown road ranks: {unknown_categories}"
    )

X["road_rank"] = pd.Categorical(
    X["road_rank"],
    categories=road_rank_categories
)

print("Feature Order:", X.columns.tolist())

print(
    "Road Rank Categories:",
    road_rank_categories
)

print(
    "Unknown Road Rank Categories:",
    len(unknown_categories)
)

print("Missing or Infinite Features: 0")


# 6. 전체 도로 링크 예측
print("\n===== Generate Link Predictions =====")

classes = model.classes_.tolist()

if 1 not in classes:

    raise ValueError(
        "Positive class is missing."
    )

positive_index = classes.index(1)

predictions = model.predict_proba(
    X
)[:, positive_index]

if len(predictions) != len(dataset):

    raise ValueError(
        "Prediction count mismatch."
    )

if not np.isfinite(predictions).all():

    raise ValueError(
        "Invalid prediction values."
    )

if (
    (predictions < 0).any()
    or (predictions > 1).any()
):

    raise ValueError(
        "Prediction values outside [0, 1]."
    )


# 기존 개발 데이터에서 결정한 임계값
threshold = float(bundle["threshold"])

baseline_threshold = float(
    config["baseline_threshold"]
)

if not 0 <= threshold <= 1:

    raise ValueError(
        "Invalid model threshold."
    )

if not np.isfinite(baseline_threshold):

    raise ValueError(
        "Invalid baseline threshold."
    )

predicted_labels = (
    predictions >= threshold
).astype(int)

baseline_values = (
    dataset["n_intersect_200m"]
    .to_numpy()
)

baseline_labels = (
    baseline_values >= baseline_threshold
).astype(int)

print("Total Predictions:", len(predictions))
print("Model Threshold:", threshold)

print(
    "Baseline Threshold:",
    baseline_threshold
)


# 7. 신규 점수 데이터 생성
print("\n===== Build Link Risk Scores =====")

result = pd.DataFrame({

    "link_id":
        dataset["link_id"].to_numpy(),

    "hotspot_score":
        predictions,

    "hotspot_predicted_label":
        predicted_labels,

    "intersection_baseline":
        baseline_values.astype(int),

    "intersection_predicted_label":
        baseline_labels

})

result = result.sort_values(
    "link_id"
).reset_index(drop=True)

print("Output Shape:", result.shape)

print("\nSample:")

print(
    result.head(10).to_string(index=False)
)


# 8. 기본 예측 점수 통계
print("\n===== Prediction Summary =====")

print(
    result["hotspot_score"].describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    ).to_string()
)

print("\nLightGBM Predicted Labels:")

print(
    result["hotspot_predicted_label"]
    .value_counts()
    .sort_index()
)

print("\nIntersection Predicted Labels:")

print(
    result["intersection_predicted_label"]
    .value_counts()
    .sort_index()
)

print(
    "\nUnique Prediction Scores:",
    result["hotspot_score"].nunique()
)


# 9. 신규 데이터 저장
print("\n===== Save Output Files =====")

result.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

records = result.to_dict(
    orient="records"
)

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        records,
        file,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":")
    )


# 10. 모델 및 데이터 생성 정보 저장
metadata = {

    "model_candidate":
        bundle["candidate"],

    "model_path":
        MODEL_PATH.relative_to(ROOT).as_posix(),

    "model_sha256":
        calculate_sha256(MODEL_PATH),

    "dataset_sha256":
        calculate_sha256(DATASET_PATH),

    "service_road_sha256":
        calculate_sha256(SERVICE_ROAD_PATH),

    "legacy_road_sha256":
        calculate_sha256(LEGACY_ROAD_PATH),

    "service_score_sha256":
        calculate_sha256(SERVICE_SCORE_PATH),

    "output_json_sha256":
        calculate_sha256(OUTPUT_JSON),

    "output_parquet_sha256":
        calculate_sha256(OUTPUT_PARQUET),

    "features":
        FEATURES,

    "road_rank_categories":
        [int(value) for value in road_rank_categories],

    "total_links":
        len(result),

    "model_threshold":
        threshold,

    "intersection_baseline_threshold":
        baseline_threshold,

    "geometry_max_difference_m":
        float(geometry_difference.max()),

    "score_meaning":
        bundle["score_meaning"],

    "deployment_status":
        "candidate_not_deployed",

    "contains_training_links":
        True,

    "limitations": [
        "Not a calibrated accident probability.",
        "Not a direct measure of driving difficulty.",
        "Contains training and evaluation links.",
        "Not an independent evaluation dataset."
    ]

}

with open(
    OUTPUT_METADATA,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=4,
        allow_nan=False
    )


# 11. 저장 결과 재검증
print("\n===== Saved Data Verification =====")

with open(
    OUTPUT_JSON,
    "r",
    encoding="utf-8"
) as file:

    reloaded_json = pd.DataFrame(
        json.load(file)
    )

reloaded_parquet = pd.read_parquet(
    OUTPUT_PARQUET
)

pd.testing.assert_frame_equal(
    result,
    reloaded_json,
    check_dtype=False,
    check_exact=True
)

pd.testing.assert_frame_equal(
    result,
    reloaded_parquet,
    check_dtype=False,
    check_exact=True
)

if set(reloaded_json["link_id"]) != dataset_ids:

    raise ValueError(
        "Saved Link IDs mismatch."
    )

print("JSON Records:", len(reloaded_json))

print(
    "Parquet Records:",
    len(reloaded_parquet)
)

print("JSON Verification: Passed")
print("Parquet Verification: Passed")
print("Saved Link IDs: Identical")


# 12. 최종 결과
print("\n===== Output Files =====")

for path in [
    OUTPUT_JSON,
    OUTPUT_PARQUET,
    OUTPUT_METADATA
]:

    print(
        f"{path.name}: "
        f"{path.stat().st_size:,} bytes"
    )

    print("Path:", path)


print("\n===== Link Risk Generation Completed =====")

print("Total Links:", len(result))
print("Model Retraining: Not Performed")
print("Existing Service Files: Not Modified")
print("Existing Model: Not Modified")
print("Model Deployment: Not Performed")