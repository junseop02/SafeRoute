
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. 기본 설정
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = (
    ROOT / "data" / "national_accident_risk" / "processed"
)

RESULTS = PROCESSED / "results"

MODEL_PATH = (
    ROOT / "models" / "national_accident_risk"
    / "national_accident_risk_lightgbm_final.joblib"
)

MODEL_CONFIG_PATH = (
    RESULTS / "national_final_frozen_config.json"
)

SCORE_PATH = (
    PROCESSED / "national_link_risk_scores.parquet"
)

SCORE_METADATA_PATH = (
    RESULTS / "national_link_risk_scores_metadata.json"
)

ROUTE_PATH = (
    RESULTS / "national_final_policy_routes.parquet"
)

SELECTION_PATH = (
    RESULTS / "national_final_policy_selections.parquet"
)

COMPARISON_CONFIG_PATH = (
    RESULTS / "national_final_policy_config.json"
)

OUTPUT_CONFIG = (
    RESULTS / "national_route_policy_P01.json"
)

OUTPUT_INTERFACE = (
    RESULTS / "national_service_interface_P01.json"
)

OUTPUT_VALIDATION = (
    RESULTS / "national_route_policy_validation.csv"
)

OUTPUT_SUMMARY = (
    RESULTS / "national_route_policy_P01_summary.csv"
)

MODEL_ID = "T09"
POLICY_ID = "P01"

W_RISK = 0.80
W_NARROW = 0.10
W_INTERSECTION = 0.10

INTERSECTION_REFERENCE = 100.0

MIN_COVERAGE = 0.80
MIN_CANDIDATES = 3

DETOUR_FACTORS = [1.5, 2.0, 3.0]

FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_branch",
    "n_intersect_200m",
    "intersection_nodes_200m"
]

VALIDATION = []


# ==================================================
# 2. 공통 함수
# ==================================================

def load_json(path):

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def check(name, condition):

    passed = bool(condition)

    VALIDATION.append({
        "check": name,
        "status": "PASS" if passed else "FAIL"
    })

    print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise AssertionError(name)


def check_outputs():

    for path in [
        OUTPUT_CONFIG,
        OUTPUT_INTERFACE,
        OUTPUT_VALIDATION,
        OUTPUT_SUMMARY
    ]:

        if path.exists():
            raise FileExistsError(path)


# ==================================================
# 3. P01 추천 점수 계산
# ==================================================

def calculate_burden(risk, wide_ratio, intersections):

    risk = np.asarray(risk, dtype=float)

    wide_ratio = np.asarray(
        wide_ratio, dtype=float
    )

    intersections = np.asarray(
        intersections, dtype=float
    )

    if not np.isfinite(risk).all():
        raise ValueError("Invalid model score")

    if not np.isfinite(wide_ratio).all():
        raise ValueError("Invalid wide road ratio")

    if not np.isfinite(intersections).all():
        raise ValueError("Invalid intersection count")

    if np.any((risk < 0) | (risk > 1)):
        raise ValueError("Model score outside 0~1")

    if np.any((wide_ratio < 0) | (wide_ratio > 1)):
        raise ValueError("Wide road ratio outside 0~1")

    if np.any(intersections < 0):
        raise ValueError("Negative intersection count")

    narrow = 1.0 - wide_ratio

    intersection_burden = np.clip(
        intersections / INTERSECTION_REFERENCE,
        0.0,
        1.0
    )

    burden = (
        W_RISK * risk
        + W_NARROW * narrow
        + W_INTERSECTION * intersection_burden
    )

    return np.clip(burden, 0.0, 1.0)


def calculate_display_score(
    risk, wide_ratio, intersections
):

    burden = calculate_burden(
        risk, wide_ratio, intersections
    )

    return 100.0 * (1.0 - burden)


# ==================================================
# 4. 모델 및 설정 검증
# ==================================================

def validate_model():

    model_config = load_json(MODEL_CONFIG_PATH)

    metadata = load_json(SCORE_METADATA_PATH)

    comparison = load_json(COMPARISON_CONFIG_PATH)

    check(
        "Model ID",
        model_config["candidate_id"] == MODEL_ID
    )

    check(
        "Model Features",
        model_config["features"] == FEATURES
    )

    check(
        "Final Iterations",
        model_config["n_estimators"] == 269
    )

    check(
        "Training Rows",
        model_config["training_rows"] == 1123804
    )

    check(
        "Model File Hash",
        sha256_file(MODEL_PATH)
        == metadata["model_sha256"]
    )

    check(
        "Score File Hash",
        sha256_file(SCORE_PATH)
        == metadata["output_sha256"]
    )

    check(
        "Score Model ID",
        metadata["model_id"] == MODEL_ID
    )

    check(
        "Score Meaning",
        metadata["score_meaning"]
        == "historical_hotspot_membership_score"
    )

    check(
        "All National Links Scored",
        metadata["all_national_links_scored"] is True
    )

    check(
        "National Link Count",
        metadata["prediction_summary"]["total_links"]
        == 1562356
    )

    check(
        "Previous Policy Reproduction",
        comparison["previous_result_reproduction"]
        == "PASS"
    )

    check(
        "Previous Detour Violations",
        comparison["detour_violations"] == 0
    )

    check(
        "P01 Weight Configuration",
        comparison["policies"]["M06_P01"]
        == [0.8, 0.1, 0.1, 100]
    )

    check(
        "Previous Policy Not Selected",
        comparison["selected_policy"] is None
    )

    return model_config, metadata


# ==================================================
# 5. 추천 수식 단위 검증
# ==================================================

def validate_formula():

    print("\n===== Formula Validation =====")

    check(
        "Weight Sum",
        np.isclose(
            W_RISK + W_NARROW + W_INTERSECTION,
            1.0
        )
    )

    check(
        "Best Case Score",
        np.isclose(
            calculate_display_score(0, 1, 0),
            100
        )
    )

    check(
        "Worst Case Score",
        np.isclose(
            calculate_display_score(1, 0, 100),
            0
        )
    )

    check(
        "Intersection Clipping",
        np.isclose(
            calculate_display_score(0, 1, 305),
            90
        )
    )

    check(
        "Manual Calculation",
        np.isclose(
            calculate_display_score(.2, .8, 30),
            79
        )
    )

    check(
        "Higher Model Burden Lowers Score",
        calculate_display_score(.1, .8, 20)
        > calculate_display_score(.2, .8, 20)
    )

    check(
        "Wider Road Raises Score",
        calculate_display_score(.2, .9, 20)
        > calculate_display_score(.2, .5, 20)
    )

    check(
        "Higher Intersection Burden Lowers Score",
        calculate_display_score(.2, .8, 10)
        > calculate_display_score(.2, .8, 50)
    )

    invalid_cases = [
        (-.1, .8, 20),
        (1.1, .8, 20),
        (.2, -.1, 20),
        (.2, 1.1, 20),
        (.2, .8, -1),
        (np.nan, .8, 20)
    ]

    for index, values in enumerate(
        invalid_cases, start=1
    ):

        rejected = False

        try:
            calculate_burden(*values)

        except ValueError:
            rejected = True

        check(
            f"Invalid Input Rejection {index}",
            rejected
        )


# ==================================================
# 6. 기존 경로 실험 재현
# ==================================================

def reproduce_policy():

    print("\n===== Route Policy Reproduction =====")

    routes = pd.read_parquet(ROUTE_PATH)

    previous = pd.read_parquet(SELECTION_PATH)

    previous = previous.loc[
        (previous["scope"] == "new_full")
        & (previous["method"] == "M06_P01")
    ].copy()

    check(
        "Route Count",
        len(routes) == 6402
    )

    check(
        "Unique Route IDs",
        routes["route_id"].is_unique
    )

    check(
        "Eligible Route Count",
        int(routes["new_eligible"].sum()) == 6303
    )

    eligible = routes.loc[
        routes["new_eligible"]
    ].copy()

    eligible["policy_burden"] = calculate_burden(
        eligible["new_model_risk"],
        eligible["new_wide_ratio"],
        eligible["new_intersections"]
    )

    eligible["display_score"] = (
        100.0 * (1.0 - eligible["policy_burden"])
    )

    check(
        "Route Burden Range",
        eligible["policy_burden"].between(0, 1).all()
    )

    check(
        "Display Score Range",
        eligible["display_score"].between(0, 100).all()
    )

    records = []

    for factor in DETOUR_FACTORS:

        for (origin, destination), group in routes.groupby(
            ["origin_id", "destination_id"],
            sort=False
        ):

            shortest = float(group["distance_m"].min())

            candidates = group.loc[
                group["new_eligible"]
                & (
                    group["distance_m"]
                    <= shortest * factor + 1e-8
                )
            ].copy()

            candidates = candidates.sort_values(
                ["distance_m", "duration_s", "route_id"],
                kind="stable"
            )

            candidates = candidates.drop_duplicates(
                subset="geometry_sha256",
                keep="first"
            )

            if len(candidates) < MIN_CANDIDATES:
                continue

            burden = calculate_burden(
                candidates["new_model_risk"],
                candidates["new_wide_ratio"],
                candidates["new_intersections"]
            )

            index = int(
                np.argmin(np.round(burden, 12))
            )

            row = candidates.iloc[index]

            records.append({
                "origin_id": origin,
                "destination_id": destination,
                "detour_factor": factor,
                "route_id": row["route_id"],
                "candidate_count": len(candidates),
                "policy_burden": float(burden[index]),
                "display_score": float(
                    100.0 * (1.0 - burden[index])
                ),
                "raw_risk": row["new_model_risk"],
                "wide_road_ratio": row["new_wide_ratio"],
                "intersection_nodes": row[
                    "new_intersections"
                ],
                "distance_m": row["distance_m"],
                "duration_s": row["duration_s"],
                "coverage": row["new_coverage"],
                "detour_ratio": (
                    row["distance_m"] / shortest
                )
            })

    selected = pd.DataFrame(records)

    keys = [
        "origin_id",
        "destination_id",
        "detour_factor"
    ]

    check(
        "Unique OD Selections",
        not selected.duplicated(keys).any()
    )

    merged = selected.merge(
        previous,
        on=keys,
        suffixes=("_new", "_previous"),
        validate="one_to_one"
    )

    check(
        "Selection Count",
        len(merged) == len(selected) == len(previous)
        == 2214
    )

    check(
        "Exact Route Selection Reproduction",
        (
            merged["route_id_new"]
            == merged["route_id_previous"]
        ).all()
    )

    for column in [
        "raw_risk",
        "wide_road_ratio",
        "intersection_nodes",
        "distance_m",
        "duration_s",
        "coverage",
        "detour_ratio"
    ]:

        check(
            f"Reproduce {column}",
            np.allclose(
                merged[f"{column}_new"],
                merged[f"{column}_previous"],
                rtol=1e-8,
                atol=1e-8
            )
        )

    check(
        "Minimum Coverage",
        (selected["coverage"] >= MIN_COVERAGE).all()
    )

    check(
        "Minimum Candidate Count",
        (
            selected["candidate_count"]
            >= MIN_CANDIDATES
        ).all()
    )

    check(
        "No Detour Violations",
        (
            selected["detour_ratio"]
            <= selected["detour_factor"] + 1e-8
        ).all()
    )

    summary = selected.groupby(
        "detour_factor",
        as_index=False
    ).agg(
        od_count=("route_id", "count"),
        mean_policy_burden=("policy_burden", "mean"),
        mean_display_score=("display_score", "mean"),
        mean_raw_risk=("raw_risk", "mean"),
        mean_wide_road_ratio=("wide_road_ratio", "mean"),
        mean_intersection_nodes=(
            "intersection_nodes", "mean"
        ),
        mean_distance_m=("distance_m", "mean"),
        mean_duration_s=("duration_s", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_detour_ratio=("detour_ratio", "mean"),
        max_detour_ratio=("detour_ratio", "max")
    )

    check(
        "OD Counts by Detour Factor",
        summary["od_count"].tolist()
        == [629, 791, 794]
    )

    print("\n===== P01 Final Summary =====")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return summary


# ==================================================
# 7. 최종 정책 설정
# ==================================================

def build_policy_config(
    model_config, metadata, source_hashes
):

    return {
        "model_id": MODEL_ID,
        "policy_id": POLICY_ID,
        "policy_version": "P01-v1.0",
        "policy_status": "selected_candidate",
        "model_type": "LightGBM_binary_classifier",
        "features": FEATURES,
        "n_estimators": 269,
        "training_rows": 1123804,
        "national_link_count": 1562356,
        "score_meaning": (
            "historical_hotspot_membership_score"
        ),
        "model_score_range": [0, 1],
        "normalization": "raw",
        "aggregation": "matched_length_weighted_mean",
        "weights": {
            "risk": W_RISK,
            "narrow": W_NARROW,
            "intersection": W_INTERSECTION
        },
        "wide_road_min_lanes": 2,
        "intersection_feature": (
            "intersection_nodes_200m"
        ),
        "intersection_reference": (
            INTERSECTION_REFERENCE
        ),
        "intersection_clipping": [0, 1],
        "burden_formula": (
            "0.8*R + 0.1*(1-W) "
            "+ 0.1*clip(C/100,0,1)"
        ),
        "display_score_formula": "100*(1-B)",
        "display_score_range": [0, 100],
        "higher_display_score_is_better": True,
        "minimum_coverage": MIN_COVERAGE,
        "minimum_distinct_candidates": MIN_CANDIDATES,
        "detour_basis": "shortest_candidate_distance",
        "detour_options": [1.5, 2.0, 3.0, "unlimited"],
        "proposed_default_detour_factor": 2.0,
        "unlimited_detour_evaluated": False,
        "ranking": [
            "burden_ascending",
            "distance_ascending",
            "duration_ascending",
            "route_id_ascending"
        ],
        "offline_burden_rounding_decimals": 12,
        "missing_score_policy": "do_not_impute_zero",
        "insufficient_coverage_policy": "no_score",
        "insufficient_candidates_policy": (
            "no_recommendation"
        ),
        "classification_threshold": (
            model_config["model_threshold"]
        ),
        "classification_threshold_used_for_ranking": False,
        "source_sha256": source_hashes,
        "model_sha256": metadata["model_sha256"],
        "policy_selected": True,
        "selection_authority": "user_confirmed",
        "route_score_formula_selected": True,
        "additional_model_training": False,
        "test_re_evaluated": False,
        "deployment_status": "not_deployed",
        "limitations": [
            "Not an actual accident probability.",
            "Not a measured novice driver difficulty.",
            "Weights are policy parameters, not learned.",
            "Route comparison used Cheonan-Asan routes.",
            "Historical hotspot data has limited types.",
            "Road network and accident periods differ.",
            "Latest JavaScript matcher not rerun.",
            "Actual nationwide service not validated."
        ]
    }


# ==================================================
# 8. 서비스 인터페이스 설정
# ==================================================

def build_interface():

    return {
        "interface_version": "T09-P01-v1.0",
        "status": "integration_candidate",
        "model_id": MODEL_ID,
        "policy_id": POLICY_ID,

        "link_data": {
            "link_id": {
                "type": "string"
            },
            "geometry": {
                "crs": "EPSG:4326",
                "coordinate_order": "longitude_latitude"
            },
            "hotspot_score": {
                "type": "number",
                "range": [0, 1]
            },
            "lanes": {
                "type": "integer",
                "minimum": 1
            },
            "intersection_nodes_200m": {
                "type": "integer",
                "minimum": 0
            },
            "legacy_hotspot_distance_m": {
                "type": "number_or_null",
                "minimum": 0,
                "used_in_P01": False
            }
        },

        "route_response": {
            "model_score": {
                "type": "number_or_null",
                "range": [0, 100],
                "meaning": "100_times_P01_burden",
                "higher_is_better": False
            },
            "model_coverage": {
                "type": "number",
                "range": [0, 1]
            },
            "model_matched_m": {
                "type": "number",
                "minimum": 0
            },
            "geometry_length_m": {
                "type": "number",
                "minimum": 0
            },
            "model_wide_ratio": {
                "type": "number_or_null",
                "range": [0, 1]
            },
            "model_intersections": {
                "type": "number_or_null",
                "minimum": 0
            },
            "model_hotspot_ratio": {
                "type": "number_or_null",
                "range": [0, 1],
                "meaning": "legacy_hotspot_proximity_ratio",
                "used_in_P01": False
            },
            "matched_link_count": {
                "type": "integer",
                "minimum": 0
            },
            "model_version": {
                "type": "string",
                "value": MODEL_ID
            },
            "policy_version": {
                "type": "string",
                "value": POLICY_ID
            },
            "model_raw_risk": {
                "type": "number_or_null",
                "range": [0, 1]
            },
            "model_narrow_burden": {
                "type": "number_or_null",
                "range": [0, 1]
            },
            "model_intersection_burden": {
                "type": "number_or_null",
                "range": [0, 1]
            }
        },

        "frontend": {
            "display_score": "100 - model_score",
            "higher_score_is_better": True,
            "recommended_label": "경로 추천 점수",
            "null_score_policy": "display_unavailable"
        },

        "integration_requirements": {
            "preserve_existing_api_endpoints": True,
            "preserve_existing_ui_structure": True,
            "preserve_existing_service_files": True,
            "new_tile_schema_version_required": True,
            "multipart_geometry_validation_required": True,
            "legacy_hotspot_distance_validation_required": True,
            "javascript_policy_reproduction_required": True,
            "live_service_validation_required": True
        },

        "original_agreement_modified": False,
        "team_interface_approval": "pending",
        "deployment_status": "not_deployed"
    }


# ==================================================
# 9. 메인 함수
# ==================================================

def main():

    start = time.perf_counter()

    print("===== Final National Route Policy =====")

    check_outputs()

    source_paths = [
        MODEL_PATH,
        MODEL_CONFIG_PATH,
        SCORE_PATH,
        SCORE_METADATA_PATH,
        ROUTE_PATH,
        SELECTION_PATH,
        COMPARISON_CONFIG_PATH
    ]

    source_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in source_paths
    }

    print("\n===== Model Validation =====")

    model_config, metadata = validate_model()

    validate_formula()

    summary = reproduce_policy()

    policy = build_policy_config(
        model_config,
        metadata,
        source_hashes
    )

    interface = build_interface()

    print("\n===== Source Integrity Verification =====")

    for path in source_paths:

        key = path.relative_to(ROOT).as_posix()

        check(
            f"Source Hash: {path.name}",
            sha256_file(path) == source_hashes[key]
        )

    with OUTPUT_CONFIG.open("w", encoding="utf-8") as f:

        json.dump(
            policy, f,
            ensure_ascii=False,
            indent=4
        )

    with OUTPUT_INTERFACE.open(
        "w", encoding="utf-8"
    ) as f:

        json.dump(
            interface, f,
            ensure_ascii=False,
            indent=4
        )

    pd.DataFrame(VALIDATION).to_csv(
        OUTPUT_VALIDATION,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== Saved Results =====")

    for path in [
        OUTPUT_CONFIG,
        OUTPUT_INTERFACE,
        OUTPUT_VALIDATION,
        OUTPUT_SUMMARY
    ]:

        print("Path:", path)
        print("Bytes:", path.stat().st_size)

    print("\n===== Processing Completed =====")

    print(
        f"Elapsed Time: "
        f"{time.perf_counter() - start:.2f}s"
    )

    print("Final Model:", MODEL_ID)
    print("Final Policy:", POLICY_ID)
    print("Weights: 0.80 / 0.10 / 0.10")
    print("Intersection Reference: 100")
    print("Policy Selected: Yes")
    print("Reproduced Route Selections: 2214")
    print("Detour Violations: 0")
    print("Validation Checks:", len(VALIDATION))
    print("All Validation Checks: PASS")
    print("Interface Candidate: Saved")
    print("Team Interface Approval: Pending")
    print("Additional Model Training: Not Performed")
    print("Test Re-evaluation: Not Performed")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":

    main()