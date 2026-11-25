from pathlib import Path
import json
import time
import importlib

import numpy as np
import pandas as pd


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

INPUT_PATH = (
    RESULTS_DIR / "national_route_aggregation.parquet"
)

INPUT_CONFIG_PATH = (
    RESULTS_DIR / "national_route_aggregation_config.json"
)

SCORE_PATH = (
    RESULTS_DIR / "national_combined_route_scores.parquet"
)

SELECTION_PATH = (
    RESULTS_DIR / "national_combined_route_selections.parquet"
)

SUMMARY_PATH = (
    RESULTS_DIR / "national_combined_scoring_summary.csv"
)

CONFIG_PATH = (
    RESULTS_DIR / "national_combined_scoring_config.json"
)

EXPECTED_ROUTES = 6402

EXPECTED_ELIGIBLE = 6303

MIN_COVERAGE = 0.8

DETOUR_FACTORS = [1.5, 2.0, 3.0]

INTERSECTION_REFERENCE = 74.0

RAW_COLUMN = "N00_raw_A02_weighted_mean"

MINMAX_COLUMN = "N01_minmax_A02_weighted_mean"

NORMALIZATIONS = {
    "RAW": RAW_COLUMN,
    "MINMAX": MINMAX_COLUMN
}

# 모델, 좁은 도로, 교차로 가중치
WEIGHTS = {
    "C01": (1.00, 0.00, 0.00),
    "C02": (0.75, 0.25, 0.00),
    "C03": (0.55, 0.45, 0.00),
    "C04": (0.50, 0.50, 0.00),
    "C05": (0.55, 0.25, 0.20),
    "C06": (0.70, 0.20, 0.10),
    "C07": (0.40, 0.40, 0.20),
    "C08": (0.00, 1.00, 0.00),
    "C09": (0.00, 0.55, 0.45)
}

REFERENCE_METHOD = "RAW_C01"

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)


# ============================================================
# 2. 데이터 불러오기 및 검증
# ============================================================

def load_data():

    with INPUT_CONFIG_PATH.open(
        "r", encoding="utf-8"
    ) as file:

        config = json.load(file)

    assert config["model_id"] == "T09"

    assert config["total_routes"] == EXPECTED_ROUTES

    assert config["eligible_routes"] == EXPECTED_ELIGIBLE

    assert config["minimum_coverage"] == MIN_COVERAGE

    assert config["route_score_formula_selected"] is False

    routes = pd.read_parquet(INPUT_PATH)

    required_columns = [
        "route_id",
        "origin_id",
        "destination_id",
        "geometry_sha256",
        "distance_m",
        "duration_s",
        "effective_coverage",
        "original_coverage",
        "scored_length_m",
        "eligible",
        RAW_COLUMN,
        MINMAX_COLUMN,
        "wide_road_ratio",
        "narrow_road_ratio",
        "weighted_intersection_nodes",
        "A04_positive_length_ratio",
        "N00_raw_A03_max",
        "legacy_rule_score",
        "legacy_model_score"
    ]

    assert set(required_columns).issubset(
        routes.columns
    )

    assert len(routes) == EXPECTED_ROUTES

    assert routes["route_id"].is_unique

    assert routes[
        [
            "origin_id",
            "destination_id",
            "geometry_sha256"
        ]
    ].notna().all().all()

    assert (routes["distance_m"] > 0).all()

    assert (routes["duration_s"] > 0).all()

    expected_eligible = (
        (routes["original_coverage"] >= MIN_COVERAGE)
        & (routes["effective_coverage"] >= MIN_COVERAGE)
        & (routes["scored_length_m"] > 0)
    )

    assert np.array_equal(
        routes["eligible"].to_numpy(),
        expected_eligible.to_numpy()
    )

    eligible = routes.loc[routes["eligible"]]

    assert len(eligible) == EXPECTED_ELIGIBLE

    numeric_columns = [
        RAW_COLUMN,
        MINMAX_COLUMN,
        "wide_road_ratio",
        "narrow_road_ratio",
        "weighted_intersection_nodes",
        "A04_positive_length_ratio",
        "N00_raw_A03_max"
    ]

    assert np.isfinite(
        eligible[numeric_columns].to_numpy()
    ).all()

    for column in numeric_columns:

        if column == "weighted_intersection_nodes":

            assert (eligible[column] >= 0).all()

        else:

            assert eligible[column].between(
                0, 1
            ).all()

    np.testing.assert_allclose(
        eligible["wide_road_ratio"]
        + eligible["narrow_road_ratio"],
        1.0,
        atol=1e-12
    )

    print("\n===== Input Data =====")

    print("Total Routes:", len(routes))

    print("Eligible Routes:", len(eligible))

    print(
        "Excluded Routes:",
        len(routes) - len(eligible)
    )

    print(
        "Unique OD:",
        routes.groupby(
            ["origin_id", "destination_id"]
        ).ngroups
    )

    print("Input Validation: PASS")

    return routes


# ============================================================
# 3. 결합 점수 생성
# ============================================================

def calculate_combined_scores(routes):

    routes = routes.copy()

    routes["intersection_burden"] = np.clip(
        routes["weighted_intersection_nodes"]
        / INTERSECTION_REFERENCE,
        0,
        1
    )

    methods = []

    print("\n===== Combined Scoring Methods =====")

    for normalization, risk_column in NORMALIZATIONS.items():

        for candidate, weights in WEIGHTS.items():

            w_r, w_n, w_i = weights

            assert np.isclose(sum(weights), 1.0)

            assert all(w >= 0 for w in weights)

            # 모델을 사용하지 않는 비교군은 한 번만 계산
            if w_r == 0 and normalization == "MINMAX":

                continue

            method = (
                f"{normalization}_{candidate}"
                if w_r > 0
                else f"POLICY_{candidate}"
            )

            burden_column = f"{method}_burden"

            score_column = f"{method}_score"

            routes[burden_column] = (
                w_r * routes[risk_column]
                + w_n * routes["narrow_road_ratio"]
                + w_i * routes["intersection_burden"]
            )

            routes[score_column] = (
                100 * (1 - routes[burden_column])
            )

            # 커버리지 미달 경로는 추천 점수 미부여
            routes.loc[
                ~routes["eligible"],
                [burden_column, score_column]
            ] = np.nan

            valid = routes.loc[routes["eligible"]]

            assert np.isfinite(
                valid[burden_column]
            ).all()

            assert valid[burden_column].between(
                -1e-12, 1 + 1e-12
            ).all()

            assert valid[score_column].between(
                -1e-10, 100 + 1e-10
            ).all()

            methods.append(method)

            print(
                f"{method:12s} | "
                f"R={w_r:.2f} | "
                f"N={w_n:.2f} | "
                f"I={w_i:.2f} | "
                f"Mean Score="
                f"{valid[score_column].mean():.6f}"
            )

    # Raw 및 Min-Max 각 7개 + 정책 비교군 2개
    assert len(methods) == 16

    valid = routes.loc[routes["eligible"]]

    np.testing.assert_allclose(
        valid["RAW_C01_burden"],
        valid[RAW_COLUMN],
        atol=1e-12
    )

    np.testing.assert_allclose(
        valid["MINMAX_C01_burden"],
        valid[MINMAX_COLUMN],
        atol=1e-12
    )

    np.testing.assert_allclose(
        valid["POLICY_C08_burden"],
        valid["narrow_road_ratio"],
        atol=1e-12
    )

    print("\n===== Intersection Burden =====")

    print("Reference:", INTERSECTION_REFERENCE)

    print(
        "Saturated Routes:",
        int(
            (valid["intersection_burden"] >= 1).sum()
        )
    )

    print(
        "Mean Intersection Burden:",
        valid["intersection_burden"].mean()
    )

    print("Combined Methods:", len(methods))

    print("Score Validation: PASS")

    return routes, methods


# ============================================================
# 4. 경로 후보 구성
# ============================================================

def prepare_candidate_groups(routes):

    groups = []

    counters = {
        factor: {
            "total_od": 0,
            "no_eligible": 0,
            "single_eligible": 0,
            "two_eligible": 0,
            "three_or_more": 0
        }
        for factor in DETOUR_FACTORS
    }

    for od, group in routes.groupby(
        ["origin_id", "destination_id"],
        sort=True
    ):

        shortest = float(
            group["distance_m"].min()
        )

        eligible = group.loc[
            group["eligible"]
        ].sort_values(
            [
                "distance_m",
                "duration_s",
                "route_id"
            ]
        ).drop_duplicates(
            subset=["geometry_sha256"]
        )

        for factor in DETOUR_FACTORS:

            candidates = eligible.loc[
                eligible["distance_m"]
                <= shortest * factor + 1e-9
            ].copy()

            count = len(candidates)

            counter = counters[factor]

            counter["total_od"] += 1

            if count == 0:

                counter["no_eligible"] += 1

            elif count == 1:

                counter["single_eligible"] += 1

            elif count == 2:

                counter["two_eligible"] += 1

            else:

                counter["three_or_more"] += 1

            # 연구 비교는 후보 2개 이상에서 수행
            if count < 2:

                continue

            groups.append({
                "origin_id": od[0],
                "destination_id": od[1],
                "detour_factor": factor,
                "shortest_distance": shortest,
                "candidate_count": count,
                "service_recommendable": count >= 3,
                "candidates": candidates
            })

    print("\n===== Candidate Group Summary =====")

    for factor, counter in counters.items():

        print(f"\nDetour Factor: {factor}")

        for key, value in counter.items():

            print(f"{key}: {value}")

        assert counter["total_od"] == (
            counter["no_eligible"]
            + counter["single_eligible"]
            + counter["two_eligible"]
            + counter["three_or_more"]
        )

    return groups, counters


# ============================================================
# 5. 경로 추천 비교
# ============================================================

def select_routes(groups, methods):

    records = []

    all_methods = [
        "M00_shortest",
        "M01_fastest",
        *methods
    ]

    print("\n===== Route Selection =====")

    for index, group in enumerate(groups, start=1):

        candidates = group["candidates"]

        for method in all_methods:

            if method == "M00_shortest":

                ordered = candidates.sort_values(
                    [
                        "distance_m",
                        "duration_s",
                        "route_id"
                    ]
                )

            elif method == "M01_fastest":

                ordered = candidates.sort_values(
                    [
                        "duration_s",
                        "distance_m",
                        "route_id"
                    ]
                )

            else:

                ordered = candidates.assign(
                    sort_burden=candidates[
                        f"{method}_burden"
                    ].round(12)
                ).sort_values(
                    [
                        "sort_burden",
                        "distance_m",
                        "duration_s",
                        "route_id"
                    ]
                )

            chosen = ordered.iloc[0]

            detour_ratio = (
                chosen["distance_m"]
                / group["shortest_distance"]
            )

            assert detour_ratio <= (
                group["detour_factor"] + 1e-9
            )

            assert chosen["effective_coverage"] >= (
                MIN_COVERAGE
            )

            records.append({

                "origin_id": group["origin_id"],

                "destination_id":
                    group["destination_id"],

                "detour_factor":
                    group["detour_factor"],

                "candidate_count":
                    group["candidate_count"],

                "service_recommendable":
                    group["service_recommendable"],

                "method": method,

                "selected_route_id":
                    chosen["route_id"],

                "distance_m":
                    float(chosen["distance_m"]),

                "duration_s":
                    float(chosen["duration_s"]),

                "detour_ratio":
                    float(detour_ratio),

                "effective_coverage":
                    float(chosen["effective_coverage"]),

                "raw_risk":
                    float(chosen[RAW_COLUMN]),

                "minmax_risk":
                    float(chosen[MINMAX_COLUMN]),

                "wide_road_ratio":
                    float(chosen["wide_road_ratio"]),

                "narrow_road_ratio":
                    float(chosen["narrow_road_ratio"]),

                "weighted_intersection_nodes":
                    float(
                        chosen["weighted_intersection_nodes"]
                    ),

                "intersection_burden":
                    float(chosen["intersection_burden"]),

                "positive_length_ratio":
                    float(
                        chosen["A04_positive_length_ratio"]
                    ),

                "maximum_risk":
                    float(chosen["N00_raw_A03_max"]),

                "combined_score":
                    float(chosen[f"{method}_score"])
                    if method in methods
                    else np.nan

            })

        if index % 500 == 0 or index == len(groups):

            print(
                f"Processed Groups: "
                f"{index:,} / {len(groups):,}",
                flush=True
            )

    selections = pd.DataFrame(records)

    assert len(selections) == (
        len(groups) * len(all_methods)
    )

    assert not selections.duplicated(
        subset=[
            "origin_id",
            "destination_id",
            "detour_factor",
            "method"
        ]
    ).any()

    violations = int(
        (
            selections["detour_ratio"]
            > selections["detour_factor"] + 1e-9
        ).sum()
    )

    assert violations == 0

    print("\nSelection Records:", len(selections))

    print("Detour Violations:", violations)

    print("Selection Validation: PASS")

    return selections


# ============================================================
# 6. 추천 결과 요약
# ============================================================

def summarize_results(selections):

    records = []

    scopes = {
        "research_2plus": selections,

        "service_3plus": selections.loc[
            selections["service_recommendable"]
        ]
    }

    od_columns = [
        "origin_id",
        "destination_id"
    ]

    for scope, frame in scopes.items():

        for factor, group in frame.groupby(
            "detour_factor"
        ):

            reference = group.loc[
                group["method"] == REFERENCE_METHOD
            ].set_index(od_columns)

            shortest = group.loc[
                group["method"] == "M00_shortest"
            ].set_index(od_columns)

            for method, selected in group.groupby(
                "method"
            ):

                selected = selected.set_index(
                    od_columns
                )

                assert selected.index.is_unique

                assert set(selected.index) == set(
                    reference.index
                )

                selected = selected.reindex(
                    reference.index
                )

                shortest_aligned = shortest.reindex(
                    reference.index
                )

                changed = (
                    selected["selected_route_id"]
                    != reference["selected_route_id"]
                )

                risk_change = (
                    selected["raw_risk"]
                    - reference["raw_risk"]
                )

                wide_change = (
                    selected["wide_road_ratio"]
                    - reference["wide_road_ratio"]
                )

                intersection_change = (
                    selected["weighted_intersection_nodes"]
                    - reference["weighted_intersection_nodes"]
                )

                distance_change = (
                    selected["distance_m"]
                    - reference["distance_m"]
                )

                duration_change = (
                    selected["duration_s"]
                    - reference["duration_s"]
                )

                shortest_risk = (
                    shortest_aligned["raw_risk"].mean()
                )

                mean_risk = selected["raw_risk"].mean()

                records.append({

                    "scope": scope,

                    "detour_factor": factor,

                    "method": method,

                    "od_count": len(selected),

                    "changed_count_vs_raw":
                        int(changed.sum()),

                    "changed_ratio_vs_raw":
                        float(changed.mean()),

                    "mean_raw_risk":
                        mean_risk,

                    "mean_wide_road_ratio":
                        selected["wide_road_ratio"].mean(),

                    "mean_narrow_road_ratio":
                        selected["narrow_road_ratio"].mean(),

                    "mean_intersection_nodes":
                        selected[
                            "weighted_intersection_nodes"
                        ].mean(),

                    "mean_positive_length_ratio":
                        selected[
                            "positive_length_ratio"
                        ].mean(),

                    "mean_maximum_risk":
                        selected["maximum_risk"].mean(),

                    "mean_distance_m":
                        selected["distance_m"].mean(),

                    "mean_duration_s":
                        selected["duration_s"].mean(),

                    "mean_detour_ratio":
                        selected["detour_ratio"].mean(),

                    "mean_coverage":
                        selected["effective_coverage"].mean(),

                    "risk_change_vs_raw":
                        risk_change.mean(),

                    "wide_change_vs_raw":
                        wide_change.mean(),

                    "intersection_change_vs_raw":
                        intersection_change.mean(),

                    "distance_change_vs_raw":
                        distance_change.mean(),

                    "duration_change_vs_raw":
                        duration_change.mean(),

                    "wide_improved_od_ratio":
                        float(
                            (wide_change > 1e-12).mean()
                        ),

                    "wide_worsened_od_ratio":
                        float(
                            (wide_change < -1e-12).mean()
                        ),

                    "risk_increased_od_ratio":
                        float(
                            (risk_change > 1e-12).mean()
                        ),

                    "risk_reduction_vs_shortest_pct":
                        (
                            (shortest_risk - mean_risk)
                            / shortest_risk * 100
                        )
                        if shortest_risk > 0
                        else np.nan,

                    "wide_change_vs_shortest":
                        (
                            selected["wide_road_ratio"]
                            - shortest_aligned[
                                "wide_road_ratio"
                            ]
                        ).mean(),

                    "distance_change_vs_shortest":
                        (
                            selected["distance_m"]
                            - shortest_aligned["distance_m"]
                        ).mean(),

                    "duration_change_vs_shortest":
                        (
                            selected["duration_s"]
                            - shortest_aligned["duration_s"]
                        ).mean()

                })

    summary = pd.DataFrame(records)

    print("\n===== Combined Scoring Comparison =====")

    columns = [
        "method",
        "od_count",
        "mean_raw_risk",
        "mean_wide_road_ratio",
        "mean_intersection_nodes",
        "mean_distance_m",
        "mean_duration_s",
        "changed_ratio_vs_raw"
    ]

    for scope in scopes:

        for factor in DETOUR_FACTORS:

            selected = summary.loc[
                (summary["scope"] == scope)
                & (summary["detour_factor"] == factor)
            ]

            print(
                f"\nScope: {scope} | "
                f"Detour Factor: {factor}"
            )

            print(
                selected[columns].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}"
                )
            )

    return summary


# ============================================================
# 7. 메인 실행
# ============================================================

def main():

    start = time.perf_counter()

    print(
        "===== National Combined Scoring Evaluation ====="
    )

    training_utils.print_memory("Initial Memory")

    output_paths = [
        SCORE_PATH,
        SELECTION_PATH,
        SUMMARY_PATH,
        CONFIG_PATH
    ]

    # 기존 실험 결과 보호
    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    source_paths = [
        INPUT_PATH,
        INPUT_CONFIG_PATH
    ]

    source_hashes = {
        path.name: baseline_utils.sha256_file(path)
        for path in source_paths
    }

    routes = load_data()

    routes, methods = calculate_combined_scores(
        routes
    )

    groups, counters = prepare_candidate_groups(
        routes
    )

    selections = select_routes(
        groups,
        methods
    )

    summary = summarize_results(
        selections
    )

    # 원본 파일 변경 여부 확인
    for path in source_paths:

        assert (
            baseline_utils.sha256_file(path)
            == source_hashes[path.name]
        )

    config = {

        "model_id": "T09",

        "experiment_scope":
            "existing_cheonan_asan_routes",

        "total_routes": len(routes),

        "eligible_routes":
            int(routes["eligible"].sum()),

        "minimum_coverage": MIN_COVERAGE,

        "research_minimum_candidates": 2,

        "service_minimum_candidates": 3,

        "detour_factors": DETOUR_FACTORS,

        "intersection_reference":
            INTERSECTION_REFERENCE,

        "normalizations": NORMALIZATIONS,

        "weights": {
            key: {
                "risk": value[0],
                "narrow": value[1],
                "intersection": value[2]
            }
            for key, value in WEIGHTS.items()
        },

        "combined_methods": methods,

        "reference_method": REFERENCE_METHOD,

        "candidate_group_summary": counters,

        "burden_formula":
            "w_R * R + w_N * N + w_I * I",

        "score_formula":
            "100 * (1 - burden)",

        "source_sha256": source_hashes,

        "selected_method": None,

        "route_score_formula_selected": False,

        "test_re_evaluated": False,

        "deployment_status": "not_deployed",

        "limitations": [

            "Existing regional route matches reused.",

            "Route-level accident outcomes unavailable.",

            "Novice driver difficulty labels unavailable.",

            "Lane and intersection features may be "
            "represented in both model and policy.",

            "Intersection reference 74 is an experimental "
            "legacy normalization reference.",

            "Two-candidate results are research-only.",

            "Three-candidate results are offline policy "
            "simulations, not live service verification.",

            "Latest national legacy model scores "
            "not directly compared.",

            "Historical hotspot membership score "
            "is not actual accident probability."

        ]

    }

    routes.to_parquet(
        SCORE_PATH,
        index=False
    )

    selections.to_parquet(
        SELECTION_PATH,
        index=False
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    with CONFIG_PATH.open(
        "x", encoding="utf-8"
    ) as file:

        json.dump(
            config,
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
        f"{time.perf_counter() - start:.2f}s"
    )

    print("Total Routes:", len(routes))

    print(
        "Eligible Routes:",
        int(routes["eligible"].sum())
    )

    print("Combined Methods:", len(methods))

    print("Baseline Methods: 2")

    print("Selection Records:", len(selections))

    print("Summary Records:", len(summary))

    print("Source Hash Verification: PASS")

    print("Detour Violations: 0")

    print("Final Weights Selected: No")

    print("Additional Model Training: Not Performed")

    print("Test Re-evaluation: Not Performed")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":

    main()