from pathlib import Path
from collections import defaultdict
from itertools import combinations
import json
import time
import importlib
import gc

import numpy as np
import pandas as pd


# ============================================================
# 1. 경로 및 실험 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = (
    DATA_DIR / "national_accident_risk" / "processed"
)

RESULTS_DIR = PROCESSED_DIR / "results"

ROUTES_PATH = DATA_DIR / "routes_scored.json"

MATCHES_PATH = DATA_DIR / "route_link_matches.json"

SCORES_PATH = (
    PROCESSED_DIR / "national_link_risk_scores.parquet"
)

METADATA_PATH = (
    RESULTS_DIR / "national_link_risk_scores_metadata.json"
)

NORMALIZATION_PATH = (
    RESULTS_DIR / "national_score_normalization_config.json"
)

OUTPUT_PATH = (
    RESULTS_DIR / "national_route_aggregation.parquet"
)

SUMMARY_PATH = (
    RESULTS_DIR / "national_route_aggregation_summary.csv"
)

RANKING_PATH = (
    RESULTS_DIR / "national_route_ranking_comparison.csv"
)

SELECTION_PATH = (
    RESULTS_DIR / "national_route_aggregation_selections.csv"
)

CONFIG_PATH = (
    RESULTS_DIR / "national_route_aggregation_config.json"
)

MIN_COVERAGE = 0.8

EXPECTED_ROUTES = 6402

EXPECTED_LINKS = 1562356

# 비교 실험용 우회 허용 배율
DETOUR_FACTORS = [1.5, 2.0, 3.0]

REFERENCE_METHOD = "N00_raw_A02_weighted_mean"

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)


# ============================================================
# 2. 공통 함수
# ============================================================

def load_json(path):

    with path.open("r", encoding="utf-8-sig") as file:

        return json.load(file)


def get_methods():

    methods = []

    for normalization in [
        "N00_raw",
        "N01_minmax",
        "N02_p95",
        "N03_percentile"
    ]:

        for aggregation in [
            "A01_mean",
            "A02_weighted_mean",
            "A03_max"
        ]:

            methods.append(
                f"{normalization}_{aggregation}"
            )

    methods.append("A04_positive_length_ratio")

    return methods


# ============================================================
# 3. 데이터 불러오기 및 해시 검증
# ============================================================

def load_data():

    metadata = load_json(METADATA_PATH)

    normalization = load_json(NORMALIZATION_PATH)

    assert metadata["model_id"] == "T09"

    assert normalization["model_id"] == "T09"

    assert normalization["selected_method"] is None

    score_hash = baseline_utils.sha256_file(
        SCORES_PATH
    )

    assert score_hash == metadata["output_sha256"]

    assert score_hash == (
        normalization["source_score_sha256"]
    )

    assert baseline_utils.sha256_file(
        METADATA_PATH
    ) == normalization["source_metadata_sha256"]

    routes = pd.DataFrame(
        load_json(ROUTES_PATH)
    )

    matches = load_json(MATCHES_PATH)

    assert len(routes) == EXPECTED_ROUTES

    assert len(matches) == EXPECTED_ROUTES

    assert routes["route_id"].is_unique

    match_lookup = {
        item["route_id"]: item
        for item in matches
    }

    assert len(match_lookup) == EXPECTED_ROUTES

    assert set(routes["route_id"]) == set(match_lookup)

    for column in ["distance_m", "duration_s"]:

        routes[column] = pd.to_numeric(
            routes[column],
            errors="raise"
        )

        assert np.isfinite(routes[column]).all()

        assert (routes[column] > 0).all()

    assert routes[
        ["origin_id", "destination_id"]
    ].notna().all().all()

    print("\n===== Source Data =====")

    print("Routes:", len(routes))
    print("Route Matches:", len(matches))
    print("Route ID Consistency: PASS")
    print("Score Hash Verification: PASS")
    print("Metadata Hash Verification: PASS")

    return (
        routes,
        match_lookup,
        metadata,
        normalization
    )


# ============================================================
# 4. 전국 점수 정규화 및 필요한 링크 추출
# ============================================================

def prepare_link_scores(
    match_lookup,
    normalization
):

    required_ids = set()

    for item in match_lookup.values():

        for pair in item["links"]:

            assert len(pair) == 2

            link_id, length = pair

            assert np.isfinite(float(length))

            assert float(length) >= 0

            if float(length) > 0:

                required_ids.add(str(link_id))

    scores = pd.read_parquet(
        SCORES_PATH,
        columns=[
            "link_id",
            "hotspot_score",
            "lanes",
            "intersection_nodes_200m"
        ]
    )

    assert len(scores) == EXPECTED_LINKS

    assert scores["link_id"].is_unique

    assert scores.notna().all().all()

    raw = scores["hotspot_score"].to_numpy()

    assert np.isfinite(raw).all()

    assert ((raw >= 0) & (raw <= 1)).all()

    minimum = normalization["minimum"]

    maximum = normalization["maximum"]

    p95 = normalization["p95"]

    assert np.isclose(raw.min(), minimum)

    assert np.isclose(raw.max(), maximum)

    assert np.isclose(
        np.quantile(raw, 0.95),
        p95
    )

    assert normalization[
        "reference_link_count"
    ] == len(scores)

    # 전국 전체 링크를 기준으로 순위 계산
    scores["N03_percentile"] = (
        scores["hotspot_score"].rank(
            method="average"
        ) - 1
    ) / (len(scores) - 1)

    # 실제 경로에서 사용되는 링크만 유지
    selected = scores.loc[
        scores["link_id"].isin(required_ids)
    ].copy()

    selected["N00_raw"] = (
        selected["hotspot_score"]
    )

    selected["N01_minmax"] = (
        selected["hotspot_score"] - minimum
    ) / (maximum - minimum)

    selected["N02_p95"] = np.clip(
        selected["hotspot_score"] / p95,
        0,
        1
    )

    selected = selected.set_index("link_id")

    missing_ids = required_ids - set(
        selected.index
    )

    print("\n===== Link Score Compatibility =====")

    print("National Links:", len(scores))
    print("Required Route Links:", len(required_ids))
    print("Available Route Links:", len(selected))
    print("Missing Route Links:", len(missing_ids))

    print(
        "Link ID Coverage:",
        len(selected) / len(required_ids)
        if required_ids else 0
    )

    del scores, raw

    gc.collect()

    return selected, missing_ids


# ============================================================
# 5. 개별 경로 점수 집계
# ============================================================

def aggregate_route(
    route,
    match,
    link_scores,
    threshold
):

    geometry_length = float(
        match["geometry_length_m"]
    )

    matched_length = float(
        match["matched_length_m"]
    )

    original_coverage = float(
        match["coverage"]
    )

    assert np.isfinite([
        geometry_length,
        matched_length,
        original_coverage
    ]).all()

    assert geometry_length >= 0

    assert matched_length >= 0

    assert 0 <= original_coverage <= 1.000001

    # 동일 링크가 여러 번 등장하면 매칭 길이 합산
    link_lengths = defaultdict(float)

    for link_id, length in match["links"]:

        if float(length) > 0:

            link_lengths[str(link_id)] += float(length)

    total_link_length = sum(
        link_lengths.values()
    )

    assert np.isclose(
        total_link_length,
        matched_length,
        rtol=1e-5,
        atol=0.1
    ), route["route_id"]

    if geometry_length > 0:

        assert np.isclose(
            matched_length / geometry_length,
            original_coverage,
            rtol=1e-5,
            atol=1e-6
        ), route["route_id"]

    available_ids = [
        link_id
        for link_id in link_lengths
        if link_id in link_scores.index
    ]

    weights = np.array([
        link_lengths[link_id]
        for link_id in available_ids
    ], dtype=np.float64)

    scored_length = float(weights.sum())

    effective_coverage = (
        scored_length / geometry_length
        if geometry_length > 0
        else 0.0
    )

    assert effective_coverage <= 1.00001

    result = {

        "route_id": route["route_id"],

        "origin_id": route["origin_id"],

        "destination_id": route["destination_id"],

        "origin_city": route["origin_city"],

        "destination_city": route["destination_city"],

        "profile": route["profile"],

        "geometry_sha256": route["geometry_sha256"],

        "distance_m": float(route["distance_m"]),

        "duration_s": float(route["duration_s"]),

        "geometry_length_m": geometry_length,

        "matched_length_m": matched_length,

        "scored_length_m": scored_length,

        "original_coverage": original_coverage,

        "effective_coverage": effective_coverage,

        "matched_link_count": len(link_lengths),

        "scored_link_count": len(available_ids),

        "missing_link_count": (
            len(link_lengths) - len(available_ids)
        ),

        "eligible": (
            original_coverage >= MIN_COVERAGE
            and effective_coverage >= MIN_COVERAGE
            and scored_length > 0
        ),

        "legacy_rule_score": route.get("score"),

        "legacy_model_score": route.get(
            "legacy_model_score"
        ),

        "legacy_wide_road_ratio": route.get(
            "wide_road_ratio"
        )

    }

    for method in get_methods():

        result[method] = np.nan

    result["wide_road_ratio"] = np.nan

    result["narrow_road_ratio"] = np.nan

    result["weighted_intersection_nodes"] = np.nan

    if scored_length <= 0:

        return result

    selected = link_scores.loc[
        available_ids
    ]

    for normalization in [
        "N00_raw",
        "N01_minmax",
        "N02_p95",
        "N03_percentile"
    ]:

        values = selected[
            normalization
        ].to_numpy()

        result[
            f"{normalization}_A01_mean"
        ] = float(values.mean())

        result[
            f"{normalization}_A02_weighted_mean"
        ] = float(
            np.average(values, weights=weights)
        )

        result[
            f"{normalization}_A03_max"
        ] = float(values.max())

    raw = selected[
        "hotspot_score"
    ].to_numpy()

    # 원본 모델 임계값 이상인 구간의 길이 비율
    result["A04_positive_length_ratio"] = float(
        np.average(
            raw >= threshold,
            weights=weights
        )
    )

    lanes = selected["lanes"].to_numpy()

    result["wide_road_ratio"] = float(
        np.average(
            lanes >= 2,
            weights=weights
        )
    )

    result["narrow_road_ratio"] = float(
        np.average(
            lanes < 2,
            weights=weights
        )
    )

    result["weighted_intersection_nodes"] = float(
        np.average(
            selected["intersection_nodes_200m"],
            weights=weights
        )
    )

    return result


# ============================================================
# 6. 전체 실제 경로 점수 생성
# ============================================================

def aggregate_all_routes(
    routes,
    match_lookup,
    link_scores,
    threshold
):

    records = []

    print("\n===== Route Aggregation =====")

    for index, route in enumerate(
        routes.to_dict("records"),
        start=1
    ):

        result = aggregate_route(
            route,
            match_lookup[route["route_id"]],
            link_scores,
            threshold
        )

        records.append(result)

        if index % 500 == 0 or index == len(routes):

            print(
                f"Processed: {index:,} / "
                f"{len(routes):,}",
                flush=True
            )

    result = pd.DataFrame(records)

    assert len(result) == EXPECTED_ROUTES

    assert result["route_id"].is_unique

    eligible = result.loc[
        result["eligible"]
    ]

    assert len(eligible) > 0

    values = eligible[
        get_methods()
    ].to_numpy()

    assert np.isfinite(values).all()

    assert (
        (values >= -1e-12)
        & (values <= 1 + 1e-12)
    ).all()

    np.testing.assert_allclose(
        eligible["wide_road_ratio"]
        + eligible["narrow_road_ratio"],
        1.0,
        atol=1e-12
    )

    print("\n===== Route Coverage Summary =====")

    print("Total Routes:", len(result))

    print(
        "Original Coverage >= 80%:",
        int(
            (
                result["original_coverage"]
                >= MIN_COVERAGE
            ).sum()
        )
    )

    print("Eligible Routes:", len(eligible))

    print(
        "Excluded Routes:",
        len(result) - len(eligible)
    )

    print(
        "Routes with Missing Link IDs:",
        int(
            (result["missing_link_count"] > 0).sum()
        )
    )

    print(
        "Mean Effective Coverage:",
        result["effective_coverage"].mean()
    )

    print("Aggregation Validation: PASS")

    return result


# ============================================================
# 7. 집계 방식별 기술통계
# ============================================================

def summarize_aggregation(result):

    eligible = result.loc[
        result["eligible"]
    ]

    records = []

    for method in get_methods():

        scores = eligible[method]

        records.append({

            "method": method,

            "route_count": len(scores),

            "minimum": scores.min(),

            "maximum": scores.max(),

            "mean": scores.mean(),

            "median": scores.median(),

            "std": scores.std(ddof=0),

            "p25": scores.quantile(0.25),

            "p75": scores.quantile(0.75),

            "p95": scores.quantile(0.95),

            "spearman_with_raw_weighted_mean":
                scores.corr(
                    eligible[REFERENCE_METHOD],
                    method="spearman"
                ),

            "spearman_with_wide_road_ratio":
                scores.corr(
                    eligible["wide_road_ratio"],
                    method="spearman"
                ),

            "spearman_with_distance":
                scores.corr(
                    eligible["distance_m"],
                    method="spearman"
                )

        })

    summary = pd.DataFrame(records)

    print("\n===== Aggregation Statistics =====")

    print(
        summary[
            [
                "method",
                "mean",
                "median",
                "std",
                "p95",
                "spearman_with_raw_weighted_mean",
                "spearman_with_wide_road_ratio"
            ]
        ].to_string(index=False)
    )

    return summary


# ============================================================
# 8. Raw와 Min-Max 선형 변환 검증
# ============================================================

def validate_linear_transform(
    result,
    normalization
):

    eligible = result.loc[
        result["eligible"]
    ]

    minimum = normalization["minimum"]

    maximum = normalization["maximum"]

    for aggregation in [
        "A01_mean",
        "A02_weighted_mean",
        "A03_max"
    ]:

        raw = eligible[
            f"N00_raw_{aggregation}"
        ].to_numpy()

        actual = eligible[
            f"N01_minmax_{aggregation}"
        ].to_numpy()

        expected = (
            raw - minimum
        ) / (maximum - minimum)

        np.testing.assert_allclose(
            expected,
            actual,
            rtol=1e-12,
            atol=1e-12
        )

    print("\n===== Linear Transform Validation =====")

    print("Raw vs Min-Max Mean: PASS")

    print("Raw vs Min-Max Weighted Mean: PASS")

    print("Raw vs Min-Max Maximum: PASS")


# ============================================================
# 9. 동일 출발지·목적지 경로 순위 비교
# ============================================================

def compare_route_rankings(result):

    selections = []

    group_columns = [
        "origin_id",
        "destination_id"
    ]

    for od, group in result.groupby(
        group_columns,
        sort=True
    ):

        # 전체 후보 중 최단거리 기준
        shortest_distance = float(
            group["distance_m"].min()
        )

        eligible = group.loc[
            group["eligible"]
        ].copy()

        # 동일 경로 형상 중복 제거
        eligible = eligible.sort_values(
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
                <= shortest_distance * factor + 1e-9
            ].copy()

            # 비교 가능한 후보가 2개 이상인 OD만 분석
            if len(candidates) < 2:

                continue

            methods = [
                "M00_shortest",
                *get_methods()
            ]

            for method in methods:

                if method == "M00_shortest":

                    ordered = candidates.sort_values(
                        [
                            "distance_m",
                            "duration_s",
                            "route_id"
                        ]
                    )

                else:

                    # 수치 오차에 의한 동점 순위 변경 완화
                    ordered = candidates.assign(
                        sort_score=candidates[
                            method
                        ].round(12)
                    ).sort_values(
                        [
                            "sort_score",
                            "distance_m",
                            "duration_s",
                            "route_id"
                        ]
                    )

                chosen = ordered.iloc[0]

                detour_ratio = (
                    chosen["distance_m"]
                    / shortest_distance
                )

                assert detour_ratio <= factor + 1e-9

                selections.append({

                    "origin_id": od[0],

                    "destination_id": od[1],

                    "detour_factor": factor,

                    "candidate_count": len(candidates),

                    "method": method,

                    "selected_route_id":
                        chosen["route_id"],

                    "distance_m":
                        chosen["distance_m"],

                    "duration_s":
                        chosen["duration_s"],

                    "detour_ratio": detour_ratio,

                    "raw_weighted_mean":
                        chosen[REFERENCE_METHOD],

                    "raw_max":
                        chosen["N00_raw_A03_max"],

                    "positive_length_ratio":
                        chosen["A04_positive_length_ratio"],

                    "wide_road_ratio":
                        chosen["wide_road_ratio"],

                    "weighted_intersection_nodes":
                        chosen["weighted_intersection_nodes"],

                    "effective_coverage":
                        chosen["effective_coverage"]

                })

    selections = pd.DataFrame(selections)

    assert not selections.empty

    print("\n===== Route Selection Summary =====")

    print(
        "Total Selection Records:",
        len(selections)
    )

    print(
        "Detour Violations:",
        int(
            (
                selections["detour_ratio"]
                > selections["detour_factor"] + 1e-9
            ).sum()
        )
    )

    return selections


# ============================================================
# 10. 방식별 선택 경로 비교
# ============================================================

def summarize_rankings(selections):

    records = []

    for factor, group in selections.groupby(
        "detour_factor"
    ):

        reference = group.loc[
            group["method"] == REFERENCE_METHOD
        ].set_index(
            ["origin_id", "destination_id"]
        )

        for method, selected in group.groupby(
            "method"
        ):

            selected = selected.set_index(
                ["origin_id", "destination_id"]
            )

            assert selected.index.is_unique

            assert set(selected.index) == set(
                reference.index
            )

            selected = selected.reindex(
                reference.index
            )

            changed = (
                selected["selected_route_id"]
                != reference["selected_route_id"]
            )

            records.append({

                "detour_factor": factor,

                "method": method,

                "comparable_od_count": len(selected),

                "changed_routes_vs_reference":
                    int(changed.sum()),

                "changed_ratio_vs_reference":
                    float(changed.mean()),

                "mean_distance_m":
                    selected["distance_m"].mean(),

                "mean_duration_s":
                    selected["duration_s"].mean(),

                "mean_detour_ratio":
                    selected["detour_ratio"].mean(),

                "mean_raw_weighted_score":
                    selected["raw_weighted_mean"].mean(),

                "mean_raw_max":
                    selected["raw_max"].mean(),

                "mean_positive_length_ratio":
                    selected["positive_length_ratio"].mean(),

                "mean_wide_road_ratio":
                    selected["wide_road_ratio"].mean(),

                "mean_weighted_intersection_nodes":
                    selected[
                        "weighted_intersection_nodes"
                    ].mean(),

                "distance_change_vs_reference":
                    (
                        selected["distance_m"]
                        - reference["distance_m"]
                    ).mean(),

                "duration_change_vs_reference":
                    (
                        selected["duration_s"]
                        - reference["duration_s"]
                    ).mean(),

                "wide_ratio_change_vs_reference":
                    (
                        selected["wide_road_ratio"]
                        - reference["wide_road_ratio"]
                    ).mean()

            })

    summary = pd.DataFrame(records)

    print("\n===== Ranking Comparison =====")

    for factor in DETOUR_FACTORS:

        print(f"\nDetour Factor: {factor}")

        selected = summary.loc[
            summary["detour_factor"] == factor
        ]

        print(
            selected[
                [
                    "method",
                    "comparable_od_count",
                    "changed_ratio_vs_reference",
                    "mean_distance_m",
                    "mean_duration_s",
                    "mean_raw_weighted_score",
                    "mean_wide_road_ratio"
                ]
            ].to_string(index=False)
        )

    return summary


# ============================================================
# 11. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Route Aggregation Evaluation =====")

    training_utils.print_memory(
        "Initial Memory"
    )

    output_paths = [
        OUTPUT_PATH,
        SUMMARY_PATH,
        RANKING_PATH,
        SELECTION_PATH,
        CONFIG_PATH
    ]

    # 기존 실험 결과 보호
    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    source_paths = [
        ROUTES_PATH,
        MATCHES_PATH,
        SCORES_PATH,
        METADATA_PATH,
        NORMALIZATION_PATH
    ]

    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)):
            baseline_utils.sha256_file(path)
        for path in source_paths
    }

    (
        routes,
        match_lookup,
        metadata,
        normalization
    ) = load_data()

    link_scores, missing_ids = prepare_link_scores(
        match_lookup,
        normalization
    )

    threshold = float(
        metadata["classification_threshold"]
    )

    result = aggregate_all_routes(
        routes,
        match_lookup,
        link_scores,
        threshold
    )

    del routes, match_lookup, link_scores

    gc.collect()

    summary = summarize_aggregation(result)

    validate_linear_transform(
        result,
        normalization
    )

    selections = compare_route_rankings(
        result
    )

    rankings = summarize_rankings(
        selections
    )

    # 원본 파일 변경 여부 검증
    for path in source_paths:

        key = str(
            path.relative_to(PROJECT_ROOT)
        )

        assert baseline_utils.sha256_file(
            path
        ) == source_hashes[key]

    config = {

        "model_id": "T09",

        "experiment_scope":
            "existing_cheonan_asan_routes",

        "total_routes": len(result),

        "eligible_routes":
            int(result["eligible"].sum()),

        "minimum_coverage": MIN_COVERAGE,

        "detour_factors": DETOUR_FACTORS,

        "aggregation_methods": get_methods(),

        "reference_method": REFERENCE_METHOD,

        "positive_score_threshold": threshold,

        "missing_link_ids":
            sorted(missing_ids),

        "source_sha256": source_hashes,

        "selected_normalization": None,

        "selected_aggregation": None,

        "route_score_formula_selected": False,

        "test_re_evaluated": False,

        "deployment_status": "not_deployed",

        "limitations": [

            "Existing regional route matches reused.",

            "National route matching not evaluated.",

            "Route-level accident outcomes unavailable.",

            "Scores include model training links.",

            "Hotspot membership score is not "
            "actual accident probability.",

            "Unmatched road segments are not "
            "assigned zero risk."

        ]

    }

    result.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    rankings.to_csv(
        RANKING_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    selections.to_csv(
        SELECTION_PATH,
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

    training_utils.print_memory(
        "Final Memory"
    )

    print("\n===== Processing Completed =====")

    print(
        "Elapsed Time:",
        f"{time.perf_counter() - start_time:.2f}s"
    )

    print("Total Routes:", len(result))

    print(
        "Eligible Routes:",
        int(result["eligible"].sum())
    )

    print(
        "Aggregation Methods:",
        len(get_methods())
    )

    print("Detour Factors:", DETOUR_FACTORS)

    print("Source Hash Verification: PASS")

    print("Linear Transform Verification: PASS")

    print("Detour Violations: 0")

    print("Final Aggregation Selected: No")

    print("Additional Model Training: Not Performed")

    print("Test Re-evaluation: Not Performed")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()