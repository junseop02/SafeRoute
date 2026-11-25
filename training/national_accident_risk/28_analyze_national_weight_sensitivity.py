
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

RESULTS_DIR = (
    ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
    / "results"
)

ROUTES_PATH = (
    RESULTS_DIR / "national_route_aggregation.parquet"
)

CONFIG_PATH = (
    RESULTS_DIR / "national_combined_scoring_config.json"
)

SUMMARY_PATH = (
    RESULTS_DIR / "national_weight_sensitivity_summary.csv"
)

PARETO_PATH = (
    RESULTS_DIR / "national_weight_sensitivity_pareto.csv"
)

CANDIDATES_PATH = (
    RESULTS_DIR / "national_weight_sensitivity_candidates.csv"
)

OUTPUT_CONFIG_PATH = (
    RESULTS_DIR / "national_weight_sensitivity_config.json"
)

RISK_COLUMN = "N00_raw_A02_weighted_mean"

DETOUR_FACTORS = [1.5, 2.0, 3.0]

INTERSECTION_REFERENCES = [50.0, 74.0, 100.0]

MINIMUM_COVERAGE = 0.8

WEIGHT_STEP = 5


# ==================================================
# 2. 원본 파일 무결성 검증
# ==================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def check_output_paths():

    paths = [
        SUMMARY_PATH,
        PARETO_PATH,
        CANDIDATES_PATH,
        OUTPUT_CONFIG_PATH
    ]

    for path in paths:

        if path.exists():

            raise FileExistsError(
                f"Output already exists: {path}"
            )


# ==================================================
# 3. 데이터 불러오기
# ==================================================

def load_data():

    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    assert config["model_id"] == "T09"

    assert config["minimum_coverage"] == 0.8

    assert config["route_score_formula_selected"] is False

    expected_hash = config["source_sha256"][
        ROUTES_PATH.name
    ]

    assert sha256_file(ROUTES_PATH) == expected_hash

    routes = pd.read_parquet(ROUTES_PATH)

    required = [
        "route_id",
        "origin_id",
        "destination_id",
        "geometry_sha256",
        "distance_m",
        "duration_s",
        "original_coverage",
        "effective_coverage",
        "eligible",
        RISK_COLUMN,
        "wide_road_ratio",
        "narrow_road_ratio",
        "weighted_intersection_nodes",
        "A04_positive_length_ratio",
        "N00_raw_A03_max"
    ]

    missing = set(required) - set(routes.columns)

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    assert routes["route_id"].is_unique

    assert len(routes) == config["total_routes"]

    assert routes[
        ["origin_id", "destination_id", "geometry_sha256"]
    ].notna().all().all()

    assert np.isfinite(
        routes[["distance_m", "duration_s"]]
        .to_numpy(dtype=float)
    ).all()

    assert (routes["distance_m"] > 0).all()

    assert (routes["duration_s"] > 0).all()

    eligible = routes.loc[routes["eligible"]].copy()

    assert len(eligible) == config["eligible_routes"]

    assert (
        eligible["original_coverage"]
        >= MINIMUM_COVERAGE
    ).all()

    assert (
        eligible["effective_coverage"]
        >= MINIMUM_COVERAGE
    ).all()

    bounded_columns = [
        RISK_COLUMN,
        "wide_road_ratio",
        "narrow_road_ratio",
        "A04_positive_length_ratio",
        "N00_raw_A03_max"
    ]

    values = eligible[
        bounded_columns
    ].to_numpy(dtype=float)

    assert np.isfinite(values).all()

    assert ((values >= 0) & (values <= 1)).all()

    assert np.allclose(
        eligible["wide_road_ratio"]
        + eligible["narrow_road_ratio"],
        1.0
    )

    intersections = eligible[
        "weighted_intersection_nodes"
    ].to_numpy(dtype=float)

    assert np.isfinite(intersections).all()

    assert (intersections >= 0).all()

    print("\n===== Dataset Validation =====")

    print(f"Total Routes: {len(routes)}")

    print(f"Eligible Routes: {len(eligible)}")

    print("Dataset Validation: PASS")

    return routes, config


# ==================================================
# 4. 가중치 후보 생성
# ==================================================

def build_weight_candidates():

    candidates = []

    for reference in INTERSECTION_REFERENCES:

        for risk in range(0, 101, WEIGHT_STEP):

            for narrow in range(
                0,
                101 - risk,
                WEIGHT_STEP
            ):

                intersection = 100 - risk - narrow

                candidate_id = (
                    f"R{risk:03d}_"
                    f"N{narrow:03d}_"
                    f"I{intersection:03d}_"
                    f"REF{int(reference)}"
                )

                candidates.append({
                    "candidate_id": candidate_id,
                    "w_risk": risk / 100,
                    "w_narrow": narrow / 100,
                    "w_intersection": intersection / 100,
                    "intersection_reference": reference
                })

    candidates = pd.DataFrame(candidates)

    assert len(candidates) == 693

    assert candidates["candidate_id"].is_unique

    assert np.allclose(
        candidates["w_risk"]
        + candidates["w_narrow"]
        + candidates["w_intersection"],
        1.0
    )

    print("\n===== Weight Candidates =====")

    print(f"Weight Step: {WEIGHT_STEP}%")

    print(f"Total Candidates: {len(candidates)}")

    return candidates


# ==================================================
# 5. OD별 추천 후보 구성
# ==================================================

def prepare_groups(routes, factor):

    groups = []

    counts = {
        "total_od": 0,
        "no_eligible": 0,
        "single_eligible": 0,
        "two_eligible": 0,
        "three_or_more": 0
    }

    for _, group in routes.groupby(
        ["origin_id", "destination_id"],
        sort=False
    ):

        counts["total_od"] += 1

        shortest = float(
            group["distance_m"].min()
        )

        eligible = group.loc[
            group["eligible"]
            & (
                group["distance_m"]
                <= shortest * factor + 1e-8
            )
        ].copy()

        eligible = eligible.sort_values(
            [
                "distance_m",
                "duration_s",
                "route_id"
            ],
            kind="stable"
        )

        eligible = eligible.drop_duplicates(
            subset="geometry_sha256",
            keep="first"
        )

        size = len(eligible)

        if size == 0:

            counts["no_eligible"] += 1

        elif size == 1:

            counts["single_eligible"] += 1

        elif size == 2:

            counts["two_eligible"] += 1

        else:

            counts["three_or_more"] += 1

        if size < 2:

            continue

        groups.append({
            "routes": eligible,
            "shortest": shortest,
            "size": size
        })

    print(f"\n===== Detour Factor {factor} =====")

    for key, value in counts.items():

        print(f"{key}: {value}")

    return groups, counts


# ==================================================
# 6. 벡터 연산용 배열 생성
# ==================================================

def build_arrays(groups):

    group_count = len(groups)

    max_candidates = max(
        group["size"]
        for group in groups
    )

    shape = (group_count, max_candidates)

    arrays = {
        name: np.zeros(shape, dtype=np.float64)
        for name in [
            "risk",
            "wide",
            "narrow",
            "intersection",
            "distance",
            "duration",
            "detour",
            "positive_ratio",
            "maximum_risk"
        ]
    }

    valid = np.zeros(shape, dtype=bool)

    sizes = np.zeros(
        group_count,
        dtype=np.int32
    )

    for i, group in enumerate(groups):

        df = group["routes"]

        size = group["size"]

        sizes[i] = size

        valid[i, :size] = True

        arrays["risk"][i, :size] = df[
            RISK_COLUMN
        ].to_numpy()

        arrays["wide"][i, :size] = df[
            "wide_road_ratio"
        ].to_numpy()

        arrays["narrow"][i, :size] = df[
            "narrow_road_ratio"
        ].to_numpy()

        arrays["intersection"][i, :size] = df[
            "weighted_intersection_nodes"
        ].to_numpy()

        arrays["distance"][i, :size] = df[
            "distance_m"
        ].to_numpy()

        arrays["duration"][i, :size] = df[
            "duration_s"
        ].to_numpy()

        arrays["detour"][i, :size] = (
            df["distance_m"].to_numpy()
            / group["shortest"]
        )

        arrays["positive_ratio"][i, :size] = df[
            "A04_positive_length_ratio"
        ].to_numpy()

        arrays["maximum_risk"][i, :size] = df[
            "N00_raw_A03_max"
        ].to_numpy()

    return arrays, valid, sizes


# ==================================================
# 7. 추천 경로 선택
# ==================================================

def select_routes(burden, valid):

    # 동일 점수일 경우 사전에 정렬된
    # 거리, 시간, route_id 순으로 선택
    masked = np.where(
        valid,
        np.round(burden, 12),
        np.inf
    )

    selected = np.argmin(
        masked,
        axis=1
    )

    rows = np.arange(len(selected))

    assert valid[rows, selected].all()

    return selected


def extract_selected(arrays, selected):

    rows = np.arange(len(selected))

    return {
        name: values[rows, selected]
        for name, values in arrays.items()
    }


# ==================================================
# 8. 후보별 추천 결과 집계
# ==================================================

def summarize_selection(
    candidate,
    selected,
    baseline_selected,
    values,
    baseline_values,
    scope_mask,
    scope,
    factor
):

    risk = values["risk"][scope_mask]

    wide = values["wide"][scope_mask]

    intersection = values[
        "intersection"
    ][scope_mask]

    base_risk = baseline_values[
        "risk"
    ][scope_mask]

    base_wide = baseline_values[
        "wide"
    ][scope_mask]

    base_intersection = baseline_values[
        "intersection"
    ][scope_mask]

    risk_change = risk - base_risk

    wide_change = wide - base_wide

    intersection_change = (
        intersection - base_intersection
    )

    detour = values["detour"][scope_mask]

    assert (detour <= factor + 1e-8).all()

    result = candidate.to_dict()

    result.update({
        "scope": scope,
        "detour_factor": factor,
        "od_count": int(scope_mask.sum()),

        "mean_raw_risk": float(risk.mean()),

        "mean_wide_road_ratio": float(wide.mean()),

        "mean_intersection_nodes": float(
            intersection.mean()
        ),

        "mean_distance_m": float(
            values["distance"][scope_mask].mean()
        ),

        "mean_duration_s": float(
            values["duration"][scope_mask].mean()
        ),

        "mean_detour_ratio": float(detour.mean()),

        "max_detour_ratio": float(detour.max()),

        "mean_positive_length_ratio": float(
            values["positive_ratio"][scope_mask].mean()
        ),

        "mean_maximum_risk": float(
            values["maximum_risk"][scope_mask].mean()
        ),

        "changed_ratio_vs_raw": float(
            (
                selected[scope_mask]
                != baseline_selected[scope_mask]
            ).mean()
        ),

        "mean_risk_change": float(
            risk_change.mean()
        ),

        "mean_wide_change": float(
            wide_change.mean()
        ),

        "mean_intersection_change": float(
            intersection_change.mean()
        ),

        "risk_increased_od_ratio": float(
            (risk_change > 1e-10).mean()
        ),

        "wide_improved_od_ratio": float(
            (wide_change > 1e-10).mean()
        ),

        "wide_worsened_od_ratio": float(
            (wide_change < -1e-10).mean()
        ),

        "intersection_improved_od_ratio": float(
            (intersection_change < -1e-10).mean()
        ),

        "intersection_worsened_od_ratio": float(
            (intersection_change > 1e-10).mean()
        ),

        "risk_increase_p95": float(
            np.quantile(risk_change, 0.95)
        ),

        "maximum_risk_increase": float(
            risk_change.max()
        )
    })

    return result


# ==================================================
# 9. 가중치 민감도 실험
# ==================================================

def run_experiments(routes, candidates, config):

    records = []

    for factor in DETOUR_FACTORS:

        groups, counts = prepare_groups(
            routes,
            factor
        )

        expected = config[
            "candidate_group_summary"
        ][str(factor)]

        assert counts == expected, (
            f"Candidate group mismatch: {factor}"
        )

        arrays, valid, sizes = build_arrays(groups)

        baseline_selected = select_routes(
            arrays["risk"],
            valid
        )

        baseline_values = extract_selected(
            arrays,
            baseline_selected
        )

        scopes = {
            "research_2plus": sizes >= 2,
            "service_3plus": sizes >= 3
        }

        normalized_intersections = {
            reference: np.clip(
                arrays["intersection"] / reference,
                0.0,
                1.0
            )
            for reference in INTERSECTION_REFERENCES
        }

        for i, candidate in candidates.iterrows():

            reference = candidate[
                "intersection_reference"
            ]

            burden = (
                candidate["w_risk"] * arrays["risk"]
                + candidate["w_narrow"] * arrays["narrow"]
                + candidate["w_intersection"]
                * normalized_intersections[reference]
            )

            assert np.isfinite(burden[valid]).all()

            assert (
                (burden[valid] >= -1e-10)
                & (burden[valid] <= 1.0 + 1e-10)
            ).all()

            selected = select_routes(
                burden,
                valid
            )

            values = extract_selected(
                arrays,
                selected
            )

            for scope, mask in scopes.items():

                result = summarize_selection(
                    candidate,
                    selected,
                    baseline_selected,
                    values,
                    baseline_values,
                    mask,
                    scope,
                    factor
                )

                records.append(result)

            if (
                (i + 1) % 100 == 0
                or i + 1 == len(candidates)
            ):

                print(
                    f"Factor {factor} | "
                    f"Candidates: {i + 1:,}"
                    f" / {len(candidates):,}"
                )

    summary = pd.DataFrame(records)

    assert len(summary) == 4158

    assert not summary.duplicated(
        [
            "candidate_id",
            "scope",
            "detour_factor"
        ]
    ).any()

    print("\n===== Experiment Validation =====")

    print(f"Summary Records: {len(summary):,}")

    print("Detour Violations: 0")

    print("Score Validation: PASS")

    return summary


# ==================================================
# 10. 기존 실험 결과 재현 검증
# ==================================================

def validate_previous_results(summary):

    previous = pd.read_csv(
        RESULTS_DIR
        / "national_combined_scoring_summary.csv"
    )

    mapping = {
        "RAW_C01": "R100_N000_I000_REF74",
        "RAW_C02": "R075_N025_I000_REF74",
        "RAW_C03": "R055_N045_I000_REF74",
        "RAW_C04": "R050_N050_I000_REF74",
        "RAW_C05": "R055_N025_I020_REF74",
        "RAW_C06": "R070_N020_I010_REF74",
        "RAW_C07": "R040_N040_I020_REF74",
        "POLICY_C08": "R000_N100_I000_REF74",
        "POLICY_C09": "R000_N055_I045_REF74"
    }

    previous = previous.loc[
        previous["method"].isin(mapping)
    ].copy()

    previous["candidate_id"] = previous[
        "method"
    ].map(mapping)

    merged = previous.merge(
        summary,
        on=[
            "candidate_id",
            "scope",
            "detour_factor"
        ],
        suffixes=("_old", "_new"),
        validate="one_to_one"
    )

    assert len(merged) == 54

    columns = [
        "od_count",
        "mean_raw_risk",
        "mean_wide_road_ratio",
        "mean_intersection_nodes",
        "mean_distance_m",
        "mean_duration_s",
        "mean_detour_ratio",
        "mean_positive_length_ratio",
        "mean_maximum_risk",
        "changed_ratio_vs_raw"
    ]

    print("\n===== Previous Result Reproduction =====")

    for column in columns:

        old = merged[
            f"{column}_old"
        ].to_numpy()

        new = merged[
            f"{column}_new"
        ].to_numpy()

        difference = np.max(
            np.abs(old - new)
        )

        print(
            f"{column}: "
            f"max difference={difference:.10f}"
        )

        assert np.allclose(
            old,
            new,
            rtol=1e-8,
            atol=1e-6
        ), f"Previous result mismatch: {column}"

    print("Previous Result Reproduction: PASS")


# ==================================================
# 11. 파레토 최적 후보 분석
# ==================================================

def pareto_mask(values):

    # 모든 목적함수를 최소화 방향으로 통일
    count = len(values)

    keep = np.ones(count, dtype=bool)

    tolerance = 1e-10

    for i in range(count):

        no_worse = np.all(
            values <= values[i] + tolerance,
            axis=1
        )

        strictly_better = np.any(
            values < values[i] - tolerance,
            axis=1
        )

        dominated = np.any(
            no_worse & strictly_better
        )

        keep[i] = not dominated

    return keep


def analyze_pareto(summary):

    result = []

    for _, group in summary.groupby(
        ["scope", "detour_factor"]
    ):

        values = np.column_stack([
            group["mean_raw_risk"].to_numpy(),

            -group["mean_wide_road_ratio"].to_numpy(),

            group["mean_intersection_nodes"].to_numpy()
        ])

        mask = pareto_mask(values)

        selected = group.loc[mask].copy()

        selected["pareto_optimal"] = True

        result.append(selected)

    pareto = pd.concat(
        result,
        ignore_index=True
    )

    print("\n===== Pareto Summary =====")

    print(
        pareto.groupby(
            ["scope", "detour_factor"]
        ).size().to_string()
    )

    return pareto


# ==================================================
# 12. 모델 부담 증가 제한별 후보 비교
# ==================================================

def analyze_tradeoffs(pareto):

    results = []

    # 모델 단독 추천 대비 평균 점수 증가 허용치
    risk_budgets = [
        0.0,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02
    ]

    for (scope, factor), group in pareto.groupby(
        ["scope", "detour_factor"]
    ):

        for budget in risk_budgets:

            eligible = group.loc[
                group["mean_risk_change"]
                <= budget + 1e-10
            ].copy()

            if eligible.empty:

                continue

            # 넓은 도로 비율 우선
            # 이후 교차로, 모델 부담, 우회 배율 비교
            eligible = eligible.sort_values(
                [
                    "mean_wide_road_ratio",
                    "mean_intersection_nodes",
                    "mean_raw_risk",
                    "mean_detour_ratio",
                    "candidate_id"
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                    True
                ],
                kind="stable"
            )

            best = eligible.iloc[0].to_dict()

            best["risk_increase_budget"] = budget

            best["eligible_pareto_candidates"] = len(
                eligible
            )

            results.append(best)

    candidates = pd.DataFrame(results)

    print("\n===== Tradeoff Candidates =====")

    columns = [
        "risk_increase_budget",
        "candidate_id",
        "w_risk",
        "w_narrow",
        "w_intersection",
        "intersection_reference",
        "mean_raw_risk",
        "mean_wide_road_ratio",
        "mean_intersection_nodes",
        "mean_risk_change",
        "mean_wide_change",
        "mean_detour_ratio",
        "risk_increased_od_ratio"
    ]

    for factor in DETOUR_FACTORS:

        selected = candidates.loc[
            (candidates["scope"] == "service_3plus")
            & (candidates["detour_factor"] == factor)
        ]

        print(f"\nService Scope | Factor {factor}")

        print(
            selected[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    return candidates


# ==================================================
# 13. 결과 저장
# ==================================================

def save_results(
    summary,
    pareto,
    candidates,
    source_hashes
):

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    pareto.to_csv(
        PARETO_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    candidates.to_csv(
        CANDIDATES_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "model_id": "T09",
        "experiment_scope": (
            "existing_cheonan_asan_routes"
        ),
        "normalization": "RAW",
        "aggregation": "length_weighted_mean",
        "weight_step_percent": WEIGHT_STEP,
        "weight_combinations": 231,
        "total_formula_candidates": 693,
        "intersection_references": (
            INTERSECTION_REFERENCES
        ),
        "detour_factors": DETOUR_FACTORS,
        "minimum_coverage": MINIMUM_COVERAGE,
        "research_minimum_candidates": 2,
        "service_minimum_candidates": 3,
        "pareto_objectives": [
            "minimize_mean_raw_risk",
            "maximize_mean_wide_road_ratio",
            "minimize_mean_intersection_nodes"
        ],
        "risk_increase_budgets": [
            0.0,
            0.001,
            0.0025,
            0.005,
            0.01,
            0.02
        ],
        "summary_records": len(summary),
        "pareto_records": len(pareto),
        "tradeoff_records": len(candidates),
        "source_sha256": source_hashes,
        "previous_result_reproduction": "PASS",
        "detour_violations": 0,
        "selected_candidate": None,
        "route_score_formula_selected": False,
        "additional_model_training": False,
        "test_re_evaluated": False,
        "deployment_status": "not_deployed",
        "limitations": [
            "Regional offline route experiment.",
            "No actual route-level accident outcomes.",
            "No novice driver difficulty labels.",
            "Model score is not accident probability.",
            "Lane and intersection information may "
            "be represented in both model and policy.",
            "Pareto objectives use aggregate means.",
            "Mean risk budgets do not guarantee "
            "individual route risk limits.",
            "Identical formula outcomes may remain "
            "under different reference values.",
            "Final policy weights are not selected."
        ]
    }

    with OUTPUT_CONFIG_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n===== Saved Results =====")

    for path in [
        SUMMARY_PATH,
        PARETO_PATH,
        CANDIDATES_PATH,
        OUTPUT_CONFIG_PATH
    ]:

        size = path.stat().st_size / (1024 ** 2)

        print(f"Path: {path}")

        print(f"File Size: {size:.4f} MiB")


# ==================================================
# 14. 메인 함수
# ==================================================

def main():

    start = time.perf_counter()

    print(
        "===== National Weight Sensitivity Analysis ====="
    )

    check_output_paths()

    source_paths = [
        ROUTES_PATH,
        CONFIG_PATH,
        RESULTS_DIR
        / "national_combined_scoring_summary.csv"
    ]

    source_hashes = {
        path.name: sha256_file(path)
        for path in source_paths
    }

    routes, config = load_data()

    candidates = build_weight_candidates()

    summary = run_experiments(
        routes,
        candidates,
        config
    )

    validate_previous_results(summary)

    pareto = analyze_pareto(summary)

    tradeoffs = analyze_tradeoffs(pareto)

    for path in source_paths:

        assert (
            sha256_file(path)
            == source_hashes[path.name]
        ), f"Source modified: {path}"

    print("\nSource Hash Verification: PASS")

    save_results(
        summary,
        pareto,
        tradeoffs,
        source_hashes
    )

    elapsed = time.perf_counter() - start

    print("\n===== Processing Completed =====")

    print(f"Elapsed Time: {elapsed:.2f}s")

    print(f"Formula Candidates: {len(candidates):,}")

    print(f"Summary Records: {len(summary):,}")

    print(f"Pareto Records: {len(pareto):,}")

    print(f"Tradeoff Records: {len(tradeoffs):,}")

    print("Detour Violations: 0")

    print("Final Weights Selected: No")

    print("Additional Model Training: Not Performed")

    print("Test Re-evaluation: Not Performed")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":

    main()