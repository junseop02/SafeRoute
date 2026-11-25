
import gzip
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

REFERENCE = ROOT.parent / "SafeRoute_새디자인_20261124"

TILE_DIR = REFERENCE / "data" / "tiles"

PROCESSED = (
    ROOT / "data" / "national_accident_risk" / "processed"
)

RESULTS = PROCESSED / "results"

MATCH_PATH = ROOT / "data" / "route_link_matches.json"

ROUTE_PATH = RESULTS / "national_route_aggregation.parquet"

SCORE_PATH = PROCESSED / "national_link_risk_scores.parquet"

CONFIG_PATH = RESULTS / "national_weight_sensitivity_config.json"

OUTPUT_ROUTES = RESULTS / "national_final_policy_routes.parquet"

OUTPUT_SELECTIONS = RESULTS / "national_final_policy_selections.parquet"

OUTPUT_SUMMARY = RESULTS / "national_final_policy_summary.csv"

OUTPUT_COVERAGE = RESULTS / "national_final_policy_coverage.csv"

OUTPUT_CONFIG = RESULTS / "national_final_policy_config.json"

MIN_COVERAGE = 0.8

DETOUR_FACTORS = [1.5, 2.0, 3.0]

RISK_COLUMN = "N00_raw_A02_weighted_mean"

POLICIES = {
    "M00_shortest": None,
    "M01_fastest": None,
    "M02_previous_national": None,
    "M03_T09_only": (1.00, 0.00, 0.00, 100),
    "M04_C05": (0.55, 0.25, 0.20, 74),
    "M05_C06": (0.70, 0.20, 0.10, 74),
    "M06_P01": (0.80, 0.10, 0.10, 100),
    "M07_P02": (0.75, 0.20, 0.05, 100),
    "M08_P03": (0.60, 0.35, 0.05, 100),
    "M09_wide_only": (0.00, 1.00, 0.00, 100)
}


# ==================================================
# 2. 파일 검증
# ==================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def check_outputs():

    for path in [
        OUTPUT_ROUTES,
        OUTPUT_SELECTIONS,
        OUTPUT_SUMMARY,
        OUTPUT_COVERAGE,
        OUTPUT_CONFIG
    ]:
        if path.exists():
            raise FileExistsError(path)


# ==================================================
# 3. 입력 데이터 불러오기
# ==================================================

def load_data():

    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = json.load(f)

    assert config["model_id"] == "T09"
    assert config["route_score_formula_selected"] is False

    routes = pd.read_parquet(ROUTE_PATH)

    with MATCH_PATH.open(encoding="utf-8") as f:
        matches = json.load(f)

    assert routes["route_id"].is_unique
    assert len(matches) == len(routes) == 6402

    match_map = {
        item["route_id"]: item
        for item in matches
    }

    assert len(match_map) == len(matches)

    assert set(match_map) == set(routes["route_id"])

    required_ids = {
        str(link_id)
        for item in matches
        for link_id, length in item["links"]
        if float(length) > 0
    }

    scores = pd.read_parquet(
        SCORE_PATH,
        columns=[
            "link_id",
            "hotspot_score",
            "hotspot_predicted_label",
            "lanes",
            "intersection_nodes_200m"
        ]
    )

    scores["link_id"] = scores["link_id"].astype(str)

    scores = scores.loc[
        scores["link_id"].isin(required_ids)
    ].copy()

    assert scores["link_id"].is_unique

    assert scores["hotspot_score"].between(0, 1).all()

    new_map = scores.set_index("link_id").to_dict("index")

    print("\n===== Input Data =====")
    print("Total Routes:", len(routes))
    print("Required Link IDs:", len(required_ids))
    print("New Model Available Links:", len(new_map))

    return routes, match_map, required_ids, new_map


# ==================================================
# 4. 기존 전국 모델 점수 추출
# ==================================================

def load_previous_scores(required_ids, tiles):

    old_map = {}
    duplicates = 0

    for i, path in enumerate(tiles, start=1):

        with gzip.open(path, "rt", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                row = json.loads(line)

                assert len(row) == 6

                link_id = str(row[0])

                if link_id not in required_ids:
                    continue

                score = float(row[2])
                lanes = float(row[3])
                intersections = float(row[4])
                accident_distance = float(row[5])

                values = np.array([
                    score,
                    lanes,
                    intersections,
                    accident_distance
                ])

                assert np.isfinite(values).all()
                assert 0 <= score <= 100
                assert lanes >= 1
                assert intersections >= 0
                assert accident_distance >= 0

                record = (
                    score,
                    lanes,
                    intersections,
                    accident_distance
                )

                if link_id in old_map:

                    assert old_map[link_id] == record

                    duplicates += 1

                else:

                    old_map[link_id] = record

        if i % 100 == 0 or i == len(tiles):

            print(
                f"Tiles: {i:,}/{len(tiles):,} | "
                f"Required Links Found: {len(old_map):,}"
            )

    print("\n===== Previous Model Scores =====")
    print("Matched Link IDs:", len(old_map))
    print("Duplicate Rows Checked:", duplicates)
    print("Missing Link IDs:", len(required_ids - set(old_map)))

    return old_map


# ==================================================
# 5. 동일 매칭 링크 기준 경로 집계
# ==================================================

def aggregate_route(item, old_map, new_map):

    geometry_length = float(item["geometry_length_m"])
    matched_length = float(item["matched_length_m"])

    assert geometry_length > 0
    assert matched_length >= 0

    old_length = 0.0
    new_length = 0.0
    common_length = 0.0

    old_sum = 0.0
    new_sum = 0.0

    common_old_sum = 0.0
    common_new_sum = 0.0

    common_wide = 0.0
    common_intersections = 0.0
    common_positive = 0.0
    common_near = 0.0
    common_maximum = 0.0

    new_wide = 0.0
    new_intersections = 0.0
    new_positive = 0.0
    new_maximum = 0.0

    link_length_sum = 0.0

    for link_id, length in item["links"]:

        link_id = str(link_id)
        length = float(length)

        assert np.isfinite(length)
        assert length >= 0

        if length == 0:
            continue

        link_length_sum += length

        old = old_map.get(link_id)
        new = new_map.get(link_id)

        if old is not None:

            old_length += length
            old_sum += length * old[0]

        if new is not None:

            risk = float(new["hotspot_score"])
            lanes = float(new["lanes"])

            intersections = float(
                new["intersection_nodes_200m"]
            )

            positive = int(
                new["hotspot_predicted_label"]
            )

            new_length += length
            new_sum += length * risk

            new_wide += length * (lanes >= 2)
            new_intersections += length * intersections
            new_positive += length * positive
            new_maximum = max(new_maximum, risk)

        if old is not None and new is not None:

            assert old[1] == lanes
            assert old[2] == intersections

            common_length += length

            common_old_sum += length * old[0]
            common_new_sum += length * risk

            common_wide += length * (lanes >= 2)

            common_intersections += (
                length * intersections
            )

            common_positive += length * positive
            common_near += length * (old[3] <= 150)

            common_maximum = max(common_maximum, risk)

    assert np.isclose(
        link_length_sum,
        matched_length,
        rtol=1e-5,
        atol=0.1
    )

    def ratio(value, denominator):

        if denominator <= 0:
            return np.nan

        return value / denominator

    old_coverage = old_length / geometry_length
    new_coverage = new_length / geometry_length
    common_coverage = common_length / geometry_length

    original_coverage = float(item["coverage"])

    assert old_coverage <= 1.00001
    assert new_coverage <= 1.00001
    assert common_coverage <= 1.00001

    return {
        "route_id": item["route_id"],

        "old_coverage": old_coverage,
        "new_coverage": new_coverage,
        "common_coverage": common_coverage,

        "old_eligible": (
            original_coverage >= MIN_COVERAGE
            and old_coverage >= MIN_COVERAGE
        ),

        "new_eligible": (
            original_coverage >= MIN_COVERAGE
            and new_coverage >= MIN_COVERAGE
        ),

        "common_eligible": (
            original_coverage >= MIN_COVERAGE
            and common_coverage >= MIN_COVERAGE
        ),

        "old_model_burden": ratio(old_sum, old_length),

        "new_model_risk": ratio(new_sum, new_length),

        "common_old_burden": ratio(
            common_old_sum, common_length
        ),

        "common_new_risk": ratio(
            common_new_sum, common_length
        ),

        "common_wide_ratio": ratio(
            common_wide, common_length
        ),

        "common_intersections": ratio(
            common_intersections, common_length
        ),

        "common_positive_ratio": ratio(
            common_positive, common_length
        ),

        "common_near_hotspot_ratio": ratio(
            common_near, common_length
        ),

        "common_maximum_risk": (
            common_maximum
            if common_length > 0
            else np.nan
        ),

        "new_wide_ratio": ratio(new_wide, new_length),

        "new_intersections": ratio(
            new_intersections, new_length
        ),

        "new_positive_ratio": ratio(
            new_positive, new_length
        ),

        "new_maximum_risk": (
            new_maximum
            if new_length > 0
            else np.nan
        )
    }


def aggregate_all_routes(
    routes, match_map, old_map, new_map
):

    records = []

    for i, item in enumerate(
        match_map.values(), start=1
    ):

        records.append(
            aggregate_route(item, old_map, new_map)
        )

        if i % 1000 == 0 or i == len(match_map):

            print(
                f"Aggregated Routes: "
                f"{i:,}/{len(match_map):,}"
            )

    aggregated = pd.DataFrame(records)

    merged = routes.merge(
        aggregated,
        on="route_id",
        validate="one_to_one"
    )

    assert len(merged) == len(routes)

    assert np.array_equal(
        merged["eligible"].to_numpy(dtype=bool),
        merged["new_eligible"].to_numpy(dtype=bool)
    )

    eligible = merged.loc[merged["new_eligible"]]

    assert np.allclose(
        eligible[RISK_COLUMN],
        eligible["new_model_risk"],
        rtol=1e-8,
        atol=1e-8
    )

    assert np.allclose(
        eligible["wide_road_ratio"],
        eligible["new_wide_ratio"],
        atol=1e-8
    )

    assert np.allclose(
        eligible["weighted_intersection_nodes"],
        eligible["new_intersections"],
        atol=1e-8
    )

    assert np.allclose(
        eligible["A04_positive_length_ratio"],
        eligible["new_positive_ratio"],
        atol=1e-8
    )

    assert np.allclose(
        eligible["N00_raw_A03_max"],
        eligible["new_maximum_risk"],
        atol=1e-8
    )

    common = merged.loc[merged["common_eligible"]]

    columns = [
        "common_old_burden",
        "common_new_risk",
        "common_wide_ratio",
        "common_intersections",
        "common_positive_ratio",
        "common_near_hotspot_ratio",
        "common_maximum_risk"
    ]

    assert np.isfinite(
        common[columns].to_numpy()
    ).all()

    print("\n===== Route Coverage =====")

    for column in [
        "old_eligible",
        "new_eligible",
        "common_eligible"
    ]:

        print(column, int(merged[column].sum()))

    print("New Aggregation Reproduction: PASS")

    return merged


# ==================================================
# 6. OD별 경로 후보 구성
# ==================================================

def prepare_candidates(group, shortest, factor, column):

    selected = group.loc[
        group[column]
        & (
            group["distance_m"]
            <= shortest * factor + 1e-8
        )
    ].copy()

    selected = selected.sort_values(
        ["distance_m", "duration_s", "route_id"],
        kind="stable"
    )

    return selected.drop_duplicates(
        subset="geometry_sha256",
        keep="first"
    )


# ==================================================
# 7. 서비스 적용 범위 비교
# ==================================================

def compare_coverage(routes):

    records = []

    groups = list(
        routes.groupby(
            ["origin_id", "destination_id"],
            sort=False
        )
    )

    for factor in DETOUR_FACTORS:

        for column in [
            "old_eligible",
            "new_eligible",
            "common_eligible"
        ]:

            counts = {
                "zero": 0,
                "one": 0,
                "two": 0,
                "three_or_more": 0
            }

            for _, group in groups:

                shortest = float(
                    group["distance_m"].min()
                )

                candidates = prepare_candidates(
                    group, shortest, factor, column
                )

                size = len(candidates)

                key = (
                    ["zero", "one", "two"][size]
                    if size < 3
                    else "three_or_more"
                )

                counts[key] += 1

            assert sum(counts.values()) == len(groups)

            records.append({
                "detour_factor": factor,
                "coverage_scope": column,
                "total_od": len(groups),
                **counts,
                "recommendable_ratio": (
                    counts["three_or_more"] / len(groups)
                )
            })

    result = pd.DataFrame(records)

    print("\n===== Recommendation Coverage =====")
    print(result.to_string(index=False))

    return result


# ==================================================
# 8. 최종 정책별 점수 계산
# ==================================================

def policy_burden(candidates, method, scope):

    if method == "M00_shortest":

        return candidates["distance_m"].to_numpy()

    if method == "M01_fastest":

        return candidates["duration_s"].to_numpy()

    if method == "M02_previous_national":

        column = (
            "common_old_burden"
            if scope == "common"
            else "old_model_burden"
        )

        return candidates[column].to_numpy() / 100

    if scope == "common":

        risk = candidates["common_new_risk"].to_numpy()

        narrow = (
            1 - candidates["common_wide_ratio"].to_numpy()
        )

        intersections = candidates[
            "common_intersections"
        ].to_numpy()

    else:

        risk = candidates["new_model_risk"].to_numpy()

        narrow = (
            1 - candidates["new_wide_ratio"].to_numpy()
        )

        intersections = candidates[
            "new_intersections"
        ].to_numpy()

    w_risk, w_narrow, w_intersection, reference = (
        POLICIES[method]
    )

    burden = (
        w_risk * risk
        + w_narrow * narrow
        + w_intersection
        * np.clip(intersections / reference, 0, 1)
    )

    assert np.isfinite(burden).all()

    assert (
        (burden >= -1e-10)
        & (burden <= 1 + 1e-10)
    ).all()

    return burden


# ==================================================
# 9. 공통 경로 기준 정책 비교
# ==================================================

def compare_policies(routes):

    records = []

    groups = list(
        routes.groupby(
            ["origin_id", "destination_id"],
            sort=False
        )
    )

    for factor in DETOUR_FACTORS:

        for scope in ["common", "new_full"]:

            eligible_column = (
                "common_eligible"
                if scope == "common"
                else "new_eligible"
            )

            methods = list(POLICIES)

            # 신규 모델 전체 범위에서는
            # 기존 모델 점수를 강제로 대입하지 않음
            if scope == "new_full":

                methods.remove("M02_previous_national")

            for (origin, destination), group in groups:

                shortest = float(
                    group["distance_m"].min()
                )

                candidates = prepare_candidates(
                    group,
                    shortest,
                    factor,
                    eligible_column
                )

                if len(candidates) < 3:
                    continue

                for method in methods:

                    burden = policy_burden(
                        candidates, method, scope
                    )

                    # 기존 실험과 동일한 동점 처리
                    index = int(
                        np.argmin(np.round(burden, 12))
                    )

                    row = candidates.iloc[index]

                    detour = float(
                        row["distance_m"] / shortest
                    )

                    assert detour <= factor + 1e-8

                    if scope == "common":

                        risk = row["common_new_risk"]
                        wide = row["common_wide_ratio"]

                        intersections = row[
                            "common_intersections"
                        ]

                        positive = row[
                            "common_positive_ratio"
                        ]

                        maximum = row[
                            "common_maximum_risk"
                        ]

                        coverage = row["common_coverage"]

                        old_burden = row[
                            "common_old_burden"
                        ]

                        near = row[
                            "common_near_hotspot_ratio"
                        ]

                    else:

                        risk = row["new_model_risk"]
                        wide = row["new_wide_ratio"]

                        intersections = row[
                            "new_intersections"
                        ]

                        positive = row["new_positive_ratio"]
                        maximum = row["new_maximum_risk"]
                        coverage = row["new_coverage"]

                        old_burden = np.nan
                        near = np.nan

                    records.append({
                        "scope": scope,
                        "detour_factor": factor,
                        "origin_id": origin,
                        "destination_id": destination,
                        "candidate_count": len(candidates),
                        "method": method,
                        "route_id": row["route_id"],
                        "raw_risk": risk,
                        "old_model_burden": old_burden,
                        "wide_road_ratio": wide,
                        "intersection_nodes": intersections,
                        "positive_length_ratio": positive,
                        "maximum_risk": maximum,
                        "near_hotspot_ratio": near,
                        "distance_m": row["distance_m"],
                        "duration_s": row["duration_s"],
                        "detour_ratio": detour,
                        "coverage": coverage
                    })

            print(
                f"Policy Comparison | "
                f"Factor {factor} | Scope {scope}: PASS"
            )

    result = pd.DataFrame(records)

    assert not result.duplicated([
        "scope",
        "detour_factor",
        "origin_id",
        "destination_id",
        "method"
    ]).any()

    assert (
        result["detour_ratio"]
        <= result["detour_factor"] + 1e-8
    ).all()

    return result


# ==================================================
# 10. 추천 결과 요약
# ==================================================

def summarize(selections):

    records = []

    keys = ["origin_id", "destination_id"]

    for (scope, factor), group in selections.groupby(
        ["scope", "detour_factor"]
    ):

        baseline = group.loc[
            group["method"] == "M03_T09_only"
        ].set_index(keys)

        previous = group.loc[
            group["method"] == "M02_previous_national"
        ].set_index(keys)

        for method, data in group.groupby("method"):

            data = data.set_index(keys)

            assert len(data) == len(baseline)

            data = data.reindex(baseline.index)

            assert data["route_id"].notna().all()

            risk_change = (
                data["raw_risk"] - baseline["raw_risk"]
            )

            wide_change = (
                data["wide_road_ratio"]
                - baseline["wide_road_ratio"]
            )

            record = {
                "scope": scope,
                "detour_factor": factor,
                "method": method,
                "od_count": len(data),

                "mean_raw_risk": data["raw_risk"].mean(),

                "mean_old_model_burden": (
                    data["old_model_burden"].mean()
                ),

                "mean_wide_road_ratio": (
                    data["wide_road_ratio"].mean()
                ),

                "mean_intersection_nodes": (
                    data["intersection_nodes"].mean()
                ),

                "mean_positive_length_ratio": (
                    data["positive_length_ratio"].mean()
                ),

                "mean_maximum_risk": (
                    data["maximum_risk"].mean()
                ),

                "mean_near_hotspot_ratio": (
                    data["near_hotspot_ratio"].mean()
                ),

                "mean_distance_m": data["distance_m"].mean(),

                "mean_duration_s": data["duration_s"].mean(),

                "mean_detour_ratio": (
                    data["detour_ratio"].mean()
                ),

                "max_detour_ratio": (
                    data["detour_ratio"].max()
                ),

                "mean_coverage": data["coverage"].mean(),

                "changed_ratio_vs_T09": (
                    data["route_id"] != baseline["route_id"]
                ).mean(),

                "mean_risk_change_vs_T09": risk_change.mean(),

                "risk_increase_p95": risk_change.quantile(.95),

                "maximum_risk_increase": risk_change.max(),

                "risk_increased_od_ratio": (
                    risk_change > 1e-10
                ).mean(),

                "wide_improved_od_ratio": (
                    wide_change > 1e-10
                ).mean(),

                "wide_worsened_od_ratio": (
                    wide_change < -1e-10
                ).mean()
            }

            if scope == "common":

                previous_aligned = previous.reindex(
                    data.index
                )

                assert previous_aligned[
                    "route_id"
                ].notna().all()

                record["changed_ratio_vs_previous"] = (
                    data["route_id"]
                    != previous_aligned["route_id"]
                ).mean()

                record["risk_change_vs_previous"] = (
                    data["raw_risk"]
                    - previous_aligned["raw_risk"]
                ).mean()

                record["wide_change_vs_previous"] = (
                    data["wide_road_ratio"]
                    - previous_aligned["wide_road_ratio"]
                ).mean()

                record["intersection_change_vs_previous"] = (
                    data["intersection_nodes"]
                    - previous_aligned["intersection_nodes"]
                ).mean()

            records.append(record)

    summary = pd.DataFrame(records)

    columns = [
        "method",
        "od_count",
        "mean_raw_risk",
        "mean_old_model_burden",
        "mean_wide_road_ratio",
        "mean_intersection_nodes",
        "mean_distance_m",
        "mean_duration_s",
        "mean_positive_length_ratio",
        "mean_near_hotspot_ratio",
        "risk_increase_p95",
        "maximum_risk_increase"
    ]

    for scope in ["common", "new_full"]:

        for factor in DETOUR_FACTORS:

            print(
                f"\n===== {scope} | "
                f"Detour {factor} ====="
            )

            selected = summary.loc[
                (summary["scope"] == scope)
                & (summary["detour_factor"] == factor)
            ]

            print(
                selected[columns].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}"
                )
            )

    return summary


# ==================================================
# 11. 기존 28번 결과 재현 검증
# ==================================================

def validate_previous(summary):

    previous = pd.read_csv(
        RESULTS / "national_weight_sensitivity_summary.csv"
    )

    mapping = {
        "M03_T09_only": "R100_N000_I000_REF100",
        "M04_C05": "R055_N025_I020_REF74",
        "M05_C06": "R070_N020_I010_REF74",
        "M06_P01": "R080_N010_I010_REF100",
        "M07_P02": "R075_N020_I005_REF100",
        "M08_P03": "R060_N035_I005_REF100",
        "M09_wide_only": "R000_N100_I000_REF100"
    }

    current = summary.loc[
        (summary["scope"] == "new_full")
        & summary["method"].isin(mapping)
    ].copy()

    current["candidate_id"] = current["method"].map(mapping)

    previous = previous.loc[
        previous["scope"] == "service_3plus"
    ]

    merged = current.merge(
        previous,
        on=["candidate_id", "detour_factor"],
        suffixes=("_new", "_old"),
        validate="one_to_one"
    )

    assert len(merged) == 21

    print("\n===== Previous Result Reproduction =====")

    for column in [
        "od_count",
        "mean_raw_risk",
        "mean_wide_road_ratio",
        "mean_intersection_nodes",
        "mean_distance_m",
        "mean_duration_s",
        "mean_positive_length_ratio",
        "mean_maximum_risk"
    ]:

        old = merged[f"{column}_old"].to_numpy()
        new = merged[f"{column}_new"].to_numpy()

        difference = np.max(np.abs(old - new))

        print(
            f"{column}: "
            f"max difference={difference:.10f}"
        )

        assert np.allclose(
            old, new, rtol=1e-8, atol=1e-6
        ), f"Mismatch: {column}"

    print("Previous Result Reproduction: PASS")


# ==================================================
# 12. 메인 함수
# ==================================================

def main():

    start = time.perf_counter()

    print("===== Final National Route Policy Comparison =====")

    check_outputs()

    tiles = sorted(TILE_DIR.glob("*.jsonl.gz"))

    assert len(tiles) == 1147

    source_paths = [
        MATCH_PATH,
        ROUTE_PATH,
        SCORE_PATH,
        CONFIG_PATH,
        RESULTS / "national_weight_sensitivity_summary.csv",
        REFERENCE / "matcher.js",
        REFERENCE / "kakao.js",
        REFERENCE / "data" / "national_manifest.json",
        *tiles
    ]

    print("\nCalculating Source Hashes...")

    source_hashes = {
        str(path.relative_to(ROOT.parent)): sha256_file(path)
        for path in source_paths
    }

    routes, match_map, required_ids, new_map = load_data()

    print("\n===== Reading Reference Tiles =====")

    old_map = load_previous_scores(required_ids, tiles)

    print("\n===== Aggregating Route Scores =====")

    routes = aggregate_all_routes(
        routes, match_map, old_map, new_map
    )

    coverage = compare_coverage(routes)

    selections = compare_policies(routes)

    summary = summarize(selections)

    validate_previous(summary)

    print("\n===== Source Integrity Verification =====")

    for path in source_paths:

        key = str(path.relative_to(ROOT.parent))

        assert sha256_file(path) == source_hashes[key]

    print("Source Hash Verification: PASS")

    routes.to_parquet(OUTPUT_ROUTES, index=False)

    selections.to_parquet(
        OUTPUT_SELECTIONS, index=False
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    coverage.to_csv(
        OUTPUT_COVERAGE,
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "model_id": "T09",
        "reference_service": REFERENCE.name,
        "reference_tile_count": len(tiles),
        "experiment_scope": "existing_cheonan_asan_routes",
        "total_routes": len(routes),
        "minimum_coverage": MIN_COVERAGE,
        "minimum_distinct_candidates": 3,
        "detour_factors": DETOUR_FACTORS,
        "policies": POLICIES,
        "comparison_scopes": [
            "common",
            "new_full"
        ],
        "common_scope_definition": (
            "Identical matched link support for both models"
        ),
        "selection_records": len(selections),
        "summary_records": len(summary),
        "source_sha256": source_hashes,
        "detour_violations": 0,
        "previous_result_reproduction": "PASS",
        "selected_policy": None,
        "route_score_formula_selected": False,
        "additional_model_training": False,
        "test_re_evaluated": False,
        "deployment_status": "not_deployed",
        "limitations": [
            "Existing offline regional route matches.",
            "Latest JavaScript matcher not rerun.",
            "No actual route-level accident outcomes.",
            "No novice driver difficulty labels.",
            "Model score is not accident probability.",
            "Previous and new model scores have "
            "different meanings.",
            "Shortest and fastest baselines are "
            "restricted to eligible candidates.",
            "Common support may reduce route coverage.",
            "Final policy has not been selected."
        ]
    }

    with OUTPUT_CONFIG.open("w", encoding="utf-8") as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n===== Saved Results =====")

    for path in [
        OUTPUT_ROUTES,
        OUTPUT_SELECTIONS,
        OUTPUT_SUMMARY,
        OUTPUT_COVERAGE,
        OUTPUT_CONFIG
    ]:

        print("Path:", path)

        print(
            f"File Size: "
            f"{path.stat().st_size / 1024**2:.4f} MiB"
        )

    print("\n===== Processing Completed =====")

    print(f"Elapsed Time: {time.perf_counter() - start:.2f}s")

    print("Reference Tiles:", len(tiles))
    print("Total Routes:", len(routes))
    print("Selection Records:", len(selections))
    print("Summary Records:", len(summary))
    print("Detour Violations: 0")
    print("Final Policy Selected: No")
    print("Additional Model Training: Not Performed")
    print("Test Re-evaluation: Not Performed")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":

    main()