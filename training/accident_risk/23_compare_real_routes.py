from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. 프로젝트 경로 및 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"
RESULT_DIR = DATA_DIR / "results"
FIGURE_DIR = RESULT_DIR / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MIN_COVERAGE = 0.8
DETOUR_CAP = 2.0

OD_COLUMNS = [
    "origin_id",
    "destination_id"
]

plt.rcParams["font.family"] = "DejaVu Sans"


# 2. 기존 경로 데이터 불러오기
print("===== Load Route Data =====")

with open(
    ROOT / "data/routes_scored.json",
    encoding="utf-8-sig"
) as f:
    routes = pd.DataFrame(json.load(f))

with open(
    ROOT / "data/route_link_matches.json",
    encoding="utf-8-sig"
) as f:
    matches = json.load(f)

links = pd.read_parquet(
    DATA_DIR / "link_risk_analysis.parquet"
)

legacy_model = joblib.load(
    ROOT / "models/route_lightgbm.joblib"
)

print("Routes:", len(routes))
print("Route Matches:", len(matches))
print("Road Links:", len(links))
print("Legacy Model:", type(legacy_model).__name__)


# 3. ID 호환성 검사
routes["route_id"] = routes["route_id"].astype(str)
links["link_id"] = links["link_id"].astype(str)

match_ids = [
    str(row["route_id"])
    for row in matches
]

assert routes["route_id"].is_unique
assert links["link_id"].is_unique
assert len(match_ids) == len(set(match_ids))

assert set(routes["route_id"]) == set(match_ids)

assert not routes[
    OD_COLUMNS + ["geometry_sha256"]
].isna().any().any()

assert not routes.duplicated(
    OD_COLUMNS + ["geometry_sha256"]
).any()

print("Route ID Compatibility: Passed")


# 4. 정규화 기준 계산
print("\n===== Normalization Reference =====")

score_min = float(
    links["hotspot_score"].min()
)

score_max = float(
    links["hotspot_score"].max()
)

score_p95 = float(
    links["hotspot_score"].quantile(0.95)
)

intersection_p95 = float(
    links["intersection_baseline"].quantile(0.95)
)

assert score_max > score_min
assert score_p95 > 0
assert intersection_p95 > 0

print("Score Min:", score_min)
print("Score Max:", score_max)
print("Score P95:", score_p95)
print("Intersection P95:", intersection_p95)


# 5. 도로별 정규화 점수
links["risk_raw"] = links["hotspot_score"]

links["risk_minmax"] = (
    (links["hotspot_score"] - score_min)
    / (score_max - score_min)
).clip(0, 1)

links["risk_p95"] = (
    links["hotspot_score"] / score_p95
).clip(0, 1)

links["risk_percentile"] = (
    links["hotspot_score"].rank(
        method="average",
        pct=True
    )
)

links["baseline_p95"] = (
    links["intersection_baseline"]
    / intersection_p95
).clip(0, 1)

score_columns = [
    "risk_raw",
    "risk_minmax",
    "risk_p95",
    "risk_percentile",
    "baseline_p95",
    "accident_burden",
    "narrow_burden",
    "intersection_burden"
]

for column in score_columns:

    assert np.isfinite(
        links[column]
    ).all()

    assert links[column].between(
        0, 1
    ).all()


# 6. 도로별 점수 조회 테이블
aggregation_columns = score_columns + [
    "n_intersect_200m",
    "target"
]

lookup = links.set_index(
    "link_id"
)[aggregation_columns].to_dict(
    orient="index"
)


# 7. 실제 경로 길이 가중평균
print("\n===== Aggregate Real Routes =====")

aggregated = []

for index, row in enumerate(matches, start=1):

    geometry_length = float(
        row["geometry_length_m"]
    )

    stored_matched_length = float(
        row["matched_length_m"]
    )

    stored_coverage = float(
        row["coverage"]
    )

    if (
        not np.isfinite(geometry_length)
        or geometry_length <= 0
    ):
        raise ValueError("Invalid geometry length.")

    totals = {
        column: 0.0
        for column in aggregation_columns
    }

    matched_length = 0.0

    for link_id, length in row["links"]:

        link_id = str(link_id)
        length = float(length)

        if (
            not np.isfinite(length)
            or length < 0
        ):
            raise ValueError("Invalid match length.")

        if link_id not in lookup:

            raise ValueError(
                f"Unknown link ID: {link_id}"
            )

        matched_length += length

        info = lookup[link_id]

        for column in aggregation_columns:

            totals[column] += (
                float(info[column]) * length
            )

    if not np.isclose(
        matched_length,
        stored_matched_length,
        rtol=0,
        atol=1e-6
    ):
        raise ValueError(
            "Matched length mismatch."
        )

    if matched_length > geometry_length + 1e-6:

        raise ValueError(
            "Matched length exceeds route length."
        )

    coverage = (
        matched_length / geometry_length
    )

    if not np.isclose(
        coverage,
        stored_coverage,
        rtol=0,
        atol=1e-8
    ):
        raise ValueError("Coverage mismatch.")

    result = {
        "route_id": str(row["route_id"]),
        "geometry_length_m": geometry_length,
        "matched_length_m": matched_length,
        "new_coverage": coverage
    }

    for column in aggregation_columns:

        result[f"agg_{column}"] = (
            totals[column] / matched_length
            if matched_length > 0
            else np.nan
        )

    aggregated.append(result)

    if index % 1000 == 0:

        print(
            f"Processed: {index}/{len(matches)}"
        )

aggregated_df = pd.DataFrame(aggregated)

print("Aggregated Routes:", len(aggregated_df))


# 8. 기존 경로 데이터와 결합
df = routes.merge(
    aggregated_df,
    on="route_id",
    validate="one_to_one"
)

assert len(df) == len(routes)

numeric_columns = [
    "distance_m",
    "duration_s",
    "road_count",
    "guide_count",
    "n_intersect_200m",
    "wide_road_ratio",
    "near_hotspot_ratio",
    "coverage",
    "accident_burden",
    "narrow_burden",
    "intersection_burden",
    "score"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

if not np.isfinite(
    df[["distance_m", "duration_s"]]
).all().all():

    raise ValueError(
        "Invalid distance or duration."
    )

assert (df["distance_m"] > 0).all()
assert (df["duration_s"] > 0).all()


# 9. 기존 피처 재현 검증
print("\n===== Legacy Feature Reproduction =====")

feature_pairs = {
    "coverage": "new_coverage",
    "accident_burden": "agg_accident_burden",
    "narrow_burden": "agg_narrow_burden",
    "intersection_burden": "agg_intersection_burden",
    "n_intersect_200m": "agg_n_intersect_200m"
}

valid_coverage = df["coverage"] >= MIN_COVERAGE

for old_column, new_column in feature_pairs.items():

    old_values = df.loc[
        valid_coverage,
        old_column
    ].to_numpy(dtype=float)

    new_values = df.loc[
        valid_coverage,
        new_column
    ].to_numpy(dtype=float)

    if not (
        np.isfinite(old_values).all()
        and np.isfinite(new_values).all()
    ):
        raise ValueError(
            f"Missing eligible feature: {old_column}"
        )

    difference = np.abs(
        old_values - new_values
    )

    print(
        old_column,
        "Mean Difference:",
        difference.mean(),
        "Max Difference:",
        difference.max()
    )

    if not np.allclose(
        old_values,
        new_values,
        rtol=0,
        atol=1e-6
    ):
        raise ValueError(
            f"Feature reproduction failed: {old_column}"
        )

print("Legacy Feature Reproduction: Passed")


# 10. 기존 규칙 기반 점수 재현
df["legacy_rule_burden"] = 100 * (
    0.55 * df["accident_burden"]
    + 0.25 * df["narrow_burden"]
    + 0.20 * df["intersection_burden"]
)

score_difference = np.abs(
    df.loc[valid_coverage, "score"]
    - df.loc[
        valid_coverage,
        "legacy_rule_burden"
    ]
)

print("\n===== Legacy Rule Score Reproduction =====")

print(
    "Mean Difference:",
    score_difference.mean()
)

print(
    "Max Difference:",
    score_difference.max()
)

if not np.allclose(
    df.loc[valid_coverage, "score"],
    df.loc[
        valid_coverage,
        "legacy_rule_burden"
    ],
    rtol=0,
    atol=1e-6
):
    raise ValueError(
        "Legacy rule score mismatch."
    )

print("Legacy Rule Score: Passed")


# 11. 기존 LightGBM 회귀 모델 예측
print("\n===== Legacy Model Prediction =====")

legacy_features = [
    "distance_m",
    "duration_s",
    "road_count",
    "guide_count",
    "n_intersect_200m",
    "wide_road_ratio",
    "near_hotspot_ratio",
    "coverage",
    "accident_burden",
    "narrow_burden",
    "intersection_burden"
]

assert list(
    legacy_model.feature_name_
) == legacy_features

finite_features = pd.Series(
    np.isfinite(
        df[legacy_features].to_numpy(dtype=float)
    ).all(axis=1),
    index=df.index
)

predict_mask = (
    valid_coverage & finite_features
)

df["legacy_model_burden"] = np.nan

df.loc[
    predict_mask,
    "legacy_model_burden"
] = legacy_model.predict(
    df.loc[
        predict_mask,
        legacy_features
    ]
)

print(
    "Predicted Routes:",
    int(predict_mask.sum())
)

print(
    df["legacy_model_burden"]
    .describe()
    .to_string()
)

outside_range = (
    (
        df["legacy_model_burden"] < 0
    )
    | (
        df["legacy_model_burden"] > 100
    )
).sum()

print(
    "Legacy Predictions Outside 0-100:",
    int(outside_range)
)


# 12. 신규 경로별 쉬움 점수
print("\n===== Calculate New Route Scores =====")

R = df["agg_risk_minmax"]
N = df["agg_narrow_burden"]
I = df["agg_intersection_burden"]
A = df["agg_accident_burden"]
B = df["agg_baseline_p95"]

df["M00_shortest"] = -df["distance_m"]

df["M01_legacy_rule"] = (
    100 - df["legacy_rule_burden"]
)

df["M02_legacy_model"] = (
    100 - df["legacy_model_burden"]
)

df["M03_new_A"] = 100 * (
    1 - 0.55 * R
    - 0.25 * N
    - 0.20 * I
)

df["M04_new_B"] = 100 * (
    1 - 0.55 * R
    - 0.45 * N
)

df["M05_new_C"] = 100 * (
    1 - 0.35 * R
    - 0.35 * N
    - 0.30 * A
)

df["M06_intersection_B"] = 100 * (
    1 - 0.55 * B
    - 0.45 * N
)

METHODS = [
    "M00_shortest",
    "M01_legacy_rule",
    "M02_legacy_model",
    "M03_new_A",
    "M04_new_B",
    "M05_new_C",
    "M06_intersection_B"
]

for method in METHODS[3:]:

    values = df.loc[
        valid_coverage,
        method
    ]

    assert np.isfinite(values).all()

    assert values.between(
        -1e-8,
        100 + 1e-8
    ).all()

print(
    df[METHODS[1:]]
    .describe()
    .round(6)
    .to_string()
)


# 13. 공통 추천 가능 조건
print("\n===== Route Eligibility =====")

df["shortest_distance_m"] = (
    df.groupby(OD_COLUMNS)["distance_m"]
    .transform("min")
)

df["detour_ratio"] = (
    df["distance_m"]
    / df["shortest_distance_m"]
)

df["coverage_ok"] = (
    df["new_coverage"] >= MIN_COVERAGE
)

df["detour_ok"] = (
    df["detour_ratio"] <= DETOUR_CAP
)

df["scores_ok"] = np.isfinite(
    df[METHODS].to_numpy(dtype=float)
).all(axis=1)

df["eligible"] = (
    df["coverage_ok"]
    & df["detour_ok"]
    & df["scores_ok"]
)

print("Total Routes:", len(df))

print(
    "Coverage Failed:",
    int((~df["coverage_ok"]).sum())
)

print(
    "Detour Failed:",
    int((~df["detour_ok"]).sum())
)

print(
    "Invalid Scores:",
    int((~df["scores_ok"]).sum())
)

print(
    "Eligible Routes:",
    int(df["eligible"].sum())
)

eligible = df[
    df["eligible"]
].copy()


# 14. 비교 가능한 OD 그룹 선정
all_od = df.groupby(
    OD_COLUMNS
).size()

eligible_od = eligible.groupby(
    OD_COLUMNS
).size()

multiple_od = eligible_od[
    eligible_od >= 2
].reset_index(
    name="eligible_route_count"
)

comparison = eligible.merge(
    multiple_od,
    on=OD_COLUMNS,
    validate="many_to_one"
)

print("\n===== Comparable OD Groups =====")

print("Total OD Groups:", len(all_od))

print(
    "OD Groups with Eligible Routes:",
    len(eligible_od)
)

print(
    "OD Groups without Eligible Routes:",
    len(all_od) - len(eligible_od)
)

print(
    "OD Groups with One Eligible Route:",
    int((eligible_od == 1).sum())
)

print(
    "OD Groups with Multiple Eligible Routes:",
    len(multiple_od)
)

print(
    "Comparable Routes:",
    len(comparison)
)

if comparison.empty:

    raise ValueError(
        "No comparable OD groups."
    )


# 15. 방식별 최종 추천 경로 선정
def select_routes(frame, method):

    selected = frame.sort_values(
        OD_COLUMNS + [
            method,
            "distance_m",
            "duration_s",
            "route_id"
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
            True
        ]
    ).drop_duplicates(
        subset=OD_COLUMNS,
        keep="first"
    ).copy()

    selected["method"] = method

    selected["selection_score"] = (
        selected[method]
    )

    return selected


recommendations = pd.concat(
    [
        select_routes(comparison, method)
        for method in METHODS
    ],
    ignore_index=True
)

assert recommendations[
    "eligible"
].all()

assert recommendations[
    "detour_ratio"
].le(DETOUR_CAP).all()

assert not recommendations.duplicated(
    OD_COLUMNS + ["method"]
).any()

for method in METHODS:

    assert (
        recommendations["method"] == method
    ).sum() == len(multiple_od)


# 16. 최단 거리 기준 추천 경로
shortest = recommendations[
    recommendations["method"] == "M00_shortest"
][
    OD_COLUMNS + [
        "route_id",
        "distance_m",
        "duration_s"
    ]
].rename(
    columns={
        "route_id": "shortest_eligible_route_id",
        "distance_m": "shortest_eligible_distance_m",
        "duration_s": "shortest_eligible_duration_s"
    }
)

recommendations = recommendations.merge(
    shortest,
    on=OD_COLUMNS,
    validate="many_to_one"
)

recommendations["extra_distance_m"] = (
    recommendations["distance_m"]
    - recommendations[
        "shortest_eligible_distance_m"
    ]
)

recommendations["extra_duration_s"] = (
    recommendations["duration_s"]
    - recommendations[
        "shortest_eligible_duration_s"
    ]
)

recommendations["extra_distance_ratio"] = (
    recommendations["extra_distance_m"]
    / recommendations[
        "shortest_eligible_distance_m"
    ]
)


# 17. 방식별 추천 결과 요약
print("\n===== Recommendation Summary =====")

summary = recommendations.groupby(
    "method"
).agg(

    od_groups=("route_id", "size"),

    mean_distance_m=("distance_m", "mean"),

    mean_duration_s=("duration_s", "mean"),

    mean_wide_ratio=("wide_road_ratio", "mean"),

    mean_narrow_burden=(
        "agg_narrow_burden", "mean"
    ),

    mean_intersection_burden=(
        "agg_intersection_burden", "mean"
    ),

    mean_legacy_accident_burden=(
        "agg_accident_burden", "mean"
    ),

    mean_hotspot_score=(
        "agg_risk_raw", "mean"
    ),

    mean_hotspot_minmax=(
        "agg_risk_minmax", "mean"
    ),

    mean_observed_hotspot_ratio=(
        "agg_target", "mean"
    ),

    mean_coverage=("new_coverage", "mean"),

    mean_detour_ratio=("detour_ratio", "mean"),

    max_detour_ratio=("detour_ratio", "max"),

    mean_extra_distance_m=(
        "extra_distance_m", "mean"
    ),

    mean_extra_duration_s=(
        "extra_duration_s", "mean"
    ),

    mean_extra_distance_ratio=(
        "extra_distance_ratio", "mean"
    )

).reindex(METHODS)

print(
    summary.round(6)
    .to_string()
)


# 18. 기존 회귀 모델 대비 추천 변경률
print("\n===== Recommendation Change Analysis =====")

legacy_selected = recommendations[
    recommendations["method"] == "M02_legacy_model"
][
    OD_COLUMNS + [
        "route_id",
        "wide_road_ratio",
        "agg_narrow_burden",
        "agg_intersection_burden",
        "agg_accident_burden",
        "agg_risk_raw",
        "agg_target",
        "distance_m",
        "duration_s"
    ]
].copy()

change_results = []

paired_results = []

for method in METHODS:

    current = recommendations[
        recommendations["method"] == method
    ].copy()

    paired = current.merge(
        legacy_selected,
        on=OD_COLUMNS,
        suffixes=("_new", "_old"),
        validate="one_to_one"
    )

    paired["route_changed"] = (
        paired["route_id_new"]
        != paired["route_id_old"]
    )

    comparison_columns = [
        "wide_road_ratio",
        "agg_narrow_burden",
        "agg_intersection_burden",
        "agg_accident_burden",
        "agg_risk_raw",
        "agg_target",
        "distance_m",
        "duration_s"
    ]

    for column in comparison_columns:

        paired[f"delta_{column}"] = (
            paired[f"{column}_new"]
            - paired[f"{column}_old"]
        )

    paired["method"] = method

    paired_results.append(paired)

    changed = paired[
        paired["route_changed"]
    ]

    change_results.append({

        "method": method,

        "od_groups": len(paired),

        "changed_routes": len(changed),

        "change_ratio": (
            len(changed) / len(paired)
        ),

        "wide_ratio_increased": int(
            (
                paired["delta_wide_road_ratio"]
                > 1e-8
            ).sum()
        ),

        "wide_ratio_decreased": int(
            (
                paired["delta_wide_road_ratio"]
                < -1e-8
            ).sum()
        ),

        "mean_wide_ratio_delta": (
            paired["delta_wide_road_ratio"]
            .mean()
        ),

        "mean_narrow_delta": (
            paired["delta_agg_narrow_burden"]
            .mean()
        ),

        "mean_intersection_delta": (
            paired[
                "delta_agg_intersection_burden"
            ].mean()
        ),

        "mean_legacy_accident_delta": (
            paired[
                "delta_agg_accident_burden"
            ].mean()
        ),

        "mean_hotspot_score_delta": (
            paired["delta_agg_risk_raw"]
            .mean()
        ),

        "mean_observed_hotspot_delta": (
            paired["delta_agg_target"]
            .mean()
        ),

        "mean_distance_delta_m": (
            paired["delta_distance_m"]
            .mean()
        ),

        "mean_duration_delta_s": (
            paired["delta_duration_s"]
            .mean()
        )
    })

changes = pd.DataFrame(change_results)

paired_df = pd.concat(
    paired_results,
    ignore_index=True
)

print(
    changes.round(6)
    .to_string(index=False)
)


# 19. 신규 B안과 교차로 B안 직접 비교
print("\n===== LightGBM B vs Intersection B =====")

new_b = recommendations[
    recommendations["method"] == "M04_new_B"
]

baseline_b = recommendations[
    recommendations["method"] == "M06_intersection_B"
]

direct = new_b.merge(
    baseline_b,
    on=OD_COLUMNS,
    suffixes=("_ai", "_baseline"),
    validate="one_to_one"
)

direct["route_changed"] = (
    direct["route_id_ai"]
    != direct["route_id_baseline"]
)

print("OD Groups:", len(direct))

print(
    "Different Recommendations:",
    int(direct["route_changed"].sum())
)

print(
    "Different Recommendation Ratio:",
    direct["route_changed"].mean()
)

for column in [
    "wide_road_ratio",
    "agg_narrow_burden",
    "agg_intersection_burden",
    "agg_accident_burden",
    "agg_risk_raw",
    "agg_target",
    "distance_m",
    "duration_s"
]:

    delta = (
        direct[f"{column}_ai"]
        - direct[f"{column}_baseline"]
    )

    direct[f"delta_{column}"] = delta

    print(
        column,
        "Mean AI-Baseline Difference:",
        delta.mean()
    )


# 20. 대표 추천 변경 사례
print("\n===== Representative Changed Routes =====")

examples = paired_df[
    (
        paired_df["method"] == "M04_new_B"
    )
    & paired_df["route_changed"]
].copy()

examples = examples.sort_values(
    "delta_wide_road_ratio",
    ascending=False
)

example_columns = [
    "origin_name",
    "destination_name",
    "route_id_old",
    "route_id_new",
    "delta_wide_road_ratio",
    "delta_agg_risk_raw",
    "delta_distance_m",
    "delta_duration_s"
]

print("\nLargest Wide Road Ratio Improvements:")

print(
    examples.head(5)[example_columns]
    .round(6)
    .to_string(index=False)
)

print("\nLargest Wide Road Ratio Reductions:")

print(
    examples.tail(5)[example_columns]
    .round(6)
    .to_string(index=False)
)


# 21. 결과 저장
print("\n===== Save Comparison Results =====")

df.to_parquet(
    DATA_DIR / "real_route_comparison.parquet",
    index=False
)

recommendations.to_csv(
    RESULT_DIR / "real_route_recommendations.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    RESULT_DIR / "real_route_method_summary.csv",
    encoding="utf-8-sig"
)

changes.to_csv(
    RESULT_DIR / "real_route_change_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

paired_df.to_parquet(
    RESULT_DIR / "real_route_paired_comparison.parquet",
    index=False
)

direct.to_csv(
    RESULT_DIR / "real_route_ai_vs_baseline.csv",
    index=False,
    encoding="utf-8-sig"
)

examples.to_csv(
    RESULT_DIR / "real_route_changed_examples.csv",
    index=False,
    encoding="utf-8-sig"
)


# 22. 시각화 저장 함수
def save_figure(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# 23. 방식별 큰길 비율
plt.figure(figsize=(11, 6))

plt.bar(
    summary.index,
    summary["mean_wide_ratio"] * 100
)

plt.ylabel("Mean Wide Road Ratio (%)")
plt.title("Wide Road Ratio by Recommendation Method")
plt.xticks(rotation=35, ha="right")

save_figure(
    "real_route_wide_ratio.png"
)


# 24. 방식별 교차로 부담도
plt.figure(figsize=(11, 6))

plt.bar(
    summary.index,
    summary["mean_intersection_burden"]
)

plt.ylabel("Mean Intersection Burden")
plt.title("Intersection Burden by Method")
plt.xticks(rotation=35, ha="right")

save_figure(
    "real_route_intersection_burden.png"
)


# 25. 방식별 추가 주행 거리
plt.figure(figsize=(11, 6))

plt.bar(
    summary.index,
    summary["mean_extra_distance_m"]
)

plt.ylabel("Mean Extra Distance (m)")
plt.title("Extra Distance from Eligible Shortest Route")
plt.xticks(rotation=35, ha="right")

save_figure(
    "real_route_extra_distance.png"
)


# 26. 방식별 추가 소요 시간
plt.figure(figsize=(11, 6))

plt.bar(
    summary.index,
    summary["mean_extra_duration_s"]
)

plt.ylabel("Mean Extra Duration (s)")
plt.title("Duration Difference from Shortest Route")
plt.xticks(rotation=35, ha="right")

save_figure(
    "real_route_extra_duration.png"
)


# 27. 기존 모델 대비 추천 변경률
plt.figure(figsize=(11, 6))

plt.bar(
    changes["method"],
    changes["change_ratio"] * 100
)

plt.ylabel("Recommendation Change (%)")
plt.title("Recommendation Changes vs Legacy Model")
plt.xticks(rotation=35, ha="right")

save_figure(
    "real_route_change_ratio.png"
)


# 28. 큰길 비율과 우회 거리 관계
plt.figure(figsize=(10, 7))

for method, row in summary.iterrows():

    x = row["mean_extra_distance_m"]
    y = row["mean_wide_ratio"] * 100

    plt.scatter(x, y, s=70)

    plt.annotate(
        method,
        (x, y),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel("Mean Extra Distance (m)")
plt.ylabel("Mean Wide Road Ratio (%)")
plt.title("Wide Road Preference vs Extra Distance")

save_figure(
    "real_route_tradeoff.png"
)


# 29. 최종 검증
print("\n===== Final Validation =====")

print(
    "Coverage Violations:",
    int(
        (
            recommendations["new_coverage"]
            < MIN_COVERAGE
        ).sum()
    )
)

print(
    "Detour Violations:",
    int(
        (
            recommendations["detour_ratio"]
            > DETOUR_CAP
        ).sum()
    )
)

print(
    "Recommendation Methods:",
    len(METHODS)
)

print(
    "Comparable OD Groups:",
    len(multiple_od)
)

print(
    "Total Recommendations:",
    len(recommendations)
)

print("Figures Generated: 6")

print("\n===== Real Route Comparison Completed =====")

print("Final Method Selected: No")
print("Final Weights Selected: No")
print("Model Retraining: Not Performed")
print("API Calls: Not Performed")
print("Existing Service Files: Not Modified")
print("Model Deployment: Not Performed")