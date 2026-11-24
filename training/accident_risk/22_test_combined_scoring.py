from pathlib import Path

import numpy as np
import pandas as pd


# 1. 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"
RESULT_DIR = DATA_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COVERAGE = 0.8


# 2. 도로 데이터 불러오기
df = pd.read_parquet(
    DATA_DIR / "link_risk_analysis.parquet"
)

df["link_id"] = df["link_id"].astype(str)

assert len(df) == 36521
assert not df["link_id"].duplicated().any()


# 3. 정규화 점수 생성
score_min = df["hotspot_score"].min()
score_max = df["hotspot_score"].max()
score_p95 = df["hotspot_score"].quantile(0.95)

intersection_p95 = (
    df["intersection_baseline"].quantile(0.95)
)

df["risk_raw"] = df["hotspot_score"]

df["risk_minmax"] = (
    (df["hotspot_score"] - score_min)
    / (score_max - score_min)
).clip(0, 1)

df["risk_p95"] = (
    df["hotspot_score"] / score_p95
).clip(0, 1)

df["risk_percentile"] = (
    df["hotspot_score"].rank(
        method="average",
        pct=True
    )
)

df["baseline_p95"] = (
    df["intersection_baseline"]
    / intersection_p95
).clip(0, 1)


# 4. 점수 범위 검사
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

    assert np.isfinite(df[column]).all()
    assert df[column].between(0, 1).all()


# 5. 도로 특성별 대표 링크 선정
def select_link(narrow, quantile):

    subset = df[
        df["narrow_burden"] == narrow
    ].copy()

    if subset.empty:
        raise ValueError("No matching links.")

    reference = subset[
        "hotspot_score"
    ].quantile(quantile)

    subset["difference"] = (
        subset["hotspot_score"] - reference
    ).abs()

    selected = subset.sort_values(
        ["difference", "link_id"]
    ).iloc[0]

    return selected["link_id"]


samples = {
    "low_narrow": select_link(1, 0.1),
    "high_narrow": select_link(1, 0.9),
    "low_wide": select_link(0, 0.1),
    "high_wide": select_link(0, 0.9)
}

lookup = df.set_index(
    "link_id"
).to_dict(orient="index")

print("===== Representative Links =====")

for name, link_id in samples.items():

    info = lookup[link_id]

    print(
        name,
        "ID:", link_id,
        "AI:", round(info["hotspot_score"], 6),
        "MinMax:", round(info["risk_minmax"], 6),
        "Lanes:", info["lanes"],
        "Narrow:", info["narrow_burden"],
        "Intersection:", info["intersection_baseline"]
    )


# 6. 경로 점수 집계 함수
def aggregate_route(links, geometry_length_m):

    if (
        not np.isfinite(geometry_length_m)
        or geometry_length_m <= 0
    ):
        raise ValueError("Invalid route length.")

    totals = {
        column: 0.0
        for column in score_columns
    }

    valid_length = 0.0
    input_length = 0.0

    for link_id, length in links:

        length = float(length)

        if not np.isfinite(length) or length < 0:
            raise ValueError("Invalid matched length.")

        input_length += length

        info = lookup.get(str(link_id))

        if info is None or length == 0:
            continue

        valid_length += length

        for column in score_columns:

            totals[column] += (
                float(info[column]) * length
            )

    if input_length > geometry_length_m + 1e-6:
        raise ValueError("Excess matched length.")

    coverage = min(
        valid_length / geometry_length_m,
        1.0
    )

    if valid_length == 0 or coverage < MIN_COVERAGE:

        return {
            "coverage": coverage,
            "eligible": False
        }

    result = {
        column: totals[column] / valid_length
        for column in score_columns
    }

    result["coverage"] = coverage
    result["eligible"] = True

    return result


# 7. 종합 점수 후보
WEIGHT_SCHEMES = {

    "A_keep_weights": {
        "risk": 0.55,
        "narrow_burden": 0.25,
        "intersection_burden": 0.20
    },

    "B_reduce_overlap": {
        "risk": 0.55,
        "narrow_burden": 0.45
    },

    "C_keep_accident_history": {
        "risk": 0.35,
        "narrow_burden": 0.35,
        "accident_burden": 0.30
    }
}

for weights in WEIGHT_SCHEMES.values():

    assert np.isclose(
        sum(weights.values()), 1.0
    )


# 8. 종합 쉬움 점수 계산
def combined_score(features, risk_column, weights):

    if not features["eligible"]:
        return None

    values = dict(features)

    values["risk"] = features[risk_column]

    burden = 100 * sum(
        values[key] * weight
        for key, weight in weights.items()
    )

    easy_score = float(
        np.clip(100 - burden, 0, 100)
    )

    return easy_score


# 9. 가상 경로 구성
# 실제 연결 경로가 아닌 결합 함수 테스트용
routes = {

    "R1_low_narrow": [
        [samples["low_narrow"], 1000]
    ],

    "R2_high_narrow": [
        [samples["high_narrow"], 1000]
    ],

    "R3_low_wide": [
        [samples["low_wide"], 1000]
    ],

    "R4_high_wide": [
        [samples["high_wide"], 1000]
    ],

    "R5_mixed_low": [
        [samples["low_narrow"], 500],
        [samples["low_wide"], 500]
    ],

    "R6_mixed_high": [
        [samples["high_narrow"], 500],
        [samples["high_wide"], 500]
    ],

    "R7_wide_dominant": [
        [samples["high_wide"], 800],
        [samples["low_narrow"], 200]
    ],

    "R8_narrow_dominant": [
        [samples["high_wide"], 200],
        [samples["low_narrow"], 800]
    ]
}

route_features = {
    name: aggregate_route(links, 1000)
    for name, links in routes.items()
}


# 10. 정규화 및 가중치 조합 비교
normalizations = [
    "risk_raw",
    "risk_minmax",
    "risk_p95",
    "risk_percentile",
    "baseline_p95"
]

results = []

for route_name, features in route_features.items():

    # 기존 규칙 기반 점수
    legacy_burden = 100 * (
        0.55 * features["accident_burden"]
        + 0.25 * features["narrow_burden"]
        + 0.20 * features["intersection_burden"]
    )

    legacy_easy = 100 - legacy_burden

    results.append({
        "route": route_name,
        "method": "legacy_rule",
        "normalization": "legacy",
        "scheme": "legacy",
        "coverage": features["coverage"],
        "narrow_burden": features["narrow_burden"],
        "easy_score": legacy_easy
    })

    # 신규 결합 점수
    for normalization in normalizations:

        for scheme, weights in WEIGHT_SCHEMES.items():

            easy = combined_score(
                features,
                normalization,
                weights
            )

            results.append({
                "route": route_name,
                "method": (
                    f"{normalization}__{scheme}"
                ),
                "normalization": normalization,
                "scheme": scheme,
                "coverage": features["coverage"],
                "narrow_burden": features["narrow_burden"],
                "easy_score": easy
            })


# 11. 결과 정리
results_df = pd.DataFrame(results)

assert results_df["easy_score"].between(
    0, 100
).all()

results_df["rank"] = (
    results_df.groupby("method")["easy_score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

print("\n===== Route Features =====")

features_df = pd.DataFrame(
    route_features
).T

print(
    features_df[
        [
            "risk_raw",
            "risk_minmax",
            "risk_p95",
            "risk_percentile",
            "baseline_p95",
            "narrow_burden"
        ]
    ].astype(float).round(6).to_string()
)


# 12. Min-Max 방식 중심 비교
print("\n===== Min-Max Score Comparison =====")

selected = results_df[
    results_df["normalization"].isin(
        ["legacy", "risk_minmax"]
    )
]

score_table = selected.pivot(
    index="route",
    columns="scheme",
    values="easy_score"
)

print(
    score_table.round(4).to_string()
)


# 13. 방식별 1위 경로
print("\n===== Best Route by Method =====")

best_routes = results_df[
    results_df["rank"] == 1
]

print(
    best_routes[
        [
            "normalization",
            "scheme",
            "route",
            "easy_score"
        ]
    ].round(4).to_string(index=False)
)


# 14. B안 정규화 방식 비교
print("\n===== Scheme B Normalization Comparison =====")

scheme_b = results_df[
    results_df["scheme"] == "B_reduce_overlap"
]

normalization_table = scheme_b.pivot(
    index="route",
    columns="normalization",
    values="easy_score"
)

print(
    normalization_table.round(4).to_string()
)


# 15. AI 가중치 민감도 분석
print("\n===== Weight Sensitivity =====")

sensitivity = []

for risk_weight in np.arange(
    0.0, 1.01, 0.1
):

    weights = {
        "risk": float(risk_weight),
        "narrow_burden": float(1 - risk_weight)
    }

    for route_name, features in route_features.items():

        easy = combined_score(
            features,
            "risk_minmax",
            weights
        )

        sensitivity.append({
            "risk_weight": round(risk_weight, 2),
            "narrow_weight": round(1 - risk_weight, 2),
            "route": route_name,
            "easy_score": easy
        })

sensitivity_df = pd.DataFrame(sensitivity)

sensitivity_table = sensitivity_df.pivot(
    index="risk_weight",
    columns="route",
    values="easy_score"
)

print(
    sensitivity_table.round(3).to_string()
)


# 16. 점수 계산 함수 검증
print("\n===== Combined Score Validation =====")

test_weights = {
    "risk": 0.55,
    "narrow_burden": 0.45
}

# 부담도 0이면 쉬움 점수 100
assert np.isclose(
    combined_score(
        {
            "eligible": True,
            "risk_minmax": 0,
            "narrow_burden": 0
        },
        "risk_minmax",
        test_weights
    ),
    100
)

# 부담도 1이면 쉬움 점수 0
assert np.isclose(
    combined_score(
        {
            "eligible": True,
            "risk_minmax": 1,
            "narrow_burden": 1
        },
        "risk_minmax",
        test_weights
    ),
    0
)

print("Score Boundary: Passed")

# AI 점수가 같으면 큰길 우선
wide_score = combined_score(
    {
        "eligible": True,
        "risk_minmax": 0.3,
        "narrow_burden": 0
    },
    "risk_minmax",
    test_weights
)

narrow_score = combined_score(
    {
        "eligible": True,
        "risk_minmax": 0.3,
        "narrow_burden": 1
    },
    "risk_minmax",
    test_weights
)

assert wide_score > narrow_score

print("Wide Road Preference: Passed")

# 도로 폭 조건이 같으면 낮은 AI 점수 우선
low_score = combined_score(
    {
        "eligible": True,
        "risk_minmax": 0.1,
        "narrow_burden": 0.5
    },
    "risk_minmax",
    test_weights
)

high_score = combined_score(
    {
        "eligible": True,
        "risk_minmax": 0.8,
        "narrow_burden": 0.5
    },
    "risk_minmax",
    test_weights
)

assert low_score > high_score

print("Lower AI Burden Preference: Passed")

# 매칭률 부족 경로 제외
invalid = aggregate_route(
    [[samples["low_wide"], 790]],
    1000
)

assert not invalid["eligible"]

assert combined_score(
    invalid,
    "risk_minmax",
    test_weights
) is None

print("Coverage Handling: Passed")


# 17. 결과 저장
results_df.to_csv(
    RESULT_DIR / "combined_scoring_results.csv",
    index=False,
    encoding="utf-8-sig"
)

features_df.to_csv(
    RESULT_DIR / "combined_route_features.csv",
    encoding="utf-8-sig"
)

sensitivity_df.to_csv(
    RESULT_DIR / "combined_weight_sensitivity.csv",
    index=False,
    encoding="utf-8-sig"
)


# 18. 최종 결과
print("\n===== Combined Scoring Test Completed =====")

print("Test Routes:", len(routes))
print("Weight Schemes:", len(WEIGHT_SCHEMES))
print("Score Methods:", results_df["method"].nunique())
print("Total Comparisons:", len(results_df))

print("All Validation Checks: Passed")
print("Final Normalization Selected: No")
print("Final Weights Selected: No")
print("Real Route Evaluation: Not Performed")
print("Existing Service Files: Not Modified")
print("Model Deployment: Not Performed")