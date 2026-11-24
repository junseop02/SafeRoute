from pathlib import Path

import numpy as np
import pandas as pd


# 1. 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"

RESULT_DIR = DATA_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COVERAGE = 0.8


# 2. 도로별 분석 데이터 불러오기
print("===== Load Link Scores =====")

df = pd.read_parquet(
    DATA_DIR / "link_risk_analysis.parquet"
)

df["link_id"] = df["link_id"].astype(str)

if df["link_id"].duplicated().any():
    raise ValueError("Duplicate link IDs.")

required_columns = [
    "hotspot_score",
    "intersection_baseline",
    "accident_burden",
    "narrow_burden",
    "intersection_burden",
    "legacy_rule_burden",
    "length"
]

if not np.isfinite(
    df[required_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Invalid link scores.")

print("Total Links:", len(df))


# 3. 정규화 기준 계산
print("\n===== Normalization Reference =====")

score_min = float(df["hotspot_score"].min())
score_max = float(df["hotspot_score"].max())

score_p95 = float(
    df["hotspot_score"].quantile(0.95)
)

intersection_p95 = float(
    df["intersection_baseline"].quantile(0.95)
)

if score_max <= score_min:
    raise ValueError("Invalid score range.")

if score_p95 <= 0 or intersection_p95 <= 0:
    raise ValueError("Invalid P95 reference.")

print("Score Min:", score_min)
print("Score Max:", score_max)
print("Score P95:", score_p95)
print("Intersection P95:", intersection_p95)


# 4. 정규화 방식별 점수 생성
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


# 5. 정규화 결과 비교
risk_columns = [
    "risk_raw",
    "risk_minmax",
    "risk_p95",
    "risk_percentile",
    "baseline_p95"
]

print("\n===== Normalization Comparison =====")

summary = df[risk_columns].describe(
    percentiles=[
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99
    ]
).T

summary["unique_scores"] = (
    df[risk_columns].nunique()
)

summary["saturated_links"] = (
    df[risk_columns] >= 1
).sum()

print(summary.round(6).to_string())

summary.to_csv(
    RESULT_DIR / "route_normalization_comparison.csv",
    encoding="utf-8-sig"
)

for column in risk_columns:

    if not df[column].between(0, 1).all():

        raise ValueError(
            f"Invalid normalized score: {column}"
        )


# 6. 도로 점수 조회 테이블
lookup = df.set_index(
    "link_id"
).to_dict(orient="index")

aggregation_columns = risk_columns + [
    "accident_burden",
    "narrow_burden",
    "intersection_burden",
    "legacy_rule_burden"
]


# 7. 경로 점수 집계 함수
def aggregate_route(links, geometry_length_m):

    geometry_length_m = float(geometry_length_m)

    if (
        not np.isfinite(geometry_length_m)
        or geometry_length_m <= 0
    ):
        raise ValueError("Invalid route length.")

    totals = {
        column: 0.0
        for column in aggregation_columns
    }

    valid_length = 0.0
    input_length = 0.0

    unknown_links = set()
    matched_ids = set()

    # 링크별 매칭 길이 가중합
    for link_id, matched_length in links:

        link_id = str(link_id)
        matched_length = float(matched_length)

        if (
            not np.isfinite(matched_length)
            or matched_length < 0
        ):
            raise ValueError(
                "Invalid matched length."
            )

        if matched_length == 0:
            continue

        input_length += matched_length

        info = lookup.get(link_id)

        if info is None:

            unknown_links.add(link_id)
            continue

        valid_length += matched_length
        matched_ids.add(link_id)

        for column in aggregation_columns:

            totals[column] += (
                float(info[column])
                * matched_length
            )

    # 전체 경로보다 매칭 길이가 긴 경우 검사
    if input_length > geometry_length_m + 1e-6:

        raise ValueError(
            "Matched length exceeds route length."
        )

    coverage = min(
        valid_length / geometry_length_m,
        1.0
    )

    eligible = (
        valid_length > 0
        and coverage >= MIN_COVERAGE
    )

    result = {
        "geometry_length_m": geometry_length_m,
        "matched_length_m": valid_length,
        "coverage": coverage,
        "matched_link_count": len(matched_ids),
        "unknown_link_count": len(unknown_links),
        "eligible": eligible
    }

    # 매칭률이 부족하면 점수 반환하지 않음
    for column in aggregation_columns:

        result[column] = (
            totals[column] / valid_length
            if eligible
            else None
        )

    return result


# 8. 테스트용 도로 선정
# 실제 도로 데이터에서 대표 점수 링크 선택
sorted_df = df.sort_values(
    ["hotspot_score", "link_id"]
).reset_index(drop=True)

low_id = sorted_df.iloc[
    int(len(sorted_df) * 0.10)
]["link_id"]

middle_id = sorted_df.iloc[
    int(len(sorted_df) * 0.50)
]["link_id"]

high_id = sorted_df.iloc[
    int(len(sorted_df) * 0.90)
]["link_id"]

print("\n===== Test Link Samples =====")

for name, link_id in [
    ("Low", low_id),
    ("Middle", middle_id),
    ("High", high_id)
]:

    info = lookup[link_id]

    print(
        name,
        "ID:", link_id,
        "Score:", info["hotspot_score"],
        "Narrow:", info["narrow_burden"]
    )


# 9. 가상 경로 테스트
# 실제 연결된 도로 경로가 아닌 함수 검증용
test_routes = [

    {
        "name": "Route_A_Low_Dominant",
        "length": 1000,
        "links": [
            [low_id, 700],
            [middle_id, 200],
            [high_id, 100]
        ]
    },

    {
        "name": "Route_B_High_Dominant",
        "length": 1000,
        "links": [
            [low_id, 100],
            [middle_id, 200],
            [high_id, 700]
        ]
    },

    {
        "name": "Route_C_Equal_Weight",
        "length": 900,
        "links": [
            [low_id, 300],
            [middle_id, 300],
            [high_id, 300]
        ]
    },

    {
        "name": "Route_D_Coverage_80",
        "length": 1000,
        "links": [
            [low_id, 400],
            [high_id, 400]
        ]
    },

    {
        "name": "Route_E_Coverage_79",
        "length": 1000,
        "links": [
            [low_id, 400],
            [high_id, 390]
        ]
    },

    {
        "name": "Route_F_Unknown_Link",
        "length": 1000,
        "links": [
            [low_id, 800],
            ["UNKNOWN_TEST_ID", 200]
        ]
    },

    {
        "name": "Route_G_No_Match",
        "length": 1000,
        "links": []
    },

    {
        "name": "Route_H_Repeated_Link",
        "length": 1000,
        "links": [
            [low_id, 300],
            [high_id, 300],
            [low_id, 400]
        ]
    }
]

results = []

for route in test_routes:

    result = aggregate_route(
        route["links"],
        route["length"]
    )

    result["route_name"] = route["name"]

    results.append(result)

results_df = pd.DataFrame(results)

print("\n===== Route Aggregation Results =====")

display_columns = [
    "route_name",
    "coverage",
    "eligible",
    "risk_raw",
    "risk_minmax",
    "risk_p95",
    "risk_percentile",
    "baseline_p95",
    "narrow_burden",
    "legacy_rule_burden"
]

print(
    results_df[display_columns]
    .round(6)
    .to_string(index=False)
)


# 10. 집계 함수 검증
print("\n===== Aggregation Validation =====")

a, b, c, d, e, f, g, h = results

# 높은 점수 도로 비중 증가 검사
assert a["risk_raw"] < b["risk_raw"]

print("Length Weighting: Passed")

# 동일 길이일 때 산술평균과 일치
expected = np.mean([
    lookup[low_id]["hotspot_score"],
    lookup[middle_id]["hotspot_score"],
    lookup[high_id]["hotspot_score"]
])

assert np.isclose(c["risk_raw"], expected)

print("Equal Weight Average: Passed")

# 매칭률 경계값 검사
assert np.isclose(d["coverage"], 0.8)
assert d["eligible"]

assert np.isclose(e["coverage"], 0.79)
assert not e["eligible"]
assert e["risk_raw"] is None

print("Coverage Boundary: Passed")

# 알 수 없는 링크 처리
assert f["unknown_link_count"] == 1
assert np.isclose(f["coverage"], 0.8)
assert f["eligible"]

assert np.isclose(
    f["risk_raw"],
    lookup[low_id]["hotspot_score"]
)

print("Unknown Link Handling: Passed")

# 매칭 링크가 없는 경우
assert not g["eligible"]
assert g["risk_raw"] is None

print("Empty Route Handling: Passed")

# 동일 링크가 여러 번 등장하는 경우
expected_repeated = (
    lookup[low_id]["hotspot_score"] * 700
    + lookup[high_id]["hotspot_score"] * 300
) / 1000

assert np.isclose(
    h["risk_raw"],
    expected_repeated
)

assert h["matched_link_count"] == 2

print("Repeated Link Handling: Passed")

# 잘못된 매칭 길이 검사
for invalid_links, route_length in [
    ([[low_id, -10]], 1000),
    ([[low_id, float("nan")]], 1000),
    ([[low_id, 1100]], 1000),
    ([[low_id, 100]], 0)
]:

    try:

        aggregate_route(
            invalid_links,
            route_length
        )

    except ValueError:
        continue

    raise AssertionError(
        "Invalid input was accepted."
    )

print("Invalid Input Handling: Passed")


# 11. 결과 저장
results_df.to_csv(
    RESULT_DIR / "route_aggregation_test_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== Route Aggregation Test Completed =====")

print("Test Routes:", len(results))
print("All Validation Checks: Passed")

print("Normalization Selected: No")
print("Final Weights Selected: No")
print("Real Route Evaluation: Not Performed")
print("Existing Service Files: Not Modified")
print("Model Deployment: Not Performed")