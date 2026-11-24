from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"

RESULT_DIR = DATA_DIR / "results"

FIGURE_DIR = RESULT_DIR / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"


# 2. 데이터 불러오기
print("===== Load Analysis Data =====")

dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_dataset.parquet"
)

predictions = pd.read_parquet(
    DATA_DIR / "link_risk_scores.parquet"
)

with open(
    ROOT / "data/link_scores.json",
    "r",
    encoding="utf-8"
) as file:

    service_scores = pd.DataFrame(
        json.load(file)
    )

with open(
    DATA_DIR / "link_risk_metadata.json",
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


# 3. 링크 ID 검사
for name, frame in [
    ("Dataset", dataset),
    ("Predictions", predictions),
    ("Service Scores", service_scores)
]:

    if frame["link_id"].isna().any():
        raise ValueError(f"{name}: Missing IDs")

    frame["link_id"] = (
        frame["link_id"].astype(str)
    )

    if frame["link_id"].duplicated().any():
        raise ValueError(f"{name}: Duplicate IDs")


if not (
    set(dataset["link_id"])
    == set(predictions["link_id"])
    == set(service_scores["link_id"])
):

    raise ValueError("Link ID sets differ.")


# 4. 분석용 데이터 결합
service_columns = [
    "link_id",
    "accident_burden",
    "narrow_burden",
    "intersection_burden",
    "accident_distance_m"
]

df = dataset.merge(
    predictions,
    on="link_id",
    validate="one_to_one"
)

df = df.merge(
    service_scores[service_columns],
    on="link_id",
    validate="one_to_one"
)

if len(df) != 36521:

    raise ValueError(
        "Unexpected number of links."
    )

if not np.array_equal(
    df["intersection_baseline"].to_numpy(),
    df["n_intersect_200m"].to_numpy()
):

    raise ValueError(
        "Intersection baseline mismatch."
    )


# 지역명 설정
REGION_NAMES = {
    "44131": "Cheonan Dongnam",
    "44133": "Cheonan Seobuk",
    "44200": "Asan"
}

df["region_code"] = (
    df["region_code"].astype(str)
)

df["region_name"] = (
    df["region_code"].map(REGION_NAMES)
)

if df["region_name"].isna().any():

    raise ValueError("Unknown region code.")


# 기존 가중치 기반 부담 점수
df["legacy_rule_burden"] = 100 * (
    0.55 * df["accident_burden"]
    + 0.25 * df["narrow_burden"]
    + 0.20 * df["intersection_burden"]
)

print("Analysis Shape:", df.shape)

print(
    "All Link IDs Identical: True"
)


# 5. 전체 예측 점수 분포
print("\n===== Overall Score Distribution =====")

print(
    df["hotspot_score"].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    ).to_string()
)

threshold = float(
    metadata["model_threshold"]
)

baseline_threshold = float(
    metadata["intersection_baseline_threshold"]
)

print("Model Threshold:", threshold)
print("Baseline Threshold:", baseline_threshold)


# 6. 지역별 및 도로 특성별 분석 함수
def summarize_groups(column):

    summary = df.groupby(
        column,
        observed=True
    ).agg(

        links=("link_id", "size"),

        mean_score=("hotspot_score", "mean"),

        median_score=("hotspot_score", "median"),

        std_score=("hotspot_score", "std"),

        min_score=("hotspot_score", "min"),

        max_score=("hotspot_score", "max"),

        actual_positive=("target", "sum"),

        predicted_positive=(
            "hotspot_predicted_label", "sum"
        ),

        baseline_positive=(
            "intersection_predicted_label", "sum"
        )
    )

    summary["actual_positive_ratio"] = (
        summary["actual_positive"]
        / summary["links"]
    )

    summary["predicted_positive_ratio"] = (
        summary["predicted_positive"]
        / summary["links"]
    )

    summary["baseline_positive_ratio"] = (
        summary["baseline_positive"]
        / summary["links"]
    )

    return summary.reset_index()


# 7. 지역별 점수 분석
print("\n===== Regional Score Distribution =====")

region_summary = summarize_groups(
    "region_name"
)

print(
    region_summary.round(6)
    .to_string(index=False)
)

region_summary.to_csv(
    RESULT_DIR / "link_risk_region_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# 8. 도로 등급별 점수 분석
print("\n===== Road Rank Distribution =====")

rank_summary = summarize_groups(
    "road_rank"
)

print(
    rank_summary.round(6)
    .to_string(index=False)
)

rank_summary.to_csv(
    RESULT_DIR / "link_risk_rank_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# 9. 차로 수별 점수 분석
print("\n===== Lane Distribution =====")

lane_summary = summarize_groups(
    "lanes"
)

print(
    lane_summary.round(6)
    .to_string(index=False)
)

lane_summary.to_csv(
    RESULT_DIR / "link_risk_lane_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# 10. 제한속도별 점수 분석
print("\n===== Speed Limit Distribution =====")

speed_summary = summarize_groups(
    "max_spd"
)

print(
    speed_summary.round(6)
    .to_string(index=False)
)

speed_summary.to_csv(
    RESULT_DIR / "link_risk_speed_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# 11. 기존 점수와 신규 점수의 상관관계
print("\n===== Correlation Analysis =====")

correlation_columns = [
    "hotspot_score",
    "intersection_baseline",
    "accident_burden",
    "narrow_burden",
    "intersection_burden",
    "legacy_rule_burden",
    "accident_distance_m",
    "lanes",
    "max_spd",
    "length",
    "n_branch"
]

pearson = df[
    correlation_columns
].corr(method="pearson")

spearman = df[
    correlation_columns
].corr(method="spearman")

correlation_summary = pd.DataFrame({

    "pearson":
        pearson["hotspot_score"],

    "spearman":
        spearman["hotspot_score"]

})

correlation_summary["abs_spearman"] = (
    correlation_summary["spearman"].abs()
)

correlation_summary = (
    correlation_summary
    .sort_values(
        "abs_spearman",
        ascending=False
    )
)

print(
    correlation_summary.round(6)
    .to_string()
)

pearson.to_csv(
    RESULT_DIR / "link_risk_pearson.csv",
    encoding="utf-8-sig"
)

spearman.to_csv(
    RESULT_DIR / "link_risk_spearman.csv",
    encoding="utf-8-sig"
)

correlation_summary.to_csv(
    RESULT_DIR / "link_risk_correlation_summary.csv",
    encoding="utf-8-sig"
)


# 12. LightGBM과 교차로 기준 분류 비교
print("\n===== Model Agreement =====")

agreement_table = pd.crosstab(

    df["hotspot_predicted_label"],

    df["intersection_predicted_label"],

    rownames=["LightGBM"],

    colnames=["Intersection"]

).reindex(
    index=[0, 1],
    columns=[0, 1],
    fill_value=0
)

print(agreement_table)

both_negative = int(
    agreement_table.loc[0, 0]
)

baseline_only = int(
    agreement_table.loc[0, 1]
)

lightgbm_only = int(
    agreement_table.loc[1, 0]
)

both_positive = int(
    agreement_table.loc[1, 1]
)

agreement_count = (
    both_negative + both_positive
)

agreement_ratio = (
    agreement_count / len(df)
)

print("Both Negative:", both_negative)
print("Both Positive:", both_positive)
print("LightGBM Only:", lightgbm_only)
print("Intersection Only:", baseline_only)
print("Agreement Count:", agreement_count)
print("Agreement Ratio:", agreement_ratio)

agreement_table.to_csv(
    RESULT_DIR / "link_risk_model_agreement.csv"
)


# 13. 분류 일치 및 불일치 그룹 분석
print("\n===== Disagreement Group Analysis =====")

conditions = [

    (
        (df["hotspot_predicted_label"] == 0)
        & (df["intersection_predicted_label"] == 0)
    ),

    (
        (df["hotspot_predicted_label"] == 1)
        & (df["intersection_predicted_label"] == 1)
    ),

    (
        (df["hotspot_predicted_label"] == 1)
        & (df["intersection_predicted_label"] == 0)
    ),

    (
        (df["hotspot_predicted_label"] == 0)
        & (df["intersection_predicted_label"] == 1)
    )

]

group_names = [
    "Both Negative",
    "Both Positive",
    "LightGBM Only",
    "Intersection Only"
]

df["agreement_group"] = np.select(
    conditions,
    group_names,
    default="Unknown"
)

if (df["agreement_group"] == "Unknown").any():

    raise ValueError(
        "Unexpected prediction labels."
    )

agreement_summary = df.groupby(
    "agreement_group"
).agg(

    links=("link_id", "size"),

    actual_positive=("target", "sum"),

    mean_score=("hotspot_score", "mean"),

    mean_lanes=("lanes", "mean"),

    mean_speed=("max_spd", "mean"),

    mean_length=("length", "mean"),

    mean_intersections=(
        "n_intersect_200m", "mean"
    ),

    mean_branch=("n_branch", "mean"),

    mean_legacy_accident_burden=(
        "accident_burden", "mean"
    )
)

agreement_summary["actual_positive_ratio"] = (
    agreement_summary["actual_positive"]
    / agreement_summary["links"]
)

print(
    agreement_summary.round(6)
    .to_string()
)

agreement_summary.to_csv(
    RESULT_DIR / "link_risk_disagreement_summary.csv",
    encoding="utf-8-sig"
)


# 14. 상위 및 하위 도로 특성 비교
print("\n===== Top and Bottom Score Groups =====")

sorted_df = df.sort_values(
    ["hotspot_score", "link_id"]
)

group_size = int(
    len(df) * 0.10
)

bottom = sorted_df.head(group_size)
top = sorted_df.tail(group_size)

compare_columns = [
    "hotspot_score",
    "lanes",
    "max_spd",
    "length",
    "n_intersect_200m",
    "n_branch",
    "accident_burden",
    "narrow_burden",
    "intersection_burden",
    "legacy_rule_burden",
    "target"
]

top_bottom_summary = pd.DataFrame({

    "bottom_10_percent":
        bottom[compare_columns].mean(),

    "top_10_percent":
        top[compare_columns].mean()

})

print("Links Per Group:", group_size)

print(
    top_bottom_summary.round(6)
    .to_string()
)

top_bottom_summary.to_csv(
    RESULT_DIR / "link_risk_top_bottom_summary.csv",
    encoding="utf-8-sig"
)


# 15. 점수가 높은 도로와 낮은 도로
display_columns = [
    "link_id",
    "region_name",
    "hotspot_score",
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_intersect_200m",
    "n_branch",
    "target"
]

print("\n===== Top 10 Links =====")

print(
    top.tail(10)[display_columns]
    .sort_values(
        "hotspot_score",
        ascending=False
    )
    .round(6)
    .to_string(index=False)
)

print("\n===== Bottom 10 Links =====")

print(
    bottom.head(10)[display_columns]
    .round(6)
    .to_string(index=False)
)


# 16. 정규화 검토용 통계
print("\n===== Normalization Reference =====")

normalization_columns = [
    "hotspot_score",
    "intersection_baseline",
    "accident_burden",
    "narrow_burden",
    "intersection_burden"
]

normalization_summary = df[
    normalization_columns
].describe(
    percentiles=[
        0.01, 0.05, 0.25,
        0.50, 0.75, 0.95, 0.99
    ]
).T

print(
    normalization_summary.round(6)
    .to_string()
)

normalization_summary.to_csv(
    RESULT_DIR / "link_risk_normalization_reference.csv",
    encoding="utf-8-sig"
)


# 17. 시각화 저장 함수
def save_figure(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# 18. 전체 예측 점수 히스토그램
plt.figure(figsize=(10, 6))

plt.hist(
    df["hotspot_score"],
    bins=50,
    edgecolor="black",
    alpha=0.8
)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

plt.axvline(
    df["hotspot_score"].mean(),
    color="green",
    linestyle="--",
    label="Mean"
)

plt.xlabel("Hotspot Membership Score")
plt.ylabel("Number of Road Links")
plt.title("LightGBM Link Score Distribution")
plt.legend()

save_figure(
    "link_risk_histogram.png"
)


# 19. 지역별 점수 분포
regions = list(
    REGION_NAMES.values()
)

plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        df.loc[
            df["region_name"] == region,
            "hotspot_score"
        ].to_numpy()

        for region in regions
    ],
    tick_labels=regions,
    showfliers=False
)

plt.ylabel("Hotspot Membership Score")
plt.title("Link Score Distribution by Region")

save_figure(
    "link_risk_region_boxplot.png"
)


# 20. 도로 등급별 평균 점수
plt.figure(figsize=(10, 6))

plt.bar(
    rank_summary["road_rank"].astype(str),
    rank_summary["mean_score"]
)

plt.xlabel("Road Rank")
plt.ylabel("Mean Hotspot Score")
plt.title("Mean Link Score by Road Rank")

save_figure(
    "link_risk_road_rank.png"
)


# 21. 차로 수별 평균 점수
plt.figure(figsize=(10, 6))

plt.bar(
    lane_summary["lanes"].astype(str),
    lane_summary["mean_score"]
)

plt.xlabel("Number of Lanes")
plt.ylabel("Mean Hotspot Score")
plt.title("Mean Link Score by Lane Count")

save_figure(
    "link_risk_lanes.png"
)


# 22. 기존 사고 부담도와 신규 점수 비교
plt.figure(figsize=(10, 6))

plt.scatter(
    df["accident_burden"],
    df["hotspot_score"],
    s=5,
    alpha=0.2
)

plt.xlabel("Legacy Accident Burden")
plt.ylabel("LightGBM Hotspot Score")
plt.title("Legacy Burden vs LightGBM Score")

save_figure(
    "link_risk_legacy_comparison.png"
)


# 23. 교차로 수와 신규 점수 비교
plt.figure(figsize=(10, 6))

plt.scatter(
    df["intersection_baseline"],
    df["hotspot_score"],
    s=5,
    alpha=0.2
)

plt.axhline(
    threshold,
    color="red",
    linestyle="--"
)

plt.axvline(
    baseline_threshold,
    color="green",
    linestyle="--"
)

plt.xlabel("Intersection Candidates within 200m")
plt.ylabel("LightGBM Hotspot Score")
plt.title("Intersection Baseline vs LightGBM")

save_figure(
    "link_risk_intersection_comparison.png"
)


# 24. 상관관계 히트맵
plt.figure(figsize=(13, 11))

image = plt.imshow(
    spearman.to_numpy(),
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    image,
    label="Spearman Correlation"
)

plt.xticks(
    range(len(correlation_columns)),
    correlation_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_columns)),
    correlation_columns
)

for i in range(len(correlation_columns)):

    for j in range(len(correlation_columns)):

        value = spearman.iloc[i, j]

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Link Score Spearman Correlation")

save_figure(
    "link_risk_correlation_heatmap.png"
)


# 25. 분석 결과 저장
df.to_parquet(
    DATA_DIR / "link_risk_analysis.parquet",
    index=False
)


# 26. 최종 결과
print("\n===== Analysis Completed =====")

print("Total Links:", len(df))

print("Results Directory:", RESULT_DIR)

print("Figures Directory:", FIGURE_DIR)

print("Figures Generated: 7")

print("Model Retraining: Not Performed")

print("Threshold Modification: Not Performed")

print("Normalization Applied: No")

print("Existing Service Files: Not Modified")

print("Model Deployment: Not Performed")