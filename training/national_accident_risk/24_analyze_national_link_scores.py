from pathlib import Path
import gc
import json
import time
import importlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 경로 및 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

RESULTS_DIR = PROCESSED_DIR / "results"

SCORE_PATH = (
    PROCESSED_DIR
    / "national_link_risk_scores.parquet"
)

METADATA_PATH = (
    RESULTS_DIR
    / "national_link_risk_scores_metadata.json"
)

STATISTICS_PATH = (
    RESULTS_DIR
    / "national_score_statistics.csv"
)

QUANTILE_PATH = (
    RESULTS_DIR
    / "national_score_quantiles.csv"
)

DISTRIBUTION_PATH = (
    RESULTS_DIR
    / "national_score_distribution.csv"
)

GROUP_PATH = (
    RESULTS_DIR
    / "national_score_group_summary.csv"
)

CORRELATION_PATH = (
    RESULTS_DIR
    / "national_score_feature_correlation.csv"
)

NORMALIZATION_PATH = (
    RESULTS_DIR
    / "national_score_normalization_comparison.csv"
)

NORMALIZATION_CONFIG_PATH = (
    RESULTS_DIR
    / "national_score_normalization_config.json"
)

VALIDATION_PATH = (
    RESULTS_DIR
    / "national_score_analysis_validation.csv"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "national_score_figures"
)

EXPECTED_ROWS = 1562356

SCORE_COLUMN = "hotspot_score"

FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "n_branch",
    "n_intersect_200m",
    "intersection_nodes_200m"
]

QUANTILES = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00
]

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)


# ============================================================
# 2. 데이터 및 메타데이터 불러오기
# ============================================================

def load_data():

    with METADATA_PATH.open(
        "r", encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    assert metadata["model_id"] == "T09"

    assert metadata["all_national_links_scored"]

    assert metadata["features"] == FEATURES

    assert metadata["score_meaning"] == (
        "historical_hotspot_membership_score"
    )

    score_hash = baseline_utils.sha256_file(
        SCORE_PATH
    )

    assert score_hash == metadata["output_sha256"]

    df = pd.read_parquet(SCORE_PATH)

    print("\n===== National Link Score Dataset =====")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Model:", metadata["model_id"])

    print(
        "Threshold:",
        metadata["classification_threshold"]
    )

    print("Score File Hash Verification: PASS")

    return df, metadata, score_hash


# ============================================================
# 3. 전체 데이터 유효성 검증
# ============================================================

def validate_data(df, metadata):

    checks = []

    def check(name, condition):

        passed = bool(condition)

        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL"
        })

        print(
            name,
            ":",
            "PASS" if passed else "FAIL"
        )

        assert passed, name

    print("\n===== Dataset Validation =====")

    scores = df[SCORE_COLUMN].to_numpy()

    threshold = metadata[
        "classification_threshold"
    ]

    summary = metadata["prediction_summary"]

    check(
        "Expected Row Count",
        len(df) == EXPECTED_ROWS
    )

    check(
        "Unique Link IDs",
        df["link_id"].is_unique
    )

    check(
        "No Missing Values",
        df.notna().all().all()
    )

    check(
        "Finite Features",
        np.isfinite(
            df[FEATURES].to_numpy(
                dtype=np.float64
            )
        ).all()
    )

    check(
        "Finite Scores",
        np.isfinite(scores).all()
    )

    check(
        "Score Range 0 to 1",
        ((scores >= 0) & (scores <= 1)).all()
    )

    check(
        "Valid Road Rank",
        df["road_rank"].isin(
            metadata["road_rank_categories"]
        ).all()
    )

    check(
        "Valid Lanes",
        (df["lanes"] > 0).all()
    )

    check(
        "Valid Speed",
        (df["max_spd"] > 0).all()
    )

    check(
        "Valid Length",
        (df["length"] > 0).all()
    )

    check(
        "Valid Geometry Length",
        (
            np.isfinite(df["geometry_length_m"])
            & (df["geometry_length_m"] > 0)
        ).all()
    )

    check(
        "Valid Intersection Counts",
        (
            df[
                [
                    "n_branch",
                    "n_intersect_200m",
                    "intersection_nodes_200m"
                ]
            ] >= 0
        ).all().all()
    )

    check(
        "Binary Predicted Labels",
        df["hotspot_predicted_label"].isin(
            [0, 1]
        ).all()
    )

    expected_labels = (
        scores >= threshold
    ).astype(np.int8)

    check(
        "Classification Consistency",
        np.array_equal(
            expected_labels,
            df[
                "hotspot_predicted_label"
            ].to_numpy()
        )
    )

    check(
        "Positive Count Consistency",
        int(expected_labels.sum())
        == summary["predicted_positive"]
    )

    check(
        "Minimum Score Consistency",
        np.isclose(
            scores.min(),
            summary["score_min"],
            rtol=1e-12,
            atol=1e-12
        )
    )

    check(
        "Maximum Score Consistency",
        np.isclose(
            scores.max(),
            summary["score_max"],
            rtol=1e-12,
            atol=1e-12
        )
    )

    check(
        "Mean Score Consistency",
        np.isclose(
            scores.mean(),
            summary["score_mean"],
            rtol=1e-12,
            atol=1e-12
        )
    )

    return pd.DataFrame(checks)


# ============================================================
# 4. 예측 점수 기술통계량
# ============================================================

def calculate_statistics(df):

    scores = df[SCORE_COLUMN]

    statistics = {

        "total_links": len(scores),

        "unique_scores": scores.nunique(),

        "minimum": scores.min(),

        "maximum": scores.max(),

        "mean": scores.mean(),

        "median": scores.median(),

        "std": scores.std(ddof=0),

        "variance": scores.var(ddof=0),

        "skewness": scores.skew(),

        "excess_kurtosis": scores.kurt(),

        "q1": scores.quantile(0.25),

        "q3": scores.quantile(0.75),

        "iqr": (
            scores.quantile(0.75)
            - scores.quantile(0.25)
        ),

        "geometry_length_weighted_mean":
            np.average(
                scores,
                weights=df["geometry_length_m"]
            )

    }

    print("\n===== Score Statistics =====")

    for key, value in statistics.items():

        print(f"{key}: {value}")

    return pd.DataFrame([statistics])


# ============================================================
# 5. 백분위수 분석
# ============================================================

def calculate_quantiles(df):

    values = df[SCORE_COLUMN].quantile(
        QUANTILES
    )

    result = pd.DataFrame({

        "quantile": QUANTILES,

        "percentile": [
            q * 100 for q in QUANTILES
        ],

        "hotspot_score": values.to_numpy()

    })

    print("\n===== Score Quantiles =====")

    print(
        result.to_string(index=False)
    )

    return result


# ============================================================
# 6. 점수 구간별 분포 분석
# ============================================================

def calculate_distribution(df):

    edges = np.linspace(
        0.0,
        1.0,
        21
    )

    counts, _ = np.histogram(
        df[SCORE_COLUMN],
        bins=edges
    )

    result = pd.DataFrame({

        "lower_bound": edges[:-1],

        "upper_bound": edges[1:],

        "link_count": counts,

        "link_ratio": counts / len(df),

        "link_percentage":
            counts / len(df) * 100

    })

    assert result["link_count"].sum() == len(df)

    print("\n===== Score Distribution =====")

    print(
        result.to_string(index=False)
    )

    return result


# ============================================================
# 7. 도로 피처별 예측 점수 분석
# ============================================================

def summarize_group(df, feature, groups):

    temp = pd.DataFrame({

        "group": groups,

        "score": df[SCORE_COLUMN],

        "predicted_label":
            df["hotspot_predicted_label"]

    })

    assert temp["group"].notna().all()

    result = temp.groupby(
        "group",
        observed=True,
        sort=True
    ).agg(

        link_count=("score", "size"),

        mean_score=("score", "mean"),

        median_score=("score", "median"),

        std_score=(
            "score",
            lambda x: x.std(ddof=0)
        ),

        min_score=("score", "min"),

        max_score=("score", "max"),

        p95_score=(
            "score",
            lambda x: x.quantile(0.95)
        ),

        predicted_positive=(
            "predicted_label",
            "sum"
        )

    ).reset_index()

    result["predicted_positive_ratio"] = (
        result["predicted_positive"]
        / result["link_count"]
    )

    result["link_ratio"] = (
        result["link_count"] / len(df)
    )

    result["feature"] = feature

    result["group"] = result["group"].astype(str)

    assert result["link_count"].sum() == len(df)

    print(f"\n===== Group Summary: {feature} =====")

    print(
        result[
            [
                "group",
                "link_count",
                "mean_score",
                "median_score",
                "p95_score",
                "predicted_positive_ratio"
            ]
        ].to_string(index=False)
    )

    return result


def analyze_feature_groups(df):

    results = []

    # 도로 등급별
    results.append(
        summarize_group(
            df,
            "road_rank",
            df["road_rank"]
        )
    )

    # 차로 수별
    lane_groups = pd.cut(
        df["lanes"],
        bins=[0, 1, 2, 3, 4, np.inf],
        labels=["1", "2", "3", "4", "5+"]
    )

    results.append(
        summarize_group(
            df,
            "lanes",
            lane_groups
        )
    )

    # 제한속도별
    speed_groups = pd.cut(
        df["max_spd"],
        bins=[
            0, 30, 50, 70, 90, np.inf
        ],
        labels=[
            "1-30",
            "31-50",
            "51-70",
            "71-90",
            "91+"
        ]
    )

    results.append(
        summarize_group(
            df,
            "max_spd",
            speed_groups
        )
    )

    # 원본 도로 길이별
    length_groups = pd.cut(
        df["length"],
        bins=[
            0, 25, 50, 100, 250,
            500, np.inf
        ],
        labels=[
            "0-25",
            "25-50",
            "50-100",
            "100-250",
            "250-500",
            "500+"
        ]
    )

    results.append(
        summarize_group(
            df,
            "length",
            length_groups
        )
    )

    # 분기 수별
    results.append(
        summarize_group(
            df,
            "n_branch",
            df["n_branch"]
        )
    )

    # 교차로 수별
    for feature in [
        "n_intersect_200m",
        "intersection_nodes_200m"
    ]:

        intersection_groups = pd.cut(
            df[feature],
            bins=[
                -1, 0, 5, 10, 20,
                40, 60, 100, np.inf
            ],
            labels=[
                "0",
                "1-5",
                "6-10",
                "11-20",
                "21-40",
                "41-60",
                "61-100",
                "101+"
            ]
        )

        results.append(
            summarize_group(
                df,
                feature,
                intersection_groups
            )
        )

    return pd.concat(
        results,
        ignore_index=True
    )


# ============================================================
# 8. 입력 피처와 예측 점수 상관관계
# ============================================================

def analyze_correlations(df):

    # road_rank는 범주형 변수이므로
    # 숫자 크기 기반 상관계수에서 제외
    numeric_features = [
        feature for feature in FEATURES
        if feature != "road_rank"
    ]

    results = []

    for feature in numeric_features:

        pearson = df[feature].corr(
            df[SCORE_COLUMN],
            method="pearson"
        )

        spearman = df[feature].corr(
            df[SCORE_COLUMN],
            method="spearman"
        )

        results.append({

            "feature": feature,

            "pearson_correlation": pearson,

            "spearman_correlation": spearman

        })

    result = pd.DataFrame(results)

    result["absolute_spearman"] = (
        result["spearman_correlation"].abs()
    )

    result = result.sort_values(
        "absolute_spearman",
        ascending=False
    ).reset_index(drop=True)

    print("\n===== Feature Score Correlations =====")

    print(
        result.to_string(index=False)
    )

    return result


# ============================================================
# 9. 점수 정규화 방식 비교
# ============================================================

def compare_normalizations(df):

    raw = df[SCORE_COLUMN].to_numpy(
        dtype=np.float64
    )

    minimum = float(raw.min())

    maximum = float(raw.max())

    p95 = float(
        np.quantile(raw, 0.95)
    )

    assert maximum > minimum

    assert p95 > 0

    # N00: 원본 점수
    raw_scores = raw.copy()

    # N01: Min-Max 정규화
    minmax_scores = (
        (raw - minimum)
        / (maximum - minimum)
    )

    # N02: P95 기준 정규화
    p95_scores = np.clip(
        raw / p95,
        0.0,
        1.0
    )

    # N03: 상대적인 백분위 순위
    # 동일 점수는 평균 순위 적용
    ranks = df[SCORE_COLUMN].rank(
        method="average"
    ).to_numpy(dtype=np.float64)

    percentile_scores = (
        (ranks - 1)
        / (len(raw) - 1)
    )

    normalized = {

        "N00_raw": raw_scores,

        "N01_minmax": minmax_scores,

        "N02_p95": p95_scores,

        "N03_percentile": percentile_scores

    }

    results = []

    for method, scores in normalized.items():

        assert np.isfinite(scores).all()

        assert (
            (scores >= 0)
            & (scores <= 1)
        ).all()

        results.append({

            "method": method,

            "minimum": scores.min(),

            "maximum": scores.max(),

            "mean": scores.mean(),

            "median": np.median(scores),

            "std": scores.std(ddof=0),

            "p25": np.quantile(scores, 0.25),

            "p75": np.quantile(scores, 0.75),

            "p90": np.quantile(scores, 0.90),

            "p95": np.quantile(scores, 0.95),

            "p99": np.quantile(scores, 0.99),

            "zero_count":
                int((scores == 0).sum()),

            "saturated_count":
                int((scores == 1).sum()),

            "saturated_ratio":
                float((scores == 1).mean()),

            "geometry_length_weighted_mean":
                np.average(
                    scores,
                    weights=df["geometry_length_m"]
                )

        })

    comparison = pd.DataFrame(results)

    config = {

        "model_id": "T09",

        "score_meaning":
            "historical_hotspot_membership_score",

        "reference_link_count": len(raw),

        "minimum": minimum,

        "maximum": maximum,

        "p95": p95,

        "normalization_methods": {

            "N00_raw": "score",

            "N01_minmax":
                "(score - minimum) / (maximum - minimum)",

            "N02_p95":
                "clip(score / p95, 0, 1)",

            "N03_percentile":
                "(average_rank - 1) / (reference_count - 1)"

        },

        "percentile_tie_method": "average",

        "selected_method": None,

        "route_score_formula_selected": False,

        "deployment_status": "not_deployed"

    }

    print("\n===== Normalization Comparison =====")

    print(
        comparison.to_string(index=False)
    )

    return comparison, config, normalized


# ============================================================
# 10. 분석 그래프 생성
# ============================================================

def save_figures(
    df,
    groups,
    normalized,
    metadata
):

    scores = df[SCORE_COLUMN].to_numpy()

    threshold = metadata[
        "classification_threshold"
    ]

    # 1. 전체 점수 히스토그램
    plt.figure(figsize=(10, 6))

    plt.hist(
        scores,
        bins=100,
        range=(0, 1),
        edgecolor="white",
        linewidth=0.3
    )

    plt.axvline(
        scores.mean(),
        linestyle="--",
        label=f"Mean: {scores.mean():.4f}"
    )

    plt.axvline(
        threshold,
        linestyle=":",
        label=f"Threshold: {threshold:.4f}"
    )

    plt.xlabel("Hotspot Membership Score")
    plt.ylabel("Number of Road Links")
    plt.title("National Link Score Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "score_histogram.png",
        dpi=200
    )

    plt.close()

    # 2. 누적분포
    sorted_scores = np.sort(scores)

    cumulative = (
        np.arange(1, len(scores) + 1)
        / len(scores)
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        sorted_scores,
        cumulative
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold: {threshold:.4f}"
    )

    plt.xlabel("Hotspot Membership Score")
    plt.ylabel("Cumulative Proportion")
    plt.title("National Score Cumulative Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "score_cdf.png",
        dpi=200
    )

    plt.close()

    del sorted_scores, cumulative

    # 3~9. 도로 피처 그룹별 평균 점수
    for feature in FEATURES:

        selected = groups.loc[
            groups["feature"] == feature
        ]

        plt.figure(figsize=(10, 6))

        plt.bar(
            selected["group"],
            selected["mean_score"]
        )

        plt.xlabel(feature)
        plt.ylabel("Mean Hotspot Score")
        plt.title(f"Mean Score by {feature}")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / f"score_by_{feature}.png",
            dpi=200
        )

        plt.close()

    # 10. 정규화 방식별 분포 비교
    plt.figure(figsize=(11, 7))

    for method, values in normalized.items():

        plt.hist(
            values,
            bins=50,
            range=(0, 1),
            density=True,
            histtype="step",
            linewidth=1.5,
            label=method
        )

    plt.xlabel("Normalized Score")
    plt.ylabel("Density")
    plt.title("Score Normalization Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "normalization_comparison.png",
        dpi=200
    )

    plt.close()


# ============================================================
# 11. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Link Score Analysis =====")

    training_utils.print_memory(
        "Initial Memory"
    )

    output_paths = [

        STATISTICS_PATH,

        QUANTILE_PATH,

        DISTRIBUTION_PATH,

        GROUP_PATH,

        CORRELATION_PATH,

        NORMALIZATION_PATH,

        NORMALIZATION_CONFIG_PATH,

        VALIDATION_PATH,

        FIGURE_DIR

    ]

    # 기존 결과 파일 보호
    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    metadata_hash = baseline_utils.sha256_file(
        METADATA_PATH
    )

    df, metadata, score_hash = load_data()

    validation = validate_data(
        df,
        metadata
    )

    statistics = calculate_statistics(df)

    quantiles = calculate_quantiles(df)

    distribution = calculate_distribution(df)

    groups = analyze_feature_groups(df)

    correlations = analyze_correlations(df)

    (
        normalization,
        normalization_config,
        normalized
    ) = compare_normalizations(df)

    # 원본 데이터 변경 여부 확인
    assert baseline_utils.sha256_file(
        SCORE_PATH
    ) == score_hash

    assert baseline_utils.sha256_file(
        METADATA_PATH
    ) == metadata_hash

    normalization_config[
        "source_score_sha256"
    ] = score_hash

    normalization_config[
        "source_metadata_sha256"
    ] = metadata_hash

    # 결과 CSV 저장
    outputs = [

        (statistics, STATISTICS_PATH),

        (quantiles, QUANTILE_PATH),

        (distribution, DISTRIBUTION_PATH),

        (groups, GROUP_PATH),

        (correlations, CORRELATION_PATH),

        (normalization, NORMALIZATION_PATH),

        (validation, VALIDATION_PATH)

    ]

    for result, path in outputs:

        result.to_csv(
            path,
            index=False,
            encoding="utf-8-sig"
        )

    # 정규화 후보 설정 저장
    with NORMALIZATION_CONFIG_PATH.open(
        "x", encoding="utf-8"
    ) as file:

        json.dump(
            normalization_config,
            file,
            ensure_ascii=False,
            indent=4
        )

    # 분석 그래프 저장
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=False
    )

    save_figures(
        df,
        groups,
        normalized,
        metadata
    )

    del df, normalized

    gc.collect()

    print("\n===== Saved Results =====")

    for path in output_paths:

        if path.is_file():

            print("Path:", path)

            print(
                "File Size:",
                f"{path.stat().st_size / 1024**2:.4f} MiB"
            )

    figures = sorted(
        FIGURE_DIR.glob("*.png")
    )

    print("\n===== Saved Figures =====")

    for path in figures:

        print("Path:", path)

    training_utils.print_memory(
        "Final Memory"
    )

    print("\n===== Processing Completed =====")

    print(
        "Elapsed Time:",
        f"{time.perf_counter() - start_time:.2f}s"
    )

    print("Total Links:", EXPECTED_ROWS)

    print(
        "Validation Checks:",
        len(validation)
    )

    print(
        "Passed Checks:",
        int(
            (validation["status"] == "PASS").sum()
        )
    )

    print("Normalization Candidates: 4")

    print("Saved Figures:", len(figures))

    print("Score File Hash Verification: PASS")

    print("Metadata Hash Verification: PASS")

    print("Normalization Method Selected: No")

    print("Additional Model Training: Not Performed")

    print("Test Re-evaluation: Not Performed")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()