from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil


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

DATASET_PATH = (
    PROCESSED_DIR
    / "national_accident_risk_dataset.parquet"
)

MATCH_PATH = (
    PROCESSED_DIR
    / "national_accident_link_matches.parquet"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "national_spatial_grid_summary.csv"
)

GRID_SIZES = [
    1000,
    2000,
    5000,
    10000,
    20000
]

EXPECTED_ROWS = 1562356
EXPECTED_POSITIVE = 207391


# ============================================================
# 2. 메모리 확인
# ============================================================

def print_memory(title):

    memory = psutil.virtual_memory()

    print(f"\n===== {title} =====")

    print(
        f"Available RAM: "
        f"{memory.available / 1024**3:.2f} GiB"
    )

    print(f"RAM Usage: {memory.percent}%")


# ============================================================
# 3. 데이터 읽기
# ============================================================

def load_data():

    frame = pd.read_parquet(
        DATASET_PATH,
        columns=[
            "link_id",
            "center_x",
            "center_y",
            "target"
        ]
    )

    matches = pd.read_parquet(
        MATCH_PATH,
        columns=[
            "link_id",
            "hotspot_id"
        ]
    )

    assert len(frame) == EXPECTED_ROWS

    assert (
        int(frame["target"].sum())
        == EXPECTED_POSITIVE
    )

    assert not frame["link_id"].duplicated().any()

    assert not matches.duplicated(
        ["link_id", "hotspot_id"]
    ).any()

    assert np.isfinite(
        frame[
            ["center_x", "center_y"]
        ].to_numpy()
    ).all()

    print("\n===== Dataset Information =====")

    print("Total Links:", len(frame))
    print("Positive Links:", frame["target"].sum())

    print(
        "Negative Links:",
        (frame["target"] == 0).sum()
    )

    print("Match Pairs:", len(matches))

    print(
        "Matched Hotspots:",
        matches["hotspot_id"].nunique()
    )

    print("\nCoordinate Range:")

    print(
        frame[
            ["center_x", "center_y"]
        ].agg(["min", "max"]).to_string()
    )

    return frame, matches


# ============================================================
# 4. 공간 격자별 분석
# ============================================================

def analyze_grid(frame, matches, grid_size):

    print(
        f"\n===== Grid Size: "
        f"{grid_size / 1000:g} km ====="
    )

    # EPSG:5179 좌표를 정수 격자로 변환
    grid_x = np.floor(
        frame["center_x"].to_numpy()
        / grid_size
    ).astype(np.int32)

    grid_y = np.floor(
        frame["center_y"].to_numpy()
        / grid_size
    ).astype(np.int32)

    grid_frame = pd.DataFrame({
        "grid_x": grid_x,
        "grid_y": grid_y,
        "target": frame["target"].to_numpy()
    })

    # 격자별 전체 도로 및 양성 도로 수
    summary = grid_frame.groupby(
        ["grid_x", "grid_y"],
        sort=False
    ).agg(
        total_links=("target", "size"),
        positive_links=("target", "sum")
    )

    summary["negative_links"] = (
        summary["total_links"]
        - summary["positive_links"]
    )

    summary["positive_ratio"] = (
        summary["positive_links"]
        / summary["total_links"]
    )

    positive_grids = summary.loc[
        summary["positive_links"] > 0
    ]

    mixed_grids = summary.loc[
        (summary["positive_links"] > 0)
        & (summary["negative_links"] > 0)
    ]

    positive_only = summary.loc[
        summary["negative_links"] == 0
    ]

    negative_only = summary.loc[
        summary["positive_links"] == 0
    ]

    assert (
        int(summary["total_links"].sum())
        == EXPECTED_ROWS
    )

    assert (
        int(summary["positive_links"].sum())
        == EXPECTED_POSITIVE
    )

    print("Total Occupied Grids:", len(summary))

    print(
        "Positive Grids:",
        len(positive_grids)
    )

    print("Mixed Grids:", len(mixed_grids))
    print("Positive Only:", len(positive_only))
    print("Negative Only:", len(negative_only))

    print("\nLinks per Grid:")

    print(
        summary["total_links"]
        .describe()
        .to_string()
    )

    print("\nPositive Links per Positive Grid:")

    print(
        positive_grids["positive_links"]
        .describe()
        .to_string()
    )

    # ========================================================
    # 5. 사고다발지역의 격자 경계 중복 검사
    # ========================================================

    link_grid = pd.DataFrame({
        "link_id": frame["link_id"],
        "grid_x": grid_x,
        "grid_y": grid_y
    })

    matched_grids = matches.merge(
        link_grid,
        on="link_id",
        how="left",
        validate="many_to_one"
    )

    assert matched_grids[
        ["grid_x", "grid_y"]
    ].notna().all().all()

    hotspot_grid_pairs = (
        matched_grids[
            [
                "hotspot_id",
                "grid_x",
                "grid_y"
            ]
        ].drop_duplicates()
    )

    grids_per_hotspot = (
        hotspot_grid_pairs
        .groupby("hotspot_id")
        .size()
    )

    cross_grid_count = int(
        (grids_per_hotspot > 1).sum()
    )

    cross_grid_ratio = (
        cross_grid_count
        / len(grids_per_hotspot)
    )

    print("\n===== Hotspot Grid Crossing =====")

    print(
        "Matched Hotspots:",
        len(grids_per_hotspot)
    )

    print(
        "Hotspots Across Multiple Grids:",
        cross_grid_count
    )

    print(
        f"Cross Grid Ratio: "
        f"{cross_grid_ratio * 100:.4f}%"
    )

    print(
        "Maximum Grids per Hotspot:",
        grids_per_hotspot.max()
    )

    print("\nGrids per Hotspot:")

    print(
        grids_per_hotspot
        .describe()
        .to_string()
    )

    # ========================================================
    # 6. 양성 데이터 집중도 분석
    # ========================================================

    sorted_positive = (
        positive_grids["positive_links"]
        .sort_values(ascending=False)
    )

    top_1_share = (
        sorted_positive.head(1).sum()
        / EXPECTED_POSITIVE
    )

    top_10_share = (
        sorted_positive.head(10).sum()
        / EXPECTED_POSITIVE
    )

    top_100_share = (
        sorted_positive.head(100).sum()
        / EXPECTED_POSITIVE
    )

    print("\n===== Positive Concentration =====")

    print(
        f"Top 1 Grid Share: "
        f"{top_1_share * 100:.4f}%"
    )

    print(
        f"Top 10 Grid Share: "
        f"{top_10_share * 100:.4f}%"
    )

    print(
        f"Top 100 Grid Share: "
        f"{top_100_share * 100:.4f}%"
    )

    print("\nTop 10 Positive Grids:")

    print(
        positive_grids.nlargest(
            10,
            "positive_links"
        ).to_string()
    )

    return {
        "grid_size_m": grid_size,
        "total_grids": len(summary),
        "positive_grids": len(positive_grids),
        "mixed_grids": len(mixed_grids),
        "positive_only_grids": len(positive_only),
        "negative_only_grids": len(negative_only),

        "mean_links_per_grid":
            summary["total_links"].mean(),

        "max_links_per_grid":
            summary["total_links"].max(),

        "mean_positive_per_grid":
            positive_grids["positive_links"].mean(),

        "max_positive_per_grid":
            positive_grids["positive_links"].max(),

        "cross_grid_hotspots":
            cross_grid_count,

        "cross_grid_ratio":
            cross_grid_ratio,

        "max_grids_per_hotspot":
            grids_per_hotspot.max(),

        "top_1_positive_share":
            top_1_share,

        "top_10_positive_share":
            top_10_share,

        "top_100_positive_share":
            top_100_share
    }


# ============================================================
# 7. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Spatial Distribution =====")

    print("Dataset:", DATASET_PATH)
    print("Matches:", MATCH_PATH)
    print("Grid Sizes:", GRID_SIZES)

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)

    frame, matches = load_data()

    print_memory("After Dataset Load")

    results = []

    for grid_size in GRID_SIZES:

        result = analyze_grid(
            frame,
            matches,
            grid_size
        )

        results.append(result)

        print_memory(
            f"After {grid_size} m Analysis"
        )

    # 격자 크기별 결과 비교
    comparison = pd.DataFrame(results)

    print("\n===== Final Grid Comparison =====")

    print(
        comparison.to_string(index=False)
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== Saved Output =====")

    print("Path:", OUTPUT_PATH)

    print_memory("Final Memory")

    print("\n===== Analysis Completed =====")

    print(
        f"Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    print("Grid Size Selected: No")
    print("Dataset Split: Not Started")
    print("Dataset Modified: No")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()