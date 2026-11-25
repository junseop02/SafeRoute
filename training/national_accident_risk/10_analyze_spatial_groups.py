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
    / "national_spatial_group_summary.csv"
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
EXPECTED_MATCHES = 485562
EXPECTED_HOTSPOTS = 12752


# ============================================================
# 2. Union-Find
# ============================================================

class UnionFind:

    def __init__(self, size):

        self.parent = np.arange(
            size,
            dtype=np.int32
        )

        self.size = np.ones(
            size,
            dtype=np.int32
        )

    def find(self, x):

        # 경로 압축
        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[self.parent[x]]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        # 작은 그룹을 큰 그룹에 병합
        if self.size[root_a] < self.size[root_b]:

            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        self.size[root_a] += self.size[root_b]

        return True


# ============================================================
# 3. 메모리 확인
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
# 4. 데이터 읽기
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

    assert len(matches) == EXPECTED_MATCHES

    assert (
        matches["hotspot_id"].nunique()
        == EXPECTED_HOTSPOTS
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

    # 매칭된 LINK의 원본 행 번호 확보
    link_index = pd.Index(
        frame["link_id"]
    )

    match_positions = link_index.get_indexer(
        matches["link_id"]
    )

    assert (match_positions >= 0).all()

    assert (
        frame["target"].to_numpy()[
            match_positions
        ] == 1
    ).all()

    print("\n===== Dataset Information =====")

    print("Total Links:", len(frame))

    print(
        "Positive Links:",
        frame["target"].sum()
    )

    print("Match Pairs:", len(matches))

    print(
        "Matched Hotspots:",
        matches["hotspot_id"].nunique()
    )

    return frame, matches, match_positions


# ============================================================
# 5. 공간 격자 생성
# ============================================================

def create_grids(frame, grid_size):

    grid_x = np.floor(
        frame["center_x"].to_numpy()
        / grid_size
    ).astype(np.int32)

    grid_y = np.floor(
        frame["center_y"].to_numpy()
        / grid_size
    ).astype(np.int32)

    grid_index = pd.MultiIndex.from_arrays(
        [grid_x, grid_y]
    )

    grid_codes, unique_grids = pd.factorize(
        grid_index,
        sort=True
    )

    grid_codes = grid_codes.astype(
        np.int32
    )

    assert (grid_codes >= 0).all()

    return grid_codes, unique_grids


# ============================================================
# 6. 사고다발지역 기반 격자 병합
# ============================================================

def merge_hotspot_grids(
    grid_codes,
    unique_grids,
    matches,
    match_positions
):

    uf = UnionFind(
        len(unique_grids)
    )

    hotspot_grids = pd.DataFrame({

        "hotspot_id":
            matches["hotspot_id"].to_numpy(),

        "grid_code":
            grid_codes[match_positions]

    }).drop_duplicates()

    merge_count = 0

    for _, group in hotspot_grids.groupby(
        "hotspot_id",
        sort=False
    ):

        codes = group[
            "grid_code"
        ].to_numpy()

        first = int(codes[0])

        for code in codes[1:]:

            if uf.union(
                first,
                int(code)
            ):

                merge_count += 1

    # 각 격자의 최종 그룹 번호
    roots = np.array(
        [
            uf.find(i)
            for i in range(len(unique_grids))
        ],
        dtype=np.int32
    )

    group_codes, unique_roots = np.unique(
        roots,
        return_inverse=False
    ), None

    # 도로별 최종 공간 그룹
    link_groups = roots[grid_codes]

    # 동일 사고다발지역이 여러 그룹에
    # 포함되는지 확인
    hotspot_grids["spatial_group"] = roots[
        hotspot_grids["grid_code"].to_numpy()
    ]

    groups_per_hotspot = (
        hotspot_grids
        .groupby("hotspot_id")["spatial_group"]
        .nunique()
    )

    violations = int(
        (groups_per_hotspot > 1).sum()
    )

    assert violations == 0

    assert (
        len(groups_per_hotspot)
        == EXPECTED_HOTSPOTS
    )

    assert (
        len(group_codes)
        == len(unique_grids) - merge_count
    )

    return (
        roots,
        link_groups,
        merge_count,
        violations
    )


# ============================================================
# 7. 공간 그룹 통계
# ============================================================

def analyze_groups(
    frame,
    matches,
    match_positions,
    grid_size
):

    print(
        f"\n===== Grid Size: "
        f"{grid_size / 1000:g} km ====="
    )

    grid_codes, unique_grids = create_grids(
        frame,
        grid_size
    )

    (
        roots,
        link_groups,
        merge_count,
        violations
    ) = merge_hotspot_grids(
        grid_codes,
        unique_grids,
        matches,
        match_positions
    )

    group_frame = pd.DataFrame({

        "spatial_group": link_groups,

        "target":
            frame["target"].to_numpy()

    })

    summary = group_frame.groupby(
        "spatial_group"
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

    grid_counts = pd.Series(
        roots
    ).value_counts()

    summary["grid_count"] = (
        grid_counts.reindex(summary.index)
    )

    positive_groups = summary.loc[
        summary["positive_links"] > 0
    ]

    negative_groups = summary.loc[
        summary["positive_links"] == 0
    ]

    mixed_groups = summary.loc[
        (summary["positive_links"] > 0)
        & (summary["negative_links"] > 0)
    ]

    positive_only = summary.loc[
        summary["negative_links"] == 0
    ]

    assert (
        int(summary["total_links"].sum())
        == EXPECTED_ROWS
    )

    assert (
        int(summary["positive_links"].sum())
        == EXPECTED_POSITIVE
    )

    assert (
        int(summary["grid_count"].sum())
        == len(unique_grids)
    )

    print("\n===== Group Summary =====")

    print("Original Grids:", len(unique_grids))
    print("Successful Merges:", merge_count)
    print("Final Groups:", len(summary))

    print(
        "Positive Groups:",
        len(positive_groups)
    )

    print(
        "Negative Only Groups:",
        len(negative_groups)
    )

    print(
        "Mixed Groups:",
        len(mixed_groups)
    )

    print(
        "Positive Only Groups:",
        len(positive_only)
    )

    print(
        "Hotspot Group Violations:",
        violations
    )

    print("\n===== Grids per Group =====")

    print(
        summary["grid_count"]
        .describe()
        .to_string()
    )

    print("\n===== Links per Group =====")

    print(
        summary["total_links"]
        .describe()
        .to_string()
    )

    print("\n===== Positive Links per Group =====")

    print(
        positive_groups["positive_links"]
        .describe()
        .to_string()
    )

    # 양성 데이터 집중도
    sorted_positive = (
        positive_groups["positive_links"]
        .sort_values(ascending=False)
    )

    top_1_share = (
        sorted_positive.head(1).sum()
        / EXPECTED_POSITIVE
    )

    top_5_share = (
        sorted_positive.head(5).sum()
        / EXPECTED_POSITIVE
    )

    top_10_share = (
        sorted_positive.head(10).sum()
        / EXPECTED_POSITIVE
    )

    print("\n===== Positive Concentration =====")

    print(
        f"Top 1 Group Share: "
        f"{top_1_share * 100:.4f}%"
    )

    print(
        f"Top 5 Group Share: "
        f"{top_5_share * 100:.4f}%"
    )

    print(
        f"Top 10 Group Share: "
        f"{top_10_share * 100:.4f}%"
    )

    print("\n===== Top 10 Positive Groups =====")

    print(
        positive_groups.nlargest(
            10,
            "positive_links"
        ).to_string()
    )

    print("\n===== Top 10 Largest Groups =====")

    print(
        summary.nlargest(
            10,
            "total_links"
        ).to_string()
    )

    return {

        "grid_size_m": grid_size,

        "original_grids":
            len(unique_grids),

        "successful_merges":
            merge_count,

        "final_groups":
            len(summary),

        "positive_groups":
            len(positive_groups),

        "negative_only_groups":
            len(negative_groups),

        "mixed_groups":
            len(mixed_groups),

        "positive_only_groups":
            len(positive_only),

        "max_grids_per_group":
            int(summary["grid_count"].max()),

        "max_links_per_group":
            int(summary["total_links"].max()),

        "max_positive_per_group":
            int(summary["positive_links"].max()),

        "top_1_positive_share":
            top_1_share,

        "top_5_positive_share":
            top_5_share,

        "top_10_positive_share":
            top_10_share,

        "hotspot_group_violations":
            violations
    }


# ============================================================
# 8. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Spatial Group Analysis =====")

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

    (
        frame,
        matches,
        match_positions
    ) = load_data()

    print_memory("After Dataset Load")

    results = []

    for grid_size in GRID_SIZES:

        result = analyze_groups(
            frame,
            matches,
            match_positions,
            grid_size
        )

        results.append(result)

        print_memory(
            f"After {grid_size} m Analysis"
        )

    comparison = pd.DataFrame(
        results
    )

    print("\n===== Final Group Comparison =====")

    print(
        comparison.to_string(
            index=False
        )
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