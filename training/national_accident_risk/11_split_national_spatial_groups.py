from pathlib import Path
import importlib
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

GRID_SIZES = [1000, 2000, 5000]

SPLIT_NAMES = [
    "train",
    "validation",
    "test"
]

TARGET_RATIOS = np.array([
    0.70,
    0.15,
    0.15
])

RANDOM_STATE = 42

N_CANDIDATES = 100

EXPECTED_ROWS = 1562356
EXPECTED_POSITIVE = 207391

SUMMARY_PATH = (
    RESULTS_DIR
    / "national_spatial_split_summary.csv"
)


# ============================================================
# 2. 기존 공간 그룹 함수 불러오기
# ============================================================

group_module = importlib.import_module(
    "10_analyze_spatial_groups"
)


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
# 4. 공간 그룹별 통계 생성
# ============================================================

def build_group_summary(frame, link_groups):

    group_frame = pd.DataFrame({

        "spatial_group": link_groups,

        "target":
            frame["target"].to_numpy()

    })

    summary = group_frame.groupby(
        "spatial_group",
        sort=True
    ).agg(

        total_links=("target", "size"),

        positive_links=("target", "sum")

    )

    summary["negative_links"] = (
        summary["total_links"]
        - summary["positive_links"]
    )

    assert (
        summary["total_links"].sum()
        == EXPECTED_ROWS
    )

    assert (
        summary["positive_links"].sum()
        == EXPECTED_POSITIVE
    )

    return summary


# ============================================================
# 5. 분할 오차 계산
# ============================================================

def calculate_split_error(
    assigned_counts,
    total_counts
):

    actual_ratios = (
        assigned_counts
        / total_counts
    )

    error = np.sum(

        (
            actual_ratios
            - TARGET_RATIOS[:, None]
        ) ** 2

    )

    return float(error)


# ============================================================
# 6. 공간 그룹 분할 후보 탐색
# ============================================================

def search_best_split(summary):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    group_values = summary[
        [
            "total_links",
            "positive_links",
            "negative_links"
        ]
    ].to_numpy(dtype=np.int64)

    total_counts = group_values.sum(
        axis=0
    )

    group_count = len(summary)

    # 전체 및 클래스별 상대적인 그룹 크기
    importance = (
        group_values
        / total_counts
    ).max(axis=1)

    best_error = np.inf
    best_assignment = None
    best_counts = None
    valid_candidates = 0

    for candidate in range(N_CANDIDATES):

        # 큰 그룹을 먼저 배정하되
        # 후보마다 순서를 조금씩 변경
        if candidate == 0:

            priority = importance

        else:

            priority = importance * rng.uniform(
                0.7,
                1.3,
                size=group_count
            )

        order = np.argsort(
            -priority,
            kind="stable"
        )

        assigned_counts = np.zeros(
            (3, 3),
            dtype=np.int64
        )

        assignment = np.full(
            group_count,
            -1,
            dtype=np.int8
        )

        target_counts = (
            TARGET_RATIOS[:, None]
            * total_counts
        )

        # 그룹별 데이터를 분할 목표에 맞게 배정
        for group_idx in order:

            values = group_values[group_idx]

            before = (
                assigned_counts
                - target_counts
            ) / total_counts

            after = (
                assigned_counts
                + values
                - target_counts
            ) / total_counts

            costs = (

                (after ** 2).sum(axis=1)

                - (before ** 2).sum(axis=1)

            )

            split_idx = int(
                np.argmin(costs)
            )

            assignment[group_idx] = split_idx

            assigned_counts[split_idx] += values

        assert (assignment >= 0).all()

        # 모든 분할에 양성과 음성이 존재해야 함
        if (assigned_counts[:, 1:] == 0).any():
            continue

        valid_candidates += 1

        error = calculate_split_error(
            assigned_counts,
            total_counts
        )

        if error < best_error:

            best_error = error

            best_assignment = assignment.copy()

            best_counts = assigned_counts.copy()

        if (candidate + 1) % 10 == 0:

            print(

                f"Candidate: "
                f"{candidate + 1}/{N_CANDIDATES}"

                f" | Best Error: "
                f"{best_error:.10f}"

            )

    if best_assignment is None:

        raise RuntimeError(
            "No valid spatial split found."
        )

    print("\nValid Candidates:", valid_candidates)
    print("Best Split Error:", best_error)

    return (
        best_assignment,
        best_counts,
        best_error
    )


# ============================================================
# 7. 격자 크기별 데이터 분할
# ============================================================

def split_dataset(
    frame,
    matches,
    match_positions,
    grid_size
):

    print(
        f"\n===== Grid Size: "
        f"{grid_size / 1000:g} km ====="
    )

    # 기존 10번 코드와 동일한 격자 생성
    grid_codes, unique_grids = (
        group_module.create_grids(
            frame,
            grid_size
        )
    )

    # 기존 10번 코드와 동일한 그룹 병합
    (
        roots,
        link_groups,
        merge_count,
        violations
    ) = group_module.merge_hotspot_grids(

        grid_codes,
        unique_grids,
        matches,
        match_positions

    )

    assert violations == 0

    summary = build_group_summary(
        frame,
        link_groups
    )

    print("Original Grids:", len(unique_grids))
    print("Final Groups:", len(summary))

    print(
        "Positive Groups:",
        (summary["positive_links"] > 0).sum()
    )

    print("Successful Merges:", merge_count)

    # 최적 그룹 배정 탐색
    (
        assignment,
        assigned_counts,
        best_error
    ) = search_best_split(summary)

    summary["split_code"] = assignment

    # 각 도로의 그룹을 분할 코드로 변환
    group_positions = summary.index.get_indexer(
        link_groups
    )

    assert (group_positions >= 0).all()

    split_codes = assignment[group_positions]

    split_frame = pd.DataFrame({

        "link_id":
            frame["link_id"],

        "grid_x":
            np.floor(
                frame["center_x"].to_numpy()
                / grid_size
            ).astype(np.int32),

        "grid_y":
            np.floor(
                frame["center_y"].to_numpy()
                / grid_size
            ).astype(np.int32),

        "spatial_group":
            link_groups,

        "target":
            frame["target"],

        "split":
            pd.Categorical.from_codes(
                split_codes,
                categories=SPLIT_NAMES
            )

    })

    # ========================================================
    # 8. 분할 결과 검증
    # ========================================================

    assert len(split_frame) == EXPECTED_ROWS

    assert not split_frame[
        "link_id"
    ].duplicated().any()

    assert split_frame["split"].notna().all()

    # 동일 공간 그룹이 여러 분할에 들어가는지 확인
    group_split_counts = (
        split_frame.groupby(
            "spatial_group"
        )["split"].nunique()
    )

    group_violations = int(
        (group_split_counts > 1).sum()
    )

    assert group_violations == 0

    # 동일 격자의 분할 중복 확인
    grid_split_counts = (
        split_frame.groupby(
            ["grid_x", "grid_y"]
        )["split"].nunique()
    )

    grid_violations = int(
        (grid_split_counts > 1).sum()
    )

    assert grid_violations == 0

    # 동일 사고다발지역의 분할 중복 확인
    hotspot_splits = pd.DataFrame({

        "hotspot_id":
            matches["hotspot_id"].to_numpy(),

        "split_code":
            split_codes[match_positions]

    })

    hotspot_split_counts = (
        hotspot_splits.groupby(
            "hotspot_id"
        )["split_code"].nunique()
    )

    hotspot_violations = int(
        (hotspot_split_counts > 1).sum()
    )

    assert hotspot_violations == 0

    print("\n===== Split Validation =====")

    print(
        "Group Violations:",
        group_violations
    )

    print(
        "Grid Violations:",
        grid_violations
    )

    print(
        "Hotspot Violations:",
        hotspot_violations
    )

    # ========================================================
    # 9. 분할별 통계
    # ========================================================

    results = []

    print("\n===== Split Distribution =====")

    for split_idx, split_name in enumerate(
        SPLIT_NAMES
    ):

        subset = split_frame.loc[
            split_codes == split_idx
        ]

        group_subset = summary.loc[
            summary["split_code"] == split_idx
        ]

        total = len(subset)

        positive = int(
            subset["target"].sum()
        )

        negative = total - positive

        assert np.array_equal(

            np.array([
                total,
                positive,
                negative
            ]),

            assigned_counts[split_idx]

        )

        positive_group_count = int(

            (
                group_subset["positive_links"]
                > 0
            ).sum()

        )

        max_positive_group = int(
            group_subset["positive_links"].max()
        )

        results.append({

            "grid_size_m": grid_size,

            "split": split_name,

            "total_links": total,

            "positive_links": positive,

            "negative_links": negative,

            "total_ratio":
                total / EXPECTED_ROWS,

            "positive_ratio":
                positive / total,

            "positive_share":
                positive / EXPECTED_POSITIVE,

            "spatial_groups":
                len(group_subset),

            "positive_groups":
                positive_group_count,

            "max_positive_group":
                max_positive_group,

            "max_positive_group_share":
                max_positive_group / positive,

            "split_error":
                best_error,

            "group_violations":
                group_violations,

            "grid_violations":
                grid_violations,

            "hotspot_violations":
                hotspot_violations

        })

    result_frame = pd.DataFrame(
        results
    )

    print(
        result_frame.to_string(
            index=False
        )
    )

    # ========================================================
    # 10. 분할 후보 저장
    # ========================================================

    output_path = (

        PROCESSED_DIR

        / f"national_spatial_split_{grid_size}m.parquet"

    )

    split_frame.to_parquet(
        output_path,
        index=False
    )

    print("\n===== Saved Output =====")

    print("Path:", output_path)

    print(
        f"File Size: "
        f"{output_path.stat().st_size / 1024**2:.2f} MiB"
    )

    return result_frame


# ============================================================
# 11. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Spatial Split =====")

    print("Grid Sizes:", GRID_SIZES)
    print("Target Ratios:", TARGET_RATIOS)
    print("Random State:", RANDOM_STATE)
    print("Candidates:", N_CANDIDATES)

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 기존 결과 덮어쓰기 방지
    output_paths = [

        PROCESSED_DIR
        / f"national_spatial_split_{size}m.parquet"

        for size in GRID_SIZES

    ]

    output_paths.append(SUMMARY_PATH)

    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    # 기존 10번 데이터 로딩 함수 재사용
    (
        frame,
        matches,
        match_positions
    ) = group_module.load_data()

    print_memory("After Dataset Load")

    results = []

    for grid_size in GRID_SIZES:

        result = split_dataset(

            frame,
            matches,
            match_positions,
            grid_size

        )

        results.append(result)

        print_memory(
            f"After {grid_size} m Split"
        )

    comparison = pd.concat(
        results,
        ignore_index=True
    )

    print("\n===== Final Split Comparison =====")

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== Saved Summary =====")

    print("Path:", SUMMARY_PATH)

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(
        f"Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    print("Grid Size Selected: No")
    print("Buffer Applied: No")
    print("Final Test Set Frozen: No")
    print("Original Dataset Modified: No")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()