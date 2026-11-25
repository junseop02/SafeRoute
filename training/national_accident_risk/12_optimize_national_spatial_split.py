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

MATCH_PATH = (
    PROCESSED_DIR
    / "national_accident_link_matches.parquet"
)

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

N_RESTARTS = 20
N_ITERATIONS = 30000

EXPECTED_ROWS = 1562356
EXPECTED_POSITIVE = 207391

SUMMARY_PATH = (
    RESULTS_DIR
    / "national_optimized_split_summary.csv"
)


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
# 3. 분할 오차 계산
# ============================================================

def calculate_error(counts, totals):

    ratios = counts / totals

    return float(
        np.sum(
            (
                ratios
                - TARGET_RATIOS[:, None]
            ) ** 2
        )
    )


# ============================================================
# 4. 그룹별 데이터 수 계산
# ============================================================

def calculate_counts(values, assignment):

    counts = np.zeros(
        (3, 3),
        dtype=np.int64
    )

    np.add.at(
        counts,
        assignment,
        values
    )

    return counts


# ============================================================
# 5. 공간 분할 최적화
# ============================================================

def optimize_assignment(
    values,
    original_assignment
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    totals = values.sum(axis=0)

    group_count = len(values)

    positive_indices = np.flatnonzero(
        values[:, 1] > 0
    )

    best_assignment = (
        original_assignment.copy()
    )

    best_counts = calculate_counts(
        values,
        best_assignment
    )

    original_error = calculate_error(
        best_counts,
        totals
    )

    best_error = original_error

    print(
        "Original Split Error:",
        original_error
    )

    # 그룹 크기를 정규화하여 반복 계산량 감소
    normalized_values = (
        values / totals
    )

    target = TARGET_RATIOS[:, None]

    for restart in range(N_RESTARTS):

        # 첫 번째 실험은 기존 분할에서 시작
        if restart == 0:

            assignment = (
                original_assignment.copy()
            )

        # 나머지는 무작위 그룹 배정
        else:

            assignment = rng.choice(
                3,
                size=group_count,
                p=TARGET_RATIOS
            ).astype(np.int8)

        counts = calculate_counts(
            values,
            assignment
        )

        ratios = counts / totals

        current_error = calculate_error(
            counts,
            totals
        )

        accepted_moves = 0
        accepted_swaps = 0

        for iteration in range(N_ITERATIONS):

            # 양성 그룹을 우선적으로 탐색
            if rng.random() < 0.7:

                i = int(
                    rng.choice(
                        positive_indices
                    )
                )

            else:

                i = int(
                    rng.integers(group_count)
                )

            source = int(assignment[i])

            # --------------------------------------------
            # 그룹 하나를 다른 분할로 이동
            # --------------------------------------------

            if rng.random() < 0.7:

                destination = (
                    source
                    + int(rng.integers(1, 3))
                ) % 3

                # 기존 분할에 양성과 음성 유지
                if (
                    counts[source, 1:]
                    - values[i, 1:]
                    <= 0
                ).any():

                    continue

                delta = normalized_values[i]

                before = (
                    np.sum(
                        (ratios[source] - target[source]) ** 2
                    )
                    + np.sum(
                        (ratios[destination] - target[destination]) ** 2
                    )
                )

                after = (
                    np.sum(
                        (
                            ratios[source]
                            - delta
                            - target[source]
                        ) ** 2
                    )
                    + np.sum(
                        (
                            ratios[destination]
                            + delta
                            - target[destination]
                        ) ** 2
                    )
                )

                if after < before:

                    assignment[i] = destination

                    counts[source] -= values[i]
                    counts[destination] += values[i]

                    ratios[source] = (
                        counts[source] / totals
                    )

                    ratios[destination] = (
                        counts[destination] / totals
                    )

                    accepted_moves += 1

            # --------------------------------------------
            # 서로 다른 분할의 그룹 교환
            # --------------------------------------------

            else:

                j = int(
                    rng.integers(group_count)
                )

                destination = int(
                    assignment[j]
                )

                if source == destination:

                    continue

                new_source = (
                    counts[source]
                    - values[i]
                    + values[j]
                )

                new_destination = (
                    counts[destination]
                    - values[j]
                    + values[i]
                )

                if (
                    (new_source[1:] <= 0).any()
                    or
                    (new_destination[1:] <= 0).any()
                ):

                    continue

                before = (
                    np.sum(
                        (ratios[source] - target[source]) ** 2
                    )
                    + np.sum(
                        (ratios[destination] - target[destination]) ** 2
                    )
                )

                after = (
                    np.sum(
                        (
                            new_source / totals
                            - target[source]
                        ) ** 2
                    )
                    + np.sum(
                        (
                            new_destination / totals
                            - target[destination]
                        ) ** 2
                    )
                )

                if after < before:

                    assignment[i] = destination
                    assignment[j] = source

                    counts[source] = new_source

                    counts[destination] = (
                        new_destination
                    )

                    ratios[source] = (
                        new_source / totals
                    )

                    ratios[destination] = (
                        new_destination / totals
                    )

                    accepted_swaps += 1

        # 최종 배정 결과 재계산
        counts = calculate_counts(
            values,
            assignment
        )

        current_error = calculate_error(
            counts,
            totals
        )

        if (
            (counts[:, 1:] > 0).all()
            and current_error < best_error
        ):

            best_error = current_error

            best_assignment = assignment.copy()

            best_counts = counts.copy()

        print(
            f"Restart: {restart + 1}/{N_RESTARTS}"
            f" | Error: {current_error:.10f}"
            f" | Best: {best_error:.10f}"
            f" | Moves: {accepted_moves}"
            f" | Swaps: {accepted_swaps}"
        )

    assert best_error <= original_error

    return (
        best_assignment,
        best_counts,
        original_error,
        best_error
    )


# ============================================================
# 6. 격자별 최적화
# ============================================================

def optimize_grid(grid_size, matches):

    print(
        f"\n===== Grid Size: "
        f"{grid_size / 1000:g} km ====="
    )

    source_path = (
        PROCESSED_DIR
        / f"national_spatial_split_{grid_size}m.parquet"
    )

    output_path = (
        PROCESSED_DIR
        / f"national_optimized_split_{grid_size}m.parquet"
    )

    frame = pd.read_parquet(
        source_path
    )

    assert len(frame) == EXPECTED_ROWS

    assert (
        frame["target"].sum()
        == EXPECTED_POSITIVE
    )

    assert not frame[
        "link_id"
    ].duplicated().any()

    # 기존 그룹 중복 확인
    assert (
        frame.groupby(
            "spatial_group"
        )["split"].nunique().max()
        == 1
    )

    # 공간 그룹별 통계
    summary = frame.groupby(
        "spatial_group",
        sort=True
    ).agg(

        total_links=("target", "size"),

        positive_links=("target", "sum"),

        original_split=("split", "first")

    )

    summary["negative_links"] = (
        summary["total_links"]
        - summary["positive_links"]
    )

    values = summary[
        [
            "total_links",
            "positive_links",
            "negative_links"
        ]
    ].to_numpy(dtype=np.int64)

    original_assignment = (
        pd.Categorical(
            summary["original_split"],
            categories=SPLIT_NAMES
        ).codes.copy()
    )

    assert (
        original_assignment >= 0
    ).all()

    print("Total Groups:", len(summary))

    print(
        "Positive Groups:",
        (summary["positive_links"] > 0).sum()
    )

    # 그룹 재배정 최적화
    (
        assignment,
        counts,
        original_error,
        optimized_error
    ) = optimize_assignment(

        values,
        original_assignment

    )

    group_positions = (
        summary.index.get_indexer(
            frame["spatial_group"]
        )
    )

    assert (
        group_positions >= 0
    ).all()

    split_codes = assignment[
        group_positions
    ]

    # 원본 파일은 변경하지 않음
    frame["split"] = (
        pd.Categorical.from_codes(
            split_codes,
            categories=SPLIT_NAMES
        )
    )

    summary["split_code"] = assignment

    # ========================================================
    # 7. 공간 분할 검증
    # ========================================================

    group_violations = int(

        (
            frame.groupby(
                "spatial_group"
            )["split"].nunique()
            > 1
        ).sum()

    )

    grid_violations = int(

        (
            frame.groupby(
                ["grid_x", "grid_y"]
            )["split"].nunique()
            > 1
        ).sum()

    )

    hotspot_splits = matches.merge(

        frame[["link_id", "split"]],

        on="link_id",

        how="left",

        validate="many_to_one"

    )

    assert hotspot_splits[
        "split"
    ].notna().all()

    hotspot_violations = int(

        (
            hotspot_splits.groupby(
                "hotspot_id"
            )["split"].nunique()
            > 1
        ).sum()

    )

    assert group_violations == 0
    assert grid_violations == 0
    assert hotspot_violations == 0

    assert len(frame) == EXPECTED_ROWS

    assert (
        frame["target"].sum()
        == EXPECTED_POSITIVE
    )

    print("\n===== Split Validation =====")

    print("Group Violations:", group_violations)
    print("Grid Violations:", grid_violations)
    print("Hotspot Violations:", hotspot_violations)

    # ========================================================
    # 8. 최적화 결과 정리
    # ========================================================

    results = []

    for i, split_name in enumerate(
        SPLIT_NAMES
    ):

        total, positive, negative = counts[i]

        group_subset = summary.loc[
            summary["split_code"] == i
        ]

        subset = frame.loc[
            split_codes == i
        ]

        assert len(subset) == total

        assert (
            subset["target"].sum()
            == positive
        )

        positive_groups = int(

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

            "total_links": int(total),

            "positive_links": int(positive),

            "negative_links": int(negative),

            "total_ratio":
                total / EXPECTED_ROWS,

            "positive_ratio":
                positive / total,

            "positive_share":
                positive / EXPECTED_POSITIVE,

            "spatial_groups":
                len(group_subset),

            "positive_groups":
                positive_groups,

            "max_positive_group":
                max_positive_group,

            "max_positive_group_share":
                max_positive_group / positive,

            "original_error":
                original_error,

            "optimized_error":
                optimized_error,

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

    print("\n===== Optimized Distribution =====")

    print(
        result_frame.to_string(
            index=False
        )
    )

    improvement = (

        (original_error - optimized_error)
        / original_error * 100

        if original_error > 0
        else 0.0

    )

    print("\n===== Optimization Result =====")

    print("Original Error:", original_error)
    print("Optimized Error:", optimized_error)

    print(
        f"Error Reduction: "
        f"{improvement:.6f}%"
    )

    changed_groups = int(
        (
            assignment
            != original_assignment
        ).sum()
    )

    print("Changed Groups:", changed_groups)

    # 새로운 파일로 저장
    frame.to_parquet(
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
# 9. 메인 실행
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Spatial Split Optimization =====")

    print("Grid Sizes:", GRID_SIZES)
    print("Target Ratios:", TARGET_RATIOS)
    print("Random State:", RANDOM_STATE)
    print("Restarts:", N_RESTARTS)
    print("Iterations per Restart:", N_ITERATIONS)

    print_memory("Initial Memory")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_paths = [

        PROCESSED_DIR
        / f"national_optimized_split_{size}m.parquet"

        for size in GRID_SIZES

    ]

    output_paths.append(SUMMARY_PATH)

    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    matches = pd.read_parquet(

        MATCH_PATH,

        columns=[
            "link_id",
            "hotspot_id"
        ]

    )

    assert len(matches) == 485562

    assert (
        matches["hotspot_id"].nunique()
        == 12752
    )

    assert not matches.duplicated(
        ["link_id", "hotspot_id"]
    ).any()

    results = []

    for grid_size in GRID_SIZES:

        result = optimize_grid(
            grid_size,
            matches
        )

        results.append(result)

        print_memory(
            f"After {grid_size} m Optimization"
        )

    comparison = pd.concat(
        results,
        ignore_index=True
    )

    print("\n===== Final Optimized Comparison =====")

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

    print("Original Split Files: Not Modified")
    print("Grid Size Selected: No")
    print("Buffer Applied: No")
    print("Final Test Set Frozen: No")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()