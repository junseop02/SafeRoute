from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/accident_risk/processed"

# 공간 분할 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

SPLITS = ["train", "validation", "test"]

# 거리 검사 기준 (m)
DISTANCES = [50, 100, 200, 400]

print("===== Input Data =====")
print("Total Links:", len(dataset))
print("Spatial Groups:", dataset["spatial_group"].nunique())

# 1. 공간 그룹별 라벨 집계
group_stats = dataset.groupby(
    ["split", "spatial_group"]
).agg(
    total=("target", "size"),
    positive=("target", "sum")
).reset_index()

group_stats["negative"] = (
    group_stats["total"] - group_stats["positive"]
)

# 2. 데이터셋별 양성 공간 그룹 검사
print("\n===== Positive Spatial Groups =====")

for name in SPLITS:

    groups = group_stats[
        group_stats["split"] == name
    ]

    positive_groups = groups[
        groups["positive"] > 0
    ]

    print(f"\n[{name.upper()}]")
    print("Total Groups:", len(groups))
    print("Positive Groups:", len(positive_groups))

    print(
        "Negative-only Groups:",
        len(groups) - len(positive_groups)
    )

    print(
        "Positive Links:",
        int(positive_groups["positive"].sum())
    )

    print("\nPositive Group Details:")

    print(
        positive_groups[
            ["spatial_group", "total", "positive"]
        ]
        .sort_values("positive", ascending=False)
        .to_string(index=False)
    )

    # 가장 큰 양성 그룹의 집중도
    total_positive = positive_groups["positive"].sum()

    if total_positive > 0:

        concentration = (
            positive_groups["positive"].max()
            / total_positive * 100
        )

        print(
            "Largest Positive Group Share (%):",
            round(concentration, 2)
        )

# 3. 데이터셋별 도로 중심점 추출
frames = {
    name: dataset[
        dataset["split"] == name
    ].copy()
    for name in SPLITS
}

coordinates = {
    name: frames[name][
        ["center_x", "center_y"]
    ].to_numpy()
    for name in SPLITS
}

# 좌표 유효성 검사
for name in SPLITS:

    if not np.isfinite(coordinates[name]).all():
        raise ValueError(
            f"Invalid coordinates in {name}"
        )

# 최근접 거리 탐색용 KDTree 생성
trees = {
    name: cKDTree(coordinates[name])
    for name in SPLITS
}

# 4. 데이터셋 간 최근접 거리 검사
print("\n===== Cross-Split Nearest Distance =====")

for first, second in combinations(SPLITS, 2):

    # 양방향 최근접 거리 검사
    for source, destination in [
        (first, second),
        (second, first)
    ]:

        distances, _ = trees[destination].query(
            coordinates[source],
            k=1
        )

        print(
            f"\n[{source.upper()} "
            f"-> {destination.upper()}]"
        )

        print(
            "Minimum Distance (m):",
            round(float(distances.min()), 4)
        )

        print(
            "Median Distance (m):",
            round(float(np.median(distances)), 2)
        )

        print(
            "Mean Distance (m):",
            round(float(distances.mean()), 2)
        )

        print(
            "Maximum Distance (m):",
            round(float(distances.max()), 2)
        )

        # 거리 기준별 인접 도로 개수
        for threshold in DISTANCES:

            count = int(
                (distances <= threshold).sum()
            )

            ratio = count / len(distances) * 100

            print(
                f"Within {threshold}m: "
                f"{count:,} links ({ratio:.2f}%)"
            )

# 5. 공간 완충구역 적용 영향 분석
# Validation과 Test는 유지하고
# 인접한 Train 데이터만 제외하는 상황을 가정

print("\n===== Training Buffer Impact =====")

train = frames["train"]

holdout_coordinates = np.vstack([
    coordinates["validation"],
    coordinates["test"]
])

holdout_tree = cKDTree(holdout_coordinates)

train_distances, _ = holdout_tree.query(
    coordinates["train"],
    k=1
)

print("Original Training Links:", len(train))
print("Original Positive Links:", int(train["target"].sum()))

for threshold in DISTANCES:

    # 완충구역 외부의 학습 데이터
    keep_mask = train_distances > threshold

    retained = train.loc[keep_mask]

    removed = train.loc[~keep_mask]

    positive = int(retained["target"].sum())
    negative = len(retained) - positive

    print(f"\n[Buffer: {threshold}m]")

    print("Removed Links:", len(removed))

    print(
        "Removed Positive Links:",
        int(removed["target"].sum())
    )

    print("Remaining Links:", len(retained))
    print("Remaining Negative:", negative)
    print("Remaining Positive:", positive)

    print(
        "Remaining Ratio (%):",
        round(len(retained) / len(train) * 100, 2)
    )

    print(
        "Remaining Positive Groups:",
        retained.loc[
            retained["target"] == 1,
            "spatial_group"
        ].nunique()
    )

    if len(retained) > 0:

        print(
            "Positive Ratio (%):",
            round(positive / len(retained) * 100, 2)
        )

print("\n===== Spatial Validation Completed =====")