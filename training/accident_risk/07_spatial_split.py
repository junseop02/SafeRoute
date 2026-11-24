from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import GroupShuffleSplit

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/accident_risk/processed"

# 실험 설정
GRID_SIZE = 1000
RANDOM_STATE = 42
N_CANDIDATES = 10000  # 공간 분할 후보 탐색 횟수

# 학습 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_dataset.parquet"
)

# 사고다발지역 매칭 결과 불러오기
matches = gpd.read_parquet(
    DATA_DIR / "accident_link_matches.parquet"
)

dataset["link_id"] = dataset["link_id"].astype(str)
matches["link_id"] = matches["link_id"].astype(str)

# 실제 중첩 길이가 있는 매칭만 선택
matches = matches[
    matches["overlap_length_m"] > 0
].copy()

print("===== Input Data =====")
print("Total Links:", len(dataset))
print("Matched Pairs:", len(matches))
print("Grid Size (m):", GRID_SIZE)


# 1. 공간 격자 생성
dataset["grid_x"] = np.floor(
    dataset["center_x"] / GRID_SIZE
).astype(int)

dataset["grid_y"] = np.floor(
    dataset["center_y"] / GRID_SIZE
).astype(int)

dataset["grid_id"] = (
    dataset["grid_x"].astype(str)
    + "_"
    + dataset["grid_y"].astype(str)
)

print("\n===== Spatial Grid =====")
print("Unique Grids:", dataset["grid_id"].nunique())


# 2. 동일 사고다발지역 격자 통합
# Union-Find 자료구조 사용
parent = {
    grid: grid
    for grid in dataset["grid_id"].unique()
}


def find(grid):

    while parent[grid] != grid:

        parent[grid] = parent[parent[grid]]
        grid = parent[grid]

    return grid


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:

        parent[root_b] = root_a


# 사고다발지역과 격자 정보 결합
hotspot_grids = matches[
    ["link_id", "hotspot_id"]
].merge(
    dataset[["link_id", "grid_id"]],
    on="link_id",
    how="left",
    validate="many_to_one"
)

if hotspot_grids["grid_id"].isna().any():
    raise ValueError("Missing grid IDs.")


# 동일 사고다발지역에 속한 격자 통합
for hotspot_id, group in hotspot_grids.groupby(
    "hotspot_id"
):

    grids = sorted(group["grid_id"].unique())

    for grid in grids[1:]:

        union(grids[0], grid)


# 최종 공간 그룹 생성
dataset["spatial_group"] = (
    dataset["grid_id"].map(find)
)

print("\n===== Spatial Groups =====")

print(
    "Unique Spatial Groups:",
    dataset["spatial_group"].nunique()
)

group_sizes = dataset.groupby(
    "spatial_group"
).size()

print("\nGroup Size Statistics:")
print(group_sizes.describe())

print("\nTop 10 Largest Groups:")
print(group_sizes.nlargest(10))


# 3. 공간 그룹 단위 데이터 분할 개선
# 그룹별 데이터 개수 및 클래스 분포 고려

group_stats = dataset.groupby(
    "spatial_group"
).agg(
    total=("target", "size"),
    positive=("target", "sum")
).reset_index()

group_stats["negative"] = (
    group_stats["total"]
    - group_stats["positive"]
)

group_ids = group_stats["spatial_group"].to_numpy()

group_counts = group_stats[
    ["total", "positive", "negative"]
].to_numpy()

total_counts = group_counts.sum(axis=0)

target_ratios = np.array([
    0.70,
    0.15,
    0.15
])

n_groups = len(group_stats)

n_test = int(np.ceil(n_groups * 0.15))
n_val = int(np.ceil(n_groups * 0.15))

rng = np.random.default_rng(RANDOM_STATE)

best_score = float("inf")
best_groups = None
valid_candidates = 0

print("\n===== Balanced Spatial Split Search =====")
print("Total Groups:", n_groups)

print(
    "Positive Groups:",
    (group_stats["positive"] > 0).sum()
)

print(
    "Negative-only Groups:",
    (group_stats["positive"] == 0).sum()
)

print("Search Candidates:", N_CANDIDATES)

# 여러 공간 분할 후보 탐색
for attempt in range(N_CANDIDATES):

    shuffled = rng.permutation(n_groups)

    test_group_idx = shuffled[:n_test]

    val_group_idx = shuffled[
        n_test:n_test + n_val
    ]

    train_group_idx = shuffled[
        n_test + n_val:
    ]

    # 데이터셋별 전체, 양성, 음성 개수
    split_counts = np.array([
        group_counts[train_group_idx].sum(axis=0),
        group_counts[val_group_idx].sum(axis=0),
        group_counts[test_group_idx].sum(axis=0)
    ])

    # 양성 또는 음성 클래스가 없으면 제외
    if (split_counts[:, 1:] == 0).any():
        continue

    valid_candidates += 1

    # 전체 데이터 및 클래스별 분할 비율
    actual_ratios = (
        split_counts / total_counts
    )

    # 목표 비율과의 제곱 오차 합
    score = np.sum(
        (
            actual_ratios
            - target_ratios[:, None]
        ) ** 2
    )

    # 가장 균형 잡힌 분할 후보 저장
    if score < best_score:

        best_score = score

        best_groups = (
            group_ids[train_group_idx].copy(),
            group_ids[val_group_idx].copy(),
            group_ids[test_group_idx].copy()
        )

# 유효한 후보가 없으면 중단
if best_groups is None:
    raise ValueError(
        "No valid spatial split found."
    )

print("\n===== Search Result =====")
print("Valid Candidates:", valid_candidates)
print("Best Balance Score:", best_score)

# 최종 공간 그룹 배정
selected_train_groups = set(best_groups[0])
selected_val_groups = set(best_groups[1])
selected_test_groups = set(best_groups[2])

# 공간 그룹 기준 데이터 분리
train = dataset[
    dataset["spatial_group"].isin(
        selected_train_groups
    )
].copy()

val = dataset[
    dataset["spatial_group"].isin(
        selected_val_groups
    )
].copy()

test = dataset[
    dataset["spatial_group"].isin(
        selected_test_groups
    )
].copy()


# 4. 데이터셋 구분 컬럼 생성
train["split"] = "train"
val["split"] = "validation"
test["split"] = "test"

splits = pd.concat(
    [train, val, test],
    ignore_index=True
)

print("\n===== Split Distribution =====")

for name, frame in [
    ("Train", train),
    ("Validation", val),
    ("Test", test)
]:

    positive = int(frame["target"].sum())
    negative = len(frame) - positive

    print(f"\n[{name}]")
    print("Rows:", len(frame))

    print(
        "Dataset Ratio (%):",
        round(len(frame) / len(dataset) * 100, 2)
    )

    print("Negative:", negative)
    print("Positive:", positive)

    print(
        "Positive Ratio (%):",
        round(positive / len(frame) * 100, 2)
    )

    print(
        "Spatial Groups:",
        frame["spatial_group"].nunique()
    )


# 5. 도로 링크 중복 검사
train_ids = set(train["link_id"])
val_ids = set(val["link_id"])
test_ids = set(test["link_id"])

print("\n===== Link Leakage Check =====")
print("Train-Val:", len(train_ids & val_ids))
print("Train-Test:", len(train_ids & test_ids))
print("Val-Test:", len(val_ids & test_ids))

if (
    train_ids & val_ids
    or train_ids & test_ids
    or val_ids & test_ids
):
    raise ValueError("Link leakage detected.")


# 6. 공간 그룹 중복 검사
train_groups = set(train["spatial_group"])
val_groups = set(val["spatial_group"])
test_groups = set(test["spatial_group"])

print("\n===== Spatial Group Leakage Check =====")

print(
    "Train-Val:",
    len(train_groups & val_groups)
)

print(
    "Train-Test:",
    len(train_groups & test_groups)
)

print(
    "Val-Test:",
    len(val_groups & test_groups)
)

if (
    train_groups & val_groups
    or train_groups & test_groups
    or val_groups & test_groups
):
    raise ValueError("Spatial group leakage detected.")


# 7. 동일 사고다발지역 분리 여부 검사
hotspot_split = matches[
    ["link_id", "hotspot_id"]
].merge(
    splits[["link_id", "split"]],
    on="link_id",
    how="left",
    validate="many_to_one"
)

if hotspot_split["split"].isna().any():
    raise ValueError("Missing split assignments.")

hotspot_split_count = hotspot_split.groupby(
    "hotspot_id"
)["split"].nunique()

cross_split_hotspots = hotspot_split_count[
    hotspot_split_count > 1
]

print("\n===== Hotspot Leakage Check =====")

print(
    "Hotspots Across Multiple Splits:",
    len(cross_split_hotspots)
)

if not cross_split_hotspots.empty:
    raise ValueError("Hotspot leakage detected.")


# 8. 지역별 분할 결과
print("\n===== Regional Split Distribution =====")

print(
    pd.crosstab(
        splits["region_code"],
        splits["split"]
    ).to_string()
)


# 9. 정답 라벨 분포
print("\n===== Target Distribution by Split =====")

print(
    pd.crosstab(
        splits["split"],
        splits["target"]
    ).to_string()
)


# 10. 최종 무결성 검사
if len(splits) != len(dataset):
    raise ValueError("Dataset row count mismatch.")

if splits["link_id"].duplicated().any():
    raise ValueError("Duplicate link IDs.")

for name, frame in [
    ("train", train),
    ("validation", val),
    ("test", test)
]:

    if set(frame["target"].unique()) != {0, 1}:
        raise ValueError(
            f"Both classes required in {name}."
        )


# 11. 분할 후보 데이터 저장
output_path = (
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

splits.to_parquet(
    output_path,
    index=False
)

print("\n===== Output =====")
print("Saved:", output_path)

print("\n===== Spatial Split Completed =====")