from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/accident_risk/processed"

# 공간 완충구역 설정
BUFFER_DISTANCE = 400

FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch"
]

TARGET = "target"

# 기존 공간 분할 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

# 기존 데이터셋 분리
train = dataset[
    dataset["split"] == "train"
].copy()

val = dataset[
    dataset["split"] == "validation"
].copy()

test = dataset[
    dataset["split"] == "test"
].copy()

print("===== Input Dataset =====")
print("Total Links:", len(dataset))
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print("Buffer Distance (m):", BUFFER_DISTANCE)


# 1. 도로 중심점 좌표 추출
def get_coordinates(frame):

    coordinates = frame[
        ["center_x", "center_y"]
    ].to_numpy()

    if not np.isfinite(coordinates).all():
        raise ValueError("Invalid spatial coordinates.")

    return coordinates


train_coords = get_coordinates(train)
val_coords = get_coordinates(val)
test_coords = get_coordinates(test)


# 2. Train 공간 완충구역 적용
# 기존 Validation과 Test 모두를 기준으로 검사

holdout_coords = np.vstack([
    val_coords,
    test_coords
])

holdout_tree = cKDTree(holdout_coords)

train_distances, _ = holdout_tree.query(
    train_coords,
    k=1
)

train_keep = train_distances > BUFFER_DISTANCE

final_train = train.loc[train_keep].copy()

removed_train = train.loc[~train_keep].copy()


# 3. Validation 공간 완충구역 적용
# Test와 인접한 Validation 도로 제외

test_tree = cKDTree(test_coords)

val_distances, _ = test_tree.query(
    val_coords,
    k=1
)

val_keep = val_distances > BUFFER_DISTANCE

final_val = val.loc[val_keep].copy()

removed_val = val.loc[~val_keep].copy()


# 4. Test 데이터는 그대로 유지
final_test = test.copy()


# 5. 제외된 데이터 집계
print("\n===== Buffer Removal Summary =====")

for name, removed in [
    ("Train", removed_train),
    ("Validation", removed_val)
]:

    print(f"\n[{name}]")
    print("Removed Links:", len(removed))
    print("Removed Positive:", int(removed[TARGET].sum()))
    print(
        "Removed Negative:",
        int((removed[TARGET] == 0).sum())
    )


# 6. 최종 데이터셋 분포 확인
final_frames = {
    "train": final_train,
    "validation": final_val,
    "test": final_test
}

print("\n===== Final Dataset Distribution =====")

for name, frame in final_frames.items():

    positive = int(frame[TARGET].sum())
    negative = len(frame) - positive

    positive_groups = frame.loc[
        frame[TARGET] == 1,
        "spatial_group"
    ].nunique()

    print(f"\n[{name.upper()}]")
    print("Rows:", len(frame))
    print("Negative:", negative)
    print("Positive:", positive)
    print("Positive Groups:", positive_groups)

    # 두 클래스가 모두 존재하는지 검사
    if set(frame[TARGET].unique()) != {0, 1}:

        raise ValueError(
            f"Both classes required in {name}."
        )

    print(
        "Positive Ratio (%):",
        round(positive / len(frame) * 100, 2)
    )

    # 입력 피처 유효성 검사
    if not np.isfinite(
        frame[FEATURES].to_numpy()
    ).all():

        raise ValueError(
            f"Invalid features in {name}."
        )


# 7. 최종 데이터셋 간 최소 거리 검사
print("\n===== Final Spatial Distance Check =====")

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]

for first, second in pairs:

    first_coords = get_coordinates(
        final_frames[first]
    )

    second_coords = get_coordinates(
        final_frames[second]
    )

    tree = cKDTree(second_coords)

    distances, _ = tree.query(
        first_coords,
        k=1
    )

    minimum = float(distances.min())

    print(
        f"{first} -> {second}: "
        f"{minimum:.4f} m"
    )

    if minimum <= BUFFER_DISTANCE:

        raise ValueError(
            "Spatial buffer violation."
        )


# 8. 최종 데이터셋 결합
final_dataset = pd.concat(
    [
        final_train,
        final_val,
        final_test
    ],
    ignore_index=True
)

# 제외 데이터 별도 보존
excluded_dataset = pd.concat(
    [
        removed_train,
        removed_val
    ],
    ignore_index=True
)


# 9. 데이터 무결성 검사
if final_dataset["link_id"].duplicated().any():

    raise ValueError("Duplicate link IDs.")

if excluded_dataset["link_id"].duplicated().any():

    raise ValueError("Duplicate excluded link IDs.")

final_ids = set(final_dataset["link_id"])
excluded_ids = set(excluded_dataset["link_id"])
original_ids = set(dataset["link_id"])

if final_ids & excluded_ids:

    raise ValueError("Excluded link overlap.")

if final_ids | excluded_ids != original_ids:

    raise ValueError("Original link IDs mismatch.")

if (
    len(final_dataset) + len(excluded_dataset)
    != len(dataset)
):

    raise ValueError("Dataset row count mismatch.")

group_split_count = final_dataset.groupby(
    "spatial_group"
)["split"].nunique()

if (group_split_count > 1).any():

    raise ValueError("Spatial group leakage.")


# 10. 학습 데이터 기준 클래스 가중치 계산
train_positive = int(final_train[TARGET].sum())

train_negative = (
    len(final_train) - train_positive
)

scale_pos_weight = (
    train_negative / train_positive
)

print("\n===== Training Class Weight =====")

print("Negative:", train_negative)
print("Positive:", train_positive)

print(
    "Scale Pos Weight:",
    round(scale_pos_weight, 6)
)


# 11. 최종 데이터 저장
output_path = (
    DATA_DIR / "accident_risk_final_dataset.parquet"
)

excluded_path = (
    DATA_DIR / "accident_risk_buffer_excluded.parquet"
)

final_dataset.to_parquet(
    output_path,
    index=False
)

excluded_dataset.to_parquet(
    excluded_path,
    index=False
)

print("\n===== Final Summary =====")
print("Original Links:", len(dataset))
print("Final Links:", len(final_dataset))
print("Excluded Links:", len(excluded_dataset))

print("\n===== Output =====")
print("Final Dataset:", output_path)
print("Excluded Dataset:", excluded_path)

print("\n===== Dataset Finalization Completed =====")