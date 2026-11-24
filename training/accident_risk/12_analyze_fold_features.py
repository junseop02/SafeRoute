from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/accident_risk/processed"
RESULT_DIR = DATA_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 실험 설정
RANDOM_STATE = 42
N_SPLITS = 3
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

# 도로 등급은 범주형이므로 단일 수치 AUC에서 제외
NUMERIC_FEATURES = [
    feature for feature in FEATURES
    if feature != "road_rank"
]

# 1. 공간 분할 데이터 불러오기
dataset = pd.read_parquet(
    DATA_DIR / "accident_risk_spatial_split.parquet"
)

development = dataset[
    dataset["split"] != "test"
].copy()

test = dataset[
    dataset["split"] == "test"
].copy()


# 2. 도로 중심점 좌표 추출
def get_coordinates(frame):

    return frame[
        ["center_x", "center_y"]
    ].to_numpy()


# 3. 기존 Test 주변 개발 데이터 제외
test_tree = cKDTree(
    get_coordinates(test)
)

distances, _ = test_tree.query(
    get_coordinates(development),
    k=1
)

development = development.loc[
    distances > BUFFER_DISTANCE
].reset_index(drop=True)

print("===== Development Dataset =====")
print("Links:", len(development))
print("Positive:", int(development[TARGET].sum()))
print("Groups:", development["spatial_group"].nunique())


# 4. 기존 교차검증 분할 재현
splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

folds = list(
    splitter.split(
        development[FEATURES],
        development[TARGET],
        groups=development["spatial_group"]
    )
)

feature_results = []
group_results = []
region_results = []
rank_results = []


# 5. Fold별 데이터 분석
for fold_number, (train_idx, val_idx) in enumerate(
    folds,
    start=1
):

    print("\n" + "=" * 60)
    print("FOLD:", fold_number)
    print("=" * 60)

    fold_train = development.iloc[
        train_idx
    ].copy()

    fold_val = development.iloc[
        val_idx
    ].copy()

    # 기존 교차검증과 동일한 완충구역 적용
    val_tree = cKDTree(
        get_coordinates(fold_val)
    )

    distances, _ = val_tree.query(
        get_coordinates(fold_train),
        k=1
    )

    fold_train = fold_train.loc[
        distances > BUFFER_DISTANCE
    ].copy()

    print("Train Links:", len(fold_train))
    print("Validation Links:", len(fold_val))

    # Fold별 학습·검증 데이터 분석
    for split_name, frame in [
        ("train", fold_train),
        ("validation", fold_val)
    ]:

        print(f"\n===== {split_name.upper()} =====")

        if set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError(
                f"Both classes required: "
                f"Fold {fold_number}, {split_name}"
            )

        # 6. 지역별 라벨 분포
        region_stats = frame.groupby(
            "region_code"
        ).agg(
            total=(TARGET, "size"),
            positive=(TARGET, "sum")
        ).reset_index()

        region_stats["negative"] = (
            region_stats["total"]
            - region_stats["positive"]
        )

        region_stats["positive_ratio_pct"] = (
            region_stats["positive"]
            / region_stats["total"] * 100
        )

        region_stats["fold"] = fold_number
        region_stats["split"] = split_name

        region_results.append(region_stats)

        print("\n--- Regional Distribution ---")
        print(
            region_stats.round(4).to_string(
                index=False
            )
        )

        # 7. 양성 공간 그룹 상세 분석
        group_stats = frame.groupby(
            "spatial_group"
        ).agg(
            total=(TARGET, "size"),
            positive=(TARGET, "sum")
        ).reset_index()

        group_stats = group_stats[
            group_stats["positive"] > 0
        ].copy()

        group_stats["positive_share_pct"] = (
            group_stats["positive"]
            / frame[TARGET].sum() * 100
        )

        group_stats["positive_ratio_pct"] = (
            group_stats["positive"]
            / group_stats["total"] * 100
        )

        group_stats["fold"] = fold_number
        group_stats["split"] = split_name

        group_results.append(group_stats)

        print("\n--- Positive Spatial Groups ---")
        print(
            group_stats.sort_values(
                "positive",
                ascending=False
            ).round(4).to_string(index=False)
        )

        # 8. 수치형 피처별 클래스 분포 및 AUC
        print("\n--- Numeric Feature Analysis ---")

        negative = frame[frame[TARGET] == 0]
        positive = frame[frame[TARGET] == 1]

        local_results = []

        for feature in NUMERIC_FEATURES:

            auc = roc_auc_score(
                frame[TARGET],
                frame[feature]
            )

            result = {
                "fold": fold_number,
                "split": split_name,
                "feature": feature,
                "negative_mean": negative[feature].mean(),
                "positive_mean": positive[feature].mean(),
                "negative_median": negative[feature].median(),
                "positive_median": positive[feature].median(),
                "raw_feature_auc": auc
            }

            feature_results.append(result)
            local_results.append(result)

        print(
            pd.DataFrame(local_results)
            .round(6)
            .to_string(index=False)
        )

        # 9. 주요 피처 분위수 비교
        for feature in [
            "curvature",
            "n_intersect_200m"
        ]:

            print(f"\n--- {feature} Quantiles ---")

            quantiles = frame.groupby(
                TARGET
            )[feature].quantile(
                [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
            ).unstack()

            print(quantiles.round(6).to_string())

        # 10. 도로 등급별 양성 비율
        rank_stats = frame.groupby(
            "road_rank"
        ).agg(
            total=(TARGET, "size"),
            positive=(TARGET, "sum")
        ).reset_index()

        rank_stats["positive_ratio_pct"] = (
            rank_stats["positive"]
            / rank_stats["total"] * 100
        )

        rank_stats["fold"] = fold_number
        rank_stats["split"] = split_name

        rank_results.append(rank_stats)

        print("\n--- Road Rank Distribution ---")

        print(
            rank_stats.round(4).to_string(
                index=False
            )
        )


# 11. 피처별 AUC 비교
feature_df = pd.DataFrame(feature_results)

auc_comparison = feature_df.pivot(
    index="feature",
    columns=["fold", "split"],
    values="raw_feature_auc"
)

print("\n===== Raw Feature AUC Comparison =====")
print(auc_comparison.round(6).to_string())


# 12. 학습·검증 AUC 방향 차이 검사
direction_results = []

for fold_number in range(1, N_SPLITS + 1):

    fold_features = feature_df[
        feature_df["fold"] == fold_number
    ]

    for feature in NUMERIC_FEATURES:

        feature_rows = fold_features[
            fold_features["feature"] == feature
        ]

        train_auc = feature_rows.loc[
            feature_rows["split"] == "train",
            "raw_feature_auc"
        ].iloc[0]

        val_auc = feature_rows.loc[
            feature_rows["split"] == "validation",
            "raw_feature_auc"
        ].iloc[0]

        reversed_direction = (
            (train_auc - 0.5)
            * (val_auc - 0.5) < 0
        )

        direction_results.append({
            "fold": fold_number,
            "feature": feature,
            "train_auc": train_auc,
            "validation_auc": val_auc,
            "auc_difference": val_auc - train_auc,
            "direction_reversed": reversed_direction
        })

direction_df = pd.DataFrame(direction_results)

print("\n===== Feature Direction Comparison =====")

print(
    direction_df.round(6).to_string(
        index=False
    )
)

print("\n===== Fold 3 Feature Direction =====")

print(
    direction_df[
        direction_df["fold"] == 3
    ].round(6).to_string(index=False)
)


# 13. 분석 결과 저장
outputs = {
    "fold_feature_statistics.csv": feature_df,
    "fold_positive_groups.csv": pd.concat(
        group_results,
        ignore_index=True
    ),
    "fold_region_statistics.csv": pd.concat(
        region_results,
        ignore_index=True
    ),
    "fold_road_rank_statistics.csv": pd.concat(
        rank_results,
        ignore_index=True
    ),
    "fold_feature_direction.csv": direction_df
}

print("\n===== Output =====")

for filename, frame in outputs.items():

    path = RESULT_DIR / filename

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", path)

print("\n===== Fold Feature Analysis Completed =====")
print("Reserved test performance was not evaluated.")