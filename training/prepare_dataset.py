from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# 1. 프로젝트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "routes_scored.json"
OUTPUT_DIR = PROJECT_ROOT / "training" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. 모델 입력 Feature
FEATURES = [
    "distance_m",
    "duration_s",
    "road_count",
    "guide_count",
    "n_intersect_200m",
    "wide_road_ratio",
    "near_hotspot_ratio",
    "coverage",
    "accident_burden",
    "narrow_burden",
    "intersection_burden",
]

TARGET = "score"


def main():
    # 3. JSON 불러오기
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        routes = json.load(f)

    df = pd.DataFrame(routes)

    print("전체 경로 수:", len(df))

    # 4. 필수 컬럼 확인
    required = FEATURES + [
        TARGET, "route_id", "origin_id", "destination_id"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"누락된 컬럼: {missing}")

    # 5. 중복 경로 ID 확인
    duplicates = df["route_id"].duplicated().sum()
    print("중복 route_id:", duplicates)

    if duplicates:
        raise ValueError("중복 route_id가 존재합니다.")

    # 6. 수치형 변환
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7. 매칭률 80% 이상 필터링
    df = df[df["coverage"] >= 0.8].copy()

    # 8. 결측치 및 무한대 제거
    df[FEATURES + [TARGET]] = df[
        FEATURES + [TARGET]
    ].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=FEATURES + [TARGET])

    print("필터링 후 경로 수:", len(df))

    # 9. 출발지-도착지 조합 그룹 생성
    df["od_group"] = (
        df["origin_id"].astype(str)
        + "__"
        + df["destination_id"].astype(str)
    )

    # 10. 동일 OD 조합이 서로 다른 세트에
    # 포함되지 않도록 그룹 단위 분할
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_val_idx, test_idx = next(
        outer_split.split(
            df, groups=df["od_group"]
        )
    )

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, val_idx = next(
        inner_split.split(
            train_val, groups=train_val["od_group"]
        )
    )

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    # 11. 그룹 분리 검증
    train_groups = set(train["od_group"])
    val_groups = set(val["od_group"])
    test_groups = set(test["od_group"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # 12. 데이터 저장
    train.to_parquet(
        OUTPUT_DIR / "train.parquet", index=False
    )
    val.to_parquet(
        OUTPUT_DIR / "val.parquet", index=False
    )
    test.to_parquet(
        OUTPUT_DIR / "test.parquet", index=False
    )

    with open(
        OUTPUT_DIR / "features.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(FEATURES, f, indent=2)

    print("\n===== 데이터 분할 결과 =====")
    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))
    print("전체:", len(train) + len(val) + len(test))

    print("\n정답 점수 통계:")
    print(df[TARGET].describe())

    print("\n학습 데이터 생성 완료!")


if __name__ == "__main__":
    main()