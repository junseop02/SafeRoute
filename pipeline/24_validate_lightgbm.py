from pathlib import Path

import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 경로 / 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_cheonan_asan.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
LABEL_COLUMN = "label"


ALL_FEATURES = [
    "region_code",
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch",
    "traffic",
    "traffic_is_proxy",
]


NO_REGION_FEATURES = [
    "lanes",
    "max_spd",
    "road_rank",
    "length",
    "curvature",
    "n_intersect_200m",
    "n_branch",
    "traffic",
    "traffic_is_proxy",
]


REGION_NAMES = {
    44131: "Cheonan Dongnam",
    44133: "Cheonan Seobuk",
    44200: "Asan",
}


print(
    "=== SafeRoute LightGBM 추가 검증 ==="
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print(
    "\n[1] 데이터 불러오기..."
)

df = pd.read_parquet(
    DATA_PATH
)


train_df = (
    df[
        df[LABEL_COLUMN].notna()
    ]
    .copy()
    .reset_index(drop=True)
)


train_df[LABEL_COLUMN] = (
    train_df[LABEL_COLUMN]
    .astype(int)
)


train_df["traffic_is_proxy"] = (
    train_df["traffic_is_proxy"]
    .astype(int)
)


print(
    "전체 LINK:",
    len(df)
)

print(
    "학습 가능 LINK:",
    len(train_df)
)


# ============================================================
# 3. Train / Test index 생성
#
# 22, 23번과 동일한 조건
# ============================================================

print(
    "\n[2] 동일 Train/Test 분할 생성..."
)


indices = train_df.index

y = train_df[
    LABEL_COLUMN
]


train_idx, test_idx = (
    train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)


print(
    "Train:",
    len(train_idx)
)

print(
    "Test:",
    len(test_idx)
)


# ============================================================
# 4. 평가 함수
# ============================================================

def calculate_metrics(
    y_true,
    pred,
    prob,
):

    accuracy = accuracy_score(
        y_true,
        pred,
    )

    precision = precision_score(
        y_true,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        pred,
        zero_division=0,
    )


    # 한 클래스만 존재할 경우
    # ROC-AUC 계산 불가능
    if len(
        pd.Series(
            y_true
        ).unique()
    ) >= 2:

        roc_auc = roc_auc_score(
            y_true,
            prob,
        )

        pr_auc = average_precision_score(
            y_true,
            prob,
        )

    else:

        roc_auc = float("nan")
        pr_auc = float("nan")


    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 5. LightGBM 생성 함수
# ============================================================

def build_model(
    y_train,
):

    negative_count = int(
        (
            y_train == 0
        ).sum()
    )

    positive_count = int(
        (
            y_train == 1
        ).sum()
    )


    scale_pos_weight = (
        negative_count
        / positive_count
    )


    model = lgb.LGBMClassifier(
        objective="binary",

        n_estimators=500,

        learning_rate=0.05,

        num_leaves=31,

        max_depth=-1,

        min_child_samples=20,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.0,

        reg_lambda=0.0,

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1,
    )


    return model


# ============================================================
# 6. 모델 학습 / 평가 함수
# ============================================================

def train_and_evaluate(
    model_name,
    features,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        model_name
    )

    print(
        "=" * 70
    )


    X_train = train_df.loc[
        train_idx,
        features,
    ]

    X_test = train_df.loc[
        test_idx,
        features,
    ]

    y_train = train_df.loc[
        train_idx,
        LABEL_COLUMN,
    ]

    y_test = train_df.loc[
        test_idx,
        LABEL_COLUMN,
    ]


    model = build_model(
        y_train
    )


    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Train 성능
    # --------------------------------------------------------

    train_pred = model.predict(
        X_train
    )

    train_prob = model.predict_proba(
        X_train
    )[:, 1]


    train_metrics = calculate_metrics(
        y_train,
        train_pred,
        train_prob,
    )


    # --------------------------------------------------------
    # Test 성능
    # --------------------------------------------------------

    test_pred = model.predict(
        X_test
    )

    test_prob = model.predict_proba(
        X_test
    )[:, 1]


    test_metrics = calculate_metrics(
        y_test,
        test_pred,
        test_prob,
    )


    print(
        "\nTrain 성능"
    )

    print(
        f"Accuracy : "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{train_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{train_metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{train_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{train_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{train_metrics['pr_auc']:.4f}"
    )


    print(
        "\nTest 성능"
    )

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{test_metrics['pr_auc']:.4f}"
    )


    print(
        "\nTrain → Test 차이"
    )

    print(
        f"ROC-AUC gap: "
        f"{train_metrics['roc_auc'] - test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC gap : "
        f"{train_metrics['pr_auc'] - test_metrics['pr_auc']:.4f}"
    )

    print(
        f"F1 gap     : "
        f"{train_metrics['f1'] - test_metrics['f1']:.4f}"
    )


    return (
        model,
        X_test,
        y_test,
        test_pred,
        test_prob,
        train_metrics,
        test_metrics,
    )


# ============================================================
# 7. region_code 포함 모델
# ============================================================

print(
    "\n[3] region_code 포함 모델 검증..."
)


(
    model_all,
    X_test_all,
    y_test_all,
    pred_all,
    prob_all,
    train_metrics_all,
    test_metrics_all,
) = train_and_evaluate(
    model_name="LightGBM - region_code 포함",
    features=ALL_FEATURES,
)


# ============================================================
# 8. region_code 제외 모델
# ============================================================

print(
    "\n[4] region_code 제외 모델 검증..."
)


(
    model_no_region,
    X_test_no_region,
    y_test_no_region,
    pred_no_region,
    prob_no_region,
    train_metrics_no_region,
    test_metrics_no_region,
) = train_and_evaluate(
    model_name="LightGBM - region_code 제외",
    features=NO_REGION_FEATURES,
)


# ============================================================
# 9. 포함 / 제외 비교
# ============================================================

print(
    "\n[5] region_code 포함/제외 비교..."
)


comparison = pd.DataFrame(
    [
        {
            "model": "LightGBM_with_region",

            **test_metrics_all,
        },

        {
            "model": "LightGBM_without_region",

            **test_metrics_no_region,
        },
    ]
)


comparison_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


print(
    comparison[
        comparison_columns
    ].to_string(
        index=False
    )
)


comparison_path = (
    OUTPUT_DIR
    / "lightgbm_region_comparison.csv"
)


comparison.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. 지역별 Test 성능
#
# region_code 포함 모델 기준
# ============================================================

print(
    "\n[6] 지역별 Test 성능..."
)


test_result_df = pd.DataFrame(
    {
        "index": test_idx,

        "region_code": train_df.loc[
            test_idx,
            "region_code",
        ].values,

        "label": y_test_all.values,

        "pred": pred_all,

        "prob": prob_all,
    }
)


region_results = []


for region_code in sorted(
    test_result_df[
        "region_code"
    ].unique()
):

    region_data = (
        test_result_df[
            test_result_df[
                "region_code"
            ] == region_code
        ]
        .copy()
    )


    region_metrics = calculate_metrics(
        region_data["label"],
        region_data["pred"],
        region_data["prob"],
    )


    positive_rate = (
        region_data[
            "label"
        ].mean()
    )


    region_name = REGION_NAMES.get(
        int(region_code),
        str(region_code),
    )


    print(
        "\n"
        + "-" * 60
    )

    print(
        f"{region_name} ({region_code})"
    )

    print(
        "-" * 60
    )

    print(
        "Test LINK:",
        len(region_data)
    )

    print(
        f"Positive rate: "
        f"{positive_rate:.4f}"
    )

    print(
        f"Accuracy : "
        f"{region_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{region_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{region_metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{region_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{region_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{region_metrics['pr_auc']:.4f}"
    )


    region_results.append(
        {
            "region_code": int(
                region_code
            ),

            "region_name": region_name,

            "test_count": len(
                region_data
            ),

            "positive_rate": (
                positive_rate
            ),

            **region_metrics,
        }
    )


region_results_df = pd.DataFrame(
    region_results
)


region_path = (
    OUTPUT_DIR
    / "lightgbm_region_metrics.csv"
)


region_results_df.to_csv(
    region_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 11. Leave-One-Region-Out 검증
#
# 한 지역 전체를 Test로 두고
# 나머지 두 지역만으로 학습
#
# Random split보다 훨씬 강한
# 지역 일반화 검사
# ============================================================

print(
    "\n[7] Leave-One-Region-Out 검증..."
)


loro_results = []


for test_region in sorted(
    train_df[
        "region_code"
    ].unique()
):

    region_name = REGION_NAMES.get(
        int(test_region),
        str(test_region),
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Test 지역: "
        f"{region_name} ({test_region})"
    )

    print(
        "=" * 70
    )


    region_train_df = (
        train_df[
            train_df[
                "region_code"
            ] != test_region
        ]
        .copy()
    )


    region_test_df = (
        train_df[
            train_df[
                "region_code"
            ] == test_region
        ]
        .copy()
    )


    # 지역 일반화 검증에서는
    # region_code 자체를 Feature에서 제외
    X_region_train = region_train_df[
        NO_REGION_FEATURES
    ]

    y_region_train = region_train_df[
        LABEL_COLUMN
    ]


    X_region_test = region_test_df[
        NO_REGION_FEATURES
    ]

    y_region_test = region_test_df[
        LABEL_COLUMN
    ]


    print(
        "Train:",
        len(
            region_train_df
        )
    )

    print(
        "Test:",
        len(
            region_test_df
        )
    )

    print(
        "Train positive rate:",
        round(
            y_region_train.mean(),
            4,
        )
    )

    print(
        "Test positive rate:",
        round(
            y_region_test.mean(),
            4,
        )
    )


    region_model = build_model(
        y_region_train
    )


    region_model.fit(
        X_region_train,
        y_region_train,
    )


    region_pred = (
        region_model.predict(
            X_region_test
        )
    )

    region_prob = (
        region_model.predict_proba(
            X_region_test
        )[:, 1]
    )


    region_metrics = calculate_metrics(
        y_region_test,
        region_pred,
        region_prob,
    )


    print(
        f"\nAccuracy : "
        f"{region_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{region_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{region_metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{region_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{region_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{region_metrics['pr_auc']:.4f}"
    )


    loro_results.append(
        {
            "test_region": int(
                test_region
            ),

            "region_name": (
                region_name
            ),

            "train_count": len(
                region_train_df
            ),

            "test_count": len(
                region_test_df
            ),

            "train_positive_rate": (
                y_region_train.mean()
            ),

            "test_positive_rate": (
                y_region_test.mean()
            ),

            **region_metrics,
        }
    )


loro_df = pd.DataFrame(
    loro_results
)


loro_path = (
    OUTPUT_DIR
    / "lightgbm_leave_one_region_out.csv"
)


loro_df.to_csv(
    loro_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. 최종 요약
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== 추가 검증 최종 요약 ==="
)

print(
    "=" * 70
)


print(
    "\n[Random Split]"
)


print(
    "\nregion_code 포함:"
)

print(
    f"ROC-AUC = "
    f"{test_metrics_all['roc_auc']:.4f}"
)

print(
    f"PR-AUC  = "
    f"{test_metrics_all['pr_auc']:.4f}"
)

print(
    f"F1      = "
    f"{test_metrics_all['f1']:.4f}"
)


print(
    "\nregion_code 제외:"
)

print(
    f"ROC-AUC = "
    f"{test_metrics_no_region['roc_auc']:.4f}"
)

print(
    f"PR-AUC  = "
    f"{test_metrics_no_region['pr_auc']:.4f}"
)

print(
    f"F1      = "
    f"{test_metrics_no_region['f1']:.4f}"
)


print(
    "\n[지역별 Random Test]"
)

print(
    region_results_df[
        [
            "region_name",
            "test_count",
            "positive_rate",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n[Leave-One-Region-Out]"
)

print(
    loro_df[
        [
            "region_name",
            "test_count",
            "test_positive_rate",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n저장 완료:"
)

print(
    "- outputs/lightgbm_region_comparison.csv"
)

print(
    "- outputs/lightgbm_region_metrics.csv"
)

print(
    "- outputs/lightgbm_leave_one_region_out.csv"
)

print(
    "\n=== 완료 ==="
)