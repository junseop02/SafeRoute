from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "training" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. 학습 및 검증 데이터 불러오기
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")

    with open(
        DATA_DIR / "features.json", "r", encoding="utf-8"
    ) as f:
        features = json.load(f)

    target = "score"

    X_train = train[features]
    y_train = train[target]

    X_val = val[features]
    y_val = val[target]

    print("===== 학습 데이터 =====")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    # 2. LightGBM 회귀 모델 생성
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=6,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    # 3. 모델 학습
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False,
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    # 4. 검증 데이터 예측
    pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    print("\n===== Validation 결과 =====")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))
    print("Best iteration:", model.best_iteration_)

    # 5. Feature 중요도 출력
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance", ascending=False
    )

    print("\n===== Feature 중요도 =====")
    print(importance.to_string(index=False))

    # 6. 학습 모델 저장
    model_path = MODEL_DIR / "route_lightgbm.joblib"
    joblib.dump(model, model_path)

    print("\n모델 저장:", model_path)
    print("학습 완료!")


if __name__ == "__main__":
    main()