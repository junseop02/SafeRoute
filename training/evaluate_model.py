from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "training" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "route_lightgbm.joblib"


def main():
    # 1. Test 데이터 불러오기
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    with open(
        DATA_DIR / "features.json", "r", encoding="utf-8"
    ) as f:
        features = json.load(f)

    # 2. 학습된 모델 불러오기
    model = joblib.load(MODEL_PATH)

    X_test = test[features]
    y_test = test["score"]

    # 3. Test 데이터 예측
    predictions = model.predict(X_test)

    # 4. 평가 지표 계산
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("===== Test 평가 결과 =====")
    print("Test 경로 수:", len(test))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

    # 5. 경로별 예측 결과 저장
    result = test[
        [
            "route_id",
            "origin_id",
            "destination_id",
            "score",
        ]
    ].copy()

    result["predicted_burden"] = predictions

    # 화면 표시용 점수: 높을수록 쉬움
    result["easy_score"] = np.clip(
        100 - predictions, 0, 100
    )

    result["absolute_error"] = np.abs(
        result["score"] - predictions
    )

    output_path = DATA_DIR / "test_predictions.csv"

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 예측 결과 예시 =====")
    print(result.head(10).to_string(index=False))

    print("\n결과 저장:", output_path)
    print("Test 평가 완료!")


if __name__ == "__main__":
    main()