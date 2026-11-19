from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd


# Windows 환경에서도 표준 입출력을 UTF-8로 처리
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "route_lightgbm.joblib"
FEATURE_PATH = PROJECT_ROOT / "training" / "processed" / "features.json"


def predict_routes(routes):
    # 학습된 모델 및 Feature 목록 불러오기
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)

    # 입력 데이터 확인
    if not isinstance(routes, list) or not routes:
        raise ValueError("routes는 비어 있지 않은 배열이어야 합니다.")

    df = pd.DataFrame(routes)

    # 필수 Feature 검사
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"누락된 Feature: {missing}")

    # 숫자형 데이터로 변환
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 결측치 및 무한대 검사
    X = df[features].replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError("Feature에 숫자가 아닌 값 또는 결측치가 있습니다.")

    # AI 부담 점수 예측
    predictions = model.predict(X)
    if not np.isfinite(predictions).all():
        raise ValueError('모델 예측값에 비정상 숫자가 있습니다.')

    results = []

    for i, prediction in enumerate(predictions):
        burden = float(prediction)

        results.append({
            "route_id": str(routes[i].get("route_id", i)),
            "predicted_burden": round(burden, 4),
            "easy_score": round(
                float(np.clip(100 - burden, 0, 100)), 4
            ),
        })

    return results


def main():
    try:
        # 표준 입력으로 JSON 받기
        request = json.load(sys.stdin)

        routes = request.get("routes")
        results = predict_routes(routes)

        # Node.js에서 읽을 수 있도록 JSON 출력
        print(json.dumps(
            {"success": True, "results": results},
            ensure_ascii=False
        ))

    except Exception as e:
        print(json.dumps(
            {"success": False, "error": str(e)},
            ensure_ascii=False
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()