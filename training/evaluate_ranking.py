from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "training"
    / "processed"
    / "test_predictions.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    # 동일 출발지·도착지 조합별 평가
    grouped = df.groupby(
        ["origin_id", "destination_id"]
    )

    total_groups = 0
    matched_groups = 0
    results = []

    for (origin, destination), group in grouped:

        # 후보 경로가 2개 이상인 조합만 평가
        if len(group) < 2:
            continue

        total_groups += 1

        # 기존 규칙 기반 최저 부담 경로
        rule_best = group.loc[
            group["score"].idxmin()
        ]

        # AI 예측 최저 부담 경로
        ai_best = group.loc[
            group["predicted_burden"].idxmin()
        ]

        is_match = (
            rule_best["route_id"] == ai_best["route_id"]
        )

        if is_match:
            matched_groups += 1

        # AI 추천 경로의 기존 규칙 점수가
        # 최적 규칙 점수보다 얼마나 높은지
        regret = (
            ai_best["score"] - rule_best["score"]
        )

        results.append({
            "origin_id": origin,
            "destination_id": destination,
            "candidate_count": len(group),
            "rule_best_route": rule_best["route_id"],
            "ai_best_route": ai_best["route_id"],
            "same_recommendation": is_match,
            "rule_score_regret": regret,
        })

    if total_groups == 0:
        print("비교 가능한 경로 조합이 없습니다.")
        return

    result_df = pd.DataFrame(results)

    agreement = matched_groups / total_groups * 100

    print("===== AI 경로 추천 순위 평가 =====")
    print("비교 가능한 OD 조합:", total_groups)
    print("동일 추천 조합:", matched_groups)
    print("추천 일치율:", round(agreement, 2), "%")
    print(
        "평균 규칙 점수 손실:",
        round(result_df["rule_score_regret"].mean(), 4)
    )
    print(
        "최대 규칙 점수 손실:",
        round(result_df["rule_score_regret"].max(), 4)
    )

    output_path = (
        PROJECT_ROOT
        / "training"
        / "processed"
        / "ranking_evaluation.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n평가 결과 저장:", output_path)
    print("추천 순위 평가 완료!")


if __name__ == "__main__":
    main()