from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RISK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "risk.parquet"
)

LINKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "links.geojson"
)

EXPECTED_LINK_COUNT = 36521

EXPECTED_RISK_COLUMNS = [
    "link_id",
    "risk",
]

EXPECTED_LINK_COLUMNS = [
    "link_id",
    "geometry",
]


print(
    "=== SafeRoute 백엔드 전달 패키지 최종 검사 ==="
)


# ============================================================
# 2. 파일 존재 확인
# ============================================================

print(
    "\n[1] 파일 존재 확인..."
)


if not RISK_PATH.exists():
    raise FileNotFoundError(
        f"risk.parquet 없음: {RISK_PATH}"
    )


if not LINKS_PATH.exists():
    raise FileNotFoundError(
        f"links.geojson 없음: {LINKS_PATH}"
    )


print(
    "risk.parquet:",
    RISK_PATH
)

print(
    "links.geojson:",
    LINKS_PATH
)

print(
    "파일 존재 검사: PASS"
)


# ============================================================
# 3. 파일 불러오기
# ============================================================

print(
    "\n[2] 파일 불러오기..."
)


risk = pd.read_parquet(
    RISK_PATH
)

links = gpd.read_file(
    LINKS_PATH
)


print(
    "Risk LINK:",
    len(risk)
)

print(
    "GeoJSON LINK:",
    len(links)
)


# ============================================================
# 4. 컬럼 규격 검사
# ============================================================

print(
    "\n[3] 컬럼 규격 검사..."
)


print(
    "Risk 컬럼:",
    risk.columns.tolist()
)

print(
    "GeoJSON 컬럼:",
    links.columns.tolist()
)


if risk.columns.tolist() != EXPECTED_RISK_COLUMNS:
    raise RuntimeError(
        "risk.parquet 컬럼 규격이 다릅니다."
    )


if links.columns.tolist() != EXPECTED_LINK_COLUMNS:
    raise RuntimeError(
        "links.geojson 컬럼 규격이 다릅니다."
    )


print(
    "컬럼 규격 검사: PASS"
)


# ============================================================
# 5. link_id 타입 통일
# ============================================================

risk["link_id"] = (
    risk["link_id"]
    .astype(str)
)

links["link_id"] = (
    links["link_id"]
    .astype(str)
)


# ============================================================
# 6. 행 수 검사
# ============================================================

print(
    "\n[4] LINK 수 검사..."
)


print(
    "기대 LINK:",
    EXPECTED_LINK_COUNT
)

print(
    "Risk LINK:",
    len(risk)
)

print(
    "GeoJSON LINK:",
    len(links)
)


if len(risk) != EXPECTED_LINK_COUNT:
    raise RuntimeError(
        "risk.parquet LINK 수 오류"
    )


if len(links) != EXPECTED_LINK_COUNT:
    raise RuntimeError(
        "links.geojson LINK 수 오류"
    )


print(
    "LINK 수 검사: PASS"
)


# ============================================================
# 7. link_id 중복 검사
# ============================================================

print(
    "\n[5] link_id 중복 검사..."
)


risk_duplicates = int(
    risk["link_id"]
    .duplicated()
    .sum()
)

link_duplicates = int(
    links["link_id"]
    .duplicated()
    .sum()
)


print(
    "Risk 중복:",
    risk_duplicates
)

print(
    "GeoJSON 중복:",
    link_duplicates
)


if risk_duplicates != 0:
    raise RuntimeError(
        "risk.parquet link_id 중복"
    )


if link_duplicates != 0:
    raise RuntimeError(
        "links.geojson link_id 중복"
    )


print(
    "중복 검사: PASS"
)


# ============================================================
# 8. link_id 집합 완전 일치 검사
# ============================================================

print(
    "\n[6] LINK 일치 검사..."
)


risk_ids = set(
    risk["link_id"]
)

link_ids = set(
    links["link_id"]
)


missing_geometry = (
    risk_ids
    - link_ids
)

missing_risk = (
    link_ids
    - risk_ids
)


print(
    "Geometry 없는 Risk LINK:",
    len(missing_geometry)
)

print(
    "Risk 없는 Geometry LINK:",
    len(missing_risk)
)


if missing_geometry:
    raise RuntimeError(
        "geometry가 없는 risk LINK 존재"
    )


if missing_risk:
    raise RuntimeError(
        "risk가 없는 geometry LINK 존재"
    )


print(
    "LINK 완전 일치: PASS"
)


# ============================================================
# 9. risk 값 검사
# ============================================================

print(
    "\n[7] Risk 검사..."
)


risk_missing = int(
    risk["risk"]
    .isna()
    .sum()
)

risk_range_error = int(
    (
        (risk["risk"] < 0)
        |
        (risk["risk"] > 1)
    )
    .sum()
)


print(
    "Risk 결측:",
    risk_missing
)

print(
    "Risk 범위 오류:",
    risk_range_error
)


if risk_missing != 0:
    raise RuntimeError(
        "Risk 결측 존재"
    )


if risk_range_error != 0:
    raise RuntimeError(
        "Risk 범위 오류"
    )


print(
    "Risk 검사: PASS"
)


# ============================================================
# 10. geometry 검사
# ============================================================

print(
    "\n[8] Geometry 검사..."
)


geometry_missing = int(
    links.geometry
    .isna()
    .sum()
)

geometry_empty = int(
    links.geometry
    .is_empty
    .sum()
)

geometry_invalid = int(
    (~links.geometry.is_valid)
    .sum()
)


print(
    "Geometry 결측:",
    geometry_missing
)

print(
    "Geometry Empty:",
    geometry_empty
)

print(
    "Geometry Invalid:",
    geometry_invalid
)


if geometry_missing != 0:
    raise RuntimeError(
        "Geometry 결측 존재"
    )


if geometry_empty != 0:
    raise RuntimeError(
        "빈 Geometry 존재"
    )


if geometry_invalid != 0:
    raise RuntimeError(
        "Invalid Geometry 존재"
    )


print(
    "Geometry 검사: PASS"
)


# ============================================================
# 11. Geometry 타입 검사
# ============================================================

print(
    "\n[9] Geometry 타입 검사..."
)


geometry_types = (
    links.geometry
    .geom_type
    .value_counts()
)


print(
    geometry_types.to_string()
)


unexpected_geometry = (
    links.geometry.geom_type
    != "LineString"
)


if unexpected_geometry.any():
    raise RuntimeError(
        "LineString이 아닌 geometry가 존재합니다."
    )


print(
    "Geometry 타입 검사: PASS"
)


# ============================================================
# 12. CRS 검사
# ============================================================

print(
    "\n[10] CRS 검사..."
)


print(
    "CRS:",
    links.crs
)


if links.crs is None:
    raise RuntimeError(
        "CRS 정보 없음"
    )


if links.crs.to_epsg() != 4326:
    raise RuntimeError(
        f"CRS 오류: {links.crs}"
    )


print(
    "CRS EPSG:4326: PASS"
)


# ============================================================
# 13. 두 파일 실제 JOIN 검사
#
# 백엔드에서 link_id 기준으로 연결하는 상황을
# 그대로 재현
# ============================================================

print(
    "\n[11] Risk + Geometry JOIN 검사..."
)


joined = links.merge(
    risk,
    on="link_id",
    how="left",
    validate="one_to_one",
)


print(
    "JOIN 결과:",
    len(joined)
)


join_risk_missing = int(
    joined["risk"]
    .isna()
    .sum()
)


print(
    "JOIN 후 Risk 결측:",
    join_risk_missing
)


if len(joined) != EXPECTED_LINK_COUNT:
    raise RuntimeError(
        "JOIN 후 LINK 수 변경"
    )


if join_risk_missing != 0:
    raise RuntimeError(
        "JOIN 후 Risk 누락"
    )


print(
    "JOIN 검사: PASS"
)


# ============================================================
# 14. Risk 분포 요약
# ============================================================

print(
    "\n[12] Risk 분포 요약..."
)


print(
    risk["risk"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


# ============================================================
# 15. Safety Score 계산 가능 여부 확인
#
# 실제 파일에는 추가하지 않음.
#
# Backend:
# safety_score = 100 * (1 - risk)
# ============================================================

print(
    "\n[13] Safety Score 계산 확인..."
)


safety_score = (
    100.0
    * (
        1.0
        - joined["risk"]
    )
)


print(
    "Safety Score 최소:",
    round(
        float(
            safety_score.min()
        ),
        6,
    )
)

print(
    "Safety Score 최대:",
    round(
        float(
            safety_score.max()
        ),
        6,
    )
)


# ============================================================
# 16. Routing Cost 계산식 안내
#
# 실제 경로 탐색은 Backend 담당.
# 여기서는 인터페이스 확인만 수행.
#
# cost =
# length * (1 + lambda * risk) * option_multiplier
# ============================================================

print(
    "\n[14] Backend Routing 인터페이스 확인..."
)


print(
    "기본 Routing Cost:"
)

print(
    "length * (1 + lambda * risk) * option_multiplier"
)


print(
    "\nLambda 후보:"
)

print(
    "0.5 / 1.0 / 2.0"
)


print(
    "\nOption Multiplier:"
)

print(
    "고속도로 우선 : 0.7"
)

print(
    "큰길 우선     : 0.8"
)

print(
    "좁은길        : 1.3"
)


# ============================================================
# 17. 최종 파일 크기 확인
# ============================================================

print(
    "\n[15] 파일 크기..."
)


risk_size_mb = (
    RISK_PATH.stat().st_size
    / 1024
    / 1024
)

links_size_mb = (
    LINKS_PATH.stat().st_size
    / 1024
    / 1024
)


print(
    f"risk.parquet : {risk_size_mb:.2f} MB"
)

print(
    f"links.geojson: {links_size_mb:.2f} MB"
)


# ============================================================
# 18. 최종 결과
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "=== SafeRoute Backend Package 최종 검사 완료 ==="
)

print(
    "=" * 70
)


print(
    "\n전달 파일:"
)

print(
    "1. outputs/risk.parquet"
)

print(
    "2. data/processed/links.geojson"
)


print(
    "\n공통 Key:"
)

print(
    "link_id"
)


print(
    "\nLINK 수:"
)

print(
    EXPECTED_LINK_COUNT
)


print(
    "\nRisk 의미:"
)

print(
    "0~1 사고다발 조건 모델 위험 점수"
)

print(
    "※ 실제 사고 발생 확률이 아님"
)


print(
    "\nSafety Score:"
)

print(
    "100 * (1 - risk)"
)


print(
    "\nRouting Cost:"
)

print(
    "length * (1 + lambda * risk) * option_multiplier"
)


print(
    "\n최종 검사 결과:"
)

print(
    "파일 존재             PASS"
)

print(
    "컬럼 규격             PASS"
)

print(
    "LINK 수               PASS"
)

print(
    "link_id 중복          PASS"
)

print(
    "LINK 완전 일치        PASS"
)

print(
    "Risk                  PASS"
)

print(
    "Geometry              PASS"
)

print(
    "CRS                   PASS"
)

print(
    "Risk + Geometry JOIN  PASS"
)


print(
    "\n=== 백엔드 전달 준비 완료 ==="
)