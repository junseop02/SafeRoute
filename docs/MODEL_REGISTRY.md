# SafeRoute Model Registry

기준일: 2026-11-26

## 1. 모델 관리 목적

기존 모델과 신규 모델을 구분하고,
학습 및 서비스 재현에 필요한 모델 파일을 보존한다.

기존 코드의 경로 의존성을 유지하기 위해
모델 파일의 물리적 이동은 수행하지 않는다.

## 2. 기존 경로 모델

- 파일: models/route_lightgbm.joblib
- 용도: 기존 경로 추천 서비스 및 비교 실험
- 상태: Legacy
- 백업: backups/models/route_lightgbm.joblib
- 검증: 기존 모델과 백업의 SHA256 일치

## 3. 천안·아산 사고다발지역 분류 모델

- models/accident_risk_lightgbm_model_a.joblib
- models/accident_risk_lightgbm_model_b.joblib
- models/accident_risk_lightgbm_final.joblib

용도: 지역 단위 사고다발지역 분류 연구
상태: Research / Not deployed

## 4. 전국 사고다발지역 실험 모델

경로: models/national_accident_risk/experiments/

- L01.joblib
- L02.joblib
- L03.joblib
- L04.joblib
- L05.joblib
- L06.joblib

용도: 전국 모델 피처 및 학습 방식 비교
상태: Research / Not deployed

## 5. 최종 전국 모델 T09

경로: models/national_accident_risk/

- national_accident_risk_lightgbm_final.joblib
- national_accident_risk_lightgbm_final.txt

알고리즘: LightGBM Binary Classification
모델 ID: T09
추천 정책: P01-v1.1
트리 개수: 269
입력 피처 개수: 7

### 입력 피처 순서

1. lanes
2. max_spd
3. road_rank
4. length
5. n_branch
6. n_intersect_200m
7. intersection_nodes_200m

### 독립 시험 성능

- AP: 0.38647249373135445
- ROC-AUC: 0.8445174908240203
- F1-score: 0.46030408330543404

### 모델 검증

- Joblib 로드 성공
- Native LightGBM 모델 로드 성공
- 입력 피처 및 순서 일치
- 트리 개수 일치
- 트리 구조 일치
- 이진분류 목적함수 확인
- 자동 검증: 9/9 PASS

### 서비스 적용 방식

T09 모델로 전국 도로 링크 점수를 사전 계산한다.
실시간 서비스는 생성된 도로 점수 타일을 읽고
P01-v1.1 경로 추천 정책을 적용한다.
최소 비교 가능 경로는 2개이며 탐색 목표는 3개이다.

상태: Local service integration verified

로컬 신규 서비스 전환은 완료되었다.
팀 인터페이스 승인 및 독립 배포 검증은 미완료 상태이다.

## 6. 모델 해석 시 주의사항

T09 모델은 제공된 과거 사고다발지역에
포함되는 도로 링크의 특성을 학습한다.

모델 점수는 실제 사고 발생 확률이나
초보 운전자의 체감 난이도를 의미하지 않는다.

P01은 모델과 분리된 서비스 추천 정책이다.

## 7. 파일 무결성 관리

전체 모델 파일의 SHA256 기준값은 다음 문서에 기록한다.

docs/MODEL_SHA256SUMS.txt

프로젝트 루트에서 다음 명령으로 검증한다.

sha256sum -c docs/MODEL_SHA256SUMS.txt

## 8. 배포 및 보존 정책

- 기존 모델 및 실험 모델을 보존한다.
- 최종 T09 모델을 임의로 재학습하거나 덮어쓰지 않는다.
- 모델 파일은 일반 GitHub 코드 공유 대상에서 제외한다.
- 연구용 프로젝트에는 학습 모델과 데이터 자산을 보존한다.
- 실행용 배포 패키지는 사전 계산된 도로 점수 타일을 사용한다.
- 모델 변경 시 성능 기록과 해시 기준값을 함께 갱신한다.
