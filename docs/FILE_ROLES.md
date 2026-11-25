# SafeRoute 프로젝트 파일 구성

## 1. 신규 웹 서비스 실행 파일

현재 서비스는 T09 LightGBM 모델과 P01-v1.1 추천 정책을 사용합니다.

| 파일 | 역할 |
|---|---|
| `START_SAFEROUTE.cmd` | 신규 서비스 실행 및 브라우저 연결 |
| `service/index.html` | 웹 서비스 화면 구조 |
| `service/style.css` | 화면 디자인 및 스타일 |
| `service/app.js` | 사용자 입력, 지도 연동, 경로 조회 및 결과 표시 |
| `service/server.js` | Node.js 웹 서버 및 API 요청 처리 |
| `service/kakao.js` | 카카오 API 연동, 경로 탐색, 중복 제거 및 추천 |
| `service/matcher.js` | 도로 링크 매칭, 모델 점수 집계 및 P01 점수 계산 |
| `service/START_SAFEROUTE.cmd` | 서비스 폴더 내부 실행 스크립트 |
| `runtime/node.exe` | Windows용 Node.js 실행 파일 |

### 1.1 신규 서비스 데이터

| 파일 및 폴더 | 역할 |
|---|---|
| `service/data/tiles/` | 전국 도로 링크 및 모델 점수를 저장한 1,172개 압축 공간 타일 |
| `service/data/national_manifest.json` | 전국 공간 타일의 구성 및 무결성 정보 |
| `service/data/national_route_policy_P01_v1_1.json` | 현재 적용 중인 경로 추천 정책 |
| `service/data/national_service_interface_P01_v1_1.json` | 현재 적용 중인 서비스 인터페이스 정의 |
| `service/data/national_route_policy_P01.json` | 이전 P01-v1.0 정책 기록 |
| `service/data/national_service_interface_P01.json` | 이전 P01-v1.0 인터페이스 기록 |

신규 서비스는 사전에 계산된 전국 도로 링크 1,562,356개의 모델 점수를 사용합니다.

서비스 실행 시 Python 모델을 직접 호출하지 않으며, 경로와 매칭된 도로 링크의 점수를 집계하여 추천 결과를 생성합니다.

## 2. 기존 웹 서비스 보존

기존 서비스는 신규 서비스와 구분하기 위해 `legacy/` 폴더에서 관리합니다.

| 파일 | 역할 |
|---|---|
| `legacy/index.html` | 기존 웹 서비스 화면 |
| `legacy/style.css` | 기존 화면 디자인 |
| `legacy/app.js` | 기존 사용자 입력 및 결과 처리 |
| `legacy/core.js` | 기존 경로 비교 및 추천 조건 처리 |
| `legacy/server.js` | 기존 Node.js 서버 및 Python 연동 |
| `legacy/live_route_features.py` | 기존 실시간 도로 링크 매칭 및 특성 추출 |
| `legacy/data.js` | 기존 수집 경로 기반 화면 데이터 |
| `tools/launch.py` | 기존 서비스 실행 도구 |

기존 서비스의 참조 경로와 관련 테스트는 변경된 폴더 구조에 맞게 수정했습니다.

기존 서비스의 알고리즘과 모델은 보존하며, 신규 서비스의 기본 실행 경로에서는 사용하지 않습니다.

## 3. AI 모델 및 학습

### 3.1 모델 파일

| 파일 및 폴더 | 역할 |
|---|---|
| `models/route_lightgbm.joblib` | 기존 경로 단위 LightGBM 회귀 모델 |
| `models/accident_risk_lightgbm_model_a.joblib` | 천안·아산 사고다발지역 분류 실험 모델 A |
| `models/accident_risk_lightgbm_model_b.joblib` | 천안·아산 사고다발지역 분류 실험 모델 B |
| `models/accident_risk_lightgbm_final.joblib` | 천안·아산 최종 실험 모델 |
| `models/national_accident_risk/experiments/` | 전국 사고다발지역 분류 실험 모델 L01~L06 |
| `models/national_accident_risk/national_accident_risk_lightgbm_final.joblib` | 최종 전국 T09 모델 및 학습 정보 |
| `models/national_accident_risk/national_accident_risk_lightgbm_final.txt` | 최종 T09 LightGBM Native Model |

기존 모델, 실험 모델 및 최종 모델을 포함하여 총 12개의 모델 파일을 보존합니다.

### 3.2 학습 및 분석 코드

| 파일 및 폴더 | 역할 |
|---|---|
| `training/` | 모델 학습, 평가 및 예측 코드 |
| `training/accident_risk/` | 천안·아산 사고다발지역 분류 연구 코드 |
| `training/national_accident_risk/` | 전국 사고다발지역 분류 연구 및 서비스 데이터 생성 코드 |
| `training/processed/features.json` | 기존 회귀 모델의 입력 특성 11개 정의 |
| `pipeline/` | 기존 데이터 전처리, 특성 생성 및 모델 실험 |
| `outputs/` | 기존 모델 비교, 평가 및 SHAP 분석 결과 |

전국 모델 학습 코드와 기존 모델 학습 코드는 구분하여 관리합니다.

### 3.3 신규 서비스 데이터 생성 및 검증

| 파일 | 역할 |
|---|---|
| `training/national_accident_risk/31_build_national_service_tiles.py` | 전국 도로 점수 기반 서비스 공간 타일 생성 |
| `training/national_accident_risk/32_validate_national_service_tiles.py` | 전국 공간 타일 독립 검증 |

## 4. 데이터 관련 폴더

| 폴더 | 역할 |
|---|---|
| `data/` | 원본 데이터 및 전처리 데이터 |
| `data/legacy/` | 기존 수집 데이터 및 원본 공공 데이터 |
| `data/accident_risk/` | 천안·아산 사고다발지역 연구 데이터 |
| `data/national_accident_risk/` | 전국 사고다발지역 연구 데이터 |
| `data/national_accident_risk/processed/` | 전국 도로 특성, 학습 데이터셋 및 링크별 모델 점수 |
| `data/national_accident_risk/processed/results/` | 전국 모델 성능 평가 및 추천 정책 실험 결과 |
| `source/` | 기존 서비스의 도로 및 사고 관련 입력 데이터 |
| `collectors/` | 카카오 경로 데이터 수집 코드 |
| `service/data/` | 신규 서비스 실행용 데이터 |

원본 데이터와 유효한 전처리 데이터는 모델 학습 및 결과 재현을 위해 보존합니다.

## 5. 개발 및 검증

| 파일 및 폴더 | 역할 |
|---|---|
| `tests/` | JavaScript 및 Python 자동화 테스트 |
| `tools/` | 환경 설정, 기존 서비스 실행 및 검증 도구 |
| `integration/validation/` | 신규 서비스 공간 타일 생성 및 검증 기록 |
| `docs/` | 프로젝트 설명, 모델 관리 및 검증 문서 |
| `documents/` | 기존 프로젝트 기획서 및 개발 합의 문서 |
| `requirements.txt` | Python 직접 의존성 |
| `requirements-lock.txt` | Python 의존성 버전 기록 |
| `SETUP_SAFEROUTE.cmd` | Python 개발 환경 설정 |
| `environment_versions.json` | 개발 환경 버전 정보 |
| `bundle_manifest.json` | 프로젝트 파일 무결성 검사 목록 |
| `backups/models/` | 기존 경로 모델 백업 |

### 5.1 주요 무결성 관리 문서

| 파일 | 역할 |
|---|---|
| `docs/MODEL_REGISTRY.md` | 기존 및 신규 모델 관리 정보 |
| `docs/MODEL_SHA256SUMS.txt` | 전체 모델 파일 12개의 SHA256 기준값 |
| `docs/SERVICE_PRE_MOVE_SHA256SUMS.txt` | 신규 서비스 폴더 이동 전 무결성 기록 |
| `docs/LEGACY_PRE_MOVE_SHA256SUMS.txt` | 기존 서비스 폴더 이동 전 무결성 기록 |

폴더 이동 전 해시 기록은 당시 파일 상태를 확인하기 위한 이력입니다. 이후 정상적으로 수정된 파일의 현재 해시와 다를 수 있습니다.

전체 프로젝트 매니페스트는 최종 프로젝트 구조에 맞게 갱신합니다.

## 6. 제외 및 관리 대상

### 6.1 GitHub 업로드 제외 대상

- `.venv/`: 로컬 Python 가상환경
- `__pycache__/`, `*.pyc`: Python 자동 생성 파일
- `.env`, `.env.*`: 환경변수 및 인증정보 파일
- `backups/`: 로컬 백업 파일
- `data/`: 대용량 원본 및 전처리 데이터
- `models/`: 학습 모델 파일
- `runtime/`: Node.js 실행 환경
- `source/`: 기존 서비스 입력 데이터
- `legacy/data.js`: 기존 서비스 대용량 경로 데이터
- `training/processed/`: 기존 학습용 전처리 데이터
- `service/data/tiles/`: 전국 서비스 공간 타일
- `service/data/national_manifest.json`: 생성된 전국 타일 매니페스트

### 6.2 보존 및 삭제 기준

- 기존 모델과 실험 모델은 보존합니다.
- 원본 데이터와 정상적인 학습 산출물은 보존합니다.
- 기존 서비스 코드 및 검증 기록은 보존합니다.
- 오류 발생 과정에서 생성된 불완전한 중간 산출물은 정상 결과 검증 후 삭제합니다.
- Python 캐시와 재생성 가능한 임시 파일은 정리할 수 있습니다.
- API 키와 인증정보는 공유 및 제출 자료에 포함하지 않습니다.

## 7. 팀원 공유 및 배포

전체 개발 프로젝트와 서비스 실행용 배포 패키지를 구분하여 관리합니다.

| 구분 | 목적 |
|---|---|
| 전체 개발 프로젝트 | 데이터 분석, 모델 학습, 서비스 개발 및 연구 재현 |
| GitHub 저장소 | 소스코드, 설정 파일 및 문서의 버전 관리 |
| 독립 실행형 배포 패키지 | 팀원 공유, 서비스 시연 및 교수님 제출 |

독립 실행형 배포 패키지에는 서비스 코드, 전국 공간 타일, 매니페스트, 추천 정책 및 인터페이스 설정, Node.js 실행 환경과 실행 스크립트가 필요합니다.

신규 서비스 실행에 불필요한 학습 데이터, 실험 모델, Python 가상환경 및 기존 서비스는 실행용 패키지에서 제외할 수 있습니다.

독립 실행형 배포 패키지의 최종 구성 및 별도 환경 실행 검증은 배포 단계에서 확정합니다.