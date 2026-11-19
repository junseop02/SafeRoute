# SafeRoute 프로젝트 파일 구성

## 1. 웹 서비스 실행 파일

| 파일 | 역할 |
|---|---|
| `index.html` | 웹 서비스 화면 구조 |
| `style.css` | 화면 디자인 및 스타일 |
| `app.js` | 사용자 입력, 경로 조회, 결과 표시 |
| `core.js` | 경로 비교 및 추천 조건 처리 |
| `server.js` | Node.js 서버, 카카오 API 및 Python 연동 |
| `live_route_features.py` | 실시간 경로와 도로 링크 매칭 및 특성 추출 |
| `data.js` | 기존 수집 경로 기반 화면 데이터 |
| `START_SAFEROUTE.cmd` | Windows 서버 실행 스크립트 |

## 2. AI 모델 및 학습

- `models/route_lightgbm.joblib`: 현재 서비스에서 사용하는 LightGBM 모델
- `training/`: 모델 학습, 평가 및 경로 예측 코드
- `training/processed/features.json`: 모델 입력 특성 11개 정의
- `pipeline/`: 기존 데이터 수집 이후 전처리, 특성 생성 및 모델 실험 코드
- `outputs/`: 모델 비교, 평가 및 SHAP 분석 결과

`pipeline/`의 이전 실험과 현재 서비스의 LightGBM 모델은 구분하여 관리합니다.

## 3. 데이터 관련 폴더

- `data/`: 서비스 데이터 및 기존 수집·전처리 데이터
- `source/`: 도로, 사고 등 입력 데이터
- `collectors/`: 카카오 경로 데이터 수집 코드

데이터 파일은 현재 서비스 실행과 모델 학습 재현에 필요할 수 있으므로 임의로 삭제하지 않습니다.

## 4. 개발 및 검증

- `tests/`: JavaScript 자동화 테스트, Python 테스트 및 테스트 데이터
- `tools/`: 환경 설정, 실행 및 환경 검사 도구
- `docs/`: 개발 및 검증 문서
- `documents/`: 프로젝트 기획서, 개발 합의 문서 등
- `requirements.txt`: Python 직접 의존성
- `requirements-lock.txt`: Python 의존성 버전 기록
- `SETUP_SAFEROUTE.cmd`: Python 환경 설정
- `bundle_manifest.json`: 프로젝트 파일 무결성 검사용 목록
- `environment_versions.json`: 개발 환경 버전 정보
- `runtime/`: Windows용 Node.js 실행 파일

## 5. 제외 및 관리 대상

- `.venv/`: 로컬 Python 가상환경
- `__pycache__/`, `*.pyc`: Python 실행 과정에서 생성되는 파일
- `.env`, `.env.*`: 환경변수 및 인증정보 파일
- 이전 버전 백업 코드 및 사용하지 않는 모델 파일: 프로젝트 정리 과정에서 삭제

대용량 데이터, 모델 및 실행 환경 일부는 `.gitignore`를 통해 일반 GitHub 업로드에서 제외합니다.

현재 서비스에 필요한 파일과 GitHub에 포함되는 파일은 서로 다를 수 있으므로 배포 시 별도로 확인해야 합니다.