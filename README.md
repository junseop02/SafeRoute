# SafeRoute — AI 기반 초보 운전자 친화 경로 추천 서비스

## 1. 프로젝트 소개

SafeRoute는 초보 운전자와 고령 운전자 등 복잡한 도로 환경에서 운전에 부담을 느낄 수 있는 사용자를 위해 개발한 AI 기반 경로 추천 서비스입니다.

기존 내비게이션의 일반적인 경로 안내와 달리, 도로 특성과 사고다발지, 교차로 밀집도 등의 정보를 활용해 운전 부담이 상대적으로 낮을 것으로 예측되는 경로를 추천하는 것을 목표로 합니다.

카카오모빌리티 API에서 자동차 경로 후보를 조회하고, 공공 도로 데이터와 매칭하여 추출한 특성을 LightGBM 모델에 입력합니다. 모델이 예측한 부담도를 0~100점의 쉬움 점수로 변환하고, 사용자가 설정한 우회 허용 범위 안에서 경로를 비교합니다.

본 서비스는 실제 사고 위험이나 운전자의 운전 능력을 판정하지 않으며, 추천 점수가 경로의 안전을 보장하지 않습니다.

## 2. 주요 기능

### 2.1 장소 검색 및 자동차 경로 탐색

- 카카오 장소 검색 API를 이용한 출발지 및 목적지 검색
- 카카오모빌리티 자동차 길찾기 API 연동
- 여러 경로 후보 조회 및 중복 경로 제거
- 경로별 거리와 예상 소요 시간 정보 처리

### 2.2 AI 기반 운전 부담도 분석

- 경로 좌표를 공공 도로 링크와 공간적으로 매칭
- 도로 폭 대용 지표, 사고다발지 근접성, 교차로 관련 특성 추출
- LightGBM 모델을 이용한 경로 부담도 예측
- AI 쉬움 점수(0~100점) 제공

### 2.3 사용자 맞춤 경로 추천

- 최단 경로 대비 우회 허용 범위 설정
- 도로 매칭률 80% 이상인 경로만 AI 추천 대상으로 사용
- 우회 허용 범위 안에서 AI 쉬움 점수가 높은 경로 추천
- 점수가 같으면 이동 거리가 짧은 경로 우선
- 경로별 추천 가능 여부와 제외 사유 표시

## 3. 시스템 구성

SafeRoute는 Node.js 기반 웹 서버와 Python 기반 AI 예측 모듈로 구성됩니다.

**서비스 처리 과정**

1. 사용자가 출발지와 목적지를 검색합니다.
2. 카카오모빌리티 API에서 자동차 경로 후보를 조회합니다.
3. 경로 좌표를 공공 도로망에 매칭합니다.
4. 경로별 11개 입력 특성을 생성합니다.
5. 학습된 LightGBM 모델로 경로 부담도를 예측합니다.
6. 예측 부담도를 AI 쉬움 점수로 변환합니다.
7. 우회 허용 범위와 데이터 매칭 조건을 적용합니다.
8. 추천 경로와 비교 결과를 웹 화면에 표시합니다.

## 4. 기술 스택

| 구분 | 기술 |
|---|---|
| 프런트엔드 | HTML, CSS, JavaScript |
| 백엔드 | Node.js |
| AI 모델 | LightGBM |
| AI 및 데이터 처리 | Python 3.12, NumPy, Pandas |
| 공간 데이터 처리 | GeoPandas, Shapely, PyProj |
| 모델 관리 | Joblib |
| 외부 API | Kakao Local API, Kakao Mobility Directions API |

## 5. AI 모델

### 5.1 모델 입력 특성

LightGBM 모델은 다음 11개 특성을 입력받습니다.

| Feature | 설명 |
|---|---|
| `distance_m` | 경로 거리 |
| `duration_s` | 예상 소요 시간 |
| `road_count` | 경로 도로 구간 관련 수치 |
| `guide_count` | 경로 안내 지점 관련 수치 |
| `n_intersect_200m` | 주변 200m 교차로 관련 지표 |
| `wide_road_ratio` | 2차로 이상 도로의 길이 비율 |
| `near_hotspot_ratio` | 사고다발지 150m 이내 도로의 길이 비율 |
| `coverage` | 유효 도로 특성 매칭 길이 비율 |
| `accident_burden` | 사고다발지 관련 부담 지표 |
| `narrow_burden` | 좁은 도로 관련 부담 지표 |
| `intersection_burden` | 교차로 관련 부담 지표 |

도로별 특성은 경로와 매칭된 도로 링크의 길이를 가중치로 사용해 집계합니다.

### 5.2 쉬움 점수 계산

모델이 예측한 부담도를 다음과 같이 변환합니다.

```python
easy_score = clip(100 - predicted_burden, 0, 100)
```

- 100점에 가까울수록 모델이 예측한 부담도가 낮습니다.
- 0점에 가까울수록 모델이 예측한 부담도가 높습니다.
- 실제 사고 확률이나 안전성을 의미하지 않습니다.

### 5.3 도로 링크 매칭

실시간 경로 분석에서는 다음 설정을 사용합니다.

- 경로 좌표 샘플링 간격: 2m
- 도로 링크 최대 매칭 거리: 10m
- 최소 유효 도로 매칭률: 80%

매칭률이 기준보다 낮거나 입력 특성을 정상적으로 생성할 수 없는 경로는 AI 추천에서 제외합니다.

## 6. 프로젝트 폴더 구조

```text
SafeRoute/
├── app.js
├── core.js
├── server.js
├── index.html
├── style.css
├── live_route_features.py
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── SETUP_SAFEROUTE.cmd
├── START_SAFEROUTE.cmd
├── collectors/
├── data/
├── docs/
├── documents/
├── models/
├── outputs/
├── pipeline/
├── runtime/
├── source/
├── tests/
├── tools/
└── training/
```

**주요 폴더 역할**

- `collectors/`: 경로 데이터 수집 코드
- `data/`: 서비스 데이터 및 수집 데이터
- `docs/`: 프로젝트 설명 및 검증 문서
- `documents/`: 프로젝트 기획 및 개발 관련 문서
- `models/`: 학습된 모델 파일
- `outputs/`: 모델 평가 및 분석 결과
- `pipeline/`: 데이터 전처리 및 특성 생성 코드
- `runtime/`: Node.js 실행 환경
- `source/`: 도로·사고 등 입력 데이터
- `tests/`: 기능 및 통합 테스트
- `tools/`: 환경 설정과 실행·검증 도구
- `training/`: 모델 학습 및 경로 예측 코드

## 7. 설치 및 실행

### 7.1 개발 환경

- Windows
- Python 3.12
- Node.js 실행 환경
- 카카오 REST API 키
- Python 패키지: `requirements.txt` 참조

### 7.2 Python 환경 설정

프로젝트에 포함된 환경 설정 스크립트를 실행합니다.

```bat
SETUP_SAFEROUTE.cmd
```

또는 Python 3.12 환경에서 직접 의존성을 설치할 수 있습니다.

```bash
python -m pip install -r requirements.txt
```

### 7.3 카카오 API 키 설정

실시간 장소 검색 및 자동차 경로 탐색에는 `KAKAO_REST_API_KEY` 환경변수가 필요합니다.

Windows 명령 프롬프트에서 실행하는 경우:

```bat
set KAKAO_REST_API_KEY=발급받은_API_키
START_SAFEROUTE.cmd
```

API 키는 소스코드에 직접 입력하거나 공개 저장소에 업로드하지 않습니다.

### 7.4 서버 실행

프로젝트 루트에서 다음 파일을 실행합니다.

```bat
START_SAFEROUTE.cmd
```

기본 접속 주소:

```text
http://127.0.0.1:8090
```

실시간 AI 분석에는 `.venv/Scripts/python.exe`와 필요한 모델·도로 데이터 파일이 존재해야 합니다.

## 8. 테스트 및 검증

프로젝트에는 JavaScript 기반 자동화 테스트와 Python 환경 검사 도구가 포함되어 있습니다.

### JavaScript 테스트

```bash
node --test tests/*.test.js
```

### Python 환경 검사

```bash
./.venv/Scripts/python.exe tools/check_environment.py
```

### 파일 무결성 검사

```bash
./.venv/Scripts/python.exe tools/check_environment.py --full-hash
```

파일 무결성 검사는 최종 프로젝트 구조에 맞게 `bundle_manifest.json`을 갱신한 후 실행해야 합니다.

## 9. 데이터 및 모델 관리

SafeRoute는 공공 도로망, 사고다발지, 교차로 및 경로 수집 데이터를 활용합니다.

데이터 처리와 학습 관련 코드는 `pipeline/`, `collectors/`, `training/`에서 관리합니다.

GitHub 코드 저장소에는 대용량 데이터와 모델 파일 일부가 포함되지 않을 수 있습니다. 실행 환경을 재구성할 때는 필요한 데이터와 모델을 별도로 준비해야 합니다.

## 10. 한계 및 향후 개선

현재 AI 쉬움 점수는 학습된 모델의 예측 결과를 변환한 지표입니다. 운전자의 실제 체감 난이도나 사고 위험을 직접 검증한 결과는 아닙니다.

향후 개선 과제는 다음과 같습니다.

- 초보 운전자 대상 실제 주행 평가
- 경로 난이도에 대한 검증 데이터 확보
- 모델 입력 특성 및 학습 목표 개선
- 도로 링크 매칭 정확도 검증
- 다양한 지역으로 서비스 적용 범위 확대
- 모델 성능 및 추천 결과의 정량적 검증

---

**SafeRoute — AI 기반 운전자 친화 경로 추천 프로젝트**