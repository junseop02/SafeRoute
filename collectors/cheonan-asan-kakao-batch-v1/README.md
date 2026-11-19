# 천안·아산 카카오내비 자동 수집기

출발지와 도착지를 직접 고르지 않아도 천안시·아산시의 읍·면·동 대표점 사이 자동차 경로를 자동으로 요청하고 파일로 저장합니다. 지도와 JavaScript 키는 필요하지 않고 카카오 REST API 키만 사용합니다.

## 수집 범위

- 천안시: 31개 읍·면·행정동
- 아산시: 17개 읍·면·행정동
- 합계 대표점: 48개 행정복지센터
- 방향을 구분한 출발–도착 조합: `48 × 47 = 2,256쌍`
- 기본 경로 조건: 추천(대안 포함), 빠른, 최단
- 기본 자동차 길찾기 요청: `2,256 × 3 = 6,768건`

좌표는 실행 시 카카오 장소 검색으로 자동 확인합니다. 대표점끼리 연결한 **표본 경로 데이터셋**이며, 도시 안의 연속된 모든 좌표·모든 도로·모든 가능한 경로를 완전 열거한 자료는 아닙니다. 출발점과 도착점은 두 도시 안에 있지만 카카오가 선택한 경로가 시 경계를 잠시 벗어날 수도 있습니다.

## 실행 방법

1. 압축을 풉니다.
2. `START_천안아산_자동수집.cmd`를 더블클릭합니다.
3. 검은 창에 본인 앱의 **REST API 키**를 붙여넣고 Enter를 누릅니다.
4. 지점 확인과 경로 수집이 자동으로 진행됩니다.
5. 완료되면 `data/processed` 폴더가 자동으로 열립니다.

수집 중 `Ctrl+C`를 누르면 현재 파일을 마무리하고 중단합니다. 같은 실행 파일을 다시 실행하면 이미 저장된 요청을 건너뛰고 이어서 진행합니다. REST API 키는 파일에 기록하지 않습니다.

## 생성 파일

### 입력과 진행 상황

- `data/points.csv`: 카카오에서 확인한 48개 대표점 좌표
- `data/points.geojson`: 대표점 지도 파일
- `data/route_requests.csv`: 전체 요청 계획과 완료·대기 상태
- `data/raw/geocoding/*.json`: 장소 검색 원본 응답
- `data/raw/routes/*.json`: 경로 요청별 카카오 원본 응답
- `data/errors/*.json`: 재시도할 오류 요청

### 가공 결과

- `data/processed/collection_manifest.json`: 전체 수집 건수와 파일 통계
- `data/processed/route_summary.csv`: 경로별 출발지·도착지·거리·시간·요금
- `data/processed/road_segments.csv`: 도로명·구간 거리·시간·교통속도·교통상태
- `data/processed/guides.csv`: 좌회전·우회전 등 경로 안내 지점
- `data/processed/routes.geojson`: 도로 구간 좌표와 교통 속성이 포함된 GeoJSON
- `data/processed/routes.ndjson`: 요청별 정규화 결과
- `data/processed/routes/*.json`: 요청별 정규화 JSON

CSV는 Excel에서 한글이 깨지지 않도록 UTF-8 BOM으로 저장합니다. `traffic_speed`와 `traffic_state`는 실제 차량 주행 로그가 아니라 **API를 호출한 시점의 교통 스냅샷**입니다.

## 호출량과 이용 범위

카카오 공식 문서의 자동차 길찾기 무료 쿼터는 현재 일 10,000건이며 변경될 수 있습니다. 기본 설정 6,768건은 이 수치보다 작지만, 같은 날 다른 앱 호출량이 있으면 도중에 쿼터 오류가 발생할 수 있습니다. 수집기는 인증 오류나 쿼터 오류에서 중단하며 다음 실행에서 이어갑니다.

`settings.json`의 요청 간격과 1회 최대 요청 수를 조정할 수 있습니다. `config/settings-extended.example.json`은 회피 조건까지 포함한 예시이지만 총 13,536건이므로 하루 무료 쿼터보다 큽니다. 확장 설정은 쿼터와 카카오의 허용 범위를 확인한 경우에만 나누어 실행하세요.

카카오 API 응답을 대량으로 영구 저장하거나 학습 데이터로 사용·재배포하려면 카카오 개발자 운영정책과 제휴 조건을 먼저 확인해야 합니다. 이 패키지는 실제 카카오 응답을 포함하지 않으며, 사용자 본인의 앱 키로 허용된 목적 안에서 실행하도록 구성되어 있습니다.

## 공식 문서

- 자동차 길찾기: https://developers.kakaomobility.com/guide/navi-api/directions
- 길찾기 시작하기: https://developers.kakaomobility.com/guide/navi-api/start.html
- 쿼터: https://developers.kakao.com/docs/ko/getting-started/quota
- 운영정책: https://developers.kakao.com/terms/ko/site-policies
