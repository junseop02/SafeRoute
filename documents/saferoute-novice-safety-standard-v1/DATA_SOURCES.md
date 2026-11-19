# 공식 데이터 출처

## 사고·사고다발지역

- TAAS 교통사고분석시스템: https://taas.koroad.or.kr/
- 전국 교통사고다발지역 CSV: https://www.data.go.kr/data/15029185/fileData.do
- 법규위반 사고다발지역 API: https://www.data.go.kr/data/15058087/openapi.do
- 보행자 사고다발지역: https://www.data.go.kr/data/1913/linkedData.do

TAAS는 사고의 기상, 노면, 도로형태, 선형, 교차로형태와 운전자 면허경과년수 등을 분석할 수 있습니다. 단순 사고건수는 교통량이 많은 도로를 과대평가하므로 도로 길이와 교통량으로 보정합니다.

## 도로망·교통량·시설

- 국가표준노드링크: https://www.its.go.kr/nodelink/nodelinkStatus?service=inquiryNodelink
- 국토부 도로 교통량 API: https://www.data.go.kr/data/15097077/openapi.do
- 전국 일방통행도로: https://www.data.go.kr/data/15028199/standard.do
- 전국 신호등: https://www.data.go.kr/data/15028198/standard.do
- 국토지리정보원 수치표고모델: https://www.data.go.kr/data/15059920/fileData.do

## 실시간 교통·돌발·통제

- UTIC 개방데이터: https://www.utic.go.kr/guide/newUtisData.do
- UTIC 실시간 돌발정보: https://www.utic.go.kr/guide/utisRefIncident.do
- UTIC 도로위험상황예보: https://www.utic.go.kr/guide/utisRefRoadAcc.do
- 국토부 실시간 구간 소통정보: https://www.data.go.kr/data/15040463/openapi.do
- 경찰청 실시간 소통정보: https://www.data.go.kr/data/15087120/openapi.do

전면 통제·재난·침수·공식 결빙 통제는 점수로 처리하지 않고 즉시 경로에서 제외합니다.

## 기상·시야

- 기상청 단기예보 API: https://www.data.go.kr/data/15084084/openapi.do
- 기상청 ASOS 관측: https://data.kma.go.kr/data/grnd/selectAsosList.do?pgmNo=34
- 일출·일몰·박명 API: https://www.data.go.kr/data/15012688/openapi.do

강수, 적설, 기온, 풍속, 시정과 곡선·내리막·교량을 함께 평가합니다. 가로등 데이터가 없다는 이유만으로 조명이 없다고 판단하지 않습니다.

## 보호구역·교통약자

- 전국 어린이보호구역: https://www.data.go.kr/data/15012891/standard.do
- TAAS 보호구역·보행자 분석: https://taas.koroad.or.kr/

보호구역은 제한속도와 안전시설이 설치된 장소일 수 있으므로 그 자체를 위험도로 단정하지 않습니다. 실제 사고이력, 운영시간, 보행자 노출과 운전 부담을 분리해 사용합니다.
