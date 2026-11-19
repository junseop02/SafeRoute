
import json
import os
import urllib.parse
import urllib.request
import urllib.error

# 카카오 REST API 키는 환경변수에서 읽습니다.
api_key = os.environ.get("KAKAO_REST_API_KEY", "").strip()

if not api_key:
    raise SystemExit("KAKAO_REST_API_KEY 환경변수를 설정해주세요.")

# 검색할 장소
query = "천안역"

# 카카오 키워드 장소 검색 API
params = urllib.parse.urlencode({
    "query": query,
    "size": 5
})

url = "https://dapi.kakao.com/v2/local/search/keyword.json?" + params

request = urllib.request.Request(
    url,
    headers={"Authorization": "KakaoAK " + api_key}
)

try:
    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.load(response)

    print("장소 검색 성공!")
    print("검색어:", query)

    for place in data.get("documents", []):
        print("----------------------")
        print("장소명:", place.get("place_name"))
        print("주소:", place.get("address_name"))
        print("경도:", place.get("x"))
        print("위도:", place.get("y"))

except urllib.error.HTTPError as error:
    print("HTTP 오류:", error.code)
    print(error.read().decode("utf-8", errors="replace"))

except urllib.error.URLError as error:
    print("연결 오류:", error.reason)