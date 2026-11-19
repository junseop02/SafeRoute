const state = { map: null, places: null, start: null, end: null, markers: {}, routeLines: [], routeData: null };
const $ = (selector) => document.querySelector(selector);
const keyDialog = $('#key-dialog');

function setMessage(message) { $('#map-message').textContent = message; }

function requestKey() {
  $('#key-input').value = sessionStorage.getItem('saferouteKakaoJsKey') || localStorage.getItem('saferouteKakaoJsKey') || '';
  keyDialog.showModal();
}

function loadKakao(key) {
  setMessage('카카오맵을 불러오는 중입니다…');
  const script = document.createElement('script');
  script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${encodeURIComponent(key)}&libraries=services&autoload=false`;
  script.onload = () => kakao.maps.load(initMap);
  script.onerror = () => { setMessage('지도를 불러오지 못했습니다. 키와 등록 도메인을 확인하세요.'); requestKey(); };
  document.head.appendChild(script);
}

function initMap() {
  state.map = new kakao.maps.Map($('#map'), { center: new kakao.maps.LatLng(36.799, 127.071), level: 8 });
  state.places = new kakao.maps.services.Places();
  setMessage('');
}

$('#key-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const key = $('#key-input').value.trim();
  if (!key) return;
  sessionStorage.setItem('saferouteKakaoJsKey', key);
  if ($('#remember-key').checked) localStorage.setItem('saferouteKakaoJsKey', key);
  else localStorage.removeItem('saferouteKakaoJsKey');
  keyDialog.close();
  location.reload();
});

$('#change-key').addEventListener('click', () => requestKey());
$('#key-cancel').addEventListener('click', () => keyDialog.close());

$('#search-form').addEventListener('submit', (event) => {
  event.preventDefault();
  if (!state.places) return setMessage('먼저 카카오맵을 불러와 주세요.');
  const keyword = $('#keyword').value.trim();
  $('#search-results').textContent = '검색 중…';
  state.places.keywordSearch(keyword, showResults, { size: 15, sort: kakao.maps.services.SortBy.ACCURACY });
});

function showResults(data, status) {
  const box = $('#search-results');
  box.replaceChildren();
  if (status === kakao.maps.services.Status.ZERO_RESULT) {
    box.textContent = '검색 결과가 없습니다. 지역 이름을 함께 입력해 보세요.';
    return;
  }
  if (status !== kakao.maps.services.Status.OK) {
    box.textContent = '검색 요청에 실패했습니다. 잠시 뒤 다시 시도해 주세요.';
    return;
  }
  data.slice(0, 8).forEach((place) => {
    const button = document.createElement('button');
    button.className = 'result';
    button.innerHTML = `${escapeHtml(place.place_name)}<small>${escapeHtml(place.road_address_name || place.address_name)}</small>`;
    button.addEventListener('click', () => choosePlace(place));
    box.appendChild(button);
  });
}

function choosePlace(place) {
  const target = $('#target').value;
  const point = { name: place.place_name, lat: Number(place.y), lng: Number(place.x) };
  state[target] = point;
  $(`#${target}-label`).textContent = point.name;
  if (state.markers[target]) state.markers[target].setMap(null);
  state.markers[target] = new kakao.maps.Marker({ map: state.map, position: new kakao.maps.LatLng(point.lat, point.lng) });
  state.map.panTo(state.markers[target].getPosition());
  $('#search-results').replaceChildren();
  if (state.start && state.end) {
    $('#route-button').disabled = false;
    $('#route-button').textContent = '서로 다른 카카오 경로 찾기';
  }
}

function formatDuration(seconds) {
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}분`;
  return `${Math.floor(minutes / 60)}시간 ${minutes % 60}분`;
}

function clearRouteLines() {
  state.routeLines.forEach(({ line }) => line.setMap(null));
  state.routeLines = [];
}

function routeAverageSpeed(route) {
  let weightedSpeed = 0;
  let distance = 0;
  for (const section of route.sections || []) {
    for (const road of section.roads || []) {
      if (Number(road.traffic_speed) > 0 && Number(road.distance) > 0) {
        weightedSpeed += Number(road.traffic_speed) * Number(road.distance);
        distance += Number(road.distance);
      }
    }
  }
  return distance ? Math.round(weightedSpeed / distance) : null;
}

function focusRoute(routeId) {
  state.routeLines.forEach(({ line, id }) => {
    line.setOptions({ strokeWeight: id === routeId ? 9 : 5, strokeOpacity: id === routeId ? .95 : .35 });
  });
  document.querySelectorAll('.route-card').forEach((card) => card.classList.toggle('active', card.dataset.routeId === routeId));
}

function addRouteCard(route) {
  const summary = route.data.summary || {};
  const card = document.createElement('button');
  card.className = 'route-card';
  card.dataset.routeId = route.id;
  card.innerHTML = `
    <span class="route-color" style="background:${route.color}"></span>
    <strong>${escapeHtml(route.label)}</strong>
    <span>${(Number(summary.distance || 0) / 1000).toFixed(1)}km</span>
    <span>${formatDuration(Number(summary.duration || 0))}</span>
    <span>${routeAverageSpeed(route.data) ? `${routeAverageSpeed(route.data)}km/h` : '속도 정보 없음'}</span>`;
  card.addEventListener('click', () => focusRoute(route.id));
  $('#route-results').appendChild(card);
}

function showDataFiles(files) {
  const labels = {
    raw_json: '카카오 원본 JSON', normalized_json: '통합 경로 JSON', routes_geojson: '경로 GeoJSON',
    summary_csv: '경로 요약 CSV', road_segments_csv: '도로 구간 CSV'
  };
  const box = $('#data-files');
  box.replaceChildren();
  for (const [key, href] of Object.entries(files || {})) {
    const link = document.createElement('a');
    link.href = href;
    link.download = '';
    link.textContent = labels[key] || key;
    box.appendChild(link);
  }
  box.hidden = !Object.keys(files || {}).length;
}

function drawKakaoRoutes(data) {
  clearRouteLines();
  $('#route-results').replaceChildren();
  if (!data.routes?.length) throw new Error('카카오가 반환한 자동차 경로가 없습니다.');

  const bounds = new kakao.maps.LatLngBounds();
  let pointCount = 0;
  const lineStyles = {
    recommend: 'solid', time: 'shortdash', distance: 'shortdot', 'main-road': 'dash',
    'no-motorway': 'longdash', 'no-toll': 'dashdot', 'no-schoolzone': 'longdashdot'
  };
  for (const route of data.routes) {
    for (const section of route.data.sections || []) {
      for (const road of section.roads || []) {
        const path = [];
        for (let index = 0; index < (road.vertexes || []).length; index += 2) {
          const point = new kakao.maps.LatLng(road.vertexes[index + 1], road.vertexes[index]);
          path.push(point);
          bounds.extend(point);
          pointCount += 1;
        }
        if (path.length > 1) {
          const line = new kakao.maps.Polyline({
            map: state.map, path, strokeWeight: 5, strokeColor: route.color,
            strokeOpacity: .7, strokeStyle: lineStyles[route.request_id] || 'solid'
          });
          kakao.maps.event.addListener(line, 'click', () => focusRoute(route.id));
          state.routeLines.push({ id: route.id, line });
        }
      }
    }
    addRouteCard(route);
  }
  if (!pointCount) throw new Error('응답에 지도에 표시할 경로 좌표가 없습니다.');
  state.map.setBounds(bounds);
  focusRoute(data.routes[0].id);
  $('#route-download').hidden = false;
  showDataFiles(data.files);
  const removed = Number(data.duplicates_removed || 0);
  $('#route-status').textContent = data.routes.length === 1
    ? `서로 다른 경로는 1개뿐입니다. 동일한 형상 ${removed}개를 제외했으며, 이 구간에는 카카오가 제공하는 대체 도로가 없을 수 있습니다.`
    : `${data.routes.length}개의 서로 다른 경로를 표시했습니다. 동일한 형상 ${removed}개는 제외했고 결과를 data 폴더에 저장했습니다.`;
  state.routeData = data;
}

$('#route-button').addEventListener('click', async () => {
  if (!state.start || !state.end) return;
  const button = $('#route-button');
  button.disabled = true;
  button.textContent = '경로 불러오는 중…';
  $('#route-status').textContent = '카카오모빌리티에 경로를 요청하고 있습니다.';
  const query = new URLSearchParams({
    startLng: state.start.lng,
    startLat: state.start.lat,
    endLng: state.end.lng,
    endLat: state.end.lat
  });
  try {
    const response = await fetch(`/api/routes?${query}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || '경로 요청에 실패했습니다.');
    drawKakaoRoutes(data);
    button.textContent = '다른 경로 다시 찾기';
  } catch (error) {
    $('#route-status').textContent = error.message;
    button.textContent = '경로 요청 다시 시도';
  } finally {
    button.disabled = false;
  }
});

$('#route-download').addEventListener('click', () => {
  if (!state.routeData) return;
  const blob = new Blob([JSON.stringify(state.routeData, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `saferoute-kakao-routes-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

const savedKey = sessionStorage.getItem('saferouteKakaoJsKey') || localStorage.getItem('saferouteKakaoJsKey');
if (savedKey) loadKakao(savedKey); else requestKey();
