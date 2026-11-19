const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');
const root = __dirname;
const port = Number(process.env.SAFEROUTE_PORT || 8080);
const dataRoot = process.env.SAFEROUTE_DATA_DIR ? path.resolve(process.env.SAFEROUTE_DATA_DIR) : path.join(root, 'data');
const kakaoDirectionsUrl = process.env.KAKAO_NAVI_BASE_URL || 'https://apis-navi.kakaomobility.com/v1/directions';
const types = { '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8', '.json':'application/json; charset=utf-8', '.geojson':'application/geo+json; charset=utf-8', '.csv':'text/csv; charset=utf-8', '.yaml':'text/yaml; charset=utf-8' };

function sendJson(response, status, body) {
  const data = Buffer.from(JSON.stringify(body));
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': data.length,
    'Cache-Control': 'no-store'
  });
  response.end(data);
}

function validCoordinate(value, min, max) {
  const number = Number(value);
  return Number.isFinite(number) && number >= min && number <= max;
}

const profiles = [
  { id: 'recommend', priority: 'RECOMMEND', label: '추천 경로', color: '#1d5cff', alternatives: true },
  { id: 'time', priority: 'TIME', label: '빠른 경로', color: '#e45826' },
  { id: 'distance', priority: 'DISTANCE', label: '최단 경로', color: '#16a36f' },
  { id: 'main-road', priority: 'MAIN_ROAD', label: '큰길 우선', color: '#8b5cf6' },
  { id: 'no-motorway', priority: 'RECOMMEND', avoid: 'motorway', label: '고속도로 제외', color: '#0891b2' },
  { id: 'no-toll', priority: 'RECOMMEND', avoid: 'toll', label: '유료도로 제외', color: '#ca8a04' },
  { id: 'no-schoolzone', priority: 'RECOMMEND', avoid: 'schoolzone', label: '스쿨존 제외', color: '#db2777' }
];

function routeSignature(route) {
  const hash = crypto.createHash('sha1');
  for (const section of route.sections || []) {
    for (const road of section.roads || []) {
      hash.update((road.vertexes || []).map((value) => Number(value).toFixed(5)).join(','));
    }
  }
  return hash.digest('hex');
}

function csvCell(value) {
  const text = String(value ?? '');
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function routeGeoJson(bundle) {
  const features = [];
  for (const route of bundle.routes) {
    for (const [sectionIndex, section] of (route.data.sections || []).entries()) {
      for (const [roadIndex, road] of (section.roads || []).entries()) {
        const coordinates = [];
        for (let i = 0; i < (road.vertexes || []).length; i += 2) {
          coordinates.push([road.vertexes[i], road.vertexes[i + 1]]);
        }
        if (coordinates.length < 2) continue;
        features.push({
          type: 'Feature',
          properties: {
            route_id: route.id, profile: route.label, priority: route.priority, avoid: route.avoid || '',
            color: route.color, section_index: sectionIndex, road_index: roadIndex,
            road_name: road.name || '', distance_m: road.distance || 0,
            duration_s: road.duration || 0, traffic_speed_kmh: road.traffic_speed ?? '',
            traffic_state: road.traffic_state ?? ''
          },
          geometry: { type: 'LineString', coordinates }
        });
      }
    }
  }
  return { type: 'FeatureCollection', generated_at: bundle.generated_at, features };
}

function summaryCsv(bundle) {
  const rows = [['route_id','profile','priority','avoid','distance_m','duration_s','fare_taxi_krw','fare_toll_krw']];
  for (const route of bundle.routes) {
    const summary = route.data.summary || {};
    rows.push([route.id, route.label, route.priority, route.avoid || '', summary.distance, summary.duration, summary.fare?.taxi, summary.fare?.toll]);
  }
  return rows.map((row) => row.map(csvCell).join(',')).join('\n');
}

function roadsCsv(bundle) {
  const rows = [['route_id','profile','priority','avoid','section_index','road_index','road_name','distance_m','duration_s','traffic_speed_kmh','traffic_state']];
  for (const route of bundle.routes) {
    for (const [sectionIndex, section] of (route.data.sections || []).entries()) {
      for (const [roadIndex, road] of (section.roads || []).entries()) {
        rows.push([route.id, route.label, route.priority, route.avoid || '', sectionIndex, roadIndex, road.name, road.distance, road.duration, road.traffic_speed, road.traffic_state]);
      }
    }
  }
  return rows.map((row) => row.map(csvCell).join(',')).join('\n');
}

async function saveRouteFiles(bundle, rawResponses) {
  const stamp = bundle.generated_at.replace(/[:.]/g, '-');
  const rawDir = path.join(dataRoot, 'raw');
  const processedDir = path.join(dataRoot, 'processed');
  await fs.promises.mkdir(rawDir, { recursive: true });
  await fs.promises.mkdir(processedDir, { recursive: true });
  const raw = JSON.stringify({ request: bundle.request, responses: rawResponses }, null, 2);
  const normalized = JSON.stringify(bundle, null, 2);
  const geojson = JSON.stringify(routeGeoJson(bundle), null, 2);
  const summary = `\uFEFF${summaryCsv(bundle)}`;
  const roads = `\uFEFF${roadsCsv(bundle)}`;
  const files = [
    [path.join(rawDir, `kakao-routes-${stamp}.json`), raw],
    [path.join(rawDir, 'latest-kakao-routes.json'), raw],
    [path.join(processedDir, `routes-${stamp}.json`), normalized],
    [path.join(processedDir, 'latest-routes.json'), normalized],
    [path.join(processedDir, `routes-${stamp}.geojson`), geojson],
    [path.join(processedDir, 'latest-routes.geojson'), geojson],
    [path.join(processedDir, `route-summary-${stamp}.csv`), summary],
    [path.join(processedDir, 'latest-route-summary.csv'), summary],
    [path.join(processedDir, `road-segments-${stamp}.csv`), roads],
    [path.join(processedDir, 'latest-road-segments.csv'), roads]
  ];
  await Promise.all(files.map(([file, content]) => fs.promises.writeFile(file, content, 'utf8')));
  return {
    raw_json: '/data/raw/latest-kakao-routes.json',
    normalized_json: '/data/processed/latest-routes.json',
    routes_geojson: '/data/processed/latest-routes.geojson',
    summary_csv: '/data/processed/latest-route-summary.csv',
    road_segments_csv: '/data/processed/latest-road-segments.csv'
  };
}

async function requestKakaoRoute(profile, coordinates, restKey) {
  const kakaoUrl = new URL(kakaoDirectionsUrl);
  kakaoUrl.searchParams.set('origin', `${coordinates.startLng},${coordinates.startLat}`);
  kakaoUrl.searchParams.set('destination', `${coordinates.endLng},${coordinates.endLat}`);
  kakaoUrl.searchParams.set('priority', profile.priority);
  kakaoUrl.searchParams.set('alternatives', profile.alternatives ? 'true' : 'false');
  if (profile.avoid) kakaoUrl.searchParams.set('avoid', profile.avoid);
  kakaoUrl.searchParams.set('road_details', 'true');
  kakaoUrl.searchParams.set('summary', 'false');
  const upstream = await fetch(kakaoUrl, {
    headers: { Authorization: `KakaoAK ${restKey}`, 'Content-Type': 'application/json' }
  });
  const body = await upstream.json();
  return { profile, ok: upstream.ok, status: upstream.status, body };
}

async function handleRoutes(url, response) {
  const restKey = (process.env.KAKAO_REST_API_KEY || '').trim();
  if (!restKey) {
    sendJson(response, 503, { message: 'REST API 키가 없습니다. 검은 실행 창을 닫고 다시 실행하여 키를 입력하세요.' });
    return;
  }

  const startLng = url.searchParams.get('startLng');
  const startLat = url.searchParams.get('startLat');
  const endLng = url.searchParams.get('endLng');
  const endLat = url.searchParams.get('endLat');
  if (![validCoordinate(startLng, -180, 180), validCoordinate(endLng, -180, 180),
        validCoordinate(startLat, -90, 90), validCoordinate(endLat, -90, 90)].every(Boolean)) {
    sendJson(response, 400, { message: '출발지 또는 도착지 좌표가 올바르지 않습니다.' });
    return;
  }

  try {
    const coordinates = { startLng, startLat, endLng, endLat };
    const results = await Promise.all(profiles.map((profile) => requestKakaoRoute(profile, coordinates, restKey)));
    const successful = results.filter((result) => result.ok && result.body.routes?.[0]);
    if (!successful.length) {
      const first = results[0];
      const detail = first?.body?.msg || first?.body?.message || first?.body?.error || `HTTP ${first?.status || 502}`;
      sendJson(response, 502, { message: `카카오 경로 요청 오류: ${detail}` });
      return;
    }
    const seen = new Set();
    const routes = [];
    let duplicatesRemoved = 0;
    for (const result of successful) {
      for (const [alternativeIndex, route] of result.body.routes.entries()) {
        const signature = routeSignature(route);
        if (seen.has(signature)) {
          duplicatesRemoved += 1;
          continue;
        }
        seen.add(signature);
        routes.push({
          id: `kakao-${routes.length + 1}-${result.profile.id}`,
          request_id: result.profile.id,
          label: alternativeIndex ? `${result.profile.label} 대안 ${alternativeIndex}` : result.profile.label,
          priority: result.profile.priority,
          avoid: result.profile.avoid || '',
          color: result.profile.color,
          data: route
        });
      }
    }
    const bundle = {
      generated_at: new Date().toISOString(),
      source: 'Kakao Mobility Directions API',
      request: coordinates,
      request_count: profiles.length,
      duplicates_removed: duplicatesRemoved,
      routes,
      errors: results.filter((result) => !result.ok).map((result) => ({ priority: result.profile.priority, status: result.status }))
    };
    bundle.files = await saveRouteFiles(bundle, results.map((result) => ({
      request_id: result.profile.id, priority: result.profile.priority,
      avoid: result.profile.avoid || '', status: result.status, body: result.body
    })));
    sendJson(response, 200, bundle);
  } catch (error) {
    sendJson(response, 502, { message: `카카오 서버 연결 실패: ${error.message}` });
  }
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  const pathname = decodeURIComponent(url.pathname);
  if (request.method === 'GET' && (pathname === '/api/routes' || pathname === '/api/route')) {
    await handleRoutes(url, response);
    return;
  }
  const requested = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const file = path.resolve(root, requested);
  if (!file.startsWith(root + path.sep)) {
    response.writeHead(403).end('Forbidden');
    return;
  }
  fs.readFile(file, (error, data) => {
    if (error) {
      response.writeHead(error.code === 'ENOENT' ? 404 : 500).end('Not found');
      return;
    }
    response.writeHead(200, {
      'Content-Type': types[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store'
    });
    response.end(data);
  });
});

server.on('error', (error) => {
  if (error.code === 'EADDRINUSE') {
    console.error(`${port} 포트를 이미 사용 중입니다. 이전 수집기 검은 창을 닫은 뒤 다시 실행하세요.`);
  } else {
    console.error(error.message);
  }
  process.exit(1);
});

server.listen(port, '127.0.0.1', () => {
  console.log(`카카오내비 데이터 수집기: http://localhost:${port}`);
  console.log('종료하려면 Ctrl+C를 누르세요.');
  if (process.env.SAFEROUTE_OPEN_BROWSER !== '0') {
    const opener = spawn('cmd.exe', ['/c', 'start', '', `http://localhost:${port}`], {
      detached: true, stdio: 'ignore', windowsHide: true
    });
    opener.unref();
  }
});
