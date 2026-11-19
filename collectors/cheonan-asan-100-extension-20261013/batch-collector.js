'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { once } = require('events');

const root = __dirname;
const dataRoot = path.resolve(process.env.KAKAO_BATCH_DATA_DIR || path.join(root, 'data'));
const locationFile = path.resolve(process.env.KAKAO_BATCH_LOCATIONS || path.join(root, 'locations.json'));
const settingsFile = path.resolve(process.env.KAKAO_BATCH_SETTINGS || path.join(root, 'settings.json'));
const directionsUrl = process.env.KAKAO_NAVI_BASE_URL || 'https://apis-navi.kakaomobility.com/v1/directions';
const localSearchUrl = process.env.KAKAO_LOCAL_BASE_URL || 'https://dapi.kakao.com/v2/local/search/keyword.json';
const restKey = String(process.env.KAKAO_REST_API_KEY || '').trim();
const settings = JSON.parse(fs.readFileSync(settingsFile, 'utf8'));
const seeds = JSON.parse(fs.readFileSync(locationFile, 'utf8'));
const requestDelayMs = Number(process.env.KAKAO_BATCH_DELAY_MS ?? settings.request_delay_ms ?? 300);
const maxRequests = Number(process.env.KAKAO_BATCH_MAX_REQUESTS ?? settings.max_requests_per_run ?? 0);
let stopRequested = false;

process.on('SIGINT', () => {
  stopRequested = true;
  console.log('\n중단 요청을 받았습니다. 현재 파일을 저장한 뒤 종료합니다.');
});

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, milliseconds)));
}

function csvCell(value) {
  const text = String(value ?? '');
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function csvRow(values) {
  return values.map(csvCell).join(',') + '\n';
}

async function ensureDirectories() {
  await Promise.all([
    'raw/geocoding', 'raw/routes', 'processed/routes', 'errors'
  ].map((directory) => fs.promises.mkdir(path.join(dataRoot, directory), { recursive: true })));
}

async function writeAtomic(file, content) {
  await fs.promises.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  await fs.promises.writeFile(temporary, content, 'utf8');
  await fs.promises.rm(file, { force: true });
  await fs.promises.rename(temporary, file);
}

async function fetchJson(url, options = {}, attempts = 3) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, options);
      const text = await response.text();
      let body;
      try { body = text ? JSON.parse(text) : {}; }
      catch { body = { raw_text: text }; }
      if (response.ok) return { ok: true, status: response.status, body, headers: response.headers };
      const error = new Error(body.message || body.msg || body.error || `HTTP ${response.status}`);
      error.status = response.status;
      error.body = body;
      if (![429, 500, 502, 503, 504].includes(response.status) || attempt === attempts) throw error;
      const retryAfter = Number(response.headers.get('retry-after'));
      await sleep(Number.isFinite(retryAfter) ? retryAfter * 1000 : attempt * 1500);
      lastError = error;
    } catch (error) {
      lastError = error;
      if (attempt === attempts || [400, 401, 403].includes(error.status)) throw error;
      await sleep(attempt * 1500);
    }
  }
  throw lastError;
}

function authorizedHeaders() {
  return { Authorization: `KakaoAK ${restKey}`, 'Content-Type': 'application/json' };
}

async function searchLocation(seed, query) {
  const url = new URL(localSearchUrl);
  url.searchParams.set('query', query);
  url.searchParams.set('size', '15');
  const result = await fetchJson(url, { headers: authorizedHeaders() });
  const documents = Array.isArray(result.body.documents) ? result.body.documents : [];
  const inCity = documents.filter((document) => {
    const address = `${document.address_name || ''} ${document.road_address_name || ''}`;
    return address.includes(seed.city);
  });
  const named = inCity.find((document) => String(document.place_name || '').includes(seed.name));
  return { body: result.body, document: named || inCity[0] || null };
}

async function resolvePoints() {
  const cacheFile = path.join(dataRoot, 'points.json');
  let cached = [];
  try { cached = JSON.parse(await fs.promises.readFile(cacheFile, 'utf8')); }
  catch { cached = []; }
  const byId = new Map(cached.map((point) => [point.id, point]));
  const errors = [];

  for (const [index, seed] of seeds.entries()) {
    if (byId.has(seed.id)) continue;
    if (stopRequested) break;
    console.log(`[지점 ${index + 1}/${seeds.length}] ${seed.city} ${seed.name}`);
    try {
      const queries = [seed.query, `충남 ${seed.city} ${seed.name}`];
      let match = null;
      let lastBody = null;
      for (const query of queries) {
        const result = await searchLocation(seed, query);
        lastBody = result.body;
        await writeAtomic(path.join(dataRoot, 'raw', 'geocoding', `${seed.id}.json`), JSON.stringify({
          collected_at: new Date().toISOString(), seed, query, response: result.body
        }, null, 2));
        if (result.document) {
          match = result.document;
          break;
        }
        await sleep(requestDelayMs);
      }
      if (!match) throw new Error(`카카오 장소 검색에서 ${seed.city} 주소를 찾지 못했습니다.`);
      const point = {
        ...seed,
        place_name: match.place_name || seed.name,
        address_name: match.road_address_name || match.address_name || '',
        longitude: Number(match.x),
        latitude: Number(match.y),
        kakao_place_id: match.id || ''
      };
      if (!Number.isFinite(point.longitude) || !Number.isFinite(point.latitude)) {
        throw new Error('검색 결과의 좌표가 올바르지 않습니다.');
      }
      byId.set(seed.id, point);
      await writeAtomic(cacheFile, JSON.stringify([...byId.values()], null, 2));
      await sleep(requestDelayMs);
    } catch (error) {
      errors.push({ id: seed.id, query: seed.query, message: error.message, status: error.status || '' });
      console.error(`  실패: ${error.message}`);
      if ([401, 403, 429].includes(error.status)) throw error;
    }
  }

  const points = seeds.map((seed) => byId.get(seed.id)).filter(Boolean);
  await writePointFiles(points);
  if (errors.length) {
    await writeAtomic(path.join(dataRoot, 'errors', 'geocoding-errors.json'), JSON.stringify(errors, null, 2));
  }
  return points;
}

async function writePointFiles(points) {
  let csv = '\uFEFF' + csvRow(['point_id','city','district','name','place_name','address_name','longitude','latitude','kakao_place_id']);
  const features = [];
  for (const point of points) {
    csv += csvRow([point.id, point.city, point.district, point.name, point.place_name, point.address_name,
      point.longitude, point.latitude, point.kakao_place_id]);
    features.push({
      type: 'Feature',
      properties: { point_id: point.id, city: point.city, district: point.district, name: point.name, address_name: point.address_name },
      geometry: { type: 'Point', coordinates: [point.longitude, point.latitude] }
    });
  }
  await Promise.all([
    writeAtomic(path.join(dataRoot, 'points.csv'), csv),
    writeAtomic(path.join(dataRoot, 'points.geojson'), JSON.stringify({ type: 'FeatureCollection', features }, null, 2))
  ]);
}

const selectedPairs = new Set(JSON.parse(fs.readFileSync(path.join(root,'selected_pairs.json'),'utf8')).map(p=>p.join('|')));
function buildTasks(points) {
  const profiles = Array.isArray(settings.directions_profiles) ? settings.directions_profiles : [];
  const tasks = [];
  for (const origin of points) {
    for (const destination of points) {
      if (origin.id === destination.id) continue;
      if (!selectedPairs.has(origin.id+'|'+destination.id)) continue;
      for (const profile of profiles) {
        tasks.push({
          id: `${origin.id}--${destination.id}--${profile.id}`,
          origin,
          destination,
          profile
        });
      }
    }
  }
  return tasks;
}

async function writeRequestPlan(tasks) {
  const completedDirectory = path.join(dataRoot, 'processed', 'routes');
  const errorDirectory = path.join(dataRoot, 'errors');
  let csv = '\uFEFF' + csvRow([
    'request_id','origin_id','origin_city','origin_name','origin_longitude','origin_latitude',
    'destination_id','destination_city','destination_name','destination_longitude','destination_latitude',
    'profile','priority','alternatives','avoid','status'
  ]);
  for (const task of tasks) {
    const completed = fs.existsSync(path.join(completedDirectory, `${task.id}.json`));
    const failed = fs.existsSync(path.join(errorDirectory, `${task.id}.json`));
    csv += csvRow([
      task.id, task.origin.id, task.origin.city, task.origin.name, task.origin.longitude, task.origin.latitude,
      task.destination.id, task.destination.city, task.destination.name, task.destination.longitude, task.destination.latitude,
      task.profile.label, task.profile.priority, Boolean(task.profile.alternatives), task.profile.avoid || '',
      completed ? 'completed' : failed ? 'retry' : 'pending'
    ]);
  }
  await writeAtomic(path.join(dataRoot, 'route_requests.csv'), csv);
}

function routeSignature(route) {
  const hash = crypto.createHash('sha1');
  for (const section of route.sections || []) {
    for (const road of section.roads || []) {
      hash.update((road.vertexes || []).map((value) => Number(value).toFixed(5)).join(','));
      hash.update('|');
    }
  }
  return hash.digest('hex');
}

async function collectTask(task) {
  const url = new URL(directionsUrl);
  url.searchParams.set('origin', `${task.origin.longitude},${task.origin.latitude},name=${task.origin.name}`);
  url.searchParams.set('destination', `${task.destination.longitude},${task.destination.latitude},name=${task.destination.name}`);
  url.searchParams.set('priority', task.profile.priority || 'RECOMMEND');
  url.searchParams.set('alternatives', task.profile.alternatives ? 'true' : 'false');
  url.searchParams.set('road_details', 'true');
  url.searchParams.set('summary', 'false');
  if (task.profile.avoid) url.searchParams.set('avoid', task.profile.avoid);

  const collectedAt = new Date().toISOString();
  let result;
  try {
    result = await fetchJson(url, { headers: authorizedHeaders() });
  } catch (error) {
    const failure = {
      request_id: task.id, collected_at: collectedAt, status: error.status || '', message: error.message,
      origin: task.origin, destination: task.destination, profile: task.profile, response: error.body || null
    };
    await writeAtomic(path.join(dataRoot, 'errors', `${task.id}.json`), JSON.stringify(failure, null, 2));
    throw error;
  }

  const rawRecord = {
    request_id: task.id, collected_at: collectedAt,
    origin: task.origin, destination: task.destination, profile: task.profile,
    response: result.body
  };
  await writeAtomic(path.join(dataRoot, 'raw', 'routes', `${task.id}.json`), JSON.stringify(rawRecord, null, 2));

  const seen = new Set();
  const routes = [];
  let duplicatesRemoved = 0;
  for (const [index, route] of (result.body.routes || []).entries()) {
    if (Number(route.result_code ?? 0) !== 0 || !(route.sections || []).length) continue;
    const signature = routeSignature(route);
    if (seen.has(signature)) {
      duplicatesRemoved += 1;
      continue;
    }
    seen.add(signature);
    routes.push({
      route_id: `${task.id}--route-${index + 1}`,
      alternative_index: index,
      geometry_signature: signature,
      data: route
    });
  }
  const normalized = {
    request_id: task.id,
    collected_at: collectedAt,
    source: 'Kakao Mobility Directions API',
    origin: task.origin,
    destination: task.destination,
    profile: task.profile,
    trans_id: result.body.trans_id || '',
    duplicates_removed: duplicatesRemoved,
    api_results: (result.body.routes || []).map((route) => ({ result_code: route.result_code, result_msg: route.result_msg })),
    routes
  };
  await writeAtomic(path.join(dataRoot, 'processed', 'routes', `${task.id}.json`), JSON.stringify(normalized, null, 2));
  await fs.promises.rm(path.join(dataRoot, 'errors', `${task.id}.json`), { force: true });
  return normalized;
}

async function writeStream(stream, text) {
  if (!stream.write(text)) await once(stream, 'drain');
}

async function closeStream(stream) {
  stream.end();
  await once(stream, 'finish');
}

async function buildAggregates(totalTasks) {
  const routeDirectory = path.join(dataRoot, 'processed', 'routes');
  const files = (await fs.promises.readdir(routeDirectory)).filter((name) => name.endsWith('.json')).sort();
  const output = path.join(dataRoot, 'processed');
  const summaryPath = path.join(output, 'route_summary.csv');
  const roadsPath = path.join(output, 'road_segments.csv');
  const guidesPath = path.join(output, 'guides.csv');
  const ndjsonPath = path.join(output, 'routes.ndjson');
  const geojsonPath = path.join(output, 'routes.geojson');
  let routeCount = 0;
  let roadCount = 0;
  let guideCount = 0;

  const summary = fs.createWriteStream(`${summaryPath}.${process.pid}.tmp`, { encoding: 'utf8' });
  const roads = fs.createWriteStream(`${roadsPath}.${process.pid}.tmp`, { encoding: 'utf8' });
  const guides = fs.createWriteStream(`${guidesPath}.${process.pid}.tmp`, { encoding: 'utf8' });
  const ndjson = fs.createWriteStream(`${ndjsonPath}.${process.pid}.tmp`, { encoding: 'utf8' });
  const geojson = fs.createWriteStream(`${geojsonPath}.${process.pid}.tmp`, { encoding: 'utf8' });

  await writeStream(summary, '\uFEFF' + csvRow(['request_id','route_id','collected_at','origin_id','origin_city','origin_name','destination_id','destination_city','destination_name','profile','priority','avoid','alternative_index','distance_m','duration_s','fare_taxi_krw','fare_toll_krw']));
  await writeStream(roads, '\uFEFF' + csvRow(['request_id','route_id','origin_id','destination_id','profile','section_index','road_index','road_name','distance_m','duration_s','traffic_speed_kmh','traffic_state']));
  await writeStream(guides, '\uFEFF' + csvRow(['request_id','route_id','section_index','guide_index','name','longitude','latitude','distance_m','duration_s','type','guidance','road_index']));
  await writeStream(geojson, '{"type":"FeatureCollection","features":[');
  let firstFeature = true;

  for (const file of files) {
    const record = JSON.parse(await fs.promises.readFile(path.join(routeDirectory, file), 'utf8'));
    await writeStream(ndjson, JSON.stringify(record) + '\n');
    for (const route of record.routes || []) {
      const data = route.data || {};
      const routeSummary = data.summary || {};
      routeCount += 1;
      await writeStream(summary, csvRow([
        record.request_id, route.route_id, record.collected_at,
        record.origin.id, record.origin.city, record.origin.name,
        record.destination.id, record.destination.city, record.destination.name,
        record.profile.label, record.profile.priority, record.profile.avoid || '', route.alternative_index,
        routeSummary.distance, routeSummary.duration, routeSummary.fare?.taxi, routeSummary.fare?.toll
      ]));
      for (const [sectionIndex, section] of (data.sections || []).entries()) {
        for (const [roadIndex, road] of (section.roads || []).entries()) {
          roadCount += 1;
          await writeStream(roads, csvRow([
            record.request_id, route.route_id, record.origin.id, record.destination.id, record.profile.label,
            sectionIndex, roadIndex, road.name, road.distance, road.duration, road.traffic_speed, road.traffic_state
          ]));
          const coordinates = [];
          for (let index = 0; index < (road.vertexes || []).length; index += 2) {
            coordinates.push([road.vertexes[index], road.vertexes[index + 1]]);
          }
          if (coordinates.length > 1) {
            const feature = {
              type: 'Feature',
              properties: {
                request_id: record.request_id, route_id: route.route_id,
                origin_id: record.origin.id, destination_id: record.destination.id,
                profile: record.profile.label, section_index: sectionIndex, road_index: roadIndex,
                road_name: road.name || '', distance_m: road.distance || 0, duration_s: road.duration || 0,
                traffic_speed_kmh: road.traffic_speed ?? null, traffic_state: road.traffic_state ?? null,
                collected_at: record.collected_at
              },
              geometry: { type: 'LineString', coordinates }
            };
            await writeStream(geojson, `${firstFeature ? '' : ','}${JSON.stringify(feature)}`);
            firstFeature = false;
          }
        }
        for (const [guideIndex, guide] of (section.guides || []).entries()) {
          guideCount += 1;
          await writeStream(guides, csvRow([
            record.request_id, route.route_id, sectionIndex, guideIndex, guide.name,
            guide.x, guide.y, guide.distance, guide.duration, guide.type, guide.guidance, guide.road_index
          ]));
        }
      }
    }
  }
  await writeStream(geojson, ']}');
  await Promise.all([closeStream(summary), closeStream(roads), closeStream(guides), closeStream(ndjson), closeStream(geojson)]);

  for (const destination of [summaryPath, roadsPath, guidesPath, ndjsonPath, geojsonPath]) {
    await fs.promises.rm(destination, { force: true });
    await fs.promises.rename(`${destination}.${process.pid}.tmp`, destination);
  }

  const errorFiles = (await fs.promises.readdir(path.join(dataRoot, 'errors'))).filter((name) => name.endsWith('.json'));
  const manifest = {
    generated_at: new Date().toISOString(),
    region: ['충청남도 천안시', '충청남도 아산시'],
    point_definition: '읍·면·동 행정복지센터 대표점',
    directed_pairs: totalTasks / Math.max(1, settings.directions_profiles.length),
    configured_requests: totalTasks,
    completed_requests: files.length,
    pending_requests: Math.max(0, totalTasks - files.length),
    error_files: errorFiles.length,
    unique_routes: routeCount,
    road_segments: roadCount,
    guides: guideCount,
    live_traffic_notice: 'traffic_speed와 traffic_state는 collected_at 시점의 스냅샷입니다.'
  };
  await writeAtomic(path.join(output, 'collection_manifest.json'), JSON.stringify(manifest, null, 2));
  return manifest;
}

async function main() {
  if (!restKey) throw new Error('KAKAO_REST_API_KEY가 없습니다. START_천안아산_자동수집.cmd를 실행하세요.');
  await ensureDirectories();
  console.log('천안·아산 카카오내비 자동 수집기');
  console.log(`대표점은 ${seeds.length}개 기존 대표점·음식점·시설이며 출발·도착 방향을 구분합니다.`);
  const points = await resolvePoints();
  if (points.length < 2) throw new Error('경로 요청에 필요한 대표점 좌표를 충분히 찾지 못했습니다.');
  const tasks = buildTasks(points);
  await writeRequestPlan(tasks);
  const completedDirectory = path.join(dataRoot, 'processed', 'routes');
  const completed = new Set((await fs.promises.readdir(completedDirectory)).filter((name) => name.endsWith('.json')).map((name) => path.basename(name, '.json')));
  const remaining = tasks.filter((task) => !completed.has(task.id));
  const runTasks = maxRequests > 0 ? remaining.slice(0, maxRequests) : remaining;
  console.log(`대표점 ${points.length}개 / 전체 요청 ${tasks.length}건 / 완료 ${completed.size}건 / 이번 실행 ${runTasks.length}건`);

  let succeeded = 0;
  for (const [index, task] of runTasks.entries()) {
    if (stopRequested) break;
    const started = Date.now();
    process.stdout.write(`[경로 ${index + 1}/${runTasks.length}] ${task.origin.name} → ${task.destination.name} (${task.profile.label}) ... `);
    try {
      const record = await collectTask(task);
      succeeded += 1;
      console.log(`${record.routes.length}개 저장`);
    } catch (error) {
      console.log(`실패: ${error.message}`);
      if ([401, 403, 429].includes(error.status)) {
        console.log('인증 또는 쿼터 오류이므로 안전하게 중단합니다. 다음 실행에서 이어집니다.');
        break;
      }
    }
    const elapsed = Date.now() - started;
    await sleep(Math.max(0, requestDelayMs - elapsed));
  }

  await writeRequestPlan(tasks);
  const manifest = await buildAggregates(tasks.length);
  console.log('');
  console.log(`이번 실행 성공: ${succeeded}건`);
  console.log(`누적 완료: ${manifest.completed_requests}/${manifest.configured_requests}건`);
  console.log(`고유 경로: ${manifest.unique_routes}개 / 도로 구간: ${manifest.road_segments}개`);
  console.log(`결과 폴더: ${path.join(dataRoot, 'processed')}`);
  if (manifest.pending_requests > 0) console.log('남은 요청은 같은 실행 파일을 다시 실행하면 자동으로 이어서 수집합니다.');
}

main().catch((error) => {
  console.error(`오류: ${error.message}`);
  process.exitCode = 1;
});
