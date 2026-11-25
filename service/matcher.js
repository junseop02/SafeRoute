'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');


// ==================================================
// 1. 기본 설정
// ==================================================

const CELL = 80;
const PAD = 20;

const COS = Math.cos(36.5 * Math.PI / 180);

const XM = 111320 * COS;
const YM = 111320;

const MAX_MATCH_DISTANCE = 18;
const MAX_SAMPLE_INTERVAL = 25;
const MIN_DIRECTION_SIMILARITY = 0.55;

const MODEL_ID = 'T09';
const POLICY_ID = 'P01';
const SCHEMA_VERSION = 'T09-P01-tiles-v1';

const DATA_DIR = path.join(__dirname, 'data');

const POLICY_PATH = path.join(
    DATA_DIR,
    'national_route_policy_P01_v1_1.json'
);

const INTERFACE_PATH = path.join(
    DATA_DIR,
    'national_service_interface_P01_v1_1.json'
);

const MANIFEST_PATH = path.join(
    DATA_DIR,
    'national_manifest.json'
);


// ==================================================
// 2. 공통 함수
// ==================================================

function requireCondition(condition, message) {

    if (!condition) {
        throw new Error(message);
    }
}


function readJson(file) {

    return JSON.parse(
        fs.readFileSync(file, 'utf8')
    );
}


function clip(value, minimum, maximum) {

    return Math.max(
        minimum,
        Math.min(maximum, value)
    );
}


const point = ([lon, lat]) => [
    lon * XM,
    lat * YM
];


const key = (x, y) => `${x},${y}`;


// ==================================================
// 3. 정책 및 인터페이스 검증
// ==================================================

function loadConfiguration() {

    const policy = readJson(POLICY_PATH);

    const manifest = readJson(MANIFEST_PATH);

    const serviceInterface = readJson(
        INTERFACE_PATH
    );

    requireCondition(
        policy.model_id === MODEL_ID,
        'Invalid policy model ID'
    );

    requireCondition(
        policy.policy_id === POLICY_ID,
        'Invalid policy ID'
    );

    requireCondition(
        policy.policy_selected === true,
        'Route policy is not selected'
    );

    requireCondition(
        policy.normalization === 'raw',
        'Invalid score normalization'
    );

    requireCondition(
        policy.aggregation ===
        'matched_length_weighted_mean',
        'Invalid aggregation method'
    );

    requireCondition(
        policy.weights.risk === 0.8 &&
        policy.weights.narrow === 0.1 &&
        policy.weights.intersection === 0.1,
        'Invalid P01 weights'
    );

    requireCondition(
        policy.minimum_coverage === 0.8,
        'Invalid minimum coverage'
    );

    requireCondition(
        policy.wide_road_min_lanes === 2,
        'Invalid wide road threshold'
    );

    requireCondition(
        policy.intersection_reference === 100,
        'Invalid intersection reference'
    );

    requireCondition(
        policy.classification_threshold_used_for_ranking
        === false,
        'Classification threshold must not be used'
    );

    requireCondition(
        manifest.model_id === MODEL_ID &&
        manifest.policy_id === POLICY_ID,
        'Invalid manifest model or policy'
    );

    requireCondition(
        manifest.schema_version === SCHEMA_VERSION,
        'Unsupported tile schema'
    );

    requireCondition(
        manifest.score_range[0] === 0 &&
        manifest.score_range[1] === 1,
        'Invalid tile score range'
    );

    requireCondition(
        manifest.coordinate_crs === 'EPSG:4326',
        'Invalid coordinate CRS'
    );

    requireCondition(
        manifest.policy_applied_to_tiles === false,
        'P01 must not be applied twice'
    );

    requireCondition(
        manifest.link_stats.scored_links === 1562356,
        'Unexpected national link count'
    );

    requireCondition(
        serviceInterface.model_id === MODEL_ID &&
        serviceInterface.policy_id === POLICY_ID,
        'Invalid service interface'
    );

    requireCondition(
        serviceInterface.route_response.model_score.meaning
        === '100_times_P01_burden',
        'Invalid model score meaning'
    );

    return {
        policy,
        manifest,
        serviceInterface
    };
}


const CONFIG = loadConfiguration();

const POLICY = CONFIG.policy;

const MIN_COVERAGE = POLICY.minimum_coverage;

const WIDE_ROAD_MIN_LANES =
    POLICY.wide_road_min_lanes;

const INTERSECTION_REFERENCE =
    POLICY.intersection_reference;


// ==================================================
// 4. 타일 데이터 읽기
// ==================================================

function readRows(file) {

    const data = fs.readFileSync(file);

    if (file.endsWith('.gz')) {

        return zlib
            .gunzipSync(data)
            .toString('utf8')
            .split('\n')
            .filter(Boolean)
            .map(line => JSON.parse(line));
    }

    return JSON.parse(
        data.toString('utf8')
    );
}


// ==================================================
// 5. 신규 타일 행 검증
// ==================================================

function validateRow(row) {

    requireCondition(
        Array.isArray(row) &&
        row.length === 7,
        'Invalid T09 tile row schema'
    );

    const [
        id,
        coords,
        score,
        lanes,
        intersections,
        accidentDistance,
        partIndex
    ] = row;

    requireCondition(
        typeof id === 'string' &&
        id.length > 0,
        'Invalid link ID'
    );

    requireCondition(
        Array.isArray(coords) &&
        coords.length >= 2,
        `Invalid geometry: ${id}`
    );

    for (const coordinate of coords) {

        requireCondition(
            Array.isArray(coordinate) &&
            coordinate.length === 2 &&
            coordinate.every(Number.isFinite) &&
            coordinate[0] >= 124 &&
            coordinate[0] <= 132 &&
            coordinate[1] >= 33 &&
            coordinate[1] <= 39,
            `Invalid coordinate: ${id}`
        );
    }

    requireCondition(
        Number.isFinite(score) &&
        score >= 0 &&
        score <= 1,
        `Invalid T09 score: ${id}`
    );

    requireCondition(
        Number.isInteger(lanes) &&
        lanes >= 1,
        `Invalid lanes: ${id}`
    );

    requireCondition(
        Number.isInteger(intersections) &&
        intersections >= 0,
        `Invalid intersections: ${id}`
    );

    requireCondition(
        accidentDistance === null ||
        (
            Number.isFinite(accidentDistance) &&
            accidentDistance >= 0
        ),
        `Invalid accident distance: ${id}`
    );

    requireCondition(
        Number.isInteger(partIndex) &&
        partIndex >= 0,
        `Invalid part index: ${id}`
    );

    return {
        id,
        coords,
        score,
        lanes,
        intersections,
        accidentDistance,
        partIndex
    };
}


// ==================================================
// 6. 공간 인덱스 생성
// ==================================================

function makeIndex(files) {

    const grid = new Map();

    const seenParts = new Set();

    const seenLinks = new Set();

    let segments = 0;

    let duplicateParts = 0;

    let zeroLengthSegments = 0;

    for (const file of files) {

        if (!fs.existsSync(file)) {
            continue;
        }

        const rows = readRows(file);

        for (const row of rows) {

            const {
                id,
                coords,
                score,
                lanes,
                intersections,
                accidentDistance,
                partIndex
            } = validateRow(row);

            // 복합 선형을 보존하면서 타일 간 중복 제거
            const partKey = `${id}:${partIndex}`;

            if (seenParts.has(partKey)) {

                duplicateParts++;

                continue;
            }

            seenParts.add(partKey);

            seenLinks.add(id);

            for (
                let i = 1;
                i < coords.length;
                i++
            ) {

                const a = point(coords[i - 1]);

                const b = point(coords[i]);

                requireCondition(
                    a.every(Number.isFinite) &&
                    b.every(Number.isFinite),
                    `Invalid projected coordinate: ${id}`
                );

                // 반올림으로 연속 좌표가 같아진 구간 제외
                if (
                    a[0] === b[0] &&
                    a[1] === b[1]
                ) {

                    zeroLengthSegments++;

                    continue;
                }

                const seg = {
                    id,
                    partIndex,
                    a,
                    b,
                    score,
                    lanes,
                    intersections,
                    accidentDistance
                };

                segments++;

                const x0 = Math.floor(
                    (
                        Math.min(a[0], b[0]) - PAD
                    ) / CELL
                );

                const x1 = Math.floor(
                    (
                        Math.max(a[0], b[0]) + PAD
                    ) / CELL
                );

                const y0 = Math.floor(
                    (
                        Math.min(a[1], b[1]) - PAD
                    ) / CELL
                );

                const y1 = Math.floor(
                    (
                        Math.max(a[1], b[1]) + PAD
                    ) / CELL
                );

                for (
                    let x = x0;
                    x <= x1;
                    x++
                ) {

                    for (
                        let y = y0;
                        y <= y1;
                        y++
                    ) {

                        const k = key(x, y);

                        if (!grid.has(k)) {
                            grid.set(k, []);
                        }

                        grid.get(k).push(seg);
                    }
                }
            }
        }
    }

    return {
        grid,
        segments,
        uniqueLinks: seenLinks.size,
        uniqueParts: seenParts.size,
        duplicateParts,
        zeroLengthSegments
    };
}


// ==================================================
// 7. 점과 도로 선분 사이의 거리
// ==================================================

function distanceToSegment(p, seg) {

    const [ax, ay] = seg.a;

    const [bx, by] = seg.b;

    const vx = bx - ax;

    const vy = by - ay;

    const denominator = vx * vx + vy * vy;

    const t = clip(
        (
            (p[0] - ax) * vx +
            (p[1] - ay) * vy
        ) / (denominator || 1),
        0,
        1
    );

    return Math.hypot(
        p[0] - ax - t * vx,
        p[1] - ay - t * vy
    );
}


// ==================================================
// 8. 가장 가까운 도로 탐색
// ==================================================

function nearest(index, p, dx, dy) {

    const bucket = index.grid.get(
        key(
            Math.floor(p[0] / CELL),
            Math.floor(p[1] / CELL)
        )
    ) || [];

    let best = null;

    let bestD = Infinity;

    const length = Math.hypot(dx, dy);

    for (const seg of bucket) {

        const sx = seg.b[0] - seg.a[0];

        const sy = seg.b[1] - seg.a[1];

        const sl = Math.hypot(sx, sy);

        // 기존 방향 유사도 조건 유지
        if (
            length > 2 &&
            sl > 2 &&
            Math.abs(
                (dx * sx + dy * sy) /
                (length * sl)
            ) < MIN_DIRECTION_SIMILARITY
        ) {

            continue;
        }

        const distance = distanceToSegment(
            p,
            seg
        );

        if (distance < bestD) {

            bestD = distance;

            best = seg;
        }
    }

    return bestD <= MAX_MATCH_DISTANCE
        ? best
        : null;
}


// ==================================================
// 9. P01 정책 계산
// ==================================================

function calculateP01(
    rawRisk,
    wideRatio,
    meanIntersections
) {

    requireCondition(
        Number.isFinite(rawRisk) &&
        rawRisk >= 0 &&
        rawRisk <= 1,
        'Invalid route raw risk'
    );

    requireCondition(
        Number.isFinite(wideRatio) &&
        wideRatio >= 0 &&
        wideRatio <= 1,
        'Invalid route wide road ratio'
    );

    requireCondition(
        Number.isFinite(meanIntersections) &&
        meanIntersections >= 0,
        'Invalid route intersection mean'
    );

    const R = rawRisk;

    const N = 1 - wideRatio;

    const I = clip(
        meanIntersections /
        INTERSECTION_REFERENCE,
        0,
        1
    );

    const burden = clip(
        POLICY.weights.risk * R +
        POLICY.weights.narrow * N +
        POLICY.weights.intersection * I,
        0,
        1
    );

    return {
        rawRisk: R,
        narrowBurden: N,
        intersectionBurden: I,
        burden,
        modelScore: 100 * burden,
        displayScore: 100 * (1 - burden)
    };
}


// ==================================================
// 10. 경로 매칭 및 점수 계산
// ==================================================

function scoreRoads(index, roads) {

    let total = 0;

    let matched = 0;

    let riskSum = 0;

    let wide = 0;

    let intersections = 0;

    let near = 0;

    let knownDistanceLength = 0;

    const ids = new Set();

    for (const road of roads) {

        const v = road.vertexes || [];

        for (
            let i = 2;
            i + 1 < v.length;
            i += 2
        ) {

            const a = point([
                Number(v[i - 2]),
                Number(v[i - 1])
            ]);

            const b = point([
                Number(v[i]),
                Number(v[i + 1])
            ]);

            const dx = b[0] - a[0];

            const dy = b[1] - a[1];

            const len = Math.hypot(dx, dy);

            // 기존 유효 경로 구간 조건 유지
            if (
                !Number.isFinite(len) ||
                len <= 0 ||
                len > 20000
            ) {

                continue;
            }

            const steps = Math.ceil(
                len / MAX_SAMPLE_INTERVAL
            );

            const step = len / steps;

            for (
                let j = 0;
                j < steps;
                j++
            ) {

                const t = (j + 0.5) / steps;

                const p = [
                    a[0] + t * dx,
                    a[1] + t * dy
                ];

                const found = nearest(
                    index,
                    p,
                    dx,
                    dy
                );

                total += step;

                if (!found) {
                    continue;
                }

                matched += step;

                // T09 원본 점수 길이 가중 집계
                riskSum += step * found.score;

                wide += step * (
                    found.lanes >=
                    WIDE_ROAD_MIN_LANES
                );

                intersections += (
                    step * found.intersections
                );

                // 기존 사고다발지 근접 정보 유지
                if (
                    Number.isFinite(
                        found.accidentDistance
                    )
                ) {

                    knownDistanceLength += step;

                    near += step * (
                        found.accidentDistance <= 150
                    );
                }

                // 복합 선형도 실제 링크 ID 기준 집계
                ids.add(found.id);
            }
        }
    }

    const coverage = total > 0
        ? clip(matched / total, 0, 1)
        : 0;

    const rawRisk = matched > 0
        ? clip(riskSum / matched, 0, 1)
        : null;

    const wideRatio = matched > 0
        ? clip(wide / matched, 0, 1)
        : null;

    const meanIntersections = matched > 0
        ? intersections / matched
        : null;

    const hotspotRatio = (
        matched > 0 &&
        knownDistanceLength === matched
    )
        ? clip(near / matched, 0, 1)
        : null;

    let result = null;

    // 매칭률 80% 미만이면 점수 미제공
    if (
        matched > 0 &&
        coverage >= MIN_COVERAGE
    ) {

        result = calculateP01(
            rawRisk,
            wideRatio,
            meanIntersections
        );
    }

    return {

        // 기존 API 응답 필드 유지
        model_score: result
            ? result.modelScore
            : null,

        model_coverage: coverage,

        model_matched_m: matched,

        geometry_length_m: total,

        model_wide_ratio: wideRatio,

        model_intersections: meanIntersections,

        model_hotspot_ratio: hotspotRatio,

        matched_link_count: ids.size,

        // 신규 인터페이스 후보 필드
        model_version: MODEL_ID,

        policy_version: POLICY_ID,

        model_raw_risk: result
            ? result.rawRisk
            : null,

        model_narrow_burden: result
            ? result.narrowBurden
            : null,

        model_intersection_burden: result
            ? result.intersectionBurden
            : null
    };
}


// ==================================================
// 11. 전체 경로 점수 계산
// ==================================================

function scoreRoute(index, route) {

    const roads = (
        route.sections || []
    ).flatMap(
        section => section.roads || []
    );

    return scoreRoads(index, roads);
}


// ==================================================
// 12. 경로에 필요한 타일 탐색
// ==================================================

function routeTiles(routes) {

    const tiles = new Set();

    for (const route of routes) {

        for (
            const section of route.sections || []
        ) {

            for (
                const road of section.roads || []
            ) {

                const v = road.vertexes || [];

                for (
                    let i = 2;
                    i + 1 < v.length;
                    i += 2
                ) {

                    const lon1 = Number(v[i - 2]);

                    const lat1 = Number(v[i - 1]);

                    const lon2 = Number(v[i]);

                    const lat2 = Number(v[i + 1]);

                    if (
                        ![
                            lon1,
                            lat1,
                            lon2,
                            lat2
                        ].every(Number.isFinite)
                    ) {

                        continue;
                    }

                    const x0 = Math.floor(
                        Math.min(lon1, lon2) * 10
                    );

                    const x1 = Math.floor(
                        Math.max(lon1, lon2) * 10
                    );

                    const y0 = Math.floor(
                        Math.min(lat1, lat2) * 10
                    );

                    const y1 = Math.floor(
                        Math.max(lat1, lat2) * 10
                    );

                    // 기존 타일 탐색 조건 유지
                    if (
                        x1 - x0 > 3 ||
                        y1 - y0 > 3
                    ) {

                        continue;
                    }

                    for (
                        let x = x0;
                        x <= x1;
                        x++
                    ) {

                        for (
                            let y = y0;
                            y <= y1;
                            y++
                        ) {

                            tiles.add(`${x}_${y}`);
                        }
                    }
                }
            }
        }
    }

    return [...tiles];
}


// ==================================================
// 13. 전국 타일 인덱스 제공
// ==================================================

function makeIndexProvider(tileDir, regional) {

    requireCondition(
        fs.existsSync(tileDir),
        'National T09 tile directory not found'
    );

    const tileFiles = fs.readdirSync(tileDir)
        .filter(
            filename =>
                filename.endsWith('.jsonl.gz')
        );

    const expectedCount =
        CONFIG.manifest.tile_stats.tile_count;

    requireCondition(
        tileFiles.length === expectedCount,
        'National T09 tile count mismatch'
    );

    const availableTiles = new Set(tileFiles);

    const manifestTiles = CONFIG.manifest.tiles;

    requireCondition(
        Object.keys(manifestTiles).length
        === expectedCount,
        'Manifest tile count mismatch'
    );

    for (
        const metadata of Object.values(manifestTiles)
    ) {

        requireCondition(
            availableTiles.has(metadata.file),
            `Missing tile: ${metadata.file}`
        );
    }

    return {

        national: true,

        forRoutes(routes) {

            const files = routeTiles(routes)

                .map(
                    tile => `${tile}.jsonl.gz`
                )

                .filter(
                    filename =>
                        availableTiles.has(filename)
                )

                .map(
                    filename =>
                        path.join(tileDir, filename)
                );

            return makeIndex(files);
        },

        status() {

            return {
                national_tiles: tileFiles.length,

                national_links:
                    CONFIG.manifest.link_stats
                        .scored_links,

                model_version: MODEL_ID,

                policy_version: POLICY_ID,

                tile_schema_version: SCHEMA_VERSION
            };
        }
    };
}


// ==================================================
// 14. 외부 모듈 제공
// ==================================================

module.exports = {

    makeIndex,

    makeIndexProvider,

    scoreRoads,

    scoreRoute,

    routeTiles,

    point,

    calculateP01
};