'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { scoreRoute } = require('./matcher');


// ==================================================
// 1. 기본 설정
// ==================================================

const BASE_LOCAL =
    'https://dapi.kakao.com/v2/local/search';

const BASE_NAVI =
    'https://apis-navi.kakaomobility.com/v1/directions';

const POLICY_PATH = path.join(
    __dirname,
    'data',
    'national_route_policy_P01_v1_1.json'
);

const POLICY = JSON.parse(
    fs.readFileSync(POLICY_PATH, 'utf8')
);

const MODEL_ID = 'T09';
const POLICY_ID = 'P01';

const MIN_COVERAGE = POLICY.minimum_coverage;

const MIN_CANDIDATES =
    POLICY.minimum_distinct_candidates;

const SEARCH_TARGET_CANDIDATES =
    POLICY.search_target_candidates;

// 기존 서비스 기본값 유지
const DEFAULT_CAP = 2;


// ==================================================
// 2. 정책 검증
// ==================================================

function requireCondition(condition, message) {

    if (!condition) {
        throw new Error(message);
    }
}


requireCondition(
    POLICY.model_id === MODEL_ID,
    'Invalid model ID'
);

requireCondition(
    POLICY.policy_id === POLICY_ID,
    'Invalid policy ID'
);

requireCondition(
    POLICY.policy_selected === true,
    'Route policy is not selected'
);

requireCondition(
    MIN_COVERAGE === 0.8,
    'Invalid minimum coverage'
);

requireCondition(
    MIN_CANDIDATES === 2 &&
    SEARCH_TARGET_CANDIDATES === 3,
    'Invalid minimum candidate count'
);

requireCondition(
    POLICY.detour_basis ===
    'shortest_candidate_distance',
    'Invalid detour basis'
);

requireCondition(
    JSON.stringify(POLICY.ranking) ===
    JSON.stringify([
        'burden_ascending',
        'distance_ascending',
        'duration_ascending',
        'route_id_ascending'
    ]),
    'Invalid ranking policy'
);

requireCondition(
    POLICY.offline_burden_rounding_decimals === 12,
    'Invalid burden rounding precision'
);

requireCondition(
    POLICY.insufficient_candidates_policy ===
    'no_recommendation',
    'Invalid insufficient candidate policy'
);

requireCondition(
    POLICY.classification_threshold_used_for_ranking
    === false,
    'Classification threshold must not be used'
);


// ==================================================
// 3. 카카오 API 요청
// ==================================================

async function fetchJson(
    url,
    key,
    fetcher = fetch
) {

    const response = await fetcher(url, {

        headers: {
            Authorization: `KakaoAK ${key}`
        },

        signal: AbortSignal.timeout(20000)
    });

    const body = await response.json()
        .catch(() => ({}));

    if (!response.ok) {

        const error = new Error(

            response.status === 401 ||
            response.status === 403

                ? '카카오 API 키 또는 서비스 사용 권한을 확인하세요.'

                : `카카오 API 응답 오류 (${response.status})`
        );

        error.status = response.status;

        throw error;
    }

    return body;
}


// ==================================================
// 4. 장소 검색
// ==================================================

async function searchPlaces(
    q,
    key,
    fetcher = fetch
) {

    const query = String(q || '').trim();

    if (
        query.length < 2 ||
        query.length > 80
    ) {

        throw new Error(
            '검색어는 2~80자로 입력하세요.'
        );
    }

    const request = async type => {

        const url = new URL(
            `${BASE_LOCAL}/${type}.json`
        );

        url.searchParams.set(
            'query',
            query
        );

        if (type === 'keyword') {

            url.searchParams.set(
                'size',
                '10'
            );
        }

        return fetchJson(
            url,
            key,
            fetcher
        );
    };

    let body = await request('keyword');

    if (!body.documents?.length) {

        body = await request('address');
    }

    return (body.documents || [])

        .slice(0, 10)

        .map(p => ({

            id: String(
                p.id ||
                p.address_name ||
                p.road_address_name
            ),

            name:
                p.place_name ||
                p.road_address?.building_name ||
                p.address_name ||
                p.road_address_name,

            address:
                p.road_address_name ||
                p.address_name ||
                p.road_address?.address_name ||
                '',

            x: Number(p.x),

            y: Number(p.y)
        }))

        .filter(p =>
            Number.isFinite(p.x) &&
            Number.isFinite(p.y)
        );
}


// ==================================================
// 5. 장소 좌표 검증
// ==================================================

function coordinate(p) {

    const x = Number(p?.x);

    const y = Number(p?.y);

    if (
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        x < 124 ||
        x > 132 ||
        y < 33 ||
        y > 39
    ) {

        throw new Error(
            '국내 장소 좌표를 선택하세요.'
        );
    }

    return `${x},${y}`;
}


// ==================================================
// 6. 경로 좌표 추출
// ==================================================

function geometries(route) {

    return (route.sections || [])

        .flatMap(section =>

            (section.roads || [])

                .map(road => {

                    const a = road.vertexes || [];

                    const points = [];

                    for (
                        let i = 0;
                        i + 1 < a.length;
                        i += 2
                    ) {

                        points.push([
                            Number(a[i]),
                            Number(a[i + 1])
                        ]);
                    }

                    return points;
                })

                .filter(a => a.length >= 2)
        );
}


function validGeometry(lines) {

    return (

        Array.isArray(lines) &&

        lines.length > 0 &&

        lines.every(line =>

            line.length >= 2 &&

            line.every(([x, y]) =>

                Number.isFinite(x) &&
                Number.isFinite(y) &&
                x >= 124 &&
                x <= 132 &&
                y >= 33 &&
                y <= 39
            )
        )
    );
}


// ==================================================
// 7. 경로 중복 확인
// ==================================================

function signature(lines) {

    const h = crypto.createHash('sha1');

    for (const line of lines) {

        for (const [x, y] of line) {

            h.update(
                `${x.toFixed(5)},${y.toFixed(5)};`
            );
        }

        h.update('|');
    }

    return h.digest('hex');
}


// ==================================================
// 8. 최대 우회 배율 검증
// ==================================================

function normalizeDetourCap(
    cap = DEFAULT_CAP
) {

    if (
        cap === 'unlimited' ||
        cap === Infinity
    ) {

        return Infinity;
    }

    if (
        typeof cap !== 'number' &&
        typeof cap !== 'string'
    ) {

        throw new Error(
            '잘못된 최대 우회 배율입니다.'
        );
    }

    const value = Number(cap);

    const allowed = POLICY.detour_options
        .filter(option =>
            typeof option === 'number'
        );

    if (!allowed.includes(value)) {

        throw new Error(
            '최대 우회 배율은 1.5배, 2배, 3배 또는 제한 없음으로 설정하세요.'
        );
    }

    return value;
}


// ==================================================
// 9. 유효 경로 판단
// ==================================================

function isEligible(route) {

    return (

        Number.isFinite(route.model_score) &&

        route.model_score >= 0 &&

        route.model_score <= 100 &&

        Number.isFinite(route.model_coverage) &&

        route.model_coverage >= MIN_COVERAGE &&

        route.model_coverage <= 1 &&

        Number.isFinite(route.distance_m) &&

        route.distance_m > 0 &&

        Number.isFinite(route.duration_s) &&

        route.duration_s >= 0 &&

        route.model_version === MODEL_ID &&

        route.policy_version === POLICY_ID
    );
}


// ==================================================
// 10. P01 경로 정렬
// ==================================================

function roundedBurden(modelScore) {

    if (!Number.isFinite(modelScore)) {

        return Infinity;
    }

    // Python 오프라인 정책의 부담 점수
    // 소수점 12자리 기준 반영
    return Number(

        (modelScore / 100).toFixed(
            POLICY.offline_burden_rounding_decimals
        )
    );
}


function compareRouteIds(a, b) {

    const x = String(a);

    const y = String(b);

    if (x < y) {
        return -1;
    }

    if (x > y) {
        return 1;
    }

    return 0;
}


function compareRoutes(a, b) {

    // 유효한 경로 우선
    const eligibilityDifference =
        Number(b.eligible) -
        Number(a.eligible);

    if (eligibilityDifference !== 0) {

        return eligibilityDifference;
    }

    // P01 부담 점수 오름차순
    const burdenA = roundedBurden(
        a.model_score
    );

    const burdenB = roundedBurden(
        b.model_score
    );

    if (burdenA !== burdenB) {

        return burdenA < burdenB ? -1 : 1;
    }

    // 동일 부담 점수이면 이동 거리 우선
    if (a.distance_m !== b.distance_m) {

        return a.distance_m - b.distance_m;
    }

    // 동일 거리이면 소요 시간 우선
    if (a.duration_s !== b.duration_s) {

        return a.duration_s - b.duration_s;
    }

    // 모두 동일하면 경로 ID 오름차순
    return compareRouteIds(a.id, b.id);
}


// ==================================================
// 11. 최단 거리 및 우회 배율 적용
// ==================================================

function shortestDistance(candidates) {

    const distances = candidates

        .map(route => route.distance_m)

        .filter(distance =>

            Number.isFinite(distance) &&
            distance > 0
        );

    if (!distances.length) {

        return null;
    }

    return Math.min(...distances);
}


function withinDetourCap(
    candidates,
    cap
) {

    const shortest = shortestDistance(
        candidates
    );

    if (shortest === null) {

        return [];
    }

    return candidates.filter(route =>

        Number.isFinite(route.distance_m) &&

        route.distance_m > 0 &&

        route.distance_m <= shortest * cap
    );
}


// ==================================================
// 12. 최종 경로 추천
// ==================================================

function selectRoutes(
    candidates,
    cap = DEFAULT_CAP
) {

    const normalizedCap =
        normalizeDetourCap(cap);

    const shortest = shortestDistance(
        candidates
    );

    const visible = withinDetourCap(
        candidates,
        normalizedCap
    ).map(route => ({

        ...route,

        eligible: isEligible(route)
    }));

    visible.sort(compareRoutes);

    const eligible = visible.filter(
        route => route.eligible
    );

    const recommended =

        eligible.length >= MIN_CANDIDATES

            ? eligible[0].id

            : null;

    return {

        routes: visible,

        recommended,

        distinct_eligible: eligible.length,

        shortest_m: shortest,

        total_distinct: candidates.length,

        outside_cap_count:
            candidates.length - visible.length,

        // 기존 API 응답 의미 유지
        min_routes_met:
            visible.length >= MIN_CANDIDATES,

        detour_cap: normalizedCap
    };
}


// ==================================================
// 13. 카카오 경로 탐색
// ==================================================

async function findRoutes(
    origin,
    destination,
    key,
    index,
    fetcher = fetch,
    cap = DEFAULT_CAP
) {

    const normalizedCap =
        normalizeDetourCap(cap);

    const from = coordinate(origin);

    const to = coordinate(destination);

    if (from === to) {

        throw new Error(
            '서로 다른 출발지와 도착지를 선택하세요.'
        );
    }

    // 기존 기본 경로 탐색 방식 유지
    const plans = [

        { profile: 'RECOMMEND' },

        { profile: 'TIME' },

        { profile: 'DISTANCE' }
    ];

    // 기존 추가 경로 탐색 방식 유지
    const extras = [

        {
            profile: 'RECOMMEND',
            avoid: 'schoolzone'
        },

        {
            profile: 'RECOMMEND',
            avoid: 'uturn'
        },

        {
            profile: 'RECOMMEND',
            avoid: 'toll'
        },

        {
            profile: 'RECOMMEND',
            avoid: 'motorway'
        }
    ];

    const request = async plan => {

        const url = new URL(BASE_NAVI);

        url.searchParams.set(
            'origin',
            from
        );

        url.searchParams.set(
            'destination',
            to
        );

        url.searchParams.set(
            'priority',
            plan.profile
        );

        url.searchParams.set(
            'alternatives',
            'true'
        );

        url.searchParams.set(
            'summary',
            'false'
        );

        url.searchParams.set(
            'road_details',
            'true'
        );

        if (plan.avoid) {

            url.searchParams.set(
                'avoid',
                plan.avoid
            );
        }

        return fetchJson(
            url,
            key,
            fetcher
        );
    };

    const settled = await Promise.allSettled(
        plans.map(request)
    );

    const candidates = [];

    const failures = [];

    const seen = new Set();


    // ==============================================
    // 경로 후보 추가 및 중복 제거
    // ==============================================

    function add(plan, body) {

        for (
            const [i, route]
            of (body.routes || []).entries()
        ) {

            if (Number(route.result_code) !== 0) {

                continue;
            }

            const geometry = geometries(route);

            if (!validGeometry(geometry)) {

                continue;
            }

            const distance = Number(
                route.summary?.distance
            );

            const duration = Number(
                route.summary?.duration
            );

            // 잘못된 거리 및 시간 데이터 제외
            if (

                route.summary?.distance == null ||

                route.summary?.duration == null ||

                !Number.isFinite(distance) ||

                distance <= 0 ||

                !Number.isFinite(duration) ||

                duration < 0
            ) {

                continue;
            }

            const hash = signature(geometry);

            if (seen.has(hash)) {

                continue;
            }

            seen.add(hash);

            candidates.push({

                _route: route,

                id:
                    `${plan.avoid || plan.profile}-${i + 1}`,

                profile:
                    plan.avoid
                        ? plan.avoid.toUpperCase()
                        : plan.profile,

                geometry_hash: hash,

                geometry,

                distance_m: distance,

                duration_s: duration
            });
        }
    }


    // ==============================================
    // 기본 경로 요청 결과 처리
    // ==============================================

    for (
        let j = 0;
        j < settled.length;
        j++
    ) {

        const entry = settled[j];

        if (entry.status === 'rejected') {

            failures.push(
                `${plans[j].profile}: ${entry.reason.message}`
            );

        } else {

            add(plans[j], entry.value);
        }
    }


    // ==============================================
    // 추가 경로 탐색
    // ==============================================

    for (const plan of extras) {

        const count = withinDetourCap(

            candidates,

            normalizedCap

        ).length;

        // 기존 추가 API 호출 중단 조건 유지
        if (count >= SEARCH_TARGET_CANDIDATES) {

            break;
        }

        try {

            add(
                plan,
                await request(plan)
            );

        } catch (error) {

            failures.push(
                `${plan.avoid}: ${error.message}`
            );
        }
    }


    // ==============================================
    // 경로 탐색 실패 처리
    // ==============================================

    if (!candidates.length) {

        throw new Error(

            failures.length

                ? failures.join(' / ')

                : '카카오에서 운전 경로를 찾지 못했습니다.'
        );
    }


    // ==============================================
    // T09 도로 매칭 및 P01 점수 계산
    // ==============================================

    const routeIndex = index.forRoutes(

        candidates.map(
            route => route._route
        )
    );

    for (const candidate of candidates) {

        Object.assign(

            candidate,

            scoreRoute(
                routeIndex,
                candidate._route
            )
        );

        delete candidate._route;
    }


    // ==============================================
    // 최종 추천 경로 선택
    // ==============================================

    const selected = selectRoutes(

        candidates,

        normalizedCap
    );

    return {

        origin,

        destination,

        ...selected,

        failures,

        model_area: index.national

            ? '전국 표준 도로망·사고다발지 자료가 매칭된 경로만 점수 표시'

            : '천안·아산 공공 도로 링크 자료가 매칭된 경로만 점수 표시'
    };
}


// ==================================================
// 14. 외부 모듈 제공
// ==================================================

module.exports = {

    // 기존 외부 함수 유지
    fetchJson,

    searchPlaces,

    findRoutes,

    coordinate,

    geometries,

    signature,

    // 신규 정책 검증용 함수
    normalizeDetourCap,

    isEligible,

    compareRoutes,

    selectRoutes
};