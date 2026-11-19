
'use strict';

// 1. 기본 데이터 및 상태
const D = window.SAFEROUTE_DATA;
const $ = id => document.getElementById(id);

const points = new Map(
    D.points.map(p => [p.id, p])
);

let comparison;
let selected;
let zoom = 1;
let pan = [0, 0];
let drag = null;

let aiEnabled = false;
let requestVersion = 0;


// 2. 장소 선택 옵션
function opt(id) {
    const p = points.get(id);

    return new Option(
        `${p?.city || ''} ${p?.place_name || p?.name || id}`,
        id
    );
}

const originIds = [
    ...new Set(D.routes.map(r => r.origin_id))
];

originIds.forEach(id => {
    $('origin').add(opt(id));
});


// 3. 도착지 목록 갱신
function destinations() {
    const routes = D.routes.filter(
        r => r.origin_id === $('origin').value
    );

    const ids = [
        ...new Set(routes.map(r => r.destination_id))
    ];

    ids.sort((a, b) =>
        routes.filter(r => r.destination_id === b).length -
        routes.filter(r => r.destination_id === a).length
    );

    $('destination').replaceChildren(
        ...ids.map(opt)
    );

    update();
}


// 4. AI 점수로 추천 순위 정렬
function rankByAI(routes) {
    const sorted = [...routes];

    sorted.sort((a, b) =>
        Number(b.eligible) - Number(a.eligible) ||
        (a.predicted_burden ?? Infinity) -
        (b.predicted_burden ?? Infinity) ||
        a.distance_m - b.distance_m
    );

    return sorted;
}


// 5. 경로 갱신 및 AI 예측
async function update() {
    const currentRequest = ++requestVersion;

    const routes = D.routes.filter(
        r =>
            r.origin_id === $('origin').value &&
            r.destination_id === $('destination').value
    );

    // 기존 규칙 기반 비교
    comparison = Novice.rank(
        routes,
        Number($('cap').value)
    );

    aiEnabled = false;
    selected = comparison.routes[0] || null;

    zoom = 1;
    pan = [0, 0];

    render();

    if (!comparison.routes.length) return;

    try {
        // AI 예측 API 호출
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                routes: comparison.routes
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        if (!data.success || !Array.isArray(data.results)) {
            throw new Error(data.error || 'AI 예측 실패');
        }

        // 요청 중 출발지 또는 도착지가 바뀌었다면 무시
        if (currentRequest !== requestVersion) return;

        const predictions = new Map(
            data.results.map(r => [r.route_id, r])
        );

        // AI 예측 결과 적용
        const updatedRoutes = comparison.routes.map(r => {
            const prediction = predictions.get(r.route_id);

            if (
                !prediction ||
                !Number.isFinite(prediction.predicted_burden) ||
                !Number.isFinite(prediction.easy_score)
            ) {
                throw new Error('일부 경로의 AI 점수가 없습니다.');
            }

            return {
                ...r,
                predicted_burden: prediction.predicted_burden,
                easy_score: prediction.easy_score
            };
        });

        comparison.routes = rankByAI(updatedRoutes);

        const eligibleRoutes = comparison.routes.filter(
            r => r.eligible
        );

        // 기존 정책 유지: 적격 후보가 2개 이상일 때 추천
        comparison.recommended =
            eligibleRoutes.length >= 2
                ? eligibleRoutes[0].route_id
                : null;

        aiEnabled = true;

        // 이전에 사용자가 선택한 경로 유지
        const previousId = selected?.route_id;

        selected = comparison.routes.find(
            r => r.route_id === previousId
        ) || comparison.routes[0];

        // 최초 선택은 AI 추천 경로로 변경
        if (previousId === routes[0]?.route_id) {
            selected = comparison.routes.find(
                r => r.route_id === comparison.recommended
            ) || comparison.routes[0];
        }

        render();

    } catch (error) {
        if (currentRequest !== requestVersion) return;

        console.error('SafeRoute AI 예측 오류:', error);

        // 기존 규칙 기반 결과 유지
        aiEnabled = false;
        render();
    }
}


// 6. 퍼센트 표시
const pct = x =>
    x == null ? '정보 없음' : (x * 100).toFixed(1) + '%';


// 7. 경로 카드 및 상세 정보 표시
function render() {
    $('cards').replaceChildren();

    for (const r of comparison.routes) {
        const card = document.createElement('button');

        card.className =
            'card' +
            (r.route_id === selected?.route_id ? ' selected' : '');

        const title = document.createElement('strong');

        const profileName = {
            distance: '최단 거리 수집안',
            time: '시간 우선 수집안',
            recommend: '카카오 추천 수집안'
        }[r.profile] || r.profile;

        const isRecommended =
            r.route_id === comparison.recommended;

        title.textContent =
            (isRecommended
                ? (aiEnabled ? 'AI 추천 · ' : '추천 · ')
                : '') +
            profileName;

        if (isRecommended) {
            title.className = 'badge';
        }

        const detail = document.createElement('span');

        if (aiEnabled) {
            detail.textContent =
                `AI 쉬운 경로 ${r.easy_score.toFixed(1)} / 100` +
                ` · ${(r.distance_m / 1000).toFixed(1)} km`;
        } else {
            detail.textContent =
                `규칙 부담 점수 ${
                    r.score == null ? '없음' : r.score.toFixed(1)
                } / 100` +
                ` · ${(r.distance_m / 1000).toFixed(1)} km`;
        }

        const note = document.createElement('small');

        note.textContent =
            `큰길 ${pct(r.wide_road_ratio)}` +
            ` · 매칭 ${pct(r.coverage)}` +
            (r.eligible
                ? ''
                : r.coverage < 0.8
                    ? ' · 매칭 부족: 추천 제외'
                    : ' · 우회 한도 초과: 추천 제외');

        card.append(title, detail, note);

        card.onclick = () => {
            selected = r;
            render();
        };

        $('cards').append(card);
    }

    if (!selected) {
        $('routeTitle').textContent = '수집 경로 없음';
        $('metrics').replaceChildren();
        $('reason').textContent = '';
        $('date').textContent = '';
        draw();
        return;
    }

    $('routeTitle').textContent =
        `${selected.origin_name} → ${selected.destination_name}`;

    $('metrics').replaceChildren();

    const metrics = [
        [
            pct(selected.near_hotspot_ratio),
            '사고다발지 150m 이내 비중'
        ],
        [
            pct(selected.wide_road_ratio),
            '차로 2개 이상 도로 비중'
        ],
        [
            selected.n_intersect_200m?.toFixed(1) ?? '없음',
            '주변 200m 교차로 수 가중 평균'
        ]
    ];

    for (const [value, label] of metrics) {
        const element = document.createElement('div');
        element.className = 'metric';

        const bold = document.createElement('b');
        bold.textContent = value;

        const small = document.createElement('small');
        small.textContent = label;

        element.append(bold, small);
        $('metrics').append(element);
    }

    const ruleDescription =
        `규칙 점수 구성: 사고다발지 ${
            ((selected.accident_burden || 0) * 55).toFixed(1)
        } + 좁은 길 ${
            ((selected.narrow_burden || 0) * 25).toFixed(1)
        } + 교차로 ${
            ((selected.intersection_burden || 0) * 20).toFixed(1)
        }.`;

    if (aiEnabled) {
        $('reason').textContent =
            `LightGBM 예측 부담 점수: ` +
            `${selected.predicted_burden.toFixed(2)} / 100. ` +
            `AI 쉬운 경로 점수: ` +
            `${selected.easy_score.toFixed(2)} / 100. ` +
            `높을수록 운전 부담이 낮은 경로로 추정합니다. ` +
            `기존 규칙 점수를 학습한 실험 모델이며, ` +
            `실제 사고 확률이나 안전성을 의미하지 않습니다.`;
    } else {
        $('reason').textContent =
            ruleDescription +
            ' AI 예측 미적용: 기존 규칙 기반 결과입니다.';
    }

    $('date').textContent =
        `경로 수집: ${selected.collected_at}` +
        ` · 당시 소요 시간 약 ${
            Math.round(selected.duration_s / 60)
        }분 (현재 교통 정보 아님)`;

    draw();
}


// 8. 기존 Canvas 지도
function draw() {
    const canvas = $('map');
    const bounds = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = bounds.width * dpr;
    canvas.height = bounds.height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    if (!selected) return;

    const all = comparison.routes
        .flatMap(r => D.geometry[r.route_id] || [])
        .flat();

    if (!all.length) return;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const [x, y] of all) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
    }

    const cos = Math.cos(
        (minY + maxY) / 2 * Math.PI / 180
    );

    const scale = Math.min(
        (bounds.width - 70) /
            Math.max((maxX - minX) * cos, 1e-5),
        (bounds.height - 70) /
            Math.max(maxY - minY, 1e-5)
    ) * zoom;

    const xy = ([x, y]) => [
        bounds.width / 2 + pan[0] +
            (x - (minX + maxX) / 2) * cos * scale,
        bounds.height / 2 + pan[1] -
            (y - (minY + maxY) / 2) * scale
    ];

    const line = coords => {
        ctx.beginPath();

        coords.forEach((p, i) => {
            const q = xy(p);

            if (i) {
                ctx.lineTo(...q);
            } else {
                ctx.moveTo(...q);
            }
        });

        ctx.stroke();
    };

    // 배경 도로
    ctx.strokeStyle = '#d9e2e3';
    ctx.lineWidth = 1;

    for (const road of D.roads) {
        const q = xy(road[0]);

        if (
            q[0] < -100 ||
            q[0] > bounds.width + 100 ||
            q[1] < -100 ||
            q[1] > bounds.height + 100
        ) {
            continue;
        }

        line(road);
    }

    // 사고다발지 표시
    if ($('hotspots').checked) {
        ctx.fillStyle = '#e8a34288';

        for (const a of D.accidents) {
            const p = xy([a['경도'], a['위도']]);

            if (
                p[0] < 0 ||
                p[0] > bounds.width ||
                p[1] < 0 ||
                p[1] > bounds.height
            ) {
                continue;
            }

            ctx.beginPath();
            ctx.arc(...p, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // 경로 표시
    const orderedRoutes = [
        ...comparison.routes.filter(
            r => r.route_id !== selected.route_id
        ),
        selected
    ];

    for (const r of orderedRoutes) {
        const active = r.route_id === selected.route_id;

        ctx.strokeStyle = active ? '#087caa' : '#9eaeb8';
        ctx.lineWidth = active ? 4 : 2;

        for (const coords of D.geometry[r.route_id] || []) {
            line(coords);
        }
    }

    // 출발지 및 도착지 표시
    for (const [id, color, label] of [
        [selected.origin_id, '#068765', '출발'],
        [selected.destination_id, '#d53f54', '도착']
    ]) {
        const p = points.get(id);
        if (!p) continue;

        const q = xy([p.longitude, p.latitude]);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(...q, 7, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = 'bold 13px sans-serif';
        ctx.fillText(label, q[0] + 10, q[1] - 10);
    }
}


// 9. 이벤트 연결
$('origin').onchange = destinations;
$('destination').onchange = update;
$('cap').onchange = () => {
    updateDetourHelp();
    if (liveRoute) refreshLivePolicy();
    else if (!awaitingLiveSearch) update();
};
// 사고다발지 체크박스 변경 시 현재 지도 다시 그리기
$('hotspots').onchange = () => draw();

$('plus').onclick = () => {
    zoom = Math.min(zoom * 1.4, 12);
    draw();
};

$('minus').onclick = () => {
    zoom = Math.max(zoom / 1.4, 0.5);
    draw();
};

$('reset').onclick = () => {
    zoom = 1;
    pan = [0, 0];
    draw();
};

// 휠 확대/축소: 입력 장치별 delta 차이를 정규화하고 프레임당 한 번만 그리기
let wheelDelta = 0;
let wheelFrame = null;
$('map').addEventListener('wheel', e => {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 160 : 1;
    // 큰 휠 이벤트도 한 번에 과도하게 확대되지 않도록 제한
    const delta = Math.max(-100, Math.min(100, e.deltaY * unit));
    wheelDelta += delta;
    if (wheelFrame !== null) return;
    wheelFrame = requestAnimationFrame(() => {
        const step = Math.max(-1, Math.min(1, wheelDelta / 100));
        zoom = Math.max(0.5, Math.min(12, zoom * Math.exp(-step * 0.10)));
        wheelDelta = 0;
        wheelFrame = null;
        draw();
    });
}, { passive: false });

$('map').onpointerdown = e => {
    drag = [e.clientX, e.clientY, ...pan];
    $('map').setPointerCapture(e.pointerId);
};

$('map').onpointermove = e => {
    if (drag) {
        pan = [
            drag[2] + e.clientX - drag[0],
            drag[3] + e.clientY - drag[1]
        ];

        draw();
    }
};

$('map').onpointerup = () => {
    drag = null;
};

$('map').onpointercancel = () => {
    drag = null;
};


// 10. JSON 결과 저장
$('export').onclick = () => {
    const obj = {
        profile: D.profile,
        detour_cap: Number($('cap').value),
        scoring_method: aiEnabled
            ? 'LightGBM predicted burden'
            : 'Rule-based burden',
        ...comparison,
        meaning:
            'Experimental driving burden estimate; ' +
            'not accident probability'
    };

    const blob = new Blob(
        [JSON.stringify(obj, null, 2)],
        { type: 'application/json' }
    );

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');

    a.href = url;
    a.download = 'saferoute_comparison.json';
    a.click();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
};


// 11. 초기 실행
window.onresize = draw;
destinations();


/* ==================================================
   12. 카카오 장소 검색
================================================== */

// 선택한 출발지와 도착지 정보
const selectedPlaces = {
    origin: null,
    destination: null
};

// 검색 요청 관리
const searchVersions = {
    origin: 0,
    destination: 0
};

// 검색 상태 메시지
function setSearchStatus(message) {
    $('searchStatus').textContent = message;
}

// 장소 검색 결과 표시
function showSearchResults(type, places) {
    const resultsId = type === 'origin'
        ? 'originResults'
        : 'destinationResults';

    const inputId = type === 'origin'
        ? 'originSearch'
        : 'destinationSearch';

    const container = $(resultsId);
    container.replaceChildren();

    if (places.length === 0) {
        const message = document.createElement('p');
        message.className = 'muted';
        message.textContent = '검색 결과가 없습니다.';
        container.append(message);
        return;
    }

    for (const place of places) {
        const button = document.createElement('button');
        button.type = 'button';

        const name = document.createElement('strong');
        name.textContent = place.name;

        const address = document.createElement('small');
        address.textContent = place.address;

        button.append(name, address);

        button.onclick = () => {
            // 선택한 장소 저장
            selectedPlaces[type] = {
                id: place.id,
                name: place.name,
                address: place.address,
                longitude: place.longitude,
                latitude: place.latitude
            };

            // 선택한 장소명을 검색창에 표시
            $(inputId).value = place.name;

            // 이전 검색 요청 결과 무효화
            searchVersions[type]++;

            // 검색 결과 목록 닫기
            container.replaceChildren();

            const label = type === 'origin'
                ? '출발지'
                : '도착지';

            setSearchStatus(
                `${label} 선택 완료: ${place.name}`
            );

            console.log(
                `SafeRoute ${label} 선택:`,
                selectedPlaces[type]
            );
        };

        container.append(button);
    }
}

// 카카오 장소 검색 API 호출
async function searchPlace(type, query) {
    const resultsId = type === 'origin'
        ? 'originResults'
        : 'destinationResults';

    const container = $(resultsId);

    // 검색 요청 번호 증가
    const currentVersion = ++searchVersions[type];

    if (!query.trim()) {
        container.replaceChildren();
        return;
    }

    setSearchStatus('장소 검색 중...');

    try {
        const response = await fetch(
            `/api/search?query=${encodeURIComponent(query)}`
        );

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        if (!data.success || !Array.isArray(data.places)) {
            throw new Error(
                data.error || '장소 검색 실패'
            );
        }

        // 오래된 검색 요청 결과는 무시
        if (currentVersion !== searchVersions[type]) {
            return;
        }

        showSearchResults(type, data.places);

        setSearchStatus(
            `검색 결과 ${data.places.length}개`
        );

    } catch (error) {
        if (currentVersion !== searchVersions[type]) {
            return;
        }

        console.error('카카오 장소 검색 오류:', error);

        container.replaceChildren();
        setSearchStatus(
            '장소 검색에 실패했습니다. 서버 연결을 확인해주세요.'
        );
    }
}

// 출발지 검색창 연결
// 연속 타이핑 중 불필요한 API 호출 방지
const searchTimers = { origin: null, destination: null };
function queuePlaceSearch(type, query) {
    clearTimeout(searchTimers[type]);
    selectedPlaces[type] = null;
    searchVersions[type]++;
    const resultsId = type === 'origin' ? 'originResults' : 'destinationResults';
    if (!query) {
        $(resultsId).replaceChildren();
        return;
    }
    searchTimers[type] = setTimeout(() => searchPlace(type, query), 300);
}
$('originSearch').addEventListener('input', event => {
    queuePlaceSearch('origin', event.target.value.trim());
});
$('destinationSearch').addEventListener('input', event => {
    queuePlaceSearch('destination', event.target.value.trim());
});


/* ==================================================
   13. 실제 자동차 경로 조회 및 표시
================================================== */

// 실시간 조회 경로 상태
let liveRoute = null;
let liveRouteRequestVersion = 0;
let liveRoutes = [];
let liveRecommendation = null;
let awaitingLiveSearch = false;

function updateDetourHelp(shortest = null) {
    const cap = Number($('cap').value);
    const limit = cap === Infinity ? '거리 제한 없이' : `조회 후보 최단 거리의 ${cap}배 이내에서`;
    const distance = Number.isFinite(shortest) && cap !== Infinity
        ? ` (최단 ${(shortest / 1000).toFixed(2)} km · 허용 ${(shortest * cap / 1000).toFixed(2)} km)` : '';
    $('detourHelp').textContent = `${limit} AI 쉬움 점수가 높은 경로를 추천합니다.${distance} 범위를 바꾸면 현재 경로에 바로 적용됩니다.`;
}

function refreshLivePolicy(preferredRoute = liveRoute) {
    const savedZoom = zoom, savedPan = pan;
    const preserveView = Boolean(liveRoute);
    const result = Novice.rankLive(liveRoutes, Number($('cap').value));
    liveRoutes = result.routes;
    liveRecommendation = result.recommended;
    updateDetourHelp(result.shortest);
    const preferred = liveRoutes.find(r => r.route_id === preferredRoute?.route_id) || liveRoutes[0];
    renderLiveRouteCards(preferred);
    if (preserveView) { zoom = savedZoom; pan = savedPan; drawLiveRoute(); }
}


function selectLiveRoute(index) {
    const route = liveRoutes[index];

    if (!route) return;

    liveRoute = route;

    // 선택한 경로 카드 강조
    const cards = $('cards').querySelectorAll('.card');

    cards.forEach((card, cardIndex) => {
        card.classList.toggle('selected', cardIndex === index);
        card.setAttribute('aria-pressed', String(cardIndex === index));
    });

    // 선택한 경로의 상세 정보
    $('metrics').replaceChildren();

    const metrics = [
        [
            (route.distance_m / 1000).toFixed(2) + ' km',
            '총 이동 거리'
        ],
        [
            Math.round(route.duration_s / 60) + '분',
            '예상 소요 시간'
        ],
        [
            String(route.road_count),
            '도로 구간 수'
        ]
    ];

    for (const [value, label] of metrics) {
        const element = document.createElement('div');
        element.className = 'metric';

        const bold = document.createElement('b');
        bold.textContent = value;

        const small = document.createElement('small');
        small.textContent = label;

        element.append(bold, small);
        $('metrics').append(element);
    }

    const routeNames = {
        RECOMMEND: '일반 추천 경로',
        TIME: '시간 우선 경로',
        DISTANCE: '거리 우선 경로'
    };

    const routeName =
        routeNames[route.priority] || '자동차 경로';

    $('reason').textContent = route.ai_pending
        ? `${routeName} · AI 경로 분석 중...`
        : Number.isFinite(route.easy_score)
        ? `${routeName} · AI 쉬움 점수 ${route.easy_score.toFixed(2)}/100점 (도로 매칭률 ${(route.coverage * 100).toFixed(1)}%).`
        : `${routeName} · AI 점수 산출 불가: ${route.ai_error || '분석 전 또는 매칭 기준 미충족'}`;

    $('reason').textContent += route.eligible
        ? (route.route_id === liveRecommendation ? ' · AI 추천 경로' : ' · 추천 가능')
        : ` · 추천 제외: ${route.exclusion_reason}`;

    $('date').textContent =
        '조회 당시의 예상 경로 및 소요 시간입니다. ' +
        '실제 교통 상황에 따라 달라질 수 있습니다.';

    // 다른 경로를 선택하면 지도 확대/이동 초기화
    zoom = 1;
    pan = [0, 0];

    drawLiveRoute();
}

// Presentation only: recommendation policy and model scores are unchanged.
function routeCardView(route, recommendedId) {
    const hasScore = !route.ai_pending && Number.isFinite(route.easy_score);
    const recommended = route.eligible && route.route_id === recommendedId;
    return {
        hasScore,
        state: route.ai_pending ? 'is-pending' : recommended ? 'is-recommended' : route.eligible ? 'is-eligible' : 'is-excluded',
        badge: route.ai_pending ? '분석 중' : recommended ? 'AI 추천' : route.eligible ? '비교 가능' : '추천 제외',
        scoreText: route.ai_pending ? '분석 중' : hasScore ? route.easy_score.toFixed(1) : '분석 불가',
        scoreLabel: route.ai_pending ? 'AI 분석 중' : hasScore ? `AI 쉬움 ${route.easy_score.toFixed(1)}점` : 'AI 점수 분석 불가',
        explanation: route.ai_pending ? '경로는 먼저 볼 수 있습니다. 점수는 분석 후 표시됩니다.'
            : recommended ? '현재 우회 범위 안에서 AI 쉬움 점수가 가장 높은 후보입니다.'
            : route.eligible ? '우회 범위와 분석 조건을 충족한 비교 후보입니다.'
            : `추천 제외: ${route.exclusion_reason || route.ai_error || '분석 조건 미충족'}`
    };
}

// 경로 카드와 지도는 카카오 경로 수신 즉시 표시하고 AI 결과는 나중에 갱신
function renderLiveRouteCards(preferredRoute = null) {
    $('cards').replaceChildren();
    const routeNames = {
        RECOMMEND: '일반 추천 경로',
        TIME: '시간 우선 경로',
        DISTANCE: '거리 우선 경로'
    };
    const pendingCount = liveRoutes.filter(r => r.ai_pending).length;
    const eligibleCount = liveRoutes.filter(r => r.eligible).length;
    $('routeSummary').textContent = pendingCount
        ? `${liveRoutes.length}개 경로를 찾았습니다. ${pendingCount}개 경로의 AI 점수를 분석하고 있습니다.`
        : liveRecommendation !== null
        ? `${liveRoutes.length}개 경로 중 ${eligibleCount}개가 추천 조건을 충족합니다. AI 추천 경로를 확인하세요.`
        : '현재 조건에서 추천 가능한 경로가 없습니다. 각 경로의 제외 사유를 확인하세요.';
    liveRoutes.forEach((route, index) => {
        const view = routeCardView(route, liveRecommendation);
        const card = document.createElement('button');
        card.type = 'button';
        card.className = `card live-card ${view.state}`;
        card.setAttribute('aria-pressed', 'false');
        card.setAttribute('aria-label', `${routeNames[route.priority] || '자동차 경로'}, ${view.badge}, ${view.scoreLabel}`);
        const node = (tag, className, text) => {
            const element = document.createElement(tag);
            element.className = className;
            element.textContent = text;
            return element;
        };
        const heading = node('span', 'card-heading', '');
        heading.append(node('strong', 'route-name', routeNames[route.priority] || '자동차 경로'),
            node('span', 'route-badge', view.badge));
        const score = node('span', 'easy-score-row', '');
        const value = node('span', 'easy-score-value', view.scoreText);
        if (view.hasScore) value.append(node('small', 'score-total', ' / 100점'));
        score.append(node('span', 'easy-score-label', 'AI 쉬움 점수'), value);
        const bar = node('span', 'easy-score-track', '');
        bar.setAttribute('aria-hidden', 'true');
        const fill = node('span', 'easy-score-fill', '');
        fill.style.width = `${view.hasScore ? route.easy_score : 0}%`;
        bar.append(fill);
        const details = node('span', 'route-facts', '');
        for (const [value, label] of [
            [`${(route.distance_m / 1000).toFixed(2)} km`, '이동 거리'],
            [`${Math.round(route.duration_s / 60)}분`, '예상 시간'],
            [route.detour_ratio == null ? '—' : `${route.detour_ratio.toFixed(2)}배`, '우회 비율']
        ]) {
            const fact = node('span', 'route-fact', '');
            fact.append(node('b', '', value), node('small', '', label));
            details.append(fact);
        }
        const explanation = node('span', 'card-explanation', view.explanation);
        const support = node('small', 'card-support', Number.isFinite(route.coverage)
            ? `도로 매칭률 ${(route.coverage * 100).toFixed(1)}% · 도로 ${route.road_count}개 구간`
            : `도로 ${route.road_count}개 구간 · 안내 ${route.guide_count}개`);
        card.append(heading, score, bar, details, explanation, support);
        card.addEventListener('click', () => selectLiveRoute(index));
        $('cards').append(card);
    });
    const selectedIndex = preferredRoute ? liveRoutes.indexOf(preferredRoute) : -1;
    selectLiveRoute(selectedIndex >= 0 ? selectedIndex : 0);
}

// 카카오 경로 조회 완료 후 AI 분석을 비동기로 진행
async function analyzeLiveRoutes(routes, currentVersion, origin, destination) {
    const analysisStarted = performance.now();
    try {
        const aiResponse = await fetch('/api/predict-live', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                routes: routes.map(route => ({ ...route, route_id: String(route.route_id) }))
            })
        });
        const aiData = await aiResponse.json();
        if (!aiResponse.ok || !aiData.success || !Array.isArray(aiData.results)) {
            throw new Error(aiData.error || 'AI 예측 요청 실패');
        }
        if (currentVersion !== liveRouteRequestVersion) return;
        console.info('[SafeRoute AI timing]', {...aiData.timings, browser_ai_roundtrip_ms: performance.now()-analysisStarted});
        const previouslySelected = liveRoute;
        liveRoutes = Novice.applyLivePredictions(routes, aiData.results);
        refreshLivePolicy(previouslySelected);
        const successCount = liveRoutes.filter(r => r.ai_eligible).length;
        const resultText = successCount === liveRoutes.length ? 'AI 분석 완료'
            : successCount ? `AI 분석 일부 완료 (${successCount}/${liveRoutes.length}개 성공)` : 'AI 분석 실패';
        const recommendationText = liveRecommendation == null && successCount ? ' · 현재 우회 범위에서 추천 가능한 경로 없음' : '';
        setSearchStatus(`${origin.name} → ${destination.name} ${resultText}${recommendationText}`);
    } catch (aiError) {
        if (currentVersion !== liveRouteRequestVersion) return;
        console.error('실시간 AI 예측 오류:', aiError);
        const previouslySelected = liveRoute;
        liveRoutes = routes.map(route => ({
            ...route, ai_pending: false, ai_eligible: false, eligible: false,
            easy_score: null, predicted_burden: null, coverage: null, ai_error: aiError.message
        }));
        const preferred = liveRoutes.find(r => r.route_id === previouslySelected?.route_id &&
            r.priority === previouslySelected?.priority) || liveRoutes[0];
        refreshLivePolicy(preferred);
        setSearchStatus('경로 조회 완료 · AI 분석 실패');
    }
}

$('searchRoute').onclick = async () => {
    const origin = selectedPlaces.origin;
    const destination = selectedPlaces.destination;
    if (!origin || !destination) {
        setSearchStatus('출발지와 도착지를 검색 결과에서 각각 선택해주세요.');
        return;
    }
    requestVersion++; // Ignore pending collected-route AI responses.
    const currentVersion = ++liveRouteRequestVersion;
    const directionsStarted = performance.now();
    // Enter live mode immediately: cap changes and failed requests must never
    // expose old collected routes or a previous direction's recommendation.
    awaitingLiveSearch = true;
    liveRoute = null; liveRoutes = []; liveRecommendation = null;
    selected = null;
    $('cards').replaceChildren(); $('metrics').replaceChildren();
    $('routeTitle').textContent = `${origin.name} → ${destination.name}`;
    $('routeSummary').textContent = '자동차 경로를 조회하고 있습니다.';
    $('reason').textContent = ''; $('date').textContent = '';
    draw();
    setSearchStatus('카카오모빌리티 자동차 경로 조회 중...');
    $('searchRoute').disabled = true;
    try {
        const params = new URLSearchParams({
            origin: `${origin.longitude},${origin.latitude}`,
            destination: `${destination.longitude},${destination.latitude}`
        });
        const response = await fetch(`/api/directions?${params.toString()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!data.success || !data.route) {
            throw new Error(data.error || '경로 조회 실패');
        }
        if (currentVersion !== liveRouteRequestVersion) return;
        console.info('[SafeRoute route timing]', {...data.timings, browser_directions_roundtrip_ms: performance.now()-directionsStarted});
        const candidates = Array.isArray(data.routes) && data.routes.length > 0
            ? data.routes : [data.route];
        const routeCandidates = candidates.filter(route => route
            && Number.isFinite(route.distance_m) && route.distance_m > 0
            && Number.isFinite(route.duration_s) && route.duration_s >= 0
            && Array.isArray(route.geometry) && route.geometry.length > 0
            && route.geometry.every(line => Array.isArray(line) && line.length >= 2
                && line.every(point => Array.isArray(point) && point.length >= 2
                    && Number.isFinite(point[0]) && Number.isFinite(point[1]))));
        if (!routeCandidates.length) throw new Error('표시할 수 있는 경로 후보가 없습니다.');
        liveRoutes = routeCandidates.map((route, index) => ({
            ...route,
            route_id: String(index),
            origin, destination,
            ai_pending: true,
            ai_eligible: false,
            eligible: false,
            easy_score: null,
            coverage: null
        }));
        awaitingLiveSearch = false;
        $('routeTitle').textContent = `${origin.name} → ${destination.name}`;
        // AI 분석을 기다리지 않고 경로 카드 및 지도부터 출력
        refreshLivePolicy();
        setSearchStatus(`${origin.name} → ${destination.name} 경로 조회 완료 · AI 분석 중...`);
        // await 하지 않으므로 화면이 먼저 갱신됨
        void analyzeLiveRoutes([...liveRoutes], currentVersion, origin, destination);
    } catch (error) {
        if (currentVersion !== liveRouteRequestVersion) return;
        console.error('자동차 경로 조회 오류:', error);
        $('routeSummary').textContent = '경로 조회에 실패했습니다. 출발지·도착지를 확인하고 다시 검색해주세요.';
        setSearchStatus(`경로 조회 실패: ${error.message}`);
    } finally {
        if (currentVersion === liveRouteRequestVersion) {
            $('searchRoute').disabled = false;
        }
    }
};


// 실제 경로 Canvas 표시
function drawLiveRoute() {
    if (!liveRoute) return;

    const canvas = $('map');
    const bounds = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = bounds.width * dpr;
    canvas.height = bounds.height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const all = liveRoute.geometry.flat();

    if (!all.length) return;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const [x, y] of all) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
    }

    const cos = Math.cos(
        (minY + maxY) / 2 * Math.PI / 180
    );

    const scale = Math.min(
        (bounds.width - 70) /
            Math.max((maxX - minX) * cos, 1e-5),
        (bounds.height - 70) /
            Math.max(maxY - minY, 1e-5)
    ) * zoom;

    const xy = ([x, y]) => [
        bounds.width / 2 + pan[0] +
            (x - (minX + maxX) / 2) * cos * scale,
        bounds.height / 2 + pan[1] -
            (y - (minY + maxY) / 2) * scale
    ];

    // 배경 도로 표시
    ctx.strokeStyle = '#d9e2e3';
    ctx.lineWidth = 1;

    for (const road of D.roads) {
        ctx.beginPath();

        road.forEach((point, index) => {
            const p = xy(point);

            if (index === 0) {
                ctx.moveTo(...p);
            } else {
                ctx.lineTo(...p);
            }
        });

        ctx.stroke();
    }


    // 사고다발지 표시
    if ($('hotspots').checked) {
        ctx.fillStyle = '#e8a34288';

        for (const a of D.accidents) {
            const longitude = Number(a['경도']);
            const latitude = Number(a['위도']);

            if (
                !Number.isFinite(longitude) ||
                !Number.isFinite(latitude)
            ) {
                continue;
            }

            const p = xy([longitude, latitude]);

            // 현재 지도 화면 밖의 사고다발지는 제외
            if (
                p[0] < 0 ||
                p[0] > bounds.width ||
                p[1] < 0 ||
                p[1] > bounds.height
            ) {
                continue;
            }

            ctx.beginPath();
            ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }


    // 실제 자동차 경로 표시
    ctx.strokeStyle = '#087caa';
    ctx.lineWidth = 4;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (const road of liveRoute.geometry) {
        ctx.beginPath();

        road.forEach((point, index) => {
            const p = xy(point);

            if (index === 0) {
                ctx.moveTo(...p);
            } else {
                ctx.lineTo(...p);
            }
        });

        ctx.stroke();
    }

    // 출발지와 도착지 표시
    const locations = [
        [liveRoute.origin, '#068765', '출발'],
        [liveRoute.destination, '#d53f54', '도착']
    ];

    for (const [place, color, label] of locations) {
        const p = xy([
            place.longitude,
            place.latitude
        ]);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(...p, 7, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = 'bold 13px sans-serif';
        ctx.fillText(label, p[0] + 10, p[1] - 10);
    }
}


// 기존 지도와 실시간 조회 지도 전환
const originalDraw = draw;

draw = function () {
    if (liveRoute) {
        drawLiveRoute();
    } else {
        originalDraw();
    }
};



/* ==================================================
   14. 기존 수집 경로 / 실시간 경로 화면 전환
================================================== */

// 기존 수집 경로 화면으로 복귀
function switchToCollectedRoutes() {
    // 진행 중인 실시간 경로 조회 결과 무효화
    liveRouteRequestVersion++;

    // 실시간 경로 표시 상태 해제
    liveRoute = null;

    // 기존 수집 경로 화면 갱신
    update();
}

// 기존 수집 경로의 출발지 변경
$('origin').addEventListener('change', () => {
    switchToCollectedRoutes();
});

// 기존 수집 경로의 도착지 변경
$('destination').addEventListener('change', () => {
    switchToCollectedRoutes();
});

// 우회 변경은 위의 onchange에서 현재 실시간 결과를 재사용합니다.


// Swap full selected-place objects (including coordinates), not just their labels.
function swapPlaces() {
    const oldOrigin = $('originSearch').value;
    $('originSearch').value = $('destinationSearch').value;
    $('destinationSearch').value = oldOrigin;
    [selectedPlaces.origin, selectedPlaces.destination] = [selectedPlaces.destination, selectedPlaces.origin];
    for (const type of ['origin', 'destination']) {
        clearTimeout(searchTimers[type]);
        searchVersions[type]++;
        $(type === 'origin' ? 'originResults' : 'destinationResults').replaceChildren();
    }
    // Old driving and AI responses belong to the previous direction.
    requestVersion++;
    liveRouteRequestVersion++;
    liveRoute = null; liveRoutes = []; liveRecommendation = null;
    awaitingLiveSearch = true;
    selected = null;
    $('cards').replaceChildren(); $('metrics').replaceChildren();
    $('routeSummary').textContent = '출발지와 도착지를 바꿨습니다. 경로 찾기를 눌러주세요.';
    $('reason').textContent = '출발지와 도착지를 바꿨습니다. 경로 찾기를 눌러 새 방향을 조회하세요.';
    $('date').textContent = '';
    $('routeTitle').textContent = '바뀐 방향의 경로를 검색해주세요';
    $('searchRoute').disabled = false;
    updateDetourHelp();
    draw();
    setSearchStatus(selectedPlaces.origin && selectedPlaces.destination
        ? `${selectedPlaces.origin.name} → ${selectedPlaces.destination.name} · 경로 찾기를 눌러주세요.`
        : '출발지와 도착지를 바꿨습니다. 검색 결과에서 장소를 선택한 뒤 경로 찾기를 눌러주세요.');
}
$('swapPlaces').addEventListener('click', swapPlaces);
