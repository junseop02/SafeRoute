'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');


// ==================================================
// 1. 실제 프론트엔드 파일 불러오기
// ==================================================

const ROOT = path.resolve(__dirname, '..');

const APP = fs.readFileSync(
    path.join(ROOT, 'service', 'app.js'),
    'utf8'
);

const HTML = fs.readFileSync(
    path.join(ROOT, 'service', 'index.html'),
    'utf8'
);


// ==================================================
// 2. 모의 DOM 구현
// ==================================================

class MockElement {

    constructor(tag = 'div') {

        this.tagName = tag.toUpperCase();
        this.children = [];
        this.attributes = {};
        this.style = {};

        this.value = '';
        this.placeholder = '';
        this.className = '';
        this.hidden = false;
        this.disabled = false;

        this.onclick = null;
        this.onchange = null;
        this.oninput = null;
        this.onkeydown = null;

        this._text = '';

        this.classList = {

            add: name => {
                const names = new Set(
                    this.className.split(/\s+/).filter(Boolean)
                );

                names.add(name);

                this.className = [...names].join(' ');
            },

            remove: name => {
                this.className = this.className
                    .split(/\s+/)
                    .filter(item => item && item !== name)
                    .join(' ');
            },

            contains: name => {
                return this.className
                    .split(/\s+/)
                    .includes(name);
            },

            toggle: (name, force) => {
                const enabled = force === undefined
                    ? !this.classList.contains(name)
                    : Boolean(force);

                if (enabled) {
                    this.classList.add(name);
                } else {
                    this.classList.remove(name);
                }

                return enabled;
            }
        };
    }

    get textContent() {

        return this._text +
            this.children.map(
                child => child.textContent
            ).join('');
    }

    set textContent(value) {

        this._text = String(value);
        this.children = [];
    }

    append(...children) {

        this.children.push(...children);
    }

    replaceChildren(...children) {

        this._text = '';
        this.children = [...children];
    }

    setAttribute(name, value) {

        this.attributes[name] = String(value);
    }

    getAttribute(name) {

        return this.attributes[name] ?? null;
    }

    scrollIntoView() {}

    remove() {

        this.removed = true;
    }

    click() {

        this.clicked = true;

        return this.onclick?.();
    }
}


// ==================================================
// 3. 테스트용 장소 및 경로 데이터
// ==================================================

const ORIGIN = {
    name: '천안역',
    address: '충청남도 천안시',
    x: 127.146,
    y: 36.810
};

const DESTINATION = {
    name: '아산역',
    address: '충청남도 아산시',
    x: 127.104,
    y: 36.793
};


function makeRoute(id, burden = 20, overrides = {}) {

    return {
        id,
        profile: 'RECOMMEND',

        distance_m: 6000,
        duration_s: 900,

        geometry: [
            [
                [127.146, 36.810],
                [127.104, 36.793]
            ]
        ],

        model_score: burden,
        model_coverage: 0.95,

        model_wide_ratio: 0.8,
        model_intersections: 20,
        model_hotspot_ratio: 0.15,

        model_version: 'T09',
        policy_version: 'P01',

        eligible: true,

        ...overrides
    };
}


function makeResult(routes, recommended = null) {

    return {
        origin: ORIGIN,
        destination: DESTINATION,

        routes,
        recommended,

        shortest_m: routes.length
            ? Math.min(...routes.map(r => r.distance_m))
            : null,

        distinct_eligible: routes.filter(
            route => route.eligible
        ).length,

        model_version: 'T09',
        policy_version: 'P01'
    };
}


function response(body, ok = true) {

    return {
        ok,

        async json() {
            return body;
        }
    };
}


function deferred() {

    let resolve;
    let reject;

    const promise = new Promise((res, rej) => {
        resolve = res;
        reject = rej;
    });

    return { promise, resolve, reject };
}


// ==================================================
// 4. 모의 지도 SDK
// ==================================================

function createMockKakao() {

    class LatLng {

        constructor(lat, lng) {

            this.lat = lat;
            this.lng = lng;
        }
    }

    class LatLngBounds {

        constructor() {

            this.points = [];
        }

        extend(point) {

            this.points.push(point);
        }
    }

    class Map {

        constructor(element, options) {

            this.element = element;
            this.level = options.level;
            this.center = options.center;
        }

        relayout() {}

        setBounds(bounds) {

            this.bounds = bounds;
        }

        setCenter(center) {

            this.center = center;
        }

        setLevel(level) {

            this.level = level;
        }

        getLevel() {

            return this.level;
        }
    }

    class Marker {

        constructor(options) {

            Object.assign(this, options);
        }

        setMap(map) {

            this.map = map;
        }
    }

    class Polyline {

        constructor(options) {

            Object.assign(this, options);
        }

        setMap(map) {

            this.map = map;
        }
    }

    return {
        maps: {
            LatLng,
            LatLngBounds,
            Map,
            Marker,
            Polyline,

            load(callback) {
                callback();
            }
        }
    };
}


// ==================================================
// 5. UI 테스트 실행 환경
// ==================================================

function createUI(options = {}) {

    const elements = new Map();
    const requests = [];
    const scripts = [];
    const downloads = [];
    const blobs = [];

    // 실제 HTML에 존재하는 ID만 생성
    for (const match of HTML.matchAll(/id="([^"]+)"/g)) {

        elements.set(
            match[1],
            new MockElement()
        );
    }

    const get = id => {

        assert.ok(
            elements.has(id),
            `Missing HTML element: ${id}`
        );

        return elements.get(id);
    };

    get('detourCap').value = '2';
    get('apiKey').value = 'TEST_KEY';

    get('recommendations').hidden = true;
    get('export').hidden = true;

    const document = {

        getElementById: get,

        createElement(tag) {

            const item = new MockElement(tag);

            if (tag === 'a') {

                item.click = () => {
                    downloads.push({
                        href: item.href,
                        download: item.download
                    });
                };
            }

            return item;
        },

        head: {
            append(script) {
                scripts.push(script);
            }
        }
    };

    const kakao = createMockKakao();

    const context = vm.createContext({

        document,

        window: options.mapReady
            ? { kakao }
            : {},

        kakao,

        location: {
            protocol: options.protocol || 'http:'
        },

        console,

        Blob,

        URL: {

            createObjectURL(blob) {

                blobs.push(blob);

                return 'blob:test-result';
            },

            revokeObjectURL() {}
        },

        setTimeout(callback) {

            callback();
        },

        async fetch(endpoint, config) {

            if (endpoint === '/api/status') {

                return response({
                    api_key_from_environment:
                        options.envKey || false
                });
            }

            const body = JSON.parse(config.body);

            requests.push({
                endpoint,
                config,
                body
            });

            if (options.fetcher) {

                return options.fetcher(
                    endpoint,
                    body
                );
            }

            return response({
                places: []
            });
        }
    });

    vm.runInContext(APP, context);

    const run = code => vm.runInContext(
        code,
        context
    );

    function selectPlaces() {

        run(
            `selectedPlace(
                'origin',
                ${JSON.stringify(ORIGIN)}
            )`
        );

        run(
            `selectedPlace(
                'destination',
                ${JSON.stringify(DESTINATION)}
            )`
        );
    }

    function renderResult(result) {

        run(
            `state.result = ${JSON.stringify(result)}`
        );

        run(`
            state.selected =
                state.result.routes.find(
                    r => r.id === state.result.recommended
                ) ||
                state.result.routes[0] ||
                null;

            render();
        `);
    }

    function score(route) {

        return run(
            `easyScore(${JSON.stringify(route)})`
        );
    }

    return {
        get,
        run,
        score,
        selectPlaces,
        renderResult,
        requests,
        scripts,
        downloads,
        blobs
    };
}


// ==================================================
// 6. 점수 표시 검증
// ==================================================

test('UI: display score is 100 minus burden', () => {

    const ui = createUI();

    assert.equal(
        ui.score(makeRoute('A', 20)),
        80
    );
});


test('UI: zero burden displays 100 points', () => {

    const ui = createUI();

    assert.equal(
        ui.score(makeRoute('A', 0)),
        100
    );
});


test('UI: maximum burden displays zero points', () => {

    const ui = createUI();

    assert.equal(
        ui.score(makeRoute('A', 100)),
        0
    );
});


test('UI: exact 80 percent coverage is accepted', () => {

    const ui = createUI();

    assert.equal(
        ui.score(makeRoute('A', 20, {
            model_coverage: 0.8
        })),
        80
    );
});


test('UI: insufficient coverage has no score', () => {

    const ui = createUI();

    assert.equal(
        ui.score(makeRoute('A', 20, {
            model_coverage: 0.799
        })),
        null
    );
});


test('UI: invalid burden values have no score', () => {

    const ui = createUI();

    for (const value of [
        null, -1, 101, '20', NaN, Infinity
    ]) {

        assert.equal(
            ui.score(makeRoute('A', value)),
            null
        );
    }
});


test('UI: invalid coverage values have no score', () => {

    const ui = createUI();

    for (const value of [
        null, -1, 1.01, '0.95', NaN
    ]) {

        assert.equal(
            ui.score(makeRoute('A', 20, {
                model_coverage: value
            })),
            null
        );
    }
});


test('UI: incorrect model version has no score', () => {

    const ui = createUI();

    assert.equal(
        ui.score(makeRoute('A', 20, {
            model_version: 'OLD'
        })),
        null
    );
});


test('UI: incorrect policy version has no score', () => {

    const ui = createUI();

    assert.equal(
        ui.score(makeRoute('A', 20, {
            policy_version: 'OLD'
        })),
        null
    );
});


// ==================================================
// 7. 경로 카드 및 상세 정보 검증
// ==================================================

test('UI: route cards are rendered', () => {

    const ui = createUI();

    ui.renderResult(makeResult([
        makeRoute('A', 20),
        makeRoute('B', 30)
    ], 'A'));

    assert.equal(
        ui.get('cards').children.length,
        2
    );
});


test('UI: recommended route displays AI badge', () => {

    const ui = createUI();

    ui.renderResult(makeResult([
        makeRoute('A', 20),
        makeRoute('B', 30)
    ], 'A'));

    assert.match(
        ui.get('cards').children[0].textContent,
        /AI 추천/
    );

    assert.doesNotMatch(
        ui.get('cards').children[1].textContent,
        /AI 추천/
    );
});


test('UI: missing score is not displayed as zero', () => {

    const ui = createUI();

    ui.renderResult(makeResult([
        makeRoute('A', null, {
            model_coverage: 0.5,
            eligible: false
        })
    ]));

    assert.match(
        ui.get('cards').textContent,
        /계산 불가/
    );

    assert.doesNotMatch(
        ui.get('cards').textContent,
        /0\.0 \/ 100점/
    );
});


test('UI: selected score is displayed correctly', () => {

    const ui = createUI();

    ui.renderResult(makeResult([
        makeRoute('A', 20)
    ]));

    assert.match(
        ui.get('selectedScore').textContent,
        /80\.0 \/ 100점/
    );
});


test('UI: selected route metrics contain four items', () => {

    const ui = createUI();

    ui.renderResult(makeResult([
        makeRoute('A', 20)
    ]));

    assert.equal(
        ui.get('metrics').children.length,
        4
    );

    assert.match(
        ui.get('metrics').textContent,
        /95\.0%/
    );
});


test('UI: route cards support button selection', () => {

    const ui = createUI();

    ui.renderResult(makeResult([
        makeRoute('A', 20),
        makeRoute('B', 30)
    ], 'A'));

    ui.get('cards').children[1].click();

    assert.equal(
        ui.run('state.selected.id'),
        'B'
    );

    assert.match(
        ui.get('selectedScore').textContent,
        /70\.0 \/ 100점/
    );

    assert.equal(
        ui.get('cards').children[1]
            .getAttribute('aria-pressed'),
        'true'
    );
});


test('UI: no routes clears selected information', () => {

    const ui = createUI();

    ui.renderResult(makeResult([]));

    assert.equal(
        ui.get('cards').children.length,
        0
    );

    assert.equal(
        ui.get('metrics').children.length,
        0
    );

    assert.match(
        ui.get('detail').textContent,
        /표시할 경로가 없습니다/
    );
});


test('UI: score explanation does not claim accident probability', () => {

    const ui = createUI();

    ui.renderResult(makeResult([
        makeRoute('A', 20)
    ]));

    assert.match(
        ui.get('selectedScore').textContent,
        /실제 사고 확률이 아닙니다/
    );
});


// ==================================================
// 8. 장소 검색 및 선택 검증
// ==================================================

test('UI: short place query is rejected locally', async () => {

    const ui = createUI();

    ui.get('originQuery').value = '천';

    await ui.run("search('origin')");

    assert.equal(ui.requests.length, 0);

    assert.match(
        ui.get('status').textContent,
        /2글자 이상/
    );
});


test('UI: place search sends correct query and key', async () => {

    const ui = createUI({

        fetcher() {

            return response({
                places: [ORIGIN]
            });
        }
    });

    ui.get('originQuery').value = '천안역';

    await ui.run("search('origin')");

    assert.equal(
        ui.requests[0].endpoint,
        '/api/places'
    );

    assert.equal(
        ui.requests[0].body.query,
        '천안역'
    );

    assert.equal(
        ui.requests[0].body.key,
        'TEST_KEY'
    );

    assert.equal(
        ui.get('originResults').children.length,
        1
    );
});


test('UI: place result can be selected', async () => {

    const ui = createUI({

        fetcher() {

            return response({
                places: [ORIGIN]
            });
        }
    });

    ui.get('originQuery').value = '천안역';

    await ui.run("search('origin')");

    ui.get('originResults').children[0].click();

    assert.equal(
        ui.run('state.origin.name'),
        '천안역'
    );

    assert.match(
        ui.get('originPicked').textContent,
        /선택: 천안역/
    );
});


test('UI: empty place search has clear feedback', async () => {

    const ui = createUI();

    ui.get('originQuery').value = '없는장소';

    await ui.run("search('origin')");

    assert.match(
        ui.get('status').textContent,
        /검색 결과가 없습니다/
    );

    assert.equal(
        ui.get('originSearch').disabled,
        false
    );
});


test('UI: editing selected place clears old routes', () => {

    const ui = createUI();

    ui.selectPlaces();

    ui.renderResult(makeResult([
        makeRoute('A', 20)
    ]));

    ui.get('originQuery').oninput();

    assert.equal(
        ui.run('state.origin'),
        null
    );

    assert.equal(
        ui.run('state.result'),
        null
    );

    assert.equal(
        ui.get('cards').children.length,
        0
    );
});


// ==================================================
// 9. 경로 탐색 및 추천 검증
// ==================================================

test('UI: route search requires selected places', async () => {

    const ui = createUI();

    await ui.run('findRoutes()');

    assert.equal(ui.requests.length, 0);

    assert.match(
        ui.get('status').textContent,
        /각각 선택하세요/
    );
});


test('UI: route request includes selected detour cap', async () => {

    const ui = createUI({

        fetcher() {

            return response(makeResult([
                makeRoute('A', 20),
                makeRoute('B', 30)
            ], 'A'));
        }
    });

    ui.selectPlaces();

    ui.get('detourCap').value = '1.5';

    await ui.run('findRoutes()');

    assert.equal(
        ui.requests[0].endpoint,
        '/api/routes'
    );

    assert.equal(
        ui.requests[0].body.detour_cap,
        '1.5'
    );

    assert.deepEqual(
        ui.requests[0].body.origin,
        ORIGIN
    );

    assert.deepEqual(
        ui.requests[0].body.destination,
        DESTINATION
    );
});


test('UI: recommended route is selected automatically', async () => {

    const ui = createUI({

        fetcher() {

            return response(makeResult([
                makeRoute('A', 30),
                makeRoute('B', 20)
            ], 'B'));
        }
    });

    ui.selectPlaces();

    await ui.run('findRoutes()');

    assert.equal(
        ui.run('state.selected.id'),
        'B'
    );

    assert.equal(
        ui.get('export').hidden,
        false
    );

    assert.equal(
        ui.get('findRoutes').disabled,
        false
    );
});


test('UI: single route shows minimum candidate warning', async () => {

    const ui = createUI({

        fetcher() {

            return response(makeResult([
                makeRoute('A', 20)
            ]));
        }
    });

    ui.selectPlaces();

    await ui.run('findRoutes()');

    assert.match(
        ui.get('status').textContent,
        /최소 2개/
    );

    assert.doesNotMatch(
        ui.get('cards').textContent,
        /AI 추천/
    );
});


test('UI: insufficient eligible routes show recommendation hold', async () => {

    const ui = createUI({

        fetcher() {

            return response(makeResult([
                makeRoute('A', 20),

                makeRoute('B', null, {
                    eligible: false,
                    model_coverage: 0.5
                })
            ]));
        }
    });

    ui.selectPlaces();

    await ui.run('findRoutes()');

    assert.match(
        ui.get('status').textContent,
        /추천을 보류/
    );
});


test('UI: failed route search clears previous result', async () => {

    const ui = createUI({

        fetcher() {

            return response({
                error: '테스트 요청 실패'
            }, false);
        }
    });

    ui.selectPlaces();

    ui.renderResult(makeResult([
        makeRoute('A', 20)
    ]));

    await ui.run('findRoutes()');

    assert.equal(
        ui.run('state.result'),
        null
    );

    assert.equal(
        ui.get('cards').children.length,
        0
    );

    assert.equal(
        ui.get('export').hidden,
        true
    );

    assert.equal(
        ui.get('findRoutes').disabled,
        false
    );

    assert.match(
        ui.get('status').textContent,
        /테스트 요청 실패/
    );
});


test('UI: detour change triggers another route request', async () => {

    const ui = createUI({

        fetcher() {

            return response(makeResult([
                makeRoute('A', 20)
            ]));
        }
    });

    ui.selectPlaces();

    await ui.run('findRoutes()');

    ui.get('detourCap').value = '3';

    ui.get('detourCap').onchange();

    // 비동기 응답 처리 완료 대기
    await new Promise(resolve => setImmediate(resolve));

    assert.equal(ui.requests.length, 2);

    assert.equal(
        ui.requests[1].body.detour_cap,
        '3'
    );
});


// ==================================================
// 10. 비동기 경로 요청 충돌 검증
// ==================================================

test('UI: old route success cannot overwrite newer result', async () => {

    const first = deferred();
    const second = deferred();

    let count = 0;

    const ui = createUI({

        fetcher() {

            return ++count === 1
                ? first.promise
                : second.promise;
        }
    });

    ui.selectPlaces();

    const oldRequest = ui.run('findRoutes()');
    const newRequest = ui.run('findRoutes()');

    second.resolve(response(makeResult([
        makeRoute('NEW', 10)
    ])));

    await newRequest;

    first.resolve(response(makeResult([
        makeRoute('OLD', 90)
    ])));

    await oldRequest;

    assert.equal(
        ui.run('state.selected.id'),
        'NEW'
    );
});


test('UI: old route failure cannot clear newer result', async () => {

    const first = deferred();
    const second = deferred();

    let count = 0;

    const ui = createUI({

        fetcher() {

            return ++count === 1
                ? first.promise
                : second.promise;
        }
    });

    ui.selectPlaces();

    const oldRequest = ui.run('findRoutes()');
    const newRequest = ui.run('findRoutes()');

    second.resolve(response(makeResult([
        makeRoute('NEW', 10)
    ])));

    await newRequest;

    first.reject(
        new Error('이전 요청 실패')
    );

    await oldRequest;

    assert.equal(
        ui.run('state.selected.id'),
        'NEW'
    );

    assert.doesNotMatch(
        ui.get('status').textContent,
        /이전 요청 실패/
    );
});


test('UI: old request cannot enable button during newer request', async () => {

    const first = deferred();
    const second = deferred();

    let count = 0;

    const ui = createUI({

        fetcher() {

            return ++count === 1
                ? first.promise
                : second.promise;
        }
    });

    ui.selectPlaces();

    const oldRequest = ui.run('findRoutes()');
    const newRequest = ui.run('findRoutes()');

    first.resolve(response(makeResult([
        makeRoute('OLD', 90)
    ])));

    await oldRequest;

    assert.equal(
        ui.get('findRoutes').disabled,
        true
    );

    second.resolve(response(makeResult([
        makeRoute('NEW', 10)
    ])));

    await newRequest;

    assert.equal(
        ui.get('findRoutes').disabled,
        false
    );
});


test('UI: editing a place invalidates pending route request', async () => {

    const pending = deferred();

    const ui = createUI({

        fetcher() {

            return pending.promise;
        }
    });

    ui.selectPlaces();

    const request = ui.run('findRoutes()');

    ui.get('originQuery').oninput();

    pending.resolve(response(makeResult([
        makeRoute('OLD', 20)
    ])));

    await request;

    assert.equal(
        ui.run('state.result'),
        null
    );

    assert.equal(
        ui.get('cards').children.length,
        0
    );
});


// ==================================================
// 11. 지도 연동 검증
// ==================================================

test('UI: missing map key shows warning', () => {

    const ui = createUI();

    ui.run('loadMap()');

    assert.match(
        ui.get('mapNotice').textContent,
        /JavaScript 키를 입력/
    );

    assert.equal(ui.scripts.length, 0);
});


test('UI: map SDK script is created with entered key', () => {

    const ui = createUI();

    ui.get('mapKey').value = 'TEST_MAP_KEY';

    ui.run('loadMap()');

    assert.equal(ui.scripts.length, 1);

    assert.match(
        ui.scripts[0].src,
        /appkey=TEST_MAP_KEY/
    );
});


test('UI: loaded map SDK initializes map', () => {

    const ui = createUI({
        mapReady: true
    });

    ui.get('mapKey').value = 'TEST_MAP_KEY';

    ui.run('loadMap()');

    assert.equal(
        ui.run('Boolean(state.map)'),
        true
    );

    assert.equal(
        ui.get('mapNotice').hidden,
        true
    );
});


test('UI: selected places create map markers', () => {

    const ui = createUI({
        mapReady: true
    });

    ui.get('mapKey').value = 'TEST_MAP_KEY';

    ui.run('loadMap()');

    ui.selectPlaces();

    assert.equal(
        ui.run('Object.keys(state.markers).length'),
        2
    );
});


test('UI: route polylines distinguish selected route', () => {

    const ui = createUI({
        mapReady: true
    });

    ui.get('mapKey').value = 'TEST_MAP_KEY';

    ui.run('loadMap()');

    ui.renderResult(makeResult([
        makeRoute('A', 20),
        makeRoute('B', 30)
    ], 'A'));

    assert.equal(
        ui.run('state.lines.length'),
        2
    );

    assert.equal(
        ui.run('state.lines[0].strokeColor'),
        '#879b9e'
    );

    assert.equal(
        ui.run('state.lines[1].strokeColor'),
        '#087ba8'
    );
});


test('UI: map zoom respects level boundaries', () => {

    const ui = createUI({
        mapReady: true
    });

    ui.get('mapKey').value = 'TEST_MAP_KEY';

    ui.run('loadMap()');

    ui.run('state.map.setLevel(1)');

    ui.get('zoomIn').click();

    assert.equal(
        ui.run('state.map.getLevel()'),
        1
    );

    ui.run('state.map.setLevel(14)');

    ui.get('zoomOut').click();

    assert.equal(
        ui.run('state.map.getLevel()'),
        14
    );
});


// ==================================================
// 12. 결과 저장 및 초기화 검증
// ==================================================

test('UI: export does nothing without route result', () => {

    const ui = createUI();

    ui.get('export').click();

    assert.equal(ui.downloads.length, 0);
});


test('UI: export creates correct JSON download', async () => {

    const ui = createUI();

    const result = makeResult([
        makeRoute('A', 20),
        makeRoute('B', 30)
    ], 'A');

    ui.renderResult(result);

    ui.get('export').click();

    assert.equal(ui.downloads.length, 1);

    assert.equal(
        ui.downloads[0].download,
        'saferoute_kakao_model_routes.json'
    );

    assert.equal(
        ui.blobs[0].type,
        'application/json'
    );

    const exported = JSON.parse(
        await ui.blobs[0].text()
    );

    assert.deepEqual(exported, result);

    assert.equal(
        Object.hasOwn(exported, 'key'),
        false
    );
});


test('UI: reset clears route state and displayed information', () => {

    const ui = createUI();

    ui.renderResult(makeResult([
        makeRoute('A', 20)
    ]));

    ui.run('resetRoutes()');

    assert.equal(ui.run('state.result'), null);
    assert.equal(ui.run('state.selected'), null);

    assert.equal(
        ui.get('cards').children.length,
        0
    );

    assert.equal(
        ui.get('selectedScore').children.length,
        0
    );

    assert.equal(
        ui.get('metrics').children.length,
        0
    );

    assert.equal(
        ui.get('recommendations').hidden,
        true
    );

    assert.equal(
        ui.get('export').hidden,
        true
    );
});


// ==================================================
// 13. HTML 및 초기 실행 상태 검증
// ==================================================

test('UI: HTML contains correct model and policy revision', () => {

    assert.match(HTML, /T09/);
    assert.match(HTML, /P01-v1\.1/);
});


test('UI: API key inputs use password fields', () => {

    assert.match(
        HTML,
        /id="apiKey"\s+type="password"/
    );

    assert.match(
        HTML,
        /id="mapKey"\s+type="password"/
    );
});


test('UI: HTML contains all supported detour options', () => {

    for (const value of [
        '1.5', '2', '3', 'unlimited'
    ]) {

        assert.ok(
            HTML.includes(`value="${value}"`)
        );
    }
});


test('UI: direct HTML execution shows server requirement', () => {

    const ui = createUI({
        protocol: 'file:'
    });

    assert.match(
        ui.get('status').textContent,
        /START_SAFEROUTE\.cmd/
    );
});


test('UI: environment API key updates input guidance', async () => {

    const ui = createUI({
        envKey: true
    });

    await new Promise(resolve => setImmediate(resolve));

    assert.equal(
        ui.run('state.envKey'),
        true
    );

    assert.match(
        ui.get('keyHint').textContent,
        /서버 환경 변수/
    );
});