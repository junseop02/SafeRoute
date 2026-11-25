'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const { spawn } = require('node:child_process');
const { once } = require('node:events');


// ==================================================
// 1. 테스트 서버 기본 설정
// ==================================================

const ROOT = path.resolve(__dirname, '..');

const SERVER_FILE = path.join(
    ROOT,
    'service',
    'server.js'
);

const NODE_EXE = process.execPath;

const HOST = '127.0.0.1';

let PORT;
let BASE_URL;
let serverProcess;

let serverOutput = '';
let serverError = '';


// 사용 가능한 테스트 포트 확인
async function getFreePort() {
    const socket = net.createServer();

    await new Promise((resolve, reject) => {
        socket.once('error', reject);
        socket.listen(0, HOST, resolve);
    });

    const port = socket.address().port;

    await new Promise((resolve, reject) => {
        socket.close(error => {
            if (error) {
                reject(error);
            } else {
                resolve();
            }
        });
    });

    return port;
}


// 서버 시작 대기
async function waitForServer() {
    const deadline = Date.now() + 30000;

    while (Date.now() < deadline) {

        if (
            serverProcess.exitCode !== null ||
            serverProcess.signalCode !== null
        ) {
            throw new Error(
                `Test server stopped:\n${serverError}`
            );
        }

        try {
            const response = await fetch(
                `${BASE_URL}/api/status`,
                {
                    signal: AbortSignal.timeout(1000)
                }
            );

            if (response.ok) {
                return;
            }

        } catch {
            // 서버 초기화 대기
        }

        await new Promise(resolve => {
            setTimeout(resolve, 100);
        });
    }

    throw new Error(
        `Server startup timeout:\n${serverOutput}\n${serverError}`
    );
}


// ==================================================
// 2. 테스트 서버 시작 및 종료
// ==================================================

test.before(async () => {

    PORT = await getFreePort();

    BASE_URL = `http://${HOST}:${PORT}`;

    serverProcess = spawn(
        NODE_EXE,
        [SERVER_FILE],
        {
            cwd: ROOT,

            env: {
                ...process.env,

                SAFEROUTE_PORT: String(PORT),

                SAFEROUTE_NO_BROWSER: '1',

                KAKAO_REST_API_KEY: ''
            },

            stdio: ['ignore', 'pipe', 'pipe'],

            windowsHide: true
        }
    );

    serverProcess.stdout.on('data', chunk => {
        serverOutput += chunk.toString();
    });

    serverProcess.stderr.on('data', chunk => {
        serverError += chunk.toString();
    });

    await waitForServer();
});


test.after(async () => {

    if (!serverProcess) {
        return;
    }

    if (
        serverProcess.exitCode === null &&
        serverProcess.signalCode === null
    ) {
        const exited = once(
            serverProcess,
            'exit'
        );

        serverProcess.kill();

        await exited;
    }
});


// ==================================================
// 3. 공통 HTTP 요청 함수
// ==================================================

async function request(endpoint, options = {}) {

    return fetch(
        `${BASE_URL}${endpoint}`,
        {
            ...options,
            signal: AbortSignal.timeout(5000)
        }
    );
}


async function postJson(
    endpoint,
    body,
    headers = {}
) {

    return request(endpoint, {
        method: 'POST',

        headers: {
            'Content-Type': 'application/json',
            ...headers
        },

        body: JSON.stringify(body)
    });
}


async function assertError(
    response,
    status,
    message
) {

    assert.equal(response.status, status);

    const body = await response.json();

    assert.equal(typeof body.error, 'string');

    if (message) {
        assert.match(body.error, message);
    }

    return body;
}


// ==================================================
// 4. 서버 상태 API 검증
// ==================================================

test('Server: status endpoint returns HTTP 200', async () => {
    const response = await request('/api/status');

    assert.equal(response.status, 200);
});


test('Server: national tile count is correct', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(body.national_tiles, 1172);
});


test('Server: national link count is correct', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(body.national_links, 1562356);
});


test('Server: model and policy versions are correct', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(body.model_version, 'T09');
    assert.equal(body.policy_version, 'P01');
    assert.equal(body.policy_revision, 'P01-v1.1');
});


test('Server: tile schema version is correct', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(
        body.tile_schema_version,
        'T09-P01-tiles-v1'
    );
});


test('Server: minimum coverage is 80 percent', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(body.minimum_coverage, 0.8);
});


test('Server: candidate requirements are correct', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(
        body.minimum_distinct_candidates,
        2
    );

    assert.equal(
        body.search_target_candidates,
        3
    );
});


test('Server: deployment and score meaning are correct', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(body.deployment_status, 'staging');

    assert.equal(
        body.score_meaning,
        'historical_hotspot_membership_score'
    );
});


test('Server: no environment API key is exposed', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(
        body.api_key_from_environment,
        false
    );

    assert.equal(
        Object.hasOwn(body, 'key'),
        false
    );
});


test('Server: Korean UTF-8 status text is preserved', async () => {
    const response = await request('/api/status');
    const body = await response.json();

    assert.equal(
        body.model_scope,
        '전국 표준 도로망이 매칭된 경로만'
    );
});


test('Server: JSON security and encoding headers', async () => {
    const response = await request('/api/status');

    assert.match(
        response.headers.get('content-type'),
        /application\/json.*utf-8/i
    );

    assert.equal(
        response.headers.get('cache-control'),
        'no-store'
    );

    assert.equal(
        response.headers.get('x-content-type-options'),
        'nosniff'
    );
});


test('Server: status endpoint rejects POST', async () => {
    const response = await postJson(
        '/api/status',
        {}
    );

    await assertError(response, 405);
});


// ==================================================
// 5. 정적 파일 제공 검증
// ==================================================

test('Server: root page returns HTML', async () => {
    const response = await request('/');

    assert.equal(response.status, 200);

    assert.match(
        response.headers.get('content-type'),
        /text\/html.*utf-8/i
    );

    const body = await response.text();

    assert.match(body, /SafeRoute/i);
});


test('Server: index.html matches the local file', async () => {
    const response = await request('/index.html');

    assert.equal(response.status, 200);

    const actual = Buffer.from(
        await response.arrayBuffer()
    );

    const expected = fs.readFileSync(
        path.join(ROOT, 'service', 'index.html')
    );

    assert.deepEqual(actual, expected);
});


test('Server: app.js is served correctly', async () => {
    const response = await request('/app.js');

    assert.equal(response.status, 200);

    assert.match(
        response.headers.get('content-type'),
        /text\/javascript.*utf-8/i
    );

    const actual = Buffer.from(
        await response.arrayBuffer()
    );

    const expected = fs.readFileSync(
        path.join(ROOT, 'service', 'app.js')
    );

    assert.deepEqual(actual, expected);
});


test('Server: style.css is served correctly', async () => {
    const response = await request('/style.css');

    assert.equal(response.status, 200);

    assert.match(
        response.headers.get('content-type'),
        /text\/css.*utf-8/i
    );

    const actual = Buffer.from(
        await response.arrayBuffer()
    );

    const expected = fs.readFileSync(
        path.join(ROOT, 'service', 'style.css')
    );

    assert.deepEqual(actual, expected);
});


test('Server: HEAD returns headers without a body', async () => {
    const response = await request('/', {
        method: 'HEAD'
    });

    assert.equal(response.status, 200);

    assert.ok(
        Number(
            response.headers.get('content-length')
        ) > 0
    );

    assert.equal(await response.text(), '');
});


test('Server: static files include security headers', async () => {
    const response = await request('/app.js');

    assert.equal(
        response.headers.get('cache-control'),
        'no-store'
    );

    assert.equal(
        response.headers.get('x-content-type-options'),
        'nosniff'
    );
});


test('Server: static POST requests are rejected', async () => {
    const response = await postJson('/', {});

    await assertError(response, 405);
});


test('Server: unknown static file returns 404', async () => {
    const response = await request(
        '/not-found.txt'
    );

    await assertError(response, 404);
});


// ==================================================
// 6. 내부 파일 접근 차단 검증
// ==================================================

const blockedPaths = [

    '/server.js',

    '/matcher.js',

    '/kakao.js',

    '/data/national_manifest.json',

    '/data/national_route_policy_P01_v1_1.json',

    '/data/national_service_interface_P01_v1_1.json',

    '/data/tiles/',

    '/.env',

    '/requirements.txt',

    '/%2e%2e/requirements.txt'
];


for (const endpoint of blockedPaths) {

    test(
        `Server: blocks internal path ${endpoint}`,
        async () => {

            const response = await request(endpoint);

            await assertError(response, 404);
        }
    );
}


// ==================================================
// 7. API 요청 검증
// ==================================================

test('Server: unknown API endpoint returns 404', async () => {
    const response = await postJson(
        '/api/unknown',
        {}
    );

    await assertError(response, 404);
});


test('Server: GET places endpoint is rejected', async () => {
    const response = await request('/api/places');

    await assertError(response, 404);
});


test('Server: GET routes endpoint is rejected', async () => {
    const response = await request('/api/routes');

    await assertError(response, 404);
});


test('Server: external origin is rejected', async () => {
    const response = await postJson(
        '/api/routes',
        {},
        {
            Origin: 'https://example.com'
        }
    );

    await assertError(
        response,
        403,
        /로컬 화면/
    );
});


test('Server: local IP origin is accepted', async () => {
    const response = await postJson(
        '/api/routes',
        {},
        {
            Origin: BASE_URL
        }
    );

    // Origin 검사를 통과한 뒤 키 미입력 오류
    await assertError(
        response,
        400,
        /REST API 키/
    );
});


test('Server: localhost origin is accepted', async () => {
    const response = await postJson(
        '/api/routes',
        {},
        {
            Origin: `http://localhost:${PORT}`
        }
    );

    await assertError(
        response,
        400,
        /REST API 키/
    );
});


test('Server: unsupported content type returns 415', async () => {
    const response = await request('/api/routes', {
        method: 'POST',

        headers: {
            'Content-Type': 'text/plain'
        },

        body: '{}'
    });

    await assertError(response, 415);
});


test('Server: missing content type returns 415', async () => {
    const response = await request('/api/routes', {
        method: 'POST'
    });

    await assertError(response, 415);
});


test('Server: malformed JSON returns 400', async () => {
    const response = await request('/api/routes', {
        method: 'POST',

        headers: {
            'Content-Type': 'application/json'
        },

        body: '{"invalid":'
    });

    await assertError(
        response,
        400,
        /JSON 요청 형식/
    );
});


test('Server: non-object JSON is rejected', async () => {

    for (const body of [null, [], 123, 'text']) {

        const response = await postJson(
            '/api/routes',
            body
        );

        await assertError(
            response,
            400,
            /JSON 객체/
        );
    }
});


test('Server: oversized request returns 413', async () => {
    const response = await postJson(
        '/api/routes',
        {
            data: 'A'.repeat(17000)
        }
    );

    await assertError(
        response,
        413,
        /요청 크기/
    );
});


test('Server: exact body size limit is accepted', async () => {

    const body = JSON.stringify({
        data: 'A'.repeat(16373)
    });

    assert.equal(
        Buffer.byteLength(body),
        16384
    );

    const response = await request('/api/routes', {
        method: 'POST',

        headers: {
            'Content-Type': 'application/json'
        },

        body
    });

    // 크기 검사를 통과한 뒤 키 미입력 오류
    await assertError(
        response,
        400,
        /REST API 키/
    );
});


// ==================================================
// 8. API 키 및 우회 배율 검증
// ==================================================

test('Server: missing places API key is rejected', async () => {
    const response = await postJson(
        '/api/places',
        {
            query: '천안역'
        }
    );

    await assertError(
        response,
        400,
        /REST API 키/
    );
});


test('Server: missing routes API key is rejected', async () => {
    const response = await postJson(
        '/api/routes',
        {}
    );

    await assertError(
        response,
        400,
        /REST API 키/
    );
});


test('Server: whitespace-only API key is rejected', async () => {
    const response = await postJson(
        '/api/routes',
        {
            key: '   '
        }
    );

    await assertError(
        response,
        400,
        /REST API 키/
    );
});


test('Server: missing detour cap is rejected', async () => {
    const response = await postJson(
        '/api/routes',
        {
            key: 'TEST_KEY'
        }
    );

    await assertError(
        response,
        400,
        /우회 범위/
    );
});


test('Server: null detour cap is rejected', async () => {
    const response = await postJson(
        '/api/routes',
        {
            key: 'TEST_KEY',
            detour_cap: null
        }
    );

    await assertError(
        response,
        400,
        /우회 범위/
    );
});


test('Server: invalid detour caps are rejected', async () => {

    const invalidCaps = [
        0, -1, 2.5, 4,
        'invalid', true, {}
    ];

    for (const cap of invalidCaps) {

        const response = await postJson(
            '/api/routes',
            {
                key: 'TEST_KEY',
                detour_cap: cap
            }
        );

        await assertError(
            response,
            400,
            /우회 배율/
        );
    }
});


test('Server: invalid coordinates are rejected before API calls', async () => {
    const response = await postJson(
        '/api/routes',
        {
            key: 'TEST_KEY',

            detour_cap: 2,

            origin: {
                x: 0,
                y: 0
            },

            destination: {
                x: 127,
                y: 36
            }
        }
    );

    await assertError(
        response,
        400,
        /국내 장소 좌표/
    );
});


test('Server: identical locations are rejected before API calls', async () => {

    const location = {
        x: 127,
        y: 36
    };

    const response = await postJson(
        '/api/routes',
        {
            key: 'TEST_KEY',

            detour_cap: 2,

            origin: location,

            destination: location
        }
    );

    await assertError(
        response,
        400,
        /서로 다른 출발지/
    );
});


// ==================================================
// 9. 비정상 요청 이후 서버 복구 검증
// ==================================================

test('Server: remains healthy after invalid requests', async () => {

    const response = await request('/api/status');

    assert.equal(response.status, 200);

    const body = await response.json();

    assert.equal(body.model_version, 'T09');

    assert.equal(
        body.policy_revision,
        'P01-v1.1'
    );

    assert.equal(body.national_tiles, 1172);
});