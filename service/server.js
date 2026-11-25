'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const {
    makeIndexProvider
} = require('./matcher');

const {
    searchPlaces,
    findRoutes,
    normalizeDetourCap
} = require('./kakao');


// ==================================================
// 1. 서버 기본 설정
// ==================================================

const ROOT = __dirname;

const PORT = Number(
    process.env.SAFEROUTE_PORT || 8090
);

const HOST = '127.0.0.1';

const ENV_KEY = (
    process.env.KAKAO_REST_API_KEY || ''
).trim();

const MAX_BODY_BYTES = 16384;

if (
    !Number.isInteger(PORT) ||
    PORT < 1 ||
    PORT > 65535
) {
    throw new Error('서버 포트 설정이 올바르지 않습니다.');
}


// ==================================================
// 2. 신규 전국 도로 타일 연결
// ==================================================

const index = makeIndexProvider(
    path.join(ROOT, 'data', 'tiles')
);

const modelStatus = index.status();

if (
    !index.national ||
    modelStatus.model_version !== 'T09' ||
    modelStatus.policy_version !== 'P01' ||
    modelStatus.tile_schema_version !== 'T09-P01-tiles-v1'
) {
    throw new Error(
        'T09/P01 전국 도로 데이터 설정이 올바르지 않습니다.'
    );
}


// ==================================================
// 3. 정적 파일 설정
// ==================================================

const staticFiles = {

    '/': {
        file: 'index.html',
        type: 'text/html; charset=utf-8'
    },

    '/index.html': {
        file: 'index.html',
        type: 'text/html; charset=utf-8'
    },

    '/app.js': {
        file: 'app.js',
        type: 'text/javascript; charset=utf-8'
    },

    '/style.css': {
        file: 'style.css',
        type: 'text/css; charset=utf-8'
    }
};

const allowedOrigins = new Set([
    `http://127.0.0.1:${PORT}`,
    `http://localhost:${PORT}`
]);


// ==================================================
// 4. JSON 응답 처리
// ==================================================

function send(res, status, obj) {

    const body = JSON.stringify(obj);

    res.writeHead(status, {

        'Content-Type':
            'application/json; charset=utf-8',

        'Cache-Control': 'no-store',

        'X-Content-Type-Options': 'nosniff',

        'Content-Length': Buffer.byteLength(body)
    });

    res.end(body);
}


// ==================================================
// 5. JSON 요청 처리
// ==================================================

async function readJson(req) {

    const contentType = (
        req.headers['content-type'] || ''
    ).split(';')[0].trim().toLowerCase();

    if (contentType !== 'application/json') {

        const error = new Error(
            'JSON 요청만 받습니다.'
        );

        error.status = 415;

        throw error;
    }

    const chunks = [];

    let size = 0;
    let oversized = false;

    for await (const chunk of req) {

        size += chunk.length;

        if (size > MAX_BODY_BYTES) {

            oversized = true;

        } else if (!oversized) {

            chunks.push(chunk);
        }
    }

    if (oversized) {

        const error = new Error(
            '요청 크기가 너무 큽니다.'
        );

        error.status = 413;

        throw error;
    }

    let body;

    try {

        body = JSON.parse(
            Buffer.concat(chunks).toString('utf8')
        );

    } catch {

        throw new Error(
            'JSON 요청 형식이 올바르지 않습니다.'
        );
    }

    if (
        body === null ||
        typeof body !== 'object' ||
        Array.isArray(body)
    ) {
        throw new Error(
            'JSON 객체를 입력하세요.'
        );
    }

    return body;
}


// ==================================================
// 6. 정적 화면 파일 제공
// ==================================================

function serveStatic(req, res, pathname) {

    if (!['GET', 'HEAD'].includes(req.method)) {

        return send(res, 405, {
            error: 'GET 또는 HEAD만 허용합니다.'
        });
    }

    if (!Object.hasOwn(staticFiles, pathname)) {

        return send(res, 404, {
            error: '파일을 찾을 수 없습니다.'
        });
    }

    const entry = staticFiles[pathname];

    const file = path.join(ROOT, entry.file);

    fs.readFile(file, (error, data) => {

        if (error) {

            return send(res, 404, {
                error: '파일을 찾을 수 없습니다.'
            });
        }

        res.writeHead(200, {

            'Content-Type': entry.type,

            'Content-Length': data.length,

            'Cache-Control': 'no-store',

            'X-Content-Type-Options': 'nosniff'
        });

        if (req.method === 'HEAD') {

            return res.end();
        }

        res.end(data);
    });
}


// ==================================================
// 7. HTTP 서버
// ==================================================

const server = http.createServer(
    async (req, res) => {

        try {

            const pathname = new URL(
                req.url,
                `http://${HOST}:${PORT}`
            ).pathname;


            // --------------------------------------
            // 7-1. 서버 상태 조회
            // --------------------------------------

            if (pathname === '/api/status') {

                if (req.method !== 'GET') {

                    return send(res, 405, {
                        error: 'GET 요청만 허용합니다.'
                    });
                }

                return send(res, 200, {

                    api_key_from_environment:
                        Boolean(ENV_KEY),

                    ...modelStatus,

                    model_scope:
                        '전국 표준 도로망이 매칭된 경로만',

                    minimum_coverage: 0.8,

                    minimum_distinct_candidates: 2,

                    search_target_candidates: 3,

                    policy_revision: 'P01-v1.1',

                    score_meaning:
                        'historical_hotspot_membership_score',

                    deployment_status:
                        'staging'
                });
            }


            // --------------------------------------
            // 7-2. 카카오 API 요청
            // --------------------------------------

            if (pathname.startsWith('/api/')) {

                if (
                    req.method !== 'POST' ||
                    ![
                        '/api/places',
                        '/api/routes'
                    ].includes(pathname)
                ) {
                    return send(res, 404, {
                        error: '없는 API입니다.'
                    });
                }

                const requestOrigin =
                    req.headers.origin;

                if (
                    requestOrigin &&
                    !allowedOrigins.has(requestOrigin)
                ) {
                    return send(res, 403, {
                        error:
                            '로컬 화면에서만 요청할 수 있습니다.'
                    });
                }

                const body = await readJson(req);

                const key = ENV_KEY || String(
                    body.key || ''
                ).trim();

                if (!key) {

                    return send(res, 400, {
                        error:
                            '카카오 REST API 키를 입력하세요.'
                    });
                }


                // ----------------------------------
                // 장소 검색
                // ----------------------------------

                if (pathname === '/api/places') {

                    const places = await searchPlaces(
                        body.query,
                        key
                    );

                    return send(res, 200, {
                        places
                    });
                }


                // ----------------------------------
                // 경로 탐색 및 P01 추천
                // ----------------------------------

                if (
                    body.detour_cap === undefined ||
                    body.detour_cap === null
                ) {
                    return send(res, 400, {
                        error: '우회 범위를 확인하세요.'
                    });
                }

                const cap = normalizeDetourCap(
                    body.detour_cap
                );

                const result = await findRoutes(
                    body.origin,
                    body.destination,
                    key,
                    index,
                    fetch,
                    cap
                );


                // Infinity의 JSON null 변환 방지
                const response = {

                    ...result,

                    detour_cap:
                        cap === Infinity
                            ? 'unlimited'
                            : cap,

                    model_version:
                        modelStatus.model_version,

                    policy_version:
                        modelStatus.policy_version,

                    tile_schema_version:
                        modelStatus.tile_schema_version
                };

                return send(res, 200, response);
            }


            // --------------------------------------
            // 7-3. 화면 파일 요청
            // --------------------------------------

            return serveStatic(
                req,
                res,
                pathname
            );

        } catch (error) {

            if (!res.headersSent) {

                const status = (
                    Number.isInteger(error.status) &&
                    error.status >= 400 &&
                    error.status <= 599
                ) ? error.status : 400;

                return send(res, status, {

                    error: error.message ||
                        '요청 처리에 실패했습니다.'
                });
            }

            res.end();
        }
    }
);


// ==================================================
// 8. 서버 실행 오류 처리
// ==================================================

server.on('error', error => {

    if (error.code === 'EADDRINUSE') {

        console.error(
            `포트 ${PORT}가 사용 중입니다.`
        );

        console.error(
            '기존 서버를 종료하거나 SAFEROUTE_PORT를 변경하세요.'
        );

    } else {

        console.error(error.message);
    }

    process.exitCode = 1;
});


// ==================================================
// 9. 서버 실행
// ==================================================

server.listen(PORT, HOST, () => {

    const url = `http://${HOST}:${PORT}`;

    console.log('');
    console.log('===== SafeRoute National Service =====');

    console.log('Server:', url);

    console.log(
        'National Links:',
        modelStatus.national_links
    );

    console.log(
        'National Tiles:',
        modelStatus.national_tiles
    );

    console.log(
        'Model:',
        modelStatus.model_version
    );

    console.log(
        'Policy:',
        modelStatus.policy_version
    );

    console.log(
        'Tile Schema:',
        modelStatus.tile_schema_version
    );

    console.log(
        'REST API Key:',
        ENV_KEY
            ? 'Environment configured'
            : 'Enter in browser'
    );

    console.log('Deployment Status: Staging');
    console.log('Server Startup: PASS');
    console.log('======================================');
    console.log('');

    if (
        process.env.SAFEROUTE_NO_BROWSER !== '1' &&
        process.platform === 'win32'
    ) {

        const browser = spawn(
            'explorer.exe',
            [url],
            { windowsHide: true }
        );

        browser.on('error', () => {

            console.log(
                `브라우저에서 ${url}을 직접 열어주세요.`
            );
        });
    }
});