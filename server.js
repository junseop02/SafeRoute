
'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const root = __dirname;
const port = Number(process.env.SAFEROUTE_PORT || 8090);

const types = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.geojson': 'application/geo+json; charset=utf-8',
    '.md': 'text/plain; charset=utf-8',
};


// 카카오모빌리티 응답을 SafeRoute 경로 형식으로 변환
function normalizeDrivingRoute(route, priority) {
    const geometry = [];
    let roadCount = 0;
    let guideCount = 0;

    for (const section of route.sections || []) {
        roadCount += section.roads?.length || 0;
        guideCount += section.guides?.length || 0;

        for (const road of section.roads || []) {
            const coordinates = [];
            const vertexes = road.vertexes || [];

            for (let i = 0; i + 1 < vertexes.length; i += 2) {
                coordinates.push([
                    vertexes[i],
                    vertexes[i + 1]
                ]);
            }

            if (coordinates.length >= 2) {
                geometry.push(coordinates);
            }
        }
    }

    return {
        priority,
        distance_m: route.summary.distance,
        duration_s: route.summary.duration,
        road_count: roadCount,
        guide_count: guideCount,
        geometry
    };
}

// 도로 좌표를 이용한 경로 중복 검사
function routeGeometryKey(route) {
    return JSON.stringify(route.geometry);
}


function fetchKakaoDrivingRoute(origin, destination, priority, apiKey) {
    return new Promise((resolve, reject) => {
        const url = new URL(
            'https://apis-navi.kakaomobility.com/v1/directions'
        );

        url.searchParams.set('origin', origin);
        url.searchParams.set('destination', destination);
        url.searchParams.set('priority', priority);

        const request = https.get(
            url,
            {
                headers: {
                    Authorization: `KakaoAK ${apiKey}`
                },
                timeout: 15000
            },
            response => {
                let body = '';

                response.setEncoding('utf8');

                response.on('data', chunk => {
                    body += chunk;
                });

                response.on('end', () => {
                    try {
                        const data = JSON.parse(body);

                        if (response.statusCode !== 200) {
                            return reject(
                                new Error(
                                    `Kakao API ${priority}: HTTP ${response.statusCode}`
                                )
                            );
                        }

                        const route = data.routes?.[0];

                        if (!route || route.result_code !== 0) {
                            return reject(
                                new Error(
                                    route?.result_msg ||
                                    `No route for ${priority}`
                                )
                            );
                        }

                        resolve(normalizeDrivingRoute(route, priority));

                    } catch (error) {
                        reject(error);
                    }
                });
            }
        );

        request.on('timeout', () => {
            request.destroy(new Error('Kakao API timeout'));
        });

        request.on('error', reject);
    });
}


// Persistent live AI worker: initialize road links/model once.
let liveWorker = null;
let liveBuffer = Buffer.alloc(0);
let liveQueue = [];
let livePending = null;
let liveReady = false;
let liveInitialization = null;
const liveTimeoutMs = 120000;
function failLiveWorker(child, error) {
    if (liveWorker !== child) return;
    liveWorker = null; liveReady = false;
    clearTimeout(child.readyTimer);
    const tasks = liveQueue.splice(0);
    if (livePending) { tasks.unshift(livePending); livePending = null; }
    for (const task of tasks) { clearTimeout(task.queueTimer); clearTimeout(task.timer); task.reject(error); }
    child.kill();
}
function startLiveWorker() {
    if (liveWorker) return;
    const python = path.join(root, '.venv', 'Scripts', 'python.exe');
    const script = path.join(root, 'training', 'predict_live_routes.py');
    const child = spawn(python, [script, '--worker'], {cwd: root, windowsHide: true, stdio: ['pipe','pipe','pipe']});
    const started = Date.now();
    liveWorker = child; liveReady = false; liveBuffer = Buffer.alloc(0);
    child.readyTimer = setTimeout(() => failLiveWorker(child, new Error('AI worker initialization timeout')), liveTimeoutMs);
    child.stderr.on('data', chunk => console.error('[AI]', chunk.toString().trim()));
    child.stdin.on('error', error => failLiveWorker(child, error));
    child.stdout.on('data', chunk => {
        if (liveWorker !== child) return;
        liveBuffer = Buffer.concat([liveBuffer, Buffer.from(chunk)]);
        let newline;
        while ((newline = liveBuffer.indexOf(10)) !== -1) {
            const line = liveBuffer.subarray(0, newline).toString('utf8').trim();
            liveBuffer = liveBuffer.subarray(newline + 1);
            if (!line) continue;
            let data;
            try { data = JSON.parse(line); } catch { failLiveWorker(child, new Error('Invalid AI worker response')); return; }
            if (data.type === 'ready') {
                clearTimeout(child.readyTimer); liveReady = true;
                liveInitialization = {...data.timings, process_ready_ms: Date.now()-started};
                dispatchLive(); continue;
            }
            if (livePending) {
                const task = livePending; livePending = null;
                clearTimeout(task.timer);
                task.resolve({...data, timings: {...data.timings, worker_initialization: liveInitialization,
                    queue_wait_ms: task.dispatchedAt-task.queuedAt, server_total_ms: Date.now()-task.queuedAt}});
                dispatchLive();
            }
        }
    });
    child.on('error', error => failLiveWorker(child, error));
    child.on('close', () => failLiveWorker(child, new Error('AI worker exited')));
}
function dispatchLive() {
    if (!liveReady || livePending || !liveQueue.length || !liveWorker) return;
    const task = liveQueue.shift(), child = liveWorker;
    clearTimeout(task.queueTimer);
    livePending = task; task.dispatchedAt = Date.now();
    task.timer = setTimeout(() => failLiveWorker(child, new Error('AI prediction timeout')), liveTimeoutMs);
    try { child.stdin.write(JSON.stringify(task.request) + '\n'); }
    catch (error) { failLiveWorker(child, error); }
}
function predictLive(request) {
    return new Promise((resolve, reject) => {
        const task = {request, resolve, reject, timer:null, queuedAt:Date.now()};
        task.queueTimer = setTimeout(() => {
            const position = liveQueue.indexOf(task);
            if (position < 0) return;
            liveQueue.splice(position, 1);
            reject(new Error('AI queue timeout'));
        }, liveTimeoutMs);
        liveQueue.push(task);
        try { startLiveWorker(); dispatchLive(); }
        catch (error) {
            clearTimeout(task.queueTimer);
            liveQueue.splice(liveQueue.indexOf(task), 1); reject(error);
        }
    });
}
process.once('exit', () => { if (liveWorker) liveWorker.kill(); });

const server = http.createServer((req, res) => {
    let pathname;

    try {
        pathname = new URL(req.url, 'http://localhost').pathname;
    } catch {
        res.writeHead(400);
        return res.end('Bad request');
    }

    // 카카오 장소 검색 API
    if (pathname === '/api/search') {
        if (req.method !== 'GET') {
            res.writeHead(405, { Allow: 'GET' });
            return res.end('Method not allowed');
        }

        const apiKey = process.env.KAKAO_REST_API_KEY;

        if (!apiKey) {
            res.writeHead(503, {
                'Content-Type': 'application/json; charset=utf-8'
            });

            return res.end(JSON.stringify({
                success: false,
                error: 'KAKAO_REST_API_KEY is not configured'
            }));
        }

        const requestUrl = new URL(req.url, 'http://localhost');
        const query = (requestUrl.searchParams.get('query') || '').trim();

        if (!query || query.length > 100) {
            res.writeHead(400, {
                'Content-Type': 'application/json; charset=utf-8'
            });

            return res.end(JSON.stringify({
                success: false,
                error: 'query must contain 1-100 characters'
            }));
        }

        const kakaoUrl = new URL(
            'https://dapi.kakao.com/v2/local/search/keyword.json'
        );

        kakaoUrl.searchParams.set('query', query);
        kakaoUrl.searchParams.set('size', '10');

        const kakaoRequest = https.get(
            kakaoUrl,
            {
                headers: {
                    Authorization: `KakaoAK ${apiKey}`
                },
                timeout: 10000
            },
            kakaoResponse => {
                let responseBody = '';

                kakaoResponse.setEncoding('utf8');

                kakaoResponse.on('data', chunk => {
                    responseBody += chunk;
                });

                kakaoResponse.on('end', () => {
                    try {
                        const data = JSON.parse(responseBody);

                        if (kakaoResponse.statusCode !== 200) {
                            res.writeHead(502, {
                                'Content-Type': 'application/json; charset=utf-8'
                            });

                            return res.end(JSON.stringify({
                                success: false,
                                error: 'Kakao API request failed',
                                status: kakaoResponse.statusCode
                            }));
                        }

                        const places = data.documents.map(place => ({
                            id: place.id,
                            name: place.place_name,
                            address: place.road_address_name ||
                                     place.address_name,
                            longitude: Number(place.x),
                            latitude: Number(place.y)
                        }));

                        res.writeHead(200, {
                            'Content-Type': 'application/json; charset=utf-8'
                        });

                        res.end(JSON.stringify({
                            success: true,
                            query,
                            places
                        }));

                    } catch (error) {
                        console.error('Kakao response error:', error.message);

                        res.writeHead(502, {
                            'Content-Type': 'application/json; charset=utf-8'
                        });

                        res.end(JSON.stringify({
                            success: false,
                            error: 'Invalid Kakao API response'
                        }));
                    }
                });
            }
        );

        kakaoRequest.on('timeout', () => {
            kakaoRequest.destroy(new Error('Kakao API timeout'));
        });

        kakaoRequest.on('error', error => {
            console.error('Kakao search error:', error.message);

            if (!res.writableEnded) {
                res.writeHead(502, {
                    'Content-Type': 'application/json; charset=utf-8'
                });

                res.end(JSON.stringify({
                    success: false,
                    error: 'Kakao search request failed'
                }));
            }
        });

        return;
    }


    
    // 2. 카카오모빌리티 자동차 경로 조회 API
    if (pathname === '/api/directions') {
        if (req.method !== 'GET') {
            res.writeHead(405, { Allow: 'GET' });
            return res.end('Method not allowed');
        }

        const apiKey = process.env.KAKAO_REST_API_KEY;

        if (!apiKey) {
            res.writeHead(503, {
                'Content-Type': 'application/json; charset=utf-8'
            });

            return res.end(JSON.stringify({
                success: false,
                error: 'KAKAO_REST_API_KEY is not configured'
            }));
        }

        const requestUrl = new URL(req.url, 'http://localhost');

        const origin = requestUrl.searchParams.get('origin');
        const destination = requestUrl.searchParams.get('destination');

        // 경도,위도 형식 검사
        function validCoordinate(value) {
            if (!value) return false;

            const parts = value.split(',');

            if (parts.length !== 2) return false;

            const longitude = Number(parts[0]);
            const latitude = Number(parts[1]);

            return Number.isFinite(longitude) &&
                Number.isFinite(latitude) &&
                longitude >= -180 &&
                longitude <= 180 &&
                latitude >= -90 &&
                latitude <= 90;
        }

        if (
            !validCoordinate(origin) ||
            !validCoordinate(destination)
        ) {
            res.writeHead(400, {
                'Content-Type': 'application/json; charset=utf-8'
            });

            return res.end(JSON.stringify({
                success: false,
                error: 'Invalid origin or destination coordinates'
            }));
        }

        // 세 가지 경로 탐색 기준
        const priorities = [
            'RECOMMEND',
            'TIME',
            'DISTANCE'
        ];

        // 하나가 실패해도 다른 경로 조회는 유지
        const directionsStarted = Date.now();
        Promise.allSettled(
            priorities.map(priority =>
                fetchKakaoDrivingRoute(
                    origin,
                    destination,
                    priority,
                    apiKey
                )
            )
        ).then(results => {
            const successfulRoutes = [];

            results.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    successfulRoutes.push(result.value);
                } else {
                    console.error(
                        `Directions ${priorities[index]} failed:`,
                        result.reason?.message
                    );
                }
            });

            if (successfulRoutes.length === 0) {
                res.writeHead(502, {
                    'Content-Type': 'application/json; charset=utf-8'
                });

                return res.end(JSON.stringify({
                    success: false,
                    error: 'All directions requests failed'
                }));
            }
            
            // 실제 도로 좌표를 기준으로 중복 경로 제거
            const uniqueRoutes = [];
            const seenGeometry = new Set();

            for (const route of successfulRoutes) {
                const key = routeGeometryKey(route);

                if (seenGeometry.has(key)) {
                    continue;
                }

                seenGeometry.add(key);
                uniqueRoutes.push(route);
            }
            
            console.log(
                '중복 제거 후 경로:',
                uniqueRoutes.map(route => ({
                    priority: route.priority,
                    distance_m: route.distance_m,
                    duration_s: route.duration_s
                }))
            );

            // 기존 app.js가 사용하는 단일 경로 필드 유지
            const primaryRoute =
                successfulRoutes.find(
                    route => route.priority === 'RECOMMEND'
                ) || successfulRoutes[0];

            res.writeHead(200, {
                'Content-Type': 'application/json; charset=utf-8'
            });

            res.end(JSON.stringify({
                success: true,
                route: primaryRoute,
                routes: uniqueRoutes,
                timings: {kakao_total_ms: Date.now()-directionsStarted}
            }));

        }).catch(error => {
            console.error(
                'Multiple directions error:',
                error.message
            );

            if (!res.writableEnded) {
                res.writeHead(502, {
                    'Content-Type': 'application/json; charset=utf-8'
                });

                res.end(JSON.stringify({
                    success: false,
                    error: 'Multiple directions request failed'
                }));
            }
        });

        return;
    }


    // 1. AI 예측 API
    if (pathname === '/api/predict' || pathname === '/api/predict-live') {
        if (req.method !== 'POST') {
            res.writeHead(405, { Allow: 'POST' });
            return res.end('Method not allowed');
        }

        let body = '';
        let size = 0;

        req.on('data', chunk => {
            size += chunk.length;

            if (size > 8 * 1024 * 1024) {
                req.destroy();
                return;
            }

            body += chunk.toString('utf8');
        });

        req.on('end', () => {
            let request;

            try {
                request = JSON.parse(body);

                if (!Array.isArray(request.routes) ||
                    request.routes.length === 0 ||
                    request.routes.length > 100) {
                    throw new Error('routes must contain 1-100 items');
                }
            } catch {
                res.writeHead(400, {
                    'Content-Type': 'application/json; charset=utf-8',
                });

                return res.end(JSON.stringify({
                    success: false,
                    error: 'Invalid request',
                }));
            }

            if (pathname === '/api/predict-live') {
                predictLive(request).then(data => {
                    if (res.writableEnded) return;
                    res.writeHead(data.success ? 200 : 500, {'Content-Type': 'application/json; charset=utf-8'});
                    res.end(JSON.stringify(data));
                }).catch(error => {
                    if (res.writableEnded) return;
                    console.error('Live AI error:', error.message);
                    res.writeHead(500, {'Content-Type': 'application/json; charset=utf-8'});
                    res.end(JSON.stringify({success: false, error: 'Live AI prediction failed'}));
                });
                return;
            }

            const python = path.join(
                root, '.venv', 'Scripts', 'python.exe'
            );

            const script = path.join(
                root, 'training', pathname === '/api/predict-live'
                    ? 'predict_live_routes.py' : 'predict_routes.py'
            );

            const child = spawn(python, [script], {
                cwd: root,
                windowsHide: true,
            });

            let output = '';
            let errorOutput = '';

            child.stdout.on('data', chunk => {
                output += chunk.toString();
            });

            child.stderr.on('data', chunk => {
                errorOutput += chunk.toString();
            });

            child.on('error', err => {
                console.error('Python execution error:', err.message);

                if (!res.headersSent) {
                    res.writeHead(500, {
                        'Content-Type': 'application/json',
                    });

                    res.end(JSON.stringify({
                        success: false,
                        error: 'Python execution failed',
                    }));
                }
            });

            child.on('close', code => {
                if (res.writableEnded) return;

                if (code !== 0) {
                    console.error(errorOutput);

                    res.writeHead(500, {
                        'Content-Type': 'application/json',
                    });

                    return res.end(JSON.stringify({
                        success: false,
                        error: 'Model prediction failed',
                    }));
                }

                res.writeHead(200, {
                    'Content-Type': 'application/json; charset=utf-8',
                });

                res.end(output);
            });

            child.stdin.end(JSON.stringify(request));
        });

        return;
    }

    // 2. 기존 정적 파일 제공
    if (!['GET', 'HEAD'].includes(req.method)) {
        res.writeHead(403);
        return res.end('Forbidden');
    }

    let file;

    try {
        const rel = decodeURIComponent(pathname);
        file = path.resolve(
            root, '.' + (rel === '/' ? '/index.html' : rel)
        );
    } catch {
        res.writeHead(400);
        return res.end('Bad request');
    }

    if (!file.startsWith(root + path.sep)) {
        res.writeHead(403);
        return res.end('Forbidden');
    }

    fs.stat(file, (err, stats) => {
        if (err || !stats.isFile()) {
            res.writeHead(404);
            return res.end('Not found');
        }

        res.writeHead(200, {
            'Content-Type':
                types[path.extname(file)] ||
                'application/octet-stream',
            'Content-Length': stats.size,
        });

        if (req.method === 'HEAD') {
            return res.end();
        }

        fs.createReadStream(file).pipe(res);
    });
});

server.on('error', err => {
    console.error(
        err.code === 'EADDRINUSE'
            ? `Port ${port} is in use.`
            : err.message
    );
    process.exitCode = 1;
});

server.listen(port, '127.0.0.1', () => {
    const url = `http://127.0.0.1:${port}`;

    console.log('SafeRoute: ' + url + ' (Ctrl+C to stop)');
    startLiveWorker();

    if (
        process.env.SAFEROUTE_NO_BROWSER !== '1' &&
        process.platform === 'win32'
    ) {
        spawn('explorer.exe', [url], {
            windowsHide: true,
        });
    }
});
