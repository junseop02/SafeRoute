const test = require('node:test');
const assert = require('node:assert/strict');

const {
    calculateP01,
    makeIndex,
    scoreRoads,
    scoreRoute
} = require('../service/matcher');


// 부동소수점 오차 허용 비교
function close(actual, expected, tolerance = 1e-9) {
    assert.ok(
        Math.abs(actual - expected) <= tolerance,
        `Expected ${expected}, received ${actual}`
    );
}


// ==================================================
// 1. P01 점수 계산 테스트
// ==================================================

test('P01: weighted score calculation', () => {
    const result = calculateP01(0.3, 0.8, 40);

    close(result.rawRisk, 0.3);
    close(result.narrowBurden, 0.2);
    close(result.intersectionBurden, 0.4);
    close(result.burden, 0.3);
    close(result.modelScore, 30);
    close(result.displayScore, 70);
});


test('P01: minimum burden and maximum display score', () => {
    const result = calculateP01(0, 1, 0);

    close(result.burden, 0);
    close(result.modelScore, 0);
    close(result.displayScore, 100);
});


test('P01: maximum burden and minimum display score', () => {
    const result = calculateP01(1, 0, 100);

    close(result.burden, 1);
    close(result.modelScore, 100);
    close(result.displayScore, 0);
});


test('P01: intersection burden clipping', () => {
    const result = calculateP01(0.2, 0.5, 150);

    close(result.intersectionBurden, 1);
    close(result.burden, 0.31);
    close(result.modelScore, 31);
    close(result.displayScore, 69);
});


test('P01: display score and burden score consistency', () => {
    const result = calculateP01(0.45, 0.75, 35);

    close(result.modelScore + result.displayScore, 100);
});


// ==================================================
// 2. 잘못된 입력값 처리
// ==================================================

test('P01: invalid raw risk is rejected', () => {
    for (const value of [-0.1, 1.1, NaN, Infinity, null, '0.3']) {
        assert.throws(
            () => calculateP01(value, 0.8, 40),
            /Invalid route raw risk/
        );
    }
});


test('P01: invalid wide road ratio is rejected', () => {
    for (const value of [-0.1, 1.1, NaN, Infinity, null, '0.8']) {
        assert.throws(
            () => calculateP01(0.3, value, 40),
            /Invalid route wide road ratio/
        );
    }
});


test('P01: invalid intersection mean is rejected', () => {
    for (const value of [-1, NaN, Infinity, null, '40']) {
        assert.throws(
            () => calculateP01(0.3, 0.8, value),
            /Invalid route intersection mean/
        );
    }
});


// ==================================================
// 3. 빈 경로 및 미매칭 처리
// ==================================================

test('Matcher: empty roads return no score', () => {
    const index = makeIndex([]);
    const result = scoreRoads(index, []);

    assert.equal(result.model_score, null);
    assert.equal(result.model_coverage, 0);
    assert.equal(result.model_matched_m, 0);
    assert.equal(result.geometry_length_m, 0);
    assert.equal(result.matched_link_count, 0);

    assert.equal(result.model_raw_risk, null);
    assert.equal(result.model_narrow_burden, null);
    assert.equal(result.model_intersection_burden, null);
});


test('Matcher: unmatched geometry returns no score', () => {
    const index = makeIndex([]);

    const roads = [
        {
            vertexes: [
                127.0, 36.0,
                127.001, 36.0
            ]
        }
    ];

    const result = scoreRoads(index, roads);

    assert.equal(result.model_score, null);
    assert.equal(result.model_coverage, 0);
    assert.equal(result.model_matched_m, 0);
    assert.equal(result.matched_link_count, 0);

    assert.ok(result.geometry_length_m > 0);
});


test('Matcher: empty route returns no score', () => {
    const index = makeIndex([]);

    const result = scoreRoute(index, {
        sections: []
    });

    assert.equal(result.model_score, null);
    assert.equal(result.model_coverage, 0);
});


test('Matcher: response contains correct model and policy IDs', () => {
    const index = makeIndex([]);
    const result = scoreRoads(index, []);

    assert.equal(result.model_version, 'T09');
    assert.equal(result.policy_version, 'P01');
});

// ==================================================
// 4. 공간 인덱스 테스트 환경
// ==================================================

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const zlib = require('node:zlib');

const {
    point
} = require('../service/matcher');


// 테스트용 임시 타일 생성
function createTile(t, rows, gzip = false) {
    const dir = fs.mkdtempSync(
        path.join(os.tmpdir(), 'saferoute-test-')
    );

    t.after(() => {
        fs.rmSync(dir, {
            recursive: true,
            force: true
        });
    });

    const file = path.join(
        dir,
        gzip ? 'tile.gz' : 'tile.json'
    );

    if (gzip) {
        const content = rows
            .map(row => JSON.stringify(row))
            .join('\n') + '\n';

        fs.writeFileSync(
            file,
            zlib.gzipSync(content)
        );
    } else {
        fs.writeFileSync(
            file,
            JSON.stringify(rows)
        );
    }

    return file;
}


// 테스트용 도로 데이터 생성
function makeRow(
    id = 'LINK-001',
    coords = [[127, 36], [127.001, 36]],
    score = 0.2,
    lanes = 2,
    intersections = 20,
    accidentDistance = 100,
    partIndex = 0
) {
    return [
        id,
        coords,
        score,
        lanes,
        intersections,
        accidentDistance,
        partIndex
    ];
}


// 카카오 경로 형식 생성
function makeRoad(coords) {
    return {
        vertexes: coords.flat()
    };
}


// ==================================================
// 5. 공간 타일 및 인덱스 검증
// ==================================================

test('Matcher: JSON tile loading', t => {
    const file = createTile(t, [makeRow()]);
    const index = makeIndex([file]);

    assert.equal(index.uniqueLinks, 1);
    assert.equal(index.uniqueParts, 1);
    assert.equal(index.segments, 1);
});


test('Matcher: gzip tile loading', t => {
    const file = createTile(t, [makeRow()], true);
    const index = makeIndex([file]);

    assert.equal(index.uniqueLinks, 1);
    assert.equal(index.uniqueParts, 1);
    assert.equal(index.segments, 1);
});


test('Matcher: duplicate tile parts are removed', t => {
    const row = makeRow();

    const file1 = createTile(t, [row]);
    const file2 = createTile(t, [row], true);

    const index = makeIndex([file1, file2]);

    assert.equal(index.uniqueLinks, 1);
    assert.equal(index.uniqueParts, 1);
    assert.equal(index.duplicateParts, 1);
    assert.equal(index.segments, 1);
});


test('Matcher: multipart geometry is preserved', t => {
    const rows = [
        makeRow(
            'MULTI-001',
            [[127, 36], [127.001, 36]]
        ),
        makeRow(
            'MULTI-001',
            [[127.002, 36], [127.003, 36]],
            0.2, 2, 20, 100, 1
        )
    ];

    const file = createTile(t, rows);
    const index = makeIndex([file]);

    assert.equal(index.uniqueLinks, 1);
    assert.equal(index.uniqueParts, 2);
    assert.equal(index.duplicateParts, 0);
    assert.equal(index.segments, 2);
});


test('Matcher: zero-length segments are skipped', t => {
    const row = makeRow(
        'ZERO-001',
        [
            [127, 36],
            [127, 36],
            [127.001, 36]
        ]
    );

    const file = createTile(t, [row]);
    const index = makeIndex([file]);

    assert.equal(index.zeroLengthSegments, 1);
    assert.equal(index.segments, 1);
});


test('Matcher: seven-decimal geometry is preserved', t => {
    const row = makeRow(
        'PRECISION-001',
        [
            [127.0000001, 36],
            [127.0000002, 36]
        ]
    );

    const file = createTile(t, [row]);
    const index = makeIndex([file]);

    assert.equal(index.segments, 1);
    assert.equal(index.zeroLengthSegments, 0);
});


test('Matcher: invalid tile rows are rejected', t => {
    const invalidRows = [
        makeRow().slice(0, 6),
        makeRow(''),
        makeRow('BAD', [[0, 0], [1, 1]]),
        makeRow('BAD', [[127, 36], [127.001, 36]], 1.1),
        makeRow('BAD', [[127, 36], [127.001, 36]], 0.2, 0),
        makeRow('BAD', [[127, 36], [127.001, 36]], 0.2, 2, -1),
        makeRow('BAD', [[127, 36], [127.001, 36]], 0.2, 2, 20, -1),
        makeRow('BAD', [[127, 36], [127.001, 36]], 0.2, 2, 20, 100, -1)
    ];

    for (const row of invalidRows) {
        const file = createTile(t, [row]);

        assert.throws(
            () => makeIndex([file]),
            /Invalid/
        );
    }
});


// ==================================================
// 6. 경로 매칭 및 점수 계산 검증
// ==================================================

test('Matcher: fully matched route receives correct score', t => {
    const coords = [
        [127, 36],
        [127.001, 36]
    ];

    const file = createTile(t, [
        makeRow('LINK-001', coords)
    ]);

    const index = makeIndex([file]);

    const result = scoreRoute(index, {
        sections: [
            {
                roads: [makeRoad(coords)]
            }
        ]
    });

    close(result.model_coverage, 1);
    close(result.model_raw_risk, 0.2);
    close(result.model_wide_ratio, 1);
    close(result.model_intersections, 20);
    close(result.model_hotspot_ratio, 1);
    close(result.model_score, 18);

    assert.equal(result.matched_link_count, 1);
});


test('Matcher: road-length weighted aggregation', t => {
    const coords1 = [
        [127, 36],
        [127.001, 36]
    ];

    const coords2 = [
        [127.01, 36],
        [127.013, 36]
    ];

    const rows = [
        makeRow('A', coords1, 0.2, 2, 20, 100),
        makeRow('B', coords2, 0.6, 1, 60, 300)
    ];

    const file = createTile(t, rows);
    const index = makeIndex([file]);

    const result = scoreRoads(index, [
        makeRoad(coords1),
        makeRoad(coords2)
    ]);

    close(result.model_coverage, 1);
    close(result.model_raw_risk, 0.5, 1e-7);
    close(result.model_wide_ratio, 0.25, 1e-7);
    close(result.model_intersections, 50, 1e-7);
    close(result.model_hotspot_ratio, 0.25, 1e-7);
    close(result.model_score, 52.5, 1e-7);

    assert.equal(result.matched_link_count, 2);
});


// ==================================================
// 7. 매칭률 경계값 검증
// ==================================================

// 동일한 길이의 경로 100개를 사용해
// 매칭된 경로의 비율을 조절
function coverageResult(t, matchedCount) {
    const matchedCoords = [
        [127, 36],
        [127.001, 36]
    ];

    const unmatchedCoords = [
        [127, 36.01],
        [127.001, 36.01]
    ];

    const file = createTile(t, [
        makeRow('COVERAGE-001', matchedCoords)
    ]);

    const index = makeIndex([file]);

    const roads = [];

    for (let i = 0; i < 100; i++) {
        roads.push(
            makeRoad(
                i < matchedCount
                    ? matchedCoords
                    : unmatchedCoords
            )
        );
    }

    return scoreRoads(index, roads);
}


test('Matcher: 50 percent coverage receives no score', t => {
    const result = coverageResult(t, 50);

    close(result.model_coverage, 0.5);

    assert.equal(result.model_score, null);
    assert.equal(result.model_raw_risk, null);
});


test('Matcher: 79 percent coverage receives no score', t => {
    const result = coverageResult(t, 79);

    close(result.model_coverage, 0.79);

    assert.equal(result.model_score, null);
    assert.equal(result.model_raw_risk, null);
    assert.equal(result.model_narrow_burden, null);
    assert.equal(result.model_intersection_burden, null);
});


test('Matcher: 80 percent coverage receives a score', t => {
    const result = coverageResult(t, 80);

    close(result.model_coverage, 0.8);

    assert.notEqual(result.model_score, null);
    close(result.model_score, 18);
});


test('Matcher: 81 percent coverage receives a score', t => {
    const result = coverageResult(t, 81);

    close(result.model_coverage, 0.81);

    assert.notEqual(result.model_score, null);
    close(result.model_score, 18);
});


// ==================================================
// 8. 결측값 및 복합 선형 집계
// ==================================================

test('Matcher: unknown hotspot distance does not affect P01', t => {
    const coords = [
        [127, 36],
        [127.001, 36]
    ];

    const file = createTile(t, [
        makeRow(
            'NULL-001',
            coords,
            0.2, 2, 20, null
        )
    ]);

    const index = makeIndex([file]);

    const result = scoreRoads(index, [
        makeRoad(coords)
    ]);

    assert.equal(result.model_hotspot_ratio, null);

    close(result.model_coverage, 1);
    close(result.model_score, 18);
});


test('Matcher: multipart route counts unique link IDs', t => {
    const coords1 = [
        [127, 36],
        [127.001, 36]
    ];

    const coords2 = [
        [127.002, 36],
        [127.003, 36]
    ];

    const file = createTile(t, [
        makeRow(
            'MULTI-001',
            coords1,
            0.2, 2, 20, 100, 0
        ),
        makeRow(
            'MULTI-001',
            coords2,
            0.2, 2, 20, 100, 1
        )
    ]);

    const index = makeIndex([file]);

    const result = scoreRoads(index, [
        makeRoad(coords1),
        makeRoad(coords2)
    ]);

    close(result.model_coverage, 1);
    close(result.model_score, 18);

    assert.equal(index.uniqueParts, 2);
    assert.equal(result.matched_link_count, 1);
});