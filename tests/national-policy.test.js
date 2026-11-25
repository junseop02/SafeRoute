const test = require('node:test');
const assert = require('node:assert/strict');

const {
    normalizeDetourCap,
    isEligible,
    compareRoutes,
    selectRoutes
} = require('../service/kakao');


// ==================================================
// 1. 테스트용 경로 생성
// ==================================================

function makeRoute(id = 'A', overrides = {}) {
    return {
        id,
        distance_m: 1000,
        duration_s: 100,
        model_score: 20,
        model_coverage: 0.95,
        model_version: 'T09',
        policy_version: 'P01',
        ...overrides
    };
}


// ==================================================
// 2. 우회 허용 배율 검증
// ==================================================

test('Policy: valid numeric detour caps', () => {
    for (const cap of [1.5, 2, 3]) {
        assert.equal(normalizeDetourCap(cap), cap);
    }
});


test('Policy: valid string detour caps', () => {
    for (const cap of ['1.5', '2', '3']) {
        assert.equal(
            normalizeDetourCap(cap),
            Number(cap)
        );
    }
});


test('Policy: unlimited detour cap', () => {
    assert.equal(
        normalizeDetourCap('unlimited'),
        Infinity
    );

    assert.equal(
        normalizeDetourCap(Infinity),
        Infinity
    );
});


test('Policy: default detour cap is two', () => {
    assert.equal(normalizeDetourCap(), 2);
});


test('Policy: invalid detour caps are rejected', () => {
    const invalid = [
        0, -1, 1, 2.5, 4,
        NaN, -Infinity,
        null, true, false,
        '', 'invalid', {}, []
    ];

    for (const cap of invalid) {
        assert.throws(
            () => normalizeDetourCap(cap)
        );
    }
});


// ==================================================
// 3. 추천 가능 경로 검증
// ==================================================

test('Policy: valid route is eligible', () => {
    assert.equal(
        isEligible(makeRoute()),
        true
    );
});


test('Policy: score boundaries are eligible', () => {
    for (const score of [0, 100]) {
        assert.equal(
            isEligible(
                makeRoute('A', {
                    model_score: score
                })
            ),
            true
        );
    }
});


test('Policy: invalid scores are ineligible', () => {
    const invalid = [
        -1, 101, NaN,
        Infinity, null,
        undefined, '20'
    ];

    for (const score of invalid) {
        assert.equal(
            isEligible(
                makeRoute('A', {
                    model_score: score
                })
            ),
            false
        );
    }
});


test('Policy: coverage boundary at 80 percent', () => {
    for (const coverage of [0.8, 0.81, 1]) {
        assert.equal(
            isEligible(
                makeRoute('A', {
                    model_coverage: coverage
                })
            ),
            true
        );
    }
});


test('Policy: invalid coverage is rejected', () => {
    const invalid = [
        0, 0.5, 0.79,
        0.799999, 1.01,
        NaN, Infinity,
        null, '0.9'
    ];

    for (const coverage of invalid) {
        assert.equal(
            isEligible(
                makeRoute('A', {
                    model_coverage: coverage
                })
            ),
            false
        );
    }
});


test('Policy: invalid route distances are rejected', () => {
    const invalid = [
        0, -1, NaN,
        Infinity, null, '1000'
    ];

    for (const distance of invalid) {
        assert.equal(
            isEligible(
                makeRoute('A', {
                    distance_m: distance
                })
            ),
            false
        );
    }
});


test('Policy: invalid route durations are rejected', () => {
    const invalid = [
        -1, NaN, Infinity,
        null, '100'
    ];

    for (const duration of invalid) {
        assert.equal(
            isEligible(
                makeRoute('A', {
                    duration_s: duration
                })
            ),
            false
        );
    }

    assert.equal(
        isEligible(
            makeRoute('A', {
                duration_s: 0
            })
        ),
        true
    );
});


test('Policy: model and policy IDs must match', () => {
    assert.equal(
        isEligible(
            makeRoute('A', {
                model_version: 'OLD'
            })
        ),
        false
    );

    assert.equal(
        isEligible(
            makeRoute('A', {
                policy_version: 'OLD'
            })
        ),
        false
    );
});


// ==================================================
// 4. 경로 정렬 및 동점 처리
// ==================================================

test('Policy: eligible route is ranked first', () => {
    const a = makeRoute('A', {
        eligible: false,
        model_score: 1
    });

    const b = makeRoute('B', {
        eligible: true,
        model_score: 90
    });

    assert.deepEqual(
        [a, b].sort(compareRoutes).map(r => r.id),
        ['B', 'A']
    );
});


test('Policy: lower burden is ranked first', () => {
    const a = makeRoute('A', {
        eligible: true,
        model_score: 30
    });

    const b = makeRoute('B', {
        eligible: true,
        model_score: 10
    });

    assert.deepEqual(
        [a, b].sort(compareRoutes).map(r => r.id),
        ['B', 'A']
    );
});


test('Policy: equal burden prefers shorter distance', () => {
    const a = makeRoute('A', {
        eligible: true,
        distance_m: 1500
    });

    const b = makeRoute('B', {
        eligible: true,
        distance_m: 1000
    });

    assert.deepEqual(
        [a, b].sort(compareRoutes).map(r => r.id),
        ['B', 'A']
    );
});


test('Policy: equal burden and distance prefer shorter duration', () => {
    const a = makeRoute('A', {
        eligible: true,
        duration_s: 200
    });

    const b = makeRoute('B', {
        eligible: true,
        duration_s: 100
    });

    assert.deepEqual(
        [a, b].sort(compareRoutes).map(r => r.id),
        ['B', 'A']
    );
});


test('Policy: complete tie uses route ID', () => {
    const a = makeRoute('B', {
        eligible: true
    });

    const b = makeRoute('A', {
        eligible: true
    });

    assert.deepEqual(
        [a, b].sort(compareRoutes).map(r => r.id),
        ['A', 'B']
    );
});


test('Policy: burden rounding to twelve decimals', () => {
    const a = makeRoute('A', {
        eligible: true,
        model_score: 20.00000000001,
        distance_m: 1000
    });

    const b = makeRoute('B', {
        eligible: true,
        model_score: 20,
        distance_m: 1500
    });

    // 반올림 후 부담 점수가 같으므로 거리 우선
    assert.deepEqual(
        [b, a].sort(compareRoutes).map(r => r.id),
        ['A', 'B']
    );
});


// ==================================================
// 5. 최소 추천 후보 검증
// ==================================================

test('Policy: no candidates returns no recommendation', () => {
    const result = selectRoutes([], 2);

    assert.equal(result.recommended, null);
    assert.equal(result.distinct_eligible, 0);
    assert.equal(result.shortest_m, null);
    assert.equal(result.total_distinct, 0);
    assert.equal(result.min_routes_met, false);
    assert.deepEqual(result.routes, []);
});


test('Policy: one candidate returns no recommendation', () => {
    const result = selectRoutes([
        makeRoute('A')
    ], 2);

    assert.equal(result.recommended, null);
    assert.equal(result.distinct_eligible, 1);
    assert.equal(result.min_routes_met, false);
});


test('Policy: two eligible candidates permit recommendation', () => {
    const result = selectRoutes([
        makeRoute('A', {
            model_score: 30
        }),
        makeRoute('B', {
            model_score: 10
        })
    ], 2);

    assert.equal(result.recommended, 'B');
    assert.equal(result.distinct_eligible, 2);
    assert.equal(result.min_routes_met, true);
});


test('Policy: three eligible candidates select lowest burden', () => {
    const result = selectRoutes([
        makeRoute('A', {
            model_score: 30
        }),
        makeRoute('B', {
            model_score: 10
        }),
        makeRoute('C', {
            model_score: 20
        })
    ], 2);

    assert.equal(result.recommended, 'B');
    assert.equal(result.distinct_eligible, 3);

    assert.deepEqual(
        result.routes.map(r => r.id),
        ['B', 'C', 'A']
    );
});


test('Policy: two visible but one eligible route gives no recommendation', () => {
    const result = selectRoutes([
        makeRoute('A'),
        makeRoute('B', {
            model_coverage: 0.79
        })
    ], 2);

    assert.equal(result.routes.length, 2);
    assert.equal(result.distinct_eligible, 1);
    assert.equal(result.recommended, null);

    // 기존 API 필드는 표시 후보 수를 기준으로 판단
    assert.equal(result.min_routes_met, true);
});


test('Policy: three visible and two eligible routes permit recommendation', () => {
    const result = selectRoutes([
        makeRoute('A', {
            model_score: 30
        }),
        makeRoute('B', {
            model_score: 20
        }),
        makeRoute('C', {
            model_score: null
        })
    ], 2);

    assert.equal(result.routes.length, 3);
    assert.equal(result.distinct_eligible, 2);
    assert.equal(result.recommended, 'B');
    assert.equal(result.routes[2].eligible, false);
});


// ==================================================
// 6. 우회 허용 범위 적용
// ==================================================

test('Policy: route outside detour cap is excluded', () => {
    const result = selectRoutes([
        makeRoute('A', {
            distance_m: 1000,
            model_score: 30
        }),
        makeRoute('B', {
            distance_m: 1500,
            model_score: 20
        }),
        makeRoute('C', {
            distance_m: 2100,
            model_score: 1
        })
    ], 2);

    assert.equal(result.recommended, 'B');
    assert.equal(result.shortest_m, 1000);
    assert.equal(result.outside_cap_count, 1);
    assert.equal(result.total_distinct, 3);

    assert.deepEqual(
        result.routes.map(r => r.id),
        ['B', 'A']
    );
});


test('Policy: exact detour cap boundary is included', () => {
    const result = selectRoutes([
        makeRoute('A', {
            distance_m: 1000
        }),
        makeRoute('B', {
            distance_m: 2000,
            model_score: 10
        }),
        makeRoute('C', {
            distance_m: 2001,
            model_score: 1
        })
    ], 2);

    assert.equal(result.recommended, 'B');
    assert.equal(result.routes.length, 2);
    assert.equal(result.outside_cap_count, 1);
});


test('Policy: all detour options apply correctly', () => {
    const routes = [
        makeRoute('A', { distance_m: 1000 }),
        makeRoute('B', { distance_m: 1500 }),
        makeRoute('C', { distance_m: 2000 }),
        makeRoute('D', { distance_m: 3000 }),
        makeRoute('E', { distance_m: 4000 })
    ];

    assert.equal(selectRoutes(routes, 1.5).routes.length, 2);
    assert.equal(selectRoutes(routes, 2).routes.length, 3);
    assert.equal(selectRoutes(routes, 3).routes.length, 4);

    assert.equal(
        selectRoutes(routes, 'unlimited').routes.length,
        5
    );
});


test('Policy: shortest candidate is used even when ineligible', () => {
    const result = selectRoutes([
        makeRoute('A', {
            distance_m: 1000,
            model_score: null
        }),
        makeRoute('B', {
            distance_m: 1800,
            model_score: 20
        }),
        makeRoute('C', {
            distance_m: 2200,
            model_score: 10
        })
    ], 2);

    assert.equal(result.shortest_m, 1000);
    assert.equal(result.routes.length, 2);
    assert.equal(result.distinct_eligible, 1);
    assert.equal(result.outside_cap_count, 1);
    assert.equal(result.recommended, null);
});


test('Policy: invalid distances do not become visible routes', () => {
    const result = selectRoutes([
        makeRoute('A', {
            distance_m: 0
        }),
        makeRoute('B', {
            distance_m: NaN
        }),
        makeRoute('C', {
            distance_m: 1000
        })
    ], 2);

    assert.equal(result.shortest_m, 1000);
    assert.equal(result.routes.length, 1);
    assert.equal(result.outside_cap_count, 2);
    assert.equal(result.recommended, null);
});


test('Policy: unlimited cap retains all valid distances', () => {
    const result = selectRoutes([
        makeRoute('A', {
            distance_m: 1000,
            model_score: 30
        }),
        makeRoute('B', {
            distance_m: 10000,
            model_score: 10
        })
    ], 'unlimited');

    assert.equal(result.detour_cap, Infinity);
    assert.equal(result.routes.length, 2);
    assert.equal(result.outside_cap_count, 0);
    assert.equal(result.recommended, 'B');
});


// ==================================================
// 7. 카카오 API 모의 호출 테스트 환경
// ==================================================

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
    findRoutes
} = require('../service/kakao');

const {
    makeIndex
} = require('../service/matcher');


// 테스트용 출발지 및 도착지
const TEST_ORIGIN = {
    x: 127,
    y: 36
};

const TEST_DESTINATION = {
    x: 127.001,
    y: 36
};


// 테스트용 도로 좌표
const TEST_GEOMETRIES = {
    A: [[127, 36], [127.001, 36]],
    B: [[127, 36.01], [127.001, 36.01]],
    C: [[127, 36.02], [127.001, 36.02]],
    D: [[127, 36.03], [127.001, 36.03]],
    E: [[127, 36.04], [127.001, 36.04]]
};


// 테스트용 공간 인덱스 생성
function createPolicyIndex(t, included = ['A', 'B', 'C', 'D']) {
    const dir = fs.mkdtempSync(
        path.join(os.tmpdir(), 'saferoute-policy-')
    );

    t.after(() => {
        fs.rmSync(dir, {
            recursive: true,
            force: true
        });
    });

    const scores = {
        A: 0.3,
        B: 0.1,
        C: 0.2,
        D: 0.4,
        E: 0.05
    };

    const rows = included.map(id => [
        id,
        TEST_GEOMETRIES[id],
        scores[id],
        2,
        20,
        100,
        0
    ]);

    const file = path.join(dir, 'tile.json');

    fs.writeFileSync(
        file,
        JSON.stringify(rows)
    );

    const index = makeIndex([file]);

    return {
        national: true,

        forRoutes() {
            return index;
        }
    };
}


// 카카오 API 경로 응답 생성
function mockRoute(
    id,
    distance = 1000,
    duration = 100
) {
    return {
        result_code: 0,

        summary: {
            distance,
            duration
        },

        sections: [
            {
                roads: [
                    {
                        vertexes:
                            TEST_GEOMETRIES[id].flat()
                    }
                ]
            }
        ]
    };
}


// 실제 API를 호출하지 않는 fetch 함수
function createMockFetcher(responses) {
    const calls = [];

    async function fetcher(url, options) {
        const requestUrl = new URL(url);

        const priority =
            requestUrl.searchParams.get('priority');

        const avoid =
            requestUrl.searchParams.get('avoid');

        const profile = avoid || priority;

        calls.push(profile);

        assert.equal(
            options.headers.Authorization,
            'KakaoAK TEST_KEY'
        );

        assert.equal(
            requestUrl.searchParams.get('alternatives'),
            'true'
        );

        assert.equal(
            requestUrl.searchParams.get('summary'),
            'false'
        );

        assert.equal(
            requestUrl.searchParams.get('road_details'),
            'true'
        );

        assert.equal(
            requestUrl.searchParams.get('origin'),
            '127,36'
        );

        assert.equal(
            requestUrl.searchParams.get('destination'),
            '127.001,36'
        );

        const response = responses[profile] || {
            routes: []
        };

        if (response instanceof Error) {
            throw response;
        }

        const status = response.status || 200;

        return {
            ok: status >= 200 && status < 300,
            status,

            async json() {
                return {
                    routes: response.routes || []
                };
            }
        };
    }

    return {
        fetcher,
        calls
    };
}


// 공통 경로 탐색 실행
async function runMockSearch(
    t,
    responses,
    cap = 2,
    included = ['A', 'B', 'C', 'D']
) {
    const index = createPolicyIndex(t, included);

    const mock = createMockFetcher(responses);

    const result = await findRoutes(
        TEST_ORIGIN,
        TEST_DESTINATION,
        'TEST_KEY',
        index,
        mock.fetcher,
        cap
    );

    return {
        result,
        calls: mock.calls
    };
}


// ==================================================
// 8. 기본 API 요청 및 추천 검증
// ==================================================

test('API: three distinct routes stop additional requests', async t => {
    const { result, calls } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A')]
        },

        TIME: {
            routes: [mockRoute('B')]
        },

        DISTANCE: {
            routes: [mockRoute('C')]
        }
    });

    assert.equal(calls.length, 3);

    assert.equal(result.total_distinct, 3);
    assert.equal(result.distinct_eligible, 3);
    assert.equal(result.recommended, 'TIME-1');

    assert.deepEqual(
        result.routes.map(r => r.id),
        ['TIME-1', 'DISTANCE-1', 'RECOMMEND-1']
    );

    assert.deepEqual(result.failures, []);
});


test('API: duplicate geometries are removed', async t => {
    const { result, calls } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A')]
        },

        TIME: {
            routes: [mockRoute('A')]
        },

        DISTANCE: {
            routes: [mockRoute('B')]
        },

        schoolzone: {
            routes: [mockRoute('C')]
        }
    });

    assert.equal(calls.length, 4);
    assert.equal(result.total_distinct, 3);
    assert.equal(result.distinct_eligible, 3);

    assert.equal(
        new Set(
            result.routes.map(r => r.geometry_hash)
        ).size,
        3
    );

    assert.equal(result.recommended, 'DISTANCE-1');
});


test('API: alternative routes in one response are retained', async t => {
    const { result, calls } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [
                mockRoute('A'),
                mockRoute('B'),
                mockRoute('C')
            ]
        }
    });

    assert.equal(calls.length, 3);
    assert.equal(result.total_distinct, 3);
    assert.equal(result.recommended, 'RECOMMEND-2');
});


// ==================================================
// 9. 추가 경로 탐색 검증
// ==================================================

test('API: additional search continues until target count', async t => {
    const { result, calls } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A')]
        },

        TIME: {
            routes: [mockRoute('A')]
        },

        DISTANCE: {
            routes: [mockRoute('A')]
        },

        schoolzone: {
            routes: [mockRoute('B')]
        },

        uturn: {
            routes: [mockRoute('C')]
        }
    });

    assert.deepEqual(calls, [
        'RECOMMEND',
        'TIME',
        'DISTANCE',
        'schoolzone',
        'uturn'
    ]);

    assert.equal(result.total_distinct, 3);
    assert.equal(result.distinct_eligible, 3);
    assert.equal(result.recommended, 'schoolzone-1');
});


test('API: two eligible routes permit recommendation after search exhaustion', async t => {
    const { result, calls } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A')]
        },

        TIME: {
            routes: [mockRoute('B')]
        }
    });

    assert.equal(calls.length, 7);

    assert.equal(result.total_distinct, 2);
    assert.equal(result.distinct_eligible, 2);
    assert.equal(result.recommended, 'TIME-1');
});


test('API: one distinct route does not permit recommendation', async t => {
    const { result, calls } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A')]
        },

        TIME: {
            routes: [mockRoute('A')]
        },

        DISTANCE: {
            routes: [mockRoute('A')]
        }
    });

    assert.equal(calls.length, 7);
    assert.equal(result.total_distinct, 1);
    assert.equal(result.distinct_eligible, 1);
    assert.equal(result.recommended, null);
});


test('API: routes outside cap do not satisfy search target', async t => {
    const { result, calls } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A', 1000)]
        },

        TIME: {
            routes: [mockRoute('B', 1500)]
        },

        DISTANCE: {
            routes: [mockRoute('C', 2500)]
        },

        schoolzone: {
            routes: [mockRoute('D', 1800)]
        }
    });

    assert.equal(calls.length, 4);

    assert.equal(result.total_distinct, 4);
    assert.equal(result.routes.length, 3);
    assert.equal(result.outside_cap_count, 1);
    assert.equal(result.recommended, 'TIME-1');
});


// ==================================================
// 10. API 실패 및 잘못된 경로 처리
// ==================================================

test('API: partial failure retains successful routes', async t => {
    const { result } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A')]
        },

        TIME: {
            status: 500
        },

        DISTANCE: {
            routes: [mockRoute('B')]
        },

        schoolzone: {
            routes: [mockRoute('C')]
        }
    });

    assert.equal(result.failures.length, 1);

    assert.match(
        result.failures[0],
        /TIME.*500/
    );

    assert.equal(result.distinct_eligible, 3);
    assert.equal(result.recommended, 'DISTANCE-1');
});


test('API: complete authorization failure is rejected', async t => {
    const profiles = [
        'RECOMMEND',
        'TIME',
        'DISTANCE',
        'schoolzone',
        'uturn',
        'toll',
        'motorway'
    ];

    const responses = Object.fromEntries(
        profiles.map(profile => [
            profile,
            { status: 401 }
        ])
    );

    await assert.rejects(
        () => runMockSearch(t, responses),
        /API 키/
    );
});


test('API: empty route responses are rejected', async t => {
    await assert.rejects(
        () => runMockSearch(t, {}),
        /운전 경로를 찾지 못했습니다/
    );
});


test('API: malformed routes are excluded', async t => {
    const invalidRoutes = [
        {
            ...mockRoute('A'),
            result_code: 1
        },

        {
            ...mockRoute('A'),
            sections: []
        },

        mockRoute('A', 0),

        mockRoute('A', 1000, -1),

        mockRoute('A', null),

        mockRoute('A', 1000, null)
    ];

    const { result } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [
                ...invalidRoutes,
                mockRoute('A')
            ]
        },

        TIME: {
            routes: [mockRoute('B')]
        }
    });

    assert.equal(result.total_distinct, 2);
    assert.equal(result.distinct_eligible, 2);
    assert.equal(result.recommended, 'TIME-1');
});


test('API: identical origin and destination are rejected', async t => {
    const index = createPolicyIndex(t);

    const mock = createMockFetcher({});

    await assert.rejects(
        () => findRoutes(
            TEST_ORIGIN,
            TEST_ORIGIN,
            'TEST_KEY',
            index,
            mock.fetcher,
            2
        ),
        /서로 다른 출발지/
    );

    assert.equal(mock.calls.length, 0);
});


// ==================================================
// 11. 모델 매칭 및 최종 응답 검증
// ==================================================

test('API: unmatched routes cannot be recommended', async t => {
    const { result } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A')]
        },

        TIME: {
            routes: [mockRoute('B')]
        },

        DISTANCE: {
            routes: [mockRoute('E')]
        }
    });

    const unmatched = result.routes.find(
        route => route.id === 'DISTANCE-1'
    );

    assert.equal(unmatched.model_score, null);
    assert.equal(unmatched.model_coverage, 0);
    assert.equal(unmatched.eligible, false);

    assert.equal(result.distinct_eligible, 2);
    assert.equal(result.recommended, 'TIME-1');
});


test('API: current search target counts visible routes before scoring', async t => {
    const { result, calls } = await runMockSearch(
        t,
        {
            RECOMMEND: {
                routes: [mockRoute('A')]
            },

            TIME: {
                routes: [mockRoute('B')]
            },

            DISTANCE: {
                routes: [mockRoute('C')]
            },

            schoolzone: {
                routes: [mockRoute('D')]
            }
        },
        2,
        ['A', 'D']
    );

    // 현재 구현의 탐색 제한 사항 기록
    assert.equal(calls.length, 3);
    assert.equal(result.routes.length, 3);
    assert.equal(result.distinct_eligible, 1);
    assert.equal(result.recommended, null);
});


test('API: final response preserves model fields', async t => {
    const { result } = await runMockSearch(t, {
        RECOMMEND: {
            routes: [mockRoute('A')]
        },

        TIME: {
            routes: [mockRoute('B')]
        },

        DISTANCE: {
            routes: [mockRoute('C')]
        }
    });

    for (const route of result.routes) {
        assert.equal(route.model_version, 'T09');
        assert.equal(route.policy_version, 'P01');

        assert.ok(
            Number.isFinite(route.model_score)
        );

        assert.ok(
            route.model_coverage >= 0.8
        );

        assert.ok(
            Number.isFinite(route.model_raw_risk)
        );

        assert.ok(
            Number.isFinite(route.model_narrow_burden)
        );

        assert.ok(
            Number.isFinite(route.model_intersection_burden)
        );

        assert.equal(
            Object.hasOwn(route, '_route'),
            false
        );
    }

    assert.deepEqual(result.origin, TEST_ORIGIN);
    assert.deepEqual(result.destination, TEST_DESTINATION);
});