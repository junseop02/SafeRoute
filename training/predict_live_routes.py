"""Kakao live routes -> road-link weighted features -> existing LightGBM.
Place in project/training/predict_live_routes.py.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live_route_features import load_road_links, match_route_geometry
import joblib
import numpy as np
import pandas as pd
import time
import hashlib
from collections import OrderedDict

from training.predict_routes import MODEL_PATH, FEATURE_PATH

def load_model():
    model = joblib.load(MODEL_PATH)
    with FEATURE_PATH.open(encoding='utf-8') as f:
        features = json.load(f)
    return model, features

def predict_cached(routes, model, features):
    df = pd.DataFrame(routes)
    missing = [key for key in features if key not in df.columns]
    if missing:
        raise ValueError(f'누락된 Feature: {missing}')
    for key in features:
        df[key] = pd.to_numeric(df[key], errors='coerce')
    X = df[features].replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError('Feature에 숫자가 아닌 값 또는 결측치가 있습니다.')
    values = model.predict(X)
    if not np.isfinite(values).all():
        raise ValueError('모델 예측값에 비정상 숫자가 있습니다.')
    return [
        {'route_id': str(route.get('route_id', i)),
         'predicted_burden': round(float(value), 4),
         'easy_score': round(float(np.clip(100 - float(value), 0, 100)), 4)}
        for i, (route, value) in enumerate(zip(routes, values))
    ]

SCORES_PATH = ROOT / 'data' / 'link_scores.json'
MIN_COVERAGE = 0.8


class MatchCache:
    """Worker-local, bounded cache; only geometry matches, never traffic inputs."""
    def __init__(self, limit=128):
        self.limit = limit
        self.items = OrderedDict()

    def match(self, geometry, roads):
        key = hashlib.sha256(json.dumps(geometry, separators=(',', ':'), allow_nan=False).encode()).digest()
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key], True
        result = match_route_geometry(roads, geometry, method='distance', interval_m=2, max_distance_m=10)
        self.items[key] = result
        if len(self.items) > self.limit:
            self.items.popitem(last=False)
        return result, False


def make_features(route, roads, lookup, timings=None, cache=None):
    groups = route.get('geometry')
    if not isinstance(groups, list) or not groups:
        raise ValueError('경로 geometry가 없습니다.')
    geometry = {'type': 'MultiLineString', 'coordinates': groups}
    match_started = time.perf_counter()
    if cache is None:
        matched = match_route_geometry(roads, geometry, method='distance', interval_m=2, max_distance_m=10)
        cache_hit = False
    else:
        matched, cache_hit = cache.match(geometry, roads)
    feature_started = time.perf_counter()
    if timings is not None:
        timings.update(match_ms=(feature_started-match_started)*1000, geometry_cache_hit=cache_hit)
    total = matched['geometry_length_m']
    valid = [(lookup[str(link_id)], float(length)) for link_id, length in matched['links'] if str(link_id) in lookup and length > 0]
    valid_length = sum(length for _, length in valid)
    if not total or not valid_length:
        raise ValueError('매칭된 도로 속성이 없습니다.')

    def average(key):
        return sum(float(info[key]) * length for info, length in valid) / valid_length

    features = {key: float(route[key]) for key in ('distance_m', 'duration_s', 'road_count', 'guide_count')}
    for key in ('n_intersect_200m', 'accident_burden', 'narrow_burden', 'intersection_burden'):
        features[key] = average(key)
    features['wide_road_ratio'] = sum(length for info, length in valid if float(info['lanes']) >= 2) / valid_length
    features['near_hotspot_ratio'] = sum(length for info, length in valid if float(info['accident_distance_m']) <= 150) / valid_length
    # Training uses length of links with valid attributes / geometric length.
    features['coverage'] = valid_length / total
    if not all(math.isfinite(v) for v in features.values()):
        raise ValueError('Feature에 비정상 숫자가 있습니다.')
    if timings is not None:
        timings["feature_ms"] = (time.perf_counter()-feature_started)*1000
    return features


def process_request(request, roads, lookup, model, feature_names, cache=None):
    started = time.perf_counter()
    routes = request.get('routes')
    if not isinstance(routes, list) or not 1 <= len(routes) <= 100:
        raise ValueError('routes는 1~100개 배열이어야 합니다.')
    eligible = []
    results = []
    route_timings = []
    predict_ms = 0.0
    for index, route in enumerate(routes):
        route_id = str(route.get('route_id', index))
        timing = {'route_id': route_id}
        route_timings.append(timing)
        try:
            features = make_features(route, roads, lookup, timings=timing, cache=cache)
            if features['coverage'] < MIN_COVERAGE:
                raise ValueError(f"매칭률 {features['coverage']:.1%} (최소 {MIN_COVERAGE:.0%})")
            eligible.append({'route_id': route_id, **features})
        except Exception as exc:
            results.append({'route_id': route_id, 'eligible': False, 'error': str(exc)})
    if eligible:
        predict_started = time.perf_counter()
        predicted = predict_cached(eligible, model, feature_names)
        predict_ms = (time.perf_counter()-predict_started)*1000
        for item, features in zip(predicted, eligible):
            results.append({**item, 'eligible': True, 'coverage': features['coverage'],
                            'features': {k: v for k, v in features.items() if k != 'route_id'}})
    print(f'[AI] prediction {time.perf_counter()-started:.3f}s; routes={len(routes)}', file=sys.stderr, flush=True)
    return {'success': True, 'results': results, 'timings': {'routes': route_timings, 'predict_ms': predict_ms, 'python_total_ms': (time.perf_counter()-started)*1000}}


def main():
    # Load expensive data only once, even across multiple live requests.
    started = time.perf_counter()
    roads = load_road_links()
    road_load_ms = (time.perf_counter()-started)*1000
    index_started = time.perf_counter()
    _ = roads.sindex
    index_ms = (time.perf_counter()-index_started)*1000
    lookup_started = time.perf_counter()
    with SCORES_PATH.open(encoding='utf-8') as f:
        lookup = {str(x['link_id']): x for x in json.load(f)}
    lookup_ms = (time.perf_counter()-lookup_started)*1000
    model_started = time.perf_counter()
    model, feature_names = load_model()
    cache = MatchCache()
    initialization = {"road_load_ms": road_load_ms, "spatial_index_ms": index_ms, "lookup_load_ms": lookup_ms, "model_load_ms": (time.perf_counter()-model_started)*1000, "worker_ready_ms": (time.perf_counter()-started)*1000}
    print(f'[AI] initialized in {time.perf_counter()-started:.3f}s', file=sys.stderr, flush=True)
    if '--worker' in sys.argv:
        print(json.dumps({'type': 'ready', 'timings': initialization}), flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                response = process_request(json.loads(line), roads, lookup, model, feature_names, cache=cache)
            except Exception as exc:
                response = {'success': False, 'error': str(exc)}
            print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
    else:
        try:
            response = process_request(json.load(sys.stdin), roads, lookup, model, feature_names, cache=cache)
            print(json.dumps(response, ensure_ascii=False, allow_nan=False))
        except Exception as exc:
            print(json.dumps({'success': False, 'error': str(exc)}, ensure_ascii=False))
            sys.exit(1)


if __name__ == '__main__':
    main()
