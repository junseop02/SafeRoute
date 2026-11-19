"""SafeRoute 실시간 경로-도로 링크 매칭.

입력 경로 좌표: EPSG:4326 (경도, 위도)
도로 링크: source/links.geojson -> EPSG:5179
기존 결과 형식: geometry_length_m, matched_length_m, coverage, links
"""

from collections import defaultdict
from pathlib import Path
import math

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import transform, line_interpolate_point, distance as geometry_distance
from shapely.geometry import LineString

PROJECT_ROOT = Path(__file__).resolve().parent
LINK_PATH = PROJECT_ROOT / "source" / "links.geojson"
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)


def load_road_links():
    """도로 링크를 읽고 미터 단위 좌표계로 변환한다."""
    roads = gpd.read_file(LINK_PATH)
    if "link_id" not in roads.columns:
        raise ValueError("도로 링크 데이터에 link_id 열이 없습니다.")
    if roads.crs is None:
        raise ValueError("도로 링크 데이터의 좌표계(CRS)가 없습니다.")
    return roads.to_crs(epsg=5179)


def _project_route(route_coordinates):
    """(경도, 위도) 좌표를 미터 단위 LineString으로 변환한다."""
    if len(route_coordinates) < 2:
        raise ValueError("경로 좌표가 최소 2개 필요합니다.")
    route = LineString(route_coordinates)
    return transform(route, TRANSFORMER.transform, interleaved=False)


def _empty_result():
    return {
        "geometry_length_m": 0.0,
        "matched_length_m": 0.0,
        "coverage": 0.0,
        "links": [],
    }


def _format_result(total_length, matched_lengths):
    matched_length = sum(matched_lengths.values())
    return {
        "geometry_length_m": round(float(total_length), 4),
        "matched_length_m": round(float(matched_length), 4),
        "coverage": round(float(matched_length / total_length), 6) if total_length else 0.0,
        "links": [
            [link_id, round(float(length), 4)]
            for link_id, length in matched_lengths.items()
        ],
    }


def _route_segments(projected, interval_m):
    """경로를 일정 길이로 나누어 (시작점, 끝점, 중간점, 길이)를 반환한다."""
    if interval_m <= 0:
        raise ValueError("interval_m은 0보다 커야 합니다.")
    total_length = projected.length
    count = max(1, int(np.ceil(total_length / interval_m)))
    boundaries = np.linspace(0, total_length, count + 1)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        yield (
            projected.interpolate(start),
            projected.interpolate(end),
            projected.interpolate((start + end) / 2),
            float(end - start),
        )


def _nearby_indices(roads, point, max_distance_m):
    if max_distance_m <= 0:
        raise ValueError("max_distance_m은 0보다 커야 합니다.")
    return roads.sindex.query(
        point.buffer(max_distance_m), predicate="intersects"
    )


def find_nearby_links(roads, route_coordinates, buffer_m=20):
    """경로 주변 후보 링크를 가까운 순으로 반환한다."""
    if buffer_m <= 0:
        raise ValueError("buffer_m은 0보다 커야 합니다.")
    route = _project_route(route_coordinates)
    indices = roads.sindex.query(route.buffer(buffer_m), predicate="intersects")
    nearby = roads.iloc[indices].copy()
    nearby["distance_to_route_m"] = nearby.geometry.distance(route)
    return nearby.sort_values("distance_to_route_m")


def calculate_direction_difference(route_segment, road_line):
    """두 선의 방향축 차이(0~90도). 역방향도 동일한 축으로 취급한다."""
    a0, a1 = route_segment.coords[0], route_segment.coords[-1]
    b0, b1 = road_line.coords[0], road_line.coords[-1]
    ax, ay = a1[0] - a0[0], a1[1] - a0[1]
    bx, by = b1[0] - b0[0], b1[1] - b0[1]
    an, bn = math.hypot(ax, ay), math.hypot(bx, by)
    if an == 0 or bn == 0:
        return None
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (an * bn)))
    angle = math.degrees(math.acos(cosine))
    return min(angle, 180 - angle)


def match_route_to_links(roads, route_coordinates, interval_m=5, max_distance_m=10):
    """기존 거리 기반 매칭. 구간 중간점에서 가장 가까운 링크를 선택한다."""
    projected = _project_route(route_coordinates)
    if projected.length <= 0:
        raise ValueError("경로 길이가 0입니다.")

    # Preserve exactly the original sample positions, index query and tie order.
    # Only the used midpoint is interpolated; no GeoDataFrame is copied per sample.
    if interval_m <= 0:
        raise ValueError("interval_m은 0보다 커야 합니다.")
    count = max(1, int(np.ceil(projected.length / interval_m)))
    boundaries = np.linspace(0, projected.length, count + 1)
    midpoints = line_interpolate_point(projected, (boundaries[:-1] + boundaries[1:]) / 2)
    lengths = np.diff(boundaries)
    geometries = roads.geometry.array
    link_ids = roads["link_id"].to_numpy()
    matched_lengths = defaultdict(float)
    for midpoint, length in zip(midpoints, lengths):
        indices = _nearby_indices(roads, midpoint, max_distance_m)
        if len(indices) == 0:
            continue
        distances = geometry_distance(geometries.take(indices), midpoint)
        best_position = np.argsort(distances, kind="quicksort")[0]
        if distances[best_position] <= max_distance_m:
            matched_lengths[str(link_ids[indices[best_position]])] += float(length)

    return _format_result(projected.length, matched_lengths)


def match_route_to_links_with_direction(
    roads, route_coordinates, interval_m=2, max_distance_m=10, max_angle=30
):
    """기존 방향 제한 매칭. 국소 방향 차이가 max_angle 이하인 최근접 링크 선택."""
    projected = _project_route(route_coordinates)
    if projected.length == 0:
        return _empty_result()

    matched_lengths = defaultdict(float)
    for start, end, midpoint, length in _route_segments(projected, interval_m):
        route_segment = LineString([start, end])
        best_id, best_distance = None, float("inf")
        for idx in _nearby_indices(roads, midpoint, max_distance_m):
            road = roads.geometry.iloc[idx]
            if road.geom_type != "LineString" or road.length == 0:
                continue
            position = road.project(midpoint)
            road_segment = LineString([
                road.interpolate(max(0, position - 2)),
                road.interpolate(min(road.length, position + 2)),
            ])
            angle = calculate_direction_difference(route_segment, road_segment)
            if angle is None or angle > max_angle:
                continue
            distance = midpoint.distance(road)
            if distance <= max_distance_m and distance < best_distance:
                best_id = str(roads.iloc[idx]["link_id"])
                best_distance = distance
        if best_id is not None:
            matched_lengths[best_id] += length

    return _format_result(projected.length, matched_lengths)


def match_route_geometry(roads, geometry, method="distance", **kwargs):
    """GeoJSON LineString/MultiLineString 경로를 하나의 매칭 결과로 합산한다.

    method='distance' 또는 'direction'.
    """
    geom_type = geometry.get("type")
    if geom_type == "LineString":
        groups = [geometry["coordinates"]]
    elif geom_type == "MultiLineString":
        groups = geometry["coordinates"]
    else:
        raise ValueError(f"지원하지 않는 경로 형식: {geom_type}")

    matcher = (
        match_route_to_links if method == "distance"
        else match_route_to_links_with_direction if method == "direction"
        else None
    )
    if matcher is None:
        raise ValueError("method는 'distance' 또는 'direction'이어야 합니다.")

    total_length = 0.0
    matched_lengths = defaultdict(float)
    for coordinates in groups:
        if len(coordinates) < 2:
            continue
        # 길이가 0인 부분 경로는 전체 경로 매칭에서 건너뛴다.
        if LineString(coordinates).length == 0:
            continue
        result = matcher(roads, coordinates, **kwargs)
        total_length += result["geometry_length_m"]
        for link_id, length in result["links"]:
            matched_lengths[str(link_id)] += length

    return _format_result(total_length, matched_lengths)


if __name__ == "__main__":
    roads = load_road_links()
    print(f"도로 링크 로딩 완료: {len(roads):,}개 / 좌표계: {roads.crs}")
    print("매칭 함수 준비 완료 (실제 경로는 함수에 전달하여 실행하세요).")
