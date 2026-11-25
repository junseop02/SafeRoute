import csv
import gzip
import hashlib
import json
import math
import time

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import shapely

from pyproj import Transformer
from shapely import STRtree


# ==================================================
# 1. 기본 설정
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = (
    ROOT / "data" / "national_accident_risk" / "processed"
)

RESULTS = PROCESSED / "results"

ROAD_PATH = PROCESSED / "national_road_features.parquet"

SCORE_PATH = PROCESSED / "national_link_risk_scores.parquet"

POLICY_PATH = RESULTS / "national_route_policy_P01.json"

INTERFACE_PATH = (
    RESULTS / "national_service_interface_P01.json"
)

ACCIDENT_DIR = ROOT / "data" / "legacy" / "raw" / "accident"

SERVICE_DATA = (
    ROOT / "service" / "data"
)

TILE_DIR = SERVICE_DATA / "tiles"

MANIFEST_PATH = SERVICE_DATA / "national_manifest.json"

VALIDATION_DIR = ROOT / "integration" / "validation"

BUILD_REPORT_PATH = (
    VALIDATION_DIR / "national_tile_build_report.json"
)

VALIDATION_REPORT_PATH = (
    VALIDATION_DIR / "national_tile_validation_report.json"
)

DISTANCE_REPORT_PATH = (
    VALIDATION_DIR / "national_legacy_distance_comparison.csv"
)

REFERENCE_DATA = (
    ROOT.parent
    / "SafeRoute_새디자인_20261124"
    / "data"
)

REFERENCE_TILE_DIR = REFERENCE_DATA / "tiles"

MODEL_ID = "T09"
POLICY_ID = "P01"

SCHEMA_VERSION = "T09-P01-tiles-v1"

EXPECTED_LINKS = 1562356
EXPECTED_PARTS = 1562357
EXPECTED_TILES = 1172
EXPECTED_TILE_ROWS = 1608323

EXPECTED_MULTIPART_LINK = "3720460800"
EXPECTED_PRECISION_LINK = "2640273901"

EXPECTED_LEGACY_LINKS = 1542745

BATCH_SIZE = 5000

TILE_PADDING = 0.0003

BASE_DECIMALS = 6
MAX_DECIMALS = 7

TO_WGS = Transformer.from_crs(
    5179, 4326, always_xy=True
)

TO_PROJECTED = Transformer.from_crs(
    4326, 5179, always_xy=True
)

CHECKS = []


# ==================================================
# 2. 공통 함수
# ==================================================

def check(name, condition):

    passed = bool(condition)

    CHECKS.append({
        "check": name,
        "passed": passed
    })

    print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise AssertionError(name)


def require(condition, message):

    if not condition:
        raise ValueError(message)


def load_json(path):

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):

    with path.open("x", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
            allow_nan=False
        )


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def row_digest(row):

    payload = json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False
    ).encode("utf-8")

    return hashlib.sha256(payload).digest()


def expected_tile_keys(coordinates):

    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]

    x0 = math.floor(
        (min(xs) - TILE_PADDING) * 10
    )

    x1 = math.floor(
        (max(xs) + TILE_PADDING) * 10
    )

    y0 = math.floor(
        (min(ys) - TILE_PADDING) * 10
    )

    y1 = math.floor(
        (max(ys) + TILE_PADDING) * 10
    )

    return {
        f"{x}_{y}"
        for x in range(x0, x1 + 1)
        for y in range(y0, y1 + 1)
    }


# ==================================================
# 3. 입력 및 Manifest 검증
# ==================================================

def validate_inputs():

    print("\n===== Input Validation =====")

    paths = [
        ROAD_PATH,
        SCORE_PATH,
        POLICY_PATH,
        INTERFACE_PATH,
        MANIFEST_PATH,
        BUILD_REPORT_PATH,
        SERVICE_DATA / POLICY_PATH.name,
        SERVICE_DATA / INTERFACE_PATH.name
    ]

    for path in paths:

        check(
            f"Input Exists: {path.name}",
            path.is_file()
        )

    check(
        "Tile Directory Exists",
        TILE_DIR.is_dir()
    )

    check(
        "Reference Tile Directory Exists",
        REFERENCE_TILE_DIR.is_dir()
    )

    check(
        "Validation Report Not Exists",
        not VALIDATION_REPORT_PATH.exists()
    )

    check(
        "Distance Report Not Exists",
        not DISTANCE_REPORT_PATH.exists()
    )

    manifest = load_json(MANIFEST_PATH)

    build_report = load_json(BUILD_REPORT_PATH)

    policy = load_json(POLICY_PATH)

    interface = load_json(INTERFACE_PATH)

    check(
        "Manifest Model ID",
        manifest["model_id"] == MODEL_ID
    )

    check(
        "Manifest Policy ID",
        manifest["policy_id"] == POLICY_ID
    )

    check(
        "Manifest Schema Version",
        manifest["schema_version"] == SCHEMA_VERSION
    )

    check(
        "Build Status",
        build_report["build_status"] == "PASS"
    )

    check(
        "Manifest and Build Link Stats",
        manifest["link_stats"]
        == build_report["link_stats"]
    )

    check(
        "Manifest and Build Tile Stats",
        manifest["tile_stats"]
        == build_report["tile_stats"]
    )

    check(
        "Manifest Coordinate CRS",
        manifest["coordinate_crs"] == "EPSG:4326"
    )

    check(
        "Source Coordinate CRS",
        manifest["source_geometry_crs"] == "EPSG:5179"
    )

    check(
        "Coordinate Order",
        manifest["coordinate_order"]
        == "longitude_latitude"
    )

    check(
        "Base Coordinate Precision",
        manifest["coordinate_decimals"] == BASE_DECIMALS
    )

    check(
        "Maximum Coordinate Precision",
        manifest["maximum_coordinate_decimals"]
        == MAX_DECIMALS
    )

    check(
        "Adaptive Precision Mode",
        manifest["coordinate_precision_mode"]
        == "adaptive"
    )

    check(
        "Precision Stats Consistency",
        manifest["coordinate_precision_stats"]
        == build_report["coordinate_precision_stats"]
    )

    check(
        "Tile Size",
        manifest["tile_size_degrees"] == 0.1
    )

    check(
        "Tile Padding",
        manifest["tile_padding_degrees"] == TILE_PADDING
    )

    check(
        "Compression",
        manifest["compression"] == "gzip"
    )

    check(
        "Raw Score Range",
        manifest["score_range"] == [0, 1]
    )

    check(
        "Raw Score Normalization",
        manifest["score_normalization"] == "raw"
    )

    check(
        "Score Meaning",
        manifest["score_meaning"]
        == "historical_hotspot_membership_score"
    )

    check(
        "Policy Not Applied to Link Tiles",
        manifest["policy_applied_to_tiles"] is False
    )

    check(
        "Tile Row Schema",
        manifest["tile_row_schema"] == [
            "link_id",
            "coordinates",
            "hotspot_score",
            "lanes",
            "intersection_nodes_200m",
            "legacy_hotspot_distance_m",
            "part_index"
        ]
    )

    check(
        "Geometry Identity",
        manifest["geometry_identity"] == [
            "link_id",
            "part_index"
        ]
    )

    check(
        "National Link Count",
        manifest["link_stats"]["all_links"]
        == EXPECTED_LINKS
    )

    check(
        "National Scored Link Count",
        manifest["link_stats"]["scored_links"]
        == EXPECTED_LINKS
    )

    check(
        "No Excluded Links",
        manifest["link_stats"]["excluded_links"] == 0
    )

    check(
        "Geometry Part Count",
        manifest["tile_stats"]["unique_geometry_parts"]
        == EXPECTED_PARTS
    )

    check(
        "Policy Model",
        policy["model_id"] == MODEL_ID
    )

    check(
        "Policy ID",
        policy["policy_id"] == POLICY_ID
    )

    check(
        "Policy Weights",
        policy["weights"] == {
            "risk": 0.8,
            "narrow": 0.1,
            "intersection": 0.1
        }
    )

    check(
        "Policy Intersection Reference",
        policy["intersection_reference"] == 100
    )

    check(
        "Interface Model",
        interface["model_id"] == MODEL_ID
    )

    check(
        "Interface Policy",
        interface["policy_id"] == POLICY_ID
    )

    check(
        "Policy Copy Hash",
        sha256_file(POLICY_PATH)
        == sha256_file(
            SERVICE_DATA / POLICY_PATH.name
        )
    )

    check(
        "Interface Copy Hash",
        sha256_file(INTERFACE_PATH)
        == sha256_file(
            SERVICE_DATA / INTERFACE_PATH.name
        )
    )

    check(
        "Legacy Accident Distance Not Used in Model",
        manifest["accident_data"]["used_in_model"]
        is False
    )

    check(
        "Legacy Accident Distance Not Used in P01",
        manifest["accident_data"]["used_in_P01"]
        is False
    )

    return manifest, build_report


# ==================================================
# 4. 원본 데이터 해시 검증
# ==================================================

def validate_source_hashes(manifest):

    print("\n===== Source File Integrity =====")

    for filename, expected_hash in (
        manifest["source_sha256"].items()
    ):

        path = Path(filename)

        check(
            f"Source Exists: {path.name}",
            path.is_file()
        )

        check(
            f"Source Hash: {path.name}",
            sha256_file(path) == expected_hash
        )


# ==================================================
# 5. 사고다발지역 좌표 독립 구축
# ==================================================

def prepare_accident_tree(manifest):

    print("\n===== Independent Accident Data =====")

    unique = {}

    counts = {}

    files = sorted(
        ACCIDENT_DIR.glob("*.csv")
    )

    check(
        "Accident CSV Count",
        len(files) == 12
    )

    for path in files:

        with path.open("rb") as f:
            prefix = f.read(4)

        encoding = (
            "utf-8-sig"
            if (
                prefix.startswith(b"\xef\xbb\xbf")
                or prefix.startswith(b'"\xec')
            )
            else "cp949"
        )

        count = 0

        with path.open(
            encoding=encoding,
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            require(
                {"경도", "위도", "사고건수"}
                .issubset(reader.fieldnames or []),
                f"Invalid accident columns: {path.name}"
            )

            for row in reader:

                try:

                    lon = float(row["경도"])
                    lat = float(row["위도"])
                    cases = int(row["사고건수"])

                except (
                    KeyError,
                    ValueError,
                    TypeError
                ):
                    continue

                if not (
                    124 <= lon <= 132
                    and 33 <= lat <= 39
                    and cases >= 1
                ):
                    continue

                key = (
                    round(lon, 6),
                    round(lat, 6)
                )

                unique[key] = max(
                    unique.get(key, 0),
                    cases
                )

                count += 1

        counts[path.name] = count

    check(
        "Accident Source Counts",
        counts
        == manifest["accident_data"]["source_files"]
    )

    check(
        "Accident Raw Rows",
        sum(counts.values()) == 35562
    )

    check(
        "Accident Unique Coordinates",
        len(unique) == 33096
    )

    coordinates = np.asarray(
        list(unique),
        dtype=np.float64
    )

    x, y = TO_PROJECTED.transform(
        coordinates[:, 0],
        coordinates[:, 1]
    )

    check(
        "Accident Projected Coordinates",
        np.isfinite(x).all()
        and np.isfinite(y).all()
    )

    points = shapely.points(x, y)

    return STRtree(points)


# ==================================================
# 6. 원본 도로 및 점수 검증 데이터 생성
# ==================================================

def prepare_expected_rows(accident_tree):

    print("\n===== Independent Source Reconstruction =====")

    road_file = pq.ParquetFile(ROAD_PATH)

    score_file = pq.ParquetFile(SCORE_PATH)

    check(
        "Source Road Row Count",
        road_file.metadata.num_rows == EXPECTED_LINKS
    )

    check(
        "Source Score Row Count",
        score_file.metadata.num_rows == EXPECTED_LINKS
    )

    scores = score_file.read(
        columns=[
            "link_id",
            "hotspot_score",
            "lanes",
            "intersection_nodes_200m"
        ]
    ).to_pandas()

    check(
        "Source Score IDs Unique",
        scores["link_id"].is_unique
    )

    check(
        "Source Scores Valid",
        np.isfinite(scores["hotspot_score"]).all()
        and scores["hotspot_score"].between(0, 1).all()
    )

    scores = scores.set_index("link_id")

    expected = {}

    source_ids = set()

    precision_counts = Counter()

    multipart_links = {}

    adjusted_links = []

    processed = 0

    columns = [
        "link_id",
        "lanes",
        "intersection_nodes_200m",
        "geometry_wkb"
    ]

    for batch in road_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=columns
    ):

        frame = batch.to_pandas()

        ids = frame["link_id"].astype(str).tolist()

        require(
            len(set(ids)) == len(ids)
            and not source_ids.intersection(ids),
            "Duplicate source road IDs"
        )

        source_ids.update(ids)

        matched = scores.loc[ids]

        lanes = frame["lanes"].to_numpy()

        intersections = frame[
            "intersection_nodes_200m"
        ].to_numpy()

        risks = matched["hotspot_score"].to_numpy()

        require(
            np.array_equal(
                lanes,
                matched["lanes"].to_numpy()
            ),
            "Source lane mismatch"
        )

        require(
            np.array_equal(
                intersections,
                matched[
                    "intersection_nodes_200m"
                ].to_numpy()
            ),
            "Source intersection mismatch"
        )

        geometries = shapely.from_wkb(
            frame["geometry_wkb"].to_numpy()
        )

        require(
            not shapely.is_empty(geometries).any()
            and shapely.is_valid(geometries).all(),
            "Invalid source geometry"
        )

        # 원본 선형과 사고다발지의 최단거리 재계산
        pairs, distances = accident_tree.query_nearest(
            geometries,
            all_matches=False,
            return_distance=True
        )

        nearest = np.full(
            len(frame),
            np.nan,
            dtype=np.float64
        )

        nearest[pairs[0]] = distances

        require(
            np.isfinite(nearest).all()
            and np.all(nearest >= 0),
            "Invalid reconstructed accident distance"
        )

        wgs_geometries = shapely.transform(
            geometries,
            TO_WGS.transform,
            interleaved=False
        )

        for i, geometry in enumerate(wgs_geometries):

            link_id = ids[i]

            if geometry.geom_type == "LineString":

                parts = [geometry]

            elif geometry.geom_type == "MultiLineString":

                parts = list(geometry.geoms)

                multipart_links[link_id] = len(parts)

            else:

                raise ValueError(
                    f"Unsupported geometry: {link_id}"
                )

            for part_index, part in enumerate(parts):

                coordinates = shapely.get_coordinates(part)

                require(
                    len(coordinates) >= 2
                    and np.isfinite(coordinates).all(),
                    f"Invalid source coordinates: {link_id}"
                )

                # 최소한의 유효 정밀도 독립 확인
                compact = None

                for decimals in range(
                    BASE_DECIMALS,
                    MAX_DECIMALS + 1
                ):

                    candidate = [
                        [
                            round(float(x), decimals),
                            round(float(y), decimals)
                        ]
                        for x, y in coordinates
                    ]

                    if len({
                        tuple(point)
                        for point in candidate
                    }) >= 2:

                        compact = candidate
                        break

                require(
                    compact is not None,
                    f"Unresolved precision: {link_id}"
                )

                precision_counts[decimals] += 1

                if decimals > BASE_DECIMALS:

                    adjusted_links.append({
                        "link_id": link_id,
                        "part_index": part_index,
                        "decimals": decimals
                    })

                row = [
                    link_id,
                    compact,
                    float(risks[i]),
                    int(lanes[i]),
                    int(intersections[i]),
                    round(float(nearest[i]), 3),
                    part_index
                ]

                # 일반 링크와 복합 선형을 구분
                key = (
                    link_id
                    if part_index == 0
                    else f"{link_id}:{part_index}"
                )

                require(
                    key not in expected,
                    f"Duplicate geometry part: {key}"
                )

                # 전체 좌표 대신 해시를 보관해 메모리 절약
                expected[key] = (
                    row_digest(row),
                    len(expected_tile_keys(compact))
                )

        processed += len(frame)

        if (
            processed % 100000 < len(frame)
            or processed == EXPECTED_LINKS
        ):

            print(
                f"Reconstructed: "
                f"{processed:,}/{EXPECTED_LINKS:,}",
                flush=True
            )

    check(
        "All Source Links Reconstructed",
        processed == EXPECTED_LINKS
    )

    check(
        "Source Link and Score IDs Match",
        source_ids == set(scores.index)
    )

    check(
        "Expected Geometry Parts",
        len(expected) == EXPECTED_PARTS
    )

    check(
        "Expected Six Decimal Parts",
        precision_counts[6] == 1562356
    )

    check(
        "Expected Seven Decimal Parts",
        precision_counts[7] == 1
    )

    check(
        "Expected Precision Adjusted Link",
        adjusted_links == [
            {
                "link_id": EXPECTED_PRECISION_LINK,
                "part_index": 0,
                "decimals": 7
            }
        ]
    )

    check(
        "Expected Multipart Link",
        multipart_links == {
            EXPECTED_MULTIPART_LINK: 2
        }
    )

    print("Expected Parts:", len(expected))
    print("Precision Counts:", dict(precision_counts))
    print("Adjusted Links:", adjusted_links)
    print("Multipart Links:", multipart_links)

    return expected, {
        "precision_counts": dict(precision_counts),
        "adjusted_links": adjusted_links,
        "multipart_links": multipart_links
    }


# ==================================================
# 7. 실제 타일 파일 독립 검증
# ==================================================

def validate_tiles(manifest, expected):

    print("\n===== Independent Tile Validation =====")

    files = sorted(
        TILE_DIR.glob("*.jsonl.gz")
    )

    check(
        "Actual Tile File Count",
        len(files) == EXPECTED_TILES
    )

    check(
        "Manifest Tile Count",
        len(manifest["tiles"]) == EXPECTED_TILES
    )

    check(
        "No Temporary Tile Files",
        not list(TILE_DIR.glob("*.jsonl"))
    )

    check(
        "Actual and Manifest Tile Keys",
        {
            path.name.removesuffix(".jsonl.gz")
            for path in files
        } == set(manifest["tiles"])
    )

    stats = Counter()

    seen = Counter()

    # 기존 서비스 거리 비교에 사용
    new_distances = {}

    for file_index, path in enumerate(
        files, start=1
    ):

        tile_key = path.name.removesuffix(".jsonl.gz")

        metadata = manifest["tiles"][tile_key]

        require(
            metadata["file"] == path.name,
            f"Tile filename mismatch: {tile_key}"
        )

        require(
            path.stat().st_size == metadata["bytes"],
            f"Tile size mismatch: {tile_key}"
        )

        require(
            sha256_file(path) == metadata["sha256"],
            f"Tile hash mismatch: {tile_key}"
        )

        stats["verified_files"] += 1

        stats["total_bytes"] += path.stat().st_size

        tile_seen = set()

        row_count = 0

        with gzip.open(
            path,
            "rt",
            encoding="utf-8"
        ) as f:

            for line in f:

                row = json.loads(line)

                require(
                    isinstance(row, list)
                    and len(row) == 7,
                    f"Invalid tile row schema: {tile_key}"
                )

                (
                    link_id,
                    coordinates,
                    risk,
                    lanes,
                    intersections,
                    distance,
                    part_index
                ) = row

                require(
                    isinstance(link_id, str)
                    and len(link_id) > 0,
                    "Invalid link ID"
                )

                require(
                    type(part_index) is int
                    and part_index >= 0,
                    f"Invalid part index: {link_id}"
                )

                require(
                    type(risk) in (int, float)
                    and math.isfinite(risk)
                    and 0 <= risk <= 1,
                    f"Invalid risk score: {link_id}"
                )

                require(
                    type(lanes) is int and lanes >= 1,
                    f"Invalid lanes: {link_id}"
                )

                require(
                    type(intersections) is int
                    and intersections >= 0,
                    f"Invalid intersections: {link_id}"
                )

                require(
                    type(distance) in (int, float)
                    and math.isfinite(distance)
                    and distance >= 0,
                    f"Invalid accident distance: {link_id}"
                )

                require(
                    isinstance(coordinates, list)
                    and len(coordinates) >= 2,
                    f"Invalid geometry: {link_id}"
                )

                for point in coordinates:

                    require(
                        isinstance(point, list)
                        and len(point) == 2,
                        f"Invalid coordinate pair: {link_id}"
                    )

                    x, y = point

                    require(
                        type(x) in (int, float)
                        and type(y) in (int, float)
                        and math.isfinite(x)
                        and math.isfinite(y)
                        and 124 <= x <= 132
                        and 33 <= y <= 39,
                        f"Invalid coordinate range: {link_id}"
                    )

                require(
                    len({
                        tuple(point)
                        for point in coordinates
                    }) >= 2,
                    f"Collapsed tile geometry: {link_id}"
                )

                key = (
                    link_id
                    if part_index == 0
                    else f"{link_id}:{part_index}"
                )

                require(
                    key in expected,
                    f"Unknown geometry part: {key}"
                )

                require(
                    key not in tile_seen,
                    f"Duplicate part within tile: {key}"
                )

                tile_seen.add(key)

                expected_digest, expected_count = (
                    expected[key]
                )

                # 좌표, 모델 점수, 차로, 교차로,
                # 사고다발지 거리 및 part_index 전체 대조
                require(
                    row_digest(row) == expected_digest,
                    f"Source row mismatch: {key}"
                )

                require(
                    tile_key in expected_tile_keys(
                        coordinates
                    ),
                    f"Incorrect tile placement: {key}"
                )

                seen[key] += 1

                require(
                    seen[key] <= expected_count,
                    f"Excess tile copies: {key}"
                )

                if link_id in new_distances:

                    require(
                        new_distances[link_id] == distance,
                        f"Inconsistent distance: {link_id}"
                    )

                else:

                    new_distances[link_id] = distance

                row_count += 1

                stats["total_rows"] += 1

        require(
            row_count == metadata["rows"],
            f"Tile row count mismatch: {tile_key}"
        )

        if (
            file_index % 100 == 0
            or file_index == len(files)
        ):

            print(
                f"Validated Tiles: "
                f"{file_index:,}/{len(files):,} | "
                f"Rows: {stats['total_rows']:,}",
                flush=True
            )

    check(
        "All Tile Hashes and Sizes Valid",
        stats["verified_files"] == EXPECTED_TILES
    )

    check(
        "All Tile Rows Valid",
        stats["total_rows"] == EXPECTED_TILE_ROWS
    )

    check(
        "All Source Geometry Parts Present",
        len(seen) == len(expected)
    )

    check(
        "All Source Links Present",
        len(new_distances) == EXPECTED_LINKS
    )

    check(
        "All Expected Tile Placements Present",
        all(
            seen[key] == expected_count
            for key, (
                digest,
                expected_count
            ) in expected.items()
        )
    )

    duplicate_rows = (
        stats["total_rows"] - len(seen)
    )

    check(
        "Duplicate Tile Row Count",
        duplicate_rows
        == manifest["tile_stats"]["duplicate_tile_rows"]
    )

    check(
        "Total Compressed Bytes",
        stats["total_bytes"]
        == manifest["tile_stats"]["total_bytes"]
    )

    check(
        "Multipart Part Zero Present",
        EXPECTED_MULTIPART_LINK in seen
    )

    check(
        "Multipart Part One Present",
        f"{EXPECTED_MULTIPART_LINK}:1" in seen
    )

    check(
        "Precision Adjusted Link Present",
        EXPECTED_PRECISION_LINK in seen
    )

    print("Unique Links:", len(new_distances))
    print("Unique Parts:", len(seen))
    print("Total Tile Rows:", stats["total_rows"])
    print("Duplicate Tile Rows:", duplicate_rows)

    return new_distances, {
        "verified_files": stats["verified_files"],
        "total_rows": stats["total_rows"],
        "unique_links": len(new_distances),
        "unique_parts": len(seen),
        "duplicate_tile_rows": duplicate_rows,
        "total_bytes": stats["total_bytes"],
        "source_row_mismatches": 0,
        "missing_geometry_parts": 0,
        "incorrect_tile_placements": 0
    }


# ==================================================
# 8. 기존 서비스 사고다발지 거리 비교
# ==================================================

def compare_legacy_distances(new_distances):

    print("\n===== Legacy Hotspot Distance Comparison =====")

    files = sorted(
        REFERENCE_TILE_DIR.glob("*.jsonl.gz")
    )

    check(
        "Reference Tile Count",
        len(files) == 1147
    )

    legacy_distances = {}

    for file_index, path in enumerate(
        files, start=1
    ):

        with gzip.open(
            path,
            "rt",
            encoding="utf-8"
        ) as f:

            for line in f:

                row = json.loads(line)

                require(
                    isinstance(row, list)
                    and len(row) == 6,
                    f"Invalid legacy row: {path.name}"
                )

                link_id = str(row[0])

                distance = float(row[5])

                require(
                    math.isfinite(distance)
                    and distance >= 0,
                    f"Invalid legacy distance: {link_id}"
                )

                if link_id in legacy_distances:

                    require(
                        legacy_distances[link_id] == distance,
                        f"Legacy duplicate mismatch: {link_id}"
                    )

                else:

                    legacy_distances[link_id] = distance

        if (
            file_index % 100 == 0
            or file_index == len(files)
        ):

            print(
                f"Reference Tiles: "
                f"{file_index:,}/{len(files):,}",
                flush=True
            )

    check(
        "Reference Unique Link Count",
        len(legacy_distances) == EXPECTED_LEGACY_LINKS
    )

    check(
        "All Reference Links in New Tiles",
        all(
            link_id in new_distances
            for link_id in legacy_distances
        )
    )

    old_values = np.fromiter(
        legacy_distances.values(),
        dtype=np.float64,
        count=len(legacy_distances)
    )

    new_values = np.fromiter(
        (
            new_distances[link_id]
            for link_id in legacy_distances
        ),
        dtype=np.float64,
        count=len(legacy_distances)
    )

    differences = new_values - old_values

    absolute = np.abs(differences)

    old_near = old_values <= 150

    new_near = new_values <= 150

    # 거리 차이는 임의로 허용하지 않고 분포를 기록
    summary = {
        "reference_links": len(legacy_distances),

        "new_links": len(new_distances),

        "additional_links": (
            len(new_distances)
            - len(legacy_distances)
        ),

        "exact_matches": int(
            np.count_nonzero(absolute == 0)
        ),

        "within_0_001m": int(
            np.count_nonzero(absolute <= 0.001)
        ),

        "within_0_01m": int(
            np.count_nonzero(absolute <= 0.01)
        ),

        "within_0_1m": int(
            np.count_nonzero(absolute <= 0.1)
        ),

        "within_1m": int(
            np.count_nonzero(absolute <= 1)
        ),

        "within_10m": int(
            np.count_nonzero(absolute <= 10)
        ),

        "over_10m": int(
            np.count_nonzero(absolute > 10)
        ),

        "mean_signed_difference_m": float(
            np.mean(differences)
        ),

        "mean_absolute_difference_m": float(
            np.mean(absolute)
        ),

        "median_absolute_difference_m": float(
            np.median(absolute)
        ),

        "p95_absolute_difference_m": float(
            np.quantile(absolute, 0.95)
        ),

        "p99_absolute_difference_m": float(
            np.quantile(absolute, 0.99)
        ),

        "maximum_absolute_difference_m": float(
            np.max(absolute)
        ),

        "legacy_near_150m_count": int(
            np.count_nonzero(old_near)
        ),

        "new_near_150m_count": int(
            np.count_nonzero(new_near)
        ),

        "near_150m_classification_changed": int(
            np.count_nonzero(old_near != new_near)
        )
    }

    # 거리 차이가 큰 상위 100개 링크 저장
    top_indices = np.argsort(absolute)[-100:][::-1]

    link_ids = list(legacy_distances)

    with DISTANCE_REPORT_PATH.open(
        "x",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "link_id",
            "legacy_distance_m",
            "new_distance_m",
            "signed_difference_m",
            "absolute_difference_m",
            "near_150m_changed"
        ])

        for i in top_indices:

            writer.writerow([
                link_ids[i],
                float(old_values[i]),
                float(new_values[i]),
                float(differences[i]),
                float(absolute[i]),
                bool(old_near[i] != new_near[i])
            ])

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


# ==================================================
# 9. 메인 함수
# ==================================================

def main():

    start = time.perf_counter()

    print("===== National Service Tile Validator =====")

    manifest, build_report = validate_inputs()

    manifest_hash_before = sha256_file(MANIFEST_PATH)

    build_report_hash_before = sha256_file(
        BUILD_REPORT_PATH
    )

    validate_source_hashes(manifest)

    accident_tree = prepare_accident_tree(manifest)

    expected, source_summary = prepare_expected_rows(
        accident_tree
    )

    new_distances, tile_summary = validate_tiles(
        manifest,
        expected
    )

    # 검증 완료된 대용량 객체 메모리 해제
    del expected
    del accident_tree

    legacy_summary = compare_legacy_distances(
        new_distances
    )

    del new_distances

    print("\n===== Final Source Integrity =====")

    validate_source_hashes(manifest)

    check(
        "Manifest Unchanged",
        sha256_file(MANIFEST_PATH)
        == manifest_hash_before
    )

    check(
        "Build Report Unchanged",
        sha256_file(BUILD_REPORT_PATH)
        == build_report_hash_before
    )

    elapsed = time.perf_counter() - start

    report = {
        "model_id": MODEL_ID,
        "policy_id": POLICY_ID,
        "schema_version": SCHEMA_VERSION,

        "validation_status": "PASS",

        "source_reconstruction": source_summary,

        "tile_validation": tile_summary,

        "legacy_distance_comparison": legacy_summary,

        "legacy_distance_review": "pending",

        "checks": CHECKS,

        "passed_checks": sum(
            item["passed"] for item in CHECKS
        ),

        "total_checks": len(CHECKS),

        "manifest_sha256": manifest_hash_before,

        "build_report_sha256": build_report_hash_before,

        "source_hashes_unchanged": True,

        "elapsed_seconds": elapsed,

        "model_retrained": False,

        "policy_modified": False,

        "tiles_modified": False,

        "deployment_status": "not_deployed"
    }

    save_json(
        VALIDATION_REPORT_PATH,
        report
    )

    print("\n===== Final Validation Summary =====")

    print("Model:", MODEL_ID)
    print("Policy:", POLICY_ID)
    print("Schema:", SCHEMA_VERSION)

    print(
        "Passed Checks:",
        f"{report['passed_checks']}/{report['total_checks']}"
    )

    print(
        "Validated Links:",
        f"{tile_summary['unique_links']:,}"
    )

    print(
        "Validated Geometry Parts:",
        f"{tile_summary['unique_parts']:,}"
    )

    print(
        "Validated Tiles:",
        f"{tile_summary['verified_files']:,}"
    )

    print(
        "Validated Tile Rows:",
        f"{tile_summary['total_rows']:,}"
    )

    print(
        "Source Row Mismatches:",
        tile_summary["source_row_mismatches"]
    )

    print(
        "Missing Geometry Parts:",
        tile_summary["missing_geometry_parts"]
    )

    print(
        "Incorrect Tile Placements:",
        tile_summary["incorrect_tile_placements"]
    )

    print(
        "Legacy Distance Mean Difference:",
        legacy_summary["mean_absolute_difference_m"]
    )

    print(
        "Legacy Distance Maximum Difference:",
        legacy_summary["maximum_absolute_difference_m"]
    )

    print(
        "Legacy Near-150m Changes:",
        legacy_summary[
            "near_150m_classification_changed"
        ]
    )

    print("\n===== Saved Results =====")

    print("Validation Report:", VALIDATION_REPORT_PATH)

    print("Distance Comparison:", DISTANCE_REPORT_PATH)

    print("\n===== Validation Completed =====")

    print(f"Elapsed Time: {elapsed:.2f}s")

    print("Independent Tile Validation: PASS")

    print("Legacy Distance Comparison: Completed")

    print("Legacy Distance Review: Pending")

    print("Model Retraining: Not Performed")

    print("Policy Modification: Not Performed")

    print("Tile Modification: Not Performed")

    print("Deployment: Not Performed")


if __name__ == "__main__":

    main()