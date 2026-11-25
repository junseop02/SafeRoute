import csv
import gzip
import hashlib
import json
import math
import shutil
import time

from collections import Counter, OrderedDict
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

ROAD_PATH = (
    PROCESSED / "national_road_features.parquet"
)

SCORE_PATH = (
    PROCESSED / "national_link_risk_scores.parquet"
)

POLICY_PATH = (
    RESULTS / "national_route_policy_P01.json"
)

INTERFACE_PATH = (
    RESULTS / "national_service_interface_P01.json"
)

ACCIDENT_DIR = (
    ROOT / "data" / "legacy" / "raw" / "accident"
)

REFERENCE = (
    ROOT.parent / "SafeRoute_새디자인_20261124"
)

REFERENCE_MANIFEST = (
    REFERENCE / "data" / "national_manifest.json"
)

OUTPUT = (
    ROOT / "service" / "data"
)

VALIDATION_DIR = (
    ROOT / "integration" / "validation"
)

TILE_DIR = OUTPUT / "tiles"

MANIFEST_PATH = OUTPUT / "national_manifest.json"

BUILD_REPORT = (
    VALIDATION_DIR / "national_tile_build_report.json"
)

MODEL_ID = "T09"
POLICY_ID = "P01"
SCHEMA_VERSION = "T09-P01-tiles-v1"

EXPECTED_LINKS = 1562356
EXPECTED_ACCIDENT_FILES = 12

BATCH_SIZE = 5000

TILE_SIZE = 0.1
TILE_PADDING = 0.0003
COORDINATE_DECIMALS = 6
MAX_COORDINATE_DECIMALS = 7
GZIP_LEVEL = 5

MAX_OPEN_FILES = 64

TO_WGS = Transformer.from_crs(
    5179, 4326, always_xy=True
)

TO_PROJECTED = Transformer.from_crs(
    4326, 5179, always_xy=True
)


# ==================================================
# 2. 공통 함수
# ==================================================

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


def check(name, condition):

    passed = bool(condition)

    print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise AssertionError(name)


def source_hashes(paths):

    return {
        str(path): sha256_file(path)
        for path in paths
    }


# ==================================================
# 3. 입력 데이터 검증
# ==================================================

def validate_inputs():

    print("\n===== Input Validation =====")

    for path in [
        ROAD_PATH,
        SCORE_PATH,
        POLICY_PATH,
        INTERFACE_PATH,
        REFERENCE_MANIFEST
    ]:

        check(
            f"Input Exists: {path.name}",
            path.is_file()
        )

    check(
        "Output Directory Not Exists",
        not OUTPUT.exists()
    )

    check(
        "Build Report Not Exists",
        not BUILD_REPORT.exists()
    )

    policy = load_json(POLICY_PATH)
    interface = load_json(INTERFACE_PATH)

    check(
        "Policy Model ID",
        policy["model_id"] == MODEL_ID
    )

    check(
        "Selected Policy",
        policy["policy_id"] == POLICY_ID
    )

    check(
        "Policy Selected",
        policy["policy_selected"] is True
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
        "Intersection Reference",
        policy["intersection_reference"] == 100
    )

    check(
        "Interface Model ID",
        interface["model_id"] == MODEL_ID
    )

    check(
        "Interface Policy ID",
        interface["policy_id"] == POLICY_ID
    )

    score_key = SCORE_PATH.relative_to(
        ROOT
    ).as_posix()

    check(
        "Frozen Score File Hash",
        sha256_file(SCORE_PATH)
        == policy["source_sha256"][score_key]
    )

    road_file = pq.ParquetFile(ROAD_PATH)
    score_file = pq.ParquetFile(SCORE_PATH)

    check(
        "Road Row Count",
        road_file.metadata.num_rows == EXPECTED_LINKS
    )

    check(
        "Score Row Count",
        score_file.metadata.num_rows == EXPECTED_LINKS
    )

    road_metadata = (
        road_file.schema_arrow.metadata or {}
    )

    check(
        "Road Geometry CRS",
        road_metadata.get(b"geometry_crs")
        == b"EPSG:5179"
    )

    score_metadata = (
        score_file.schema_arrow.metadata or {}
    )

    check(
        "Score Metadata Model",
        score_metadata.get(b"model_id") == b"T09"
    )

    check(
        "Score Metadata Meaning",
        score_metadata.get(b"score_meaning")
        == b"historical_hotspot_membership_score"
    )

    accident_files = sorted(
        ACCIDENT_DIR.glob("*.csv")
    )

    check(
        "Accident CSV Count",
        len(accident_files) == EXPECTED_ACCIDENT_FILES
    )

    return road_file, score_file, accident_files


# ==================================================
# 4. 기존 사고다발지역 좌표 준비
# ==================================================

def load_accident_points(files):

    print("\n===== Legacy Accident Data =====")

    unique = {}
    file_counts = {}

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

            required = {
                "경도", "위도", "사고건수"
            }

            check(
                f"CSV Columns: {path.name}",
                required.issubset(
                    reader.fieldnames or []
                )
            )

            for row in reader:

                try:

                    lon = float(row["경도"])
                    lat = float(row["위도"])
                    cases = int(row["사고건수"])

                except (
                    KeyError,
                    TypeError,
                    ValueError
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

        file_counts[path.name] = count

        print(
            f"{path.name}: "
            f"{count:,} valid rows"
        )

    raw_count = sum(file_counts.values())

    check(
        "Legacy Accident Raw Count",
        raw_count == 35562
    )

    check(
        "Legacy Accident Unique Coordinates",
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

    points = shapely.points(x, y)

    check(
        "Accident Coordinates Valid",
        np.isfinite(x).all()
        and np.isfinite(y).all()
    )

    print("Raw Accident Rows:", raw_count)
    print("Unique Coordinates:", len(points))

    return STRtree(points), {
        "source_files": file_counts,
        "raw_rows": raw_count,
        "unique_coordinates": len(points),
        "coordinate_crs": "EPSG:5179",
        "used_in_model": False,
        "used_in_P01": False,
        "purpose": "legacy_hotspot_proximity_display"
    }


# ==================================================
# 5. 전국 타일 저장 클래스
# ==================================================

class TileWriter:

    def __init__(self, folder):

        self.folder = folder

        self.folder.mkdir(
            parents=True,
            exist_ok=False
        )

        self.handles = OrderedDict()
        self.counts = Counter()

        self.geometry_parts = 0
        self.total_tile_rows = 0

    def write(self, row):

        coordinates = row[1]

        xs = [p[0] for p in coordinates]
        ys = [p[1] for p in coordinates]

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

        tile_count = (
            (x1 - x0 + 1)
            * (y1 - y0 + 1)
        )

        if tile_count > 80:

            raise ValueError(
                f"Abnormally long link: {row[0]}"
            )

        payload = (
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False
            ) + "\n"
        ).encode("utf-8")

        self.geometry_parts += 1

        for x in range(x0, x1 + 1):

            for y in range(y0, y1 + 1):

                key = f"{x}_{y}"

                if key not in self.handles:

                    path = (
                        self.folder / f"{key}.jsonl"
                    )

                    self.handles[key] = path.open("ab")

                self.handles.move_to_end(key)

                self.handles[key].write(payload)

                self.counts[key] += 1
                self.total_tile_rows += 1

                if len(self.handles) > MAX_OPEN_FILES:

                    _, handle = (
                        self.handles.popitem(last=False)
                    )

                    handle.close()

    def close(self):

        for handle in self.handles.values():
            handle.close()

        self.handles.clear()

    def finish(self):

        self.close()

        files = sorted(
            self.folder.glob("*.jsonl")
        )

        print("\n===== Gzip Compression =====")

        for index, path in enumerate(
            files, start=1
        ):

            target = Path(str(path) + ".gz")

            with path.open("rb") as source:

                with gzip.open(
                    target,
                    "wb",
                    compresslevel=GZIP_LEVEL
                ) as dest:

                    shutil.copyfileobj(
                        source,
                        dest,
                        length=1024 * 1024
                    )

            # 압축 완료된 임시 JSONL만 삭제
            path.unlink()

            if index % 100 == 0 or index == len(files):

                print(
                    f"Compressed: "
                    f"{index:,}/{len(files):,}"
                )


# ==================================================
# 6. 전국 도로 점수 조회 데이터
# ==================================================

def load_scores(score_file):

    print("\n===== National Link Scores =====")

    scores = score_file.read(
        columns=[
            "link_id",
            "lanes",
            "intersection_nodes_200m",
            "hotspot_score"
        ]
    ).to_pandas()

    check(
        "Score Link IDs Unique",
        scores["link_id"].is_unique
    )

    check(
        "Score Missing Values",
        not scores.isna().any().any()
    )

    check(
        "Score Finite Values",
        np.isfinite(
            scores["hotspot_score"]
        ).all()
    )

    check(
        "Score Range",
        scores["hotspot_score"].between(0, 1).all()
    )

    # 기존 코드: scores = scores.set_index("link_id", verify_integrity=True)
    scores = scores.set_index("link_id")

    # 인덱스 중복 여부를 직접 검증
    check(
        "Score Index Unique",
        scores.index.is_unique
    )

    print("Loaded Scores:", len(scores))

    return scores


# ==================================================
# 7. 전국 도로 타일 생성
# ==================================================

def build_tiles(road_file, scores, accident_tree):

    print("\n===== National Tile Generation =====")

    writer = TileWriter(TILE_DIR)

    stats = Counter()

    seen = set()

    multipart_links = []

    columns = [
        "link_id",
        "lanes",
        "intersection_nodes_200m",
        "geometry_wkb"
    ]

    try:

        for batch in road_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=columns
        ):

            frame = batch.to_pandas()

            ids = frame["link_id"].astype(str).tolist()

            if (
                len(set(ids)) != len(ids)
                or seen.intersection(ids)
            ):
                raise ValueError("Duplicate road link ID")

            seen.update(ids)

            # 링크 ID 기준으로 점수 결합
            matched = scores.loc[ids]

            if not np.array_equal(
                frame["lanes"].to_numpy(),
                matched["lanes"].to_numpy()
            ):
                raise ValueError("Lane mismatch")

            if not np.array_equal(
                frame[
                    "intersection_nodes_200m"
                ].to_numpy(),
                matched[
                    "intersection_nodes_200m"
                ].to_numpy()
            ):
                raise ValueError(
                    "Intersection feature mismatch"
                )

            geometries = shapely.from_wkb(
                frame["geometry_wkb"].to_numpy()
            )

            if (
                shapely.is_empty(geometries).any()
                or not shapely.is_valid(geometries).all()
            ):
                raise ValueError("Invalid road geometry")

            # 기존 사고다발지까지의 최단거리
            pairs, distances = (
                accident_tree.query_nearest(
                    geometries,
                    all_matches=False,
                    return_distance=True
                )
            )

            nearest = np.full(
                len(frame),
                np.nan,
                dtype=np.float64
            )

            nearest[pairs[0]] = distances

            if (
                not np.isfinite(nearest).all()
                or np.any(nearest < 0)
            ):
                raise ValueError(
                    "Invalid legacy hotspot distance"
                )

            # 전체 도로 좌표계 변환
            wgs_geometries = shapely.transform(
                geometries,
                TO_WGS.transform,
                interleaved=False
            )

            lanes = frame["lanes"].to_numpy()

            intersections = frame[
                "intersection_nodes_200m"
            ].to_numpy()

            risk = matched[
                "hotspot_score"
            ].to_numpy()

            for i, geometry in enumerate(
                wgs_geometries
            ):

                link_id = ids[i]

                if geometry.geom_type == "LineString":

                    parts = [geometry]

                    stats["linestring_links"] += 1

                elif (
                    geometry.geom_type
                    == "MultiLineString"
                ):

                    parts = list(geometry.geoms)

                    stats["multipart_links"] += 1

                    multipart_links.append({
                        "link_id": link_id,
                        "part_count": len(parts)
                    })

                else:

                    raise ValueError(
                        f"Unsupported geometry: "
                        f"{link_id}"
                    )

                if (
                    not np.isfinite(lanes[i])
                    or lanes[i] < 1
                    or lanes[i] != int(lanes[i])
                ):
                    raise ValueError("Invalid lanes")

                if (
                    intersections[i] < 0
                    or intersections[i]
                    != int(intersections[i])
                ):
                    raise ValueError(
                        "Invalid intersection count"
                    )

                for part_index, part in enumerate(parts):

                    coordinates = (
                        shapely.get_coordinates(part)
                    )

                    if (
                        len(coordinates) < 2
                        or not np.isfinite(
                            coordinates
                        ).all()
                    ):
                        raise ValueError(
                            f"Invalid coordinates: "
                            f"{link_id}"
                        )

                    # 기존 고정 정밀도 방식 (선형 소실 오류 발생)
                    # compact = [
                    # [
                    # round(
                    # float(x),
                    # COORDINATE_DECIMALS
                    # ),
                    # round(
                    # float(y),
                    # COORDINATE_DECIMALS
                    # )
                    # ]
                    # for x, y in coordinates
                    # ]

                    # if not all(
                    # 124 <= x <= 132
                    # and 33 <= y <= 39
                    # for x, y in compact
                    # ):
                    # raise ValueError(
                    # f"Coordinate out of bounds: "
                    # f"{link_id}"
                    # )

                    # if len({
                    # tuple(p) for p in compact
                    # }) < 2:

                    # raise ValueError(
                    # f"Geometry collapsed after "
                    # f"rounding: {link_id}"
                    # )


                    # 기본 6자리, 선형 소실 시에만 7자리 적용
                    used_decimals = COORDINATE_DECIMALS
                    while True:
                        compact = [
                            [round(float(x), used_decimals),
                             round(float(y), used_decimals)]
                            for x, y in coordinates
                        ]
                        if len({tuple(p) for p in compact}) >= 2:
                            break
                        if used_decimals >= MAX_COORDINATE_DECIMALS:
                            raise ValueError(
                                f"Geometry collapsed at maximum precision: {link_id}"
                            )
                        used_decimals += 1

                    if used_decimals > COORDINATE_DECIMALS:
                        stats["precision_adjusted_parts"] += 1
                        print(
                            f"Precision adjusted: link_id={link_id}, "
                            f"part_index={part_index}, decimals={used_decimals}",
                            flush=True
                        )
                    stats[f"coordinate_decimals_{used_decimals}"] += 1

                    if not all(
                        124 <= x <= 132 and 33 <= y <= 39
                        for x, y in compact
                    ):
                        raise ValueError(f"Coordinate out of bounds: {link_id}")

                    # 신규 타일 스키마
                    row = [
                        link_id,
                        compact,
                        float(risk[i]),
                        int(lanes[i]),
                        int(intersections[i]),
                        round(float(nearest[i]), 3),
                        part_index
                    ]

                    writer.write(row)

                    stats["geometry_parts"] += 1

                stats["scored_links"] += 1

            stats["processed_links"] += len(frame)

            if (
                stats["processed_links"] % 50000
                < len(frame)
                or stats["processed_links"]
                == EXPECTED_LINKS
            ):

                print(
                    f"Processed: "
                    f"{stats['processed_links']:,}"
                    f"/{EXPECTED_LINKS:,}",
                    flush=True
                )

        check(
            "All National Links Processed",
            stats["processed_links"] == EXPECTED_LINKS
        )

        check(
            "All National Links Scored",
            stats["scored_links"] == EXPECTED_LINKS
        )

        check(
            "All Score IDs Used",
            len(seen) == len(scores)
            and seen == set(scores.index)
        )

        check(
            "Geometry Part Count",
            stats["geometry_parts"]
            == writer.geometry_parts
        )

        check(
            "Coordinate Precision Counts",
            stats["coordinate_decimals_6"] + stats["coordinate_decimals_7"]
            == writer.geometry_parts
        )
        writer.finish()

    finally:

        writer.close()

    return writer, stats, multipart_links


# ==================================================
# 8. 타일 Manifest 생성
# ==================================================

def build_manifest(
    writer,
    stats,
    multipart_links,
    accident_info,
    hashes
):

    tile_files = sorted(
        TILE_DIR.glob("*.jsonl.gz")
    )

    check(
        "Compressed Tile Count",
        len(tile_files) == len(writer.counts)
    )

    check(
        "No Temporary JSONL Files",
        not list(TILE_DIR.glob("*.jsonl"))
    )

    print("\n===== Tile File Hashes =====")

    tile_metadata = {}

    for index, path in enumerate(
        tile_files, start=1
    ):

        key = path.name.removesuffix(".jsonl.gz")

        tile_metadata[key] = {
            "file": path.name,
            "rows": writer.counts[key],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path)
        }

        if index % 100 == 0 or index == len(tile_files):

            print(
                f"Hashed: "
                f"{index:,}/{len(tile_files):,}"
            )

    total_bytes = sum(
        item["bytes"]
        for item in tile_metadata.values()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "policy_id": POLICY_ID,
        "score_meaning": (
            "historical_hotspot_membership_score"
        ),
        "score_range": [0, 1],
        "score_normalization": "raw",
        "policy_applied_to_tiles": False,

        "coordinate_crs": "EPSG:4326",
        "coordinate_order": "longitude_latitude",
        "source_geometry_crs": "EPSG:5179",
        "coordinate_decimals": COORDINATE_DECIMALS,
        "coordinate_precision_mode": "adaptive",
        "maximum_coordinate_decimals": MAX_COORDINATE_DECIMALS,
        "coordinate_precision_stats": {
            "decimals_6": stats["coordinate_decimals_6"],
            "decimals_7": stats["coordinate_decimals_7"],
            "adjusted_parts": stats["precision_adjusted_parts"]
        },

        "tile_size_degrees": TILE_SIZE,
        "tile_padding_degrees": TILE_PADDING,
        "compression": "gzip",
        "gzip_level": GZIP_LEVEL,

        "tile_row_schema": [
            "link_id",
            "coordinates",
            "hotspot_score",
            "lanes",
            "intersection_nodes_200m",
            "legacy_hotspot_distance_m",
            "part_index"
        ],

        "geometry_identity": [
            "link_id",
            "part_index"
        ],

        "link_stats": {
            "all_links": EXPECTED_LINKS,
            "scored_links": stats["scored_links"],
            "linestring_links": stats[
                "linestring_links"
            ],
            "multipart_links": stats[
                "multipart_links"
            ],
            "scored_lines": stats["geometry_parts"],
            "excluded_links": 0
        },

        "multipart_link_details": multipart_links,

        "tile_stats": {
            "tile_count": len(tile_files),
            "total_rows": writer.total_tile_rows,
            "unique_geometry_parts": (
                writer.geometry_parts
            ),
            "duplicate_tile_rows": (
                writer.total_tile_rows
                - writer.geometry_parts
            ),
            "total_bytes": total_bytes,
            "total_mib": total_bytes / 1024**2
        },

        "accident_data": accident_info,

        "tiles": tile_metadata,

        "source_sha256": hashes,

        "legacy_matcher_compatible": False,
        "independent_validation_completed": False,
        "deployment_status": "staging_not_deployed"
    }


# ==================================================
# 9. 메인 함수
# ==================================================

def main():

    start = time.perf_counter()

    print("===== National Service Tile Builder =====")

    road_file, score_file, accident_files = (
        validate_inputs()
    )

    paths = [
        ROAD_PATH,
        SCORE_PATH,
        POLICY_PATH,
        INTERFACE_PATH,
        REFERENCE_MANIFEST,
        *accident_files
    ]

    before = source_hashes(paths)

    accident_tree, accident_info = (
        load_accident_points(accident_files)
    )

    scores = load_scores(score_file)

    writer, stats, multipart_links = build_tiles(
        road_file,
        scores,
        accident_tree
    )

    manifest = build_manifest(
        writer,
        stats,
        multipart_links,
        accident_info,
        before
    )

    print("\n===== Source Integrity =====")

    after = source_hashes(paths)

    check(
        "All Source Hashes Unchanged",
        before == after
    )

    save_json(
        MANIFEST_PATH,
        manifest
    )

    # 확정된 정책과 인터페이스 후보 복사
    shutil.copy2(
        POLICY_PATH,
        OUTPUT / "national_route_policy_P01.json"
    )

    shutil.copy2(
        INTERFACE_PATH,
        OUTPUT / "national_service_interface_P01.json"
    )

    check(
        "Policy Copy Hash",
        sha256_file(POLICY_PATH)
        == sha256_file(
            OUTPUT / "national_route_policy_P01.json"
        )
    )

    check(
        "Interface Copy Hash",
        sha256_file(INTERFACE_PATH)
        == sha256_file(
            OUTPUT / "national_service_interface_P01.json"
        )
    )

    report = {
        "model_id": MODEL_ID,
        "policy_id": POLICY_ID,
        "schema_version": SCHEMA_VERSION,
        "link_stats": manifest["link_stats"],
        "tile_stats": manifest["tile_stats"],
        "multipart_links": multipart_links,
        "coordinate_precision_stats": manifest["coordinate_precision_stats"],
        "accident_data": accident_info,
        "source_hashes_unchanged": True,
        "elapsed_seconds": (
            time.perf_counter() - start
        ),
        "build_status": "PASS",
        "independent_validation": "pending",
        "deployment_status": "not_deployed"
    }

    save_json(BUILD_REPORT, report)

    print("\n===== Final Build Summary =====")

    print("Model:", MODEL_ID)
    print("Policy:", POLICY_ID)
    print("Schema:", SCHEMA_VERSION)

    print(
        "Processed Links:",
        f"{stats['processed_links']:,}"
    )

    print(
        "Scored Links:",
        f"{stats['scored_links']:,}"
    )

    print(
        "LineString Links:",
        f"{stats['linestring_links']:,}"
    )

    print(
        "MultiLineString Links:",
        stats["multipart_links"]
    )

    print(
        "Geometry Parts:",
        f"{writer.geometry_parts:,}"
    )

    print("Coordinate Precision (6 Decimals):", f"{stats['coordinate_decimals_6']:,}")
    print("Coordinate Precision (7 Decimals):", f"{stats['coordinate_decimals_7']:,}")
    print("Precision Adjusted Parts:", stats["precision_adjusted_parts"])

    print(
        "Tile Count:",
        manifest["tile_stats"]["tile_count"]
    )

    print(
        "Total Tile Rows:",
        f"{writer.total_tile_rows:,}"
    )

    print(
        "Duplicate Tile Rows:",
        f"{manifest['tile_stats']['duplicate_tile_rows']:,}"
    )

    print(
        "Tile Size:",
        f"{manifest['tile_stats']['total_mib']:.4f} MiB"
    )

    print("\nMultipart Link Details:")

    for item in multipart_links:
        print(item)

    print("\n===== Saved Results =====")

    print("Tile Directory:", TILE_DIR)
    print("Manifest:", MANIFEST_PATH)
    print("Build Report:", BUILD_REPORT)

    print("\n===== Processing Completed =====")

    print(
        f"Elapsed Time: "
        f"{time.perf_counter() - start:.2f}s"
    )

    print("Build Status: PASS")
    print("Independent Tile Validation: Pending")
    print("Additional Model Training: Not Performed")
    print("Original Service: Not Modified")
    print("Reference Service: Not Modified")
    print("Deployment: Not Performed")


if __name__ == "__main__":

    main()