from pathlib import Path
from collections import Counter
import json
import gzip
import time
import gc
import importlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

from pyproj import Transformer
from shapely.ops import transform


# ============================================================
# 1. 경로 및 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_ROOT = (
    PROJECT_ROOT.parent / "SafeRoute_새디자인_20261124"
)

TILE_DIR = REFERENCE_ROOT / "data" / "tiles"

MANIFEST_PATH = (
    REFERENCE_ROOT / "data" / "national_manifest.json"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

RESULTS_DIR = PROCESSED_DIR / "results"

ROAD_PATH = (
    PROCESSED_DIR / "national_road_features.parquet"
)

SCORE_PATH = (
    PROCESSED_DIR / "national_link_risk_scores.parquet"
)

SUMMARY_PATH = (
    RESULTS_DIR / "national_service_compatibility.json"
)

DIFFERENCE_PATH = (
    RESULTS_DIR / "national_service_feature_differences.csv"
)

NEW_LINKS_PATH = (
    RESULTS_DIR / "national_service_additional_links.parquet"
)

MISSING_LINKS_PATH = (
    RESULTS_DIR / "national_service_missing_links.csv"
)

COORDINATE_PATH = (
    RESULTS_DIR / "national_service_coordinate_samples.csv"
)

EXPECTED_NEW_LINKS = 1562356
EXPECTED_OLD_LINKS = 1542745
EXPECTED_TILES = 1147

BATCH_SIZE = 50000

baseline_utils = importlib.import_module(
    "16_evaluate_national_baselines"
)

training_utils = importlib.import_module(
    "17_train_national_lightgbm"
)


# ============================================================
# 2. 원본 파일 해시
# ============================================================

def get_source_hashes(tile_paths):

    paths = [
        ROAD_PATH,
        SCORE_PATH,
        MANIFEST_PATH,
        *tile_paths
    ]

    return {
        str(path): baseline_utils.sha256_file(path)
        for path in paths
    }


# ============================================================
# 3. 기존 전국 서비스 확인
# ============================================================

def inspect_reference_service():

    assert REFERENCE_ROOT.is_dir()
    assert TILE_DIR.is_dir()
    assert MANIFEST_PATH.is_file()

    tile_paths = sorted(
        TILE_DIR.rglob("*.gz")
    )

    assert len(tile_paths) == EXPECTED_TILES

    with MANIFEST_PATH.open(
        "r", encoding="utf-8-sig"
    ) as file:

        manifest = json.load(file)

    print("\n===== Reference National Service =====")

    print("Reference Root:", REFERENCE_ROOT)
    print("Tile Directory:", TILE_DIR)
    print("Tile Count:", len(tile_paths))

    print("\nManifest:")

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=4
        )
    )

    print("\nReference Structure: PASS")

    return tile_paths


# ============================================================
# 4. 기존 타일 데이터 읽기
# ============================================================

def read_tile_rows(path):

    with gzip.open(
        path, "rt", encoding="utf-8"
    ) as file:

        content = file.read().strip()

    if not content:
        return

    # JSON 배열 또는 JSON Lines 구조 지원
    try:

        parsed = json.loads(content)

    except json.JSONDecodeError:

        for line in content.splitlines():

            if line.strip():

                yield json.loads(line)

        return

    if isinstance(parsed, list):

        # 개별 링크 한 행
        if parsed and not isinstance(parsed[0], list):

            yield parsed

        # 여러 링크를 포함한 배열
        else:

            yield from parsed

    else:

        raise ValueError(
            f"Unsupported tile structure: {path}"
        )


# ============================================================
# 5. 기존 전국 링크 수집 및 검증
# ============================================================

def load_reference_links(tile_paths):

    # link_id: (score, lanes, intersections, distance)
    links = {}

    counters = Counter()

    print("\n===== Reading Reference Tiles =====")

    for index, path in enumerate(
        tile_paths, start=1
    ):

        for row in read_tile_rows(path):

            counters["total_tile_rows"] += 1

            # 기존 타일 구조
            # [
            #   link_id,
            #   coordinates,
            #   burden_score,
            #   lanes,
            #   intersection_count,
            #   nearest_hotspot_distance
            # ]

            if not isinstance(row, list) or len(row) != 6:

                raise ValueError(
                    f"Unexpected tile row: {path}"
                )

            (
                link_id,
                coordinates,
                score,
                lanes,
                intersections,
                distance
            ) = row

            link_id = str(link_id)

            score = float(score)
            lanes = float(lanes)
            intersections = float(intersections)
            distance = float(distance)

            if not np.isfinite([
                score,
                lanes,
                intersections,
                distance
            ]).all():

                counters["invalid_numeric_rows"] += 1

            if not 0 <= score <= 100:

                counters["invalid_score_rows"] += 1

            if (
                lanes < 1
                or not lanes.is_integer()
                or intersections < 0
                or not intersections.is_integer()
                or distance < 0
            ):

                counters["invalid_feature_rows"] += 1

            coords = np.asarray(
                coordinates,
                dtype=np.float64
            )

            if (
                coords.ndim != 2
                or coords.shape[1] != 2
                or len(coords) < 2
                or not np.isfinite(coords).all()
            ):

                counters["invalid_geometry_rows"] += 1

            else:

                valid_bounds = (
                    (coords[:, 0] >= 124)
                    & (coords[:, 0] <= 132)
                    & (coords[:, 1] >= 33)
                    & (coords[:, 1] <= 39)
                )

                if not valid_bounds.all():

                    counters[
                        "coordinate_outside_check_bounds"
                    ] += 1

            attributes = (
                score,
                lanes,
                intersections,
                distance
            )

            if link_id in links:

                counters["duplicate_tile_rows"] += 1

                if links[link_id] != attributes:

                    counters[
                        "duplicate_attribute_conflicts"
                    ] += 1

            else:

                links[link_id] = attributes

        if index % 100 == 0 or index == len(tile_paths):

            print(
                f"Tiles: {index:,} / "
                f"{len(tile_paths):,} | "
                f"Unique Links: {len(links):,}",
                flush=True
            )

    counters["unique_links"] = len(links)

    print("\n===== Reference Tile Summary =====")

    keys = [
        "total_tile_rows",
        "unique_links",
        "duplicate_tile_rows",
        "duplicate_attribute_conflicts",
        "invalid_numeric_rows",
        "invalid_score_rows",
        "invalid_feature_rows",
        "invalid_geometry_rows",
        "coordinate_outside_check_bounds"
    ]

    for key in keys:

        print(f"{key}: {counters[key]}")

    assert len(links) == EXPECTED_OLD_LINKS

    assert (
        counters["total_tile_rows"]
        == len(links) + counters["duplicate_tile_rows"]
    )

    for key in [
        "duplicate_attribute_conflicts",
        "invalid_numeric_rows",
        "invalid_score_rows",
        "invalid_feature_rows",
        "invalid_geometry_rows"
    ]:

        assert counters[key] == 0, key

    return links, counters


# ============================================================
# 6. 신규 데이터 스키마 검증
# ============================================================

def validate_new_schema():

    road = pq.ParquetFile(ROAD_PATH)

    scores = pq.ParquetFile(SCORE_PATH)

    assert road.metadata.num_rows == EXPECTED_NEW_LINKS

    assert scores.metadata.num_rows == EXPECTED_NEW_LINKS

    required_road = {
        "link_id",
        "lanes",
        "intersection_nodes_200m",
        "geometry_wkb"
    }

    required_scores = {
        "link_id",
        "lanes",
        "intersection_nodes_200m",
        "hotspot_score",
        "hotspot_predicted_label"
    }

    assert required_road.issubset(
        road.schema_arrow.names
    )

    assert required_scores.issubset(
        scores.schema_arrow.names
    )

    road_metadata = road.schema_arrow.metadata

    score_metadata = scores.schema_arrow.metadata

    assert road_metadata[b"geometry_crs"] == b"EPSG:5179"

    assert road_metadata[b"geometry_encoding"] == b"WKB"

    assert score_metadata[b"model_id"] == b"T09"

    assert score_metadata[b"score_range"] == b"0_to_1"

    assert (
        score_metadata[b"score_meaning"]
        == b"historical_hotspot_membership_score"
    )

    print("\n===== New Dataset Schema =====")

    print("Road Rows:", road.metadata.num_rows)
    print("Score Rows:", scores.metadata.num_rows)
    print("Geometry CRS: EPSG:5179")
    print("Model: T09")
    print("Score Range: 0 to 1")
    print("Schema Validation: PASS")


# ============================================================
# 7. 신규·기존 링크 비교
# ============================================================

def compare_links(reference_links):

    counters = Counter()

    differences = []

    additional = []

    seen_ids = set()

    score_file = pq.ParquetFile(SCORE_PATH)

    print("\n===== National Link Comparison =====")

    for batch_index, batch in enumerate(
        score_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[
                "link_id",
                "lanes",
                "intersection_nodes_200m",
                "hotspot_score"
            ]
        ),
        start=1
    ):

        frame = batch.to_pandas()

        assert frame.notna().all().all()

        numeric = frame[
            [
                "lanes",
                "intersection_nodes_200m",
                "hotspot_score"
            ]
        ].to_numpy()

        assert np.isfinite(numeric).all()

        assert frame["hotspot_score"].between(
            0, 1
        ).all()

        assert (frame["lanes"] >= 1).all()

        assert (
            frame["intersection_nodes_200m"] >= 0
        ).all()

        for row in frame.itertuples(index=False):

            link_id = str(row.link_id)

            assert link_id not in seen_ids

            seen_ids.add(link_id)

            counters["new_total_links"] += 1

            old = reference_links.get(link_id)

            if old is None:

                counters["additional_new_links"] += 1

                additional.append({

                    "link_id": link_id,

                    "lanes": row.lanes,

                    "intersection_nodes_200m":
                        row.intersection_nodes_200m,

                    "hotspot_score": row.hotspot_score

                })

                continue

            counters["common_links"] += 1

            old_score, old_lanes, old_intersections, _ = old

            lanes_match = (
                old_lanes == row.lanes
            )

            intersections_match = (
                old_intersections
                == row.intersection_nodes_200m
            )

            if not lanes_match:

                counters["lane_differences"] += 1

            if not intersections_match:

                counters["intersection_differences"] += 1

            if lanes_match and intersections_match:

                counters["both_features_match"] += 1

            else:

                differences.append({

                    "link_id": link_id,

                    "old_lanes": old_lanes,

                    "new_lanes": row.lanes,

                    "lane_difference":
                        row.lanes - old_lanes,

                    "old_intersections":
                        old_intersections,

                    "new_intersections":
                        row.intersection_nodes_200m,

                    "intersection_difference":
                        row.intersection_nodes_200m
                        - old_intersections,

                    "old_burden_score": old_score,

                    "new_hotspot_score":
                        row.hotspot_score

                })

        print(
            f"Batch {batch_index:02d} | "
            f"Processed: "
            f"{counters['new_total_links']:,} / "
            f"{EXPECTED_NEW_LINKS:,}",
            flush=True
        )

    missing_ids = sorted(
        set(reference_links) - seen_ids
    )

    counters["missing_old_links"] = len(missing_ids)

    assert len(seen_ids) == EXPECTED_NEW_LINKS

    assert (
        counters["common_links"]
        + counters["additional_new_links"]
        == EXPECTED_NEW_LINKS
    )

    assert (
        counters["common_links"]
        + counters["missing_old_links"]
        == EXPECTED_OLD_LINKS
    )

    difference_columns = [
        "link_id",
        "old_lanes",
        "new_lanes",
        "lane_difference",
        "old_intersections",
        "new_intersections",
        "intersection_difference",
        "old_burden_score",
        "new_hotspot_score"
    ]

    difference_frame = pd.DataFrame(
        differences,
        columns=difference_columns
    )

    additional_frame = pd.DataFrame(
        additional,
        columns=[
            "link_id",
            "lanes",
            "intersection_nodes_200m",
            "hotspot_score"
        ]
    )

    missing_frame = pd.DataFrame({
        "link_id": missing_ids
    })

    print("\n===== Link Comparison Summary =====")

    for key in [
        "new_total_links",
        "common_links",
        "additional_new_links",
        "missing_old_links",
        "lane_differences",
        "intersection_differences",
        "both_features_match"
    ]:

        print(f"{key}: {counters[key]}")

    common = counters["common_links"]

    assert common > 0

    counters["old_link_coverage"] = (
        common / EXPECTED_OLD_LINKS
    )

    counters["lane_match_ratio"] = (
        1 - counters["lane_differences"] / common
    )

    counters["intersection_match_ratio"] = (
        1 - counters["intersection_differences"] / common
    )

    print(
        "Old Link Coverage:",
        counters["old_link_coverage"]
    )

    print(
        "Lane Match Ratio:",
        counters["lane_match_ratio"]
    )

    print(
        "Intersection Match Ratio:",
        counters["intersection_match_ratio"]
    )

    if not difference_frame.empty:

        print("\n===== Feature Difference Statistics =====")

        print(
            difference_frame[
                [
                    "lane_difference",
                    "intersection_difference"
                ]
            ].describe().to_string()
        )

    return (
        counters,
        difference_frame,
        additional_frame,
        missing_frame
    )


# ============================================================
# 8. 전국 도로 좌표계 변환 표본 검증
# ============================================================

def validate_coordinate_transform():

    road = pq.ParquetFile(ROAD_PATH)

    transformer = Transformer.from_crs(
        "EPSG:5179",
        "EPSG:4326",
        always_xy=True
    )

    records = []

    # 전국 파일에 걸쳐 최대 20개 Row Group 선택
    row_groups = np.unique(
        np.linspace(
            0,
            road.num_row_groups - 1,
            num=20,
            dtype=int
        )
    )

    print("\n===== Coordinate Transformation =====")

    for row_group in row_groups:

        frame = road.read_row_group(
            int(row_group),
            columns=[
                "link_id",
                "geometry_wkb"
            ]
        ).to_pandas()

        sample = frame.sample(
            n=min(10, len(frame)),
            random_state=42
        )

        for row in sample.itertuples(index=False):

            geometry = shapely.from_wkb(
                row.geometry_wkb
            )

            assert not geometry.is_empty

            assert geometry.is_valid

            converted = transform(
                transformer.transform,
                geometry
            )

            assert not converted.is_empty

            assert converted.is_valid

            coordinates = shapely.get_coordinates(
                converted
            )

            assert len(coordinates) >= 2

            assert np.isfinite(coordinates).all()

            min_lon, min_lat, max_lon, max_lat = (
                converted.bounds
            )

            within_check_bounds = (
                min_lon >= 124
                and max_lon <= 132
                and min_lat >= 33
                and max_lat <= 39
            )

            records.append({

                "link_id": row.link_id,

                "row_group": int(row_group),

                "geometry_type":
                    converted.geom_type,

                "coordinate_count":
                    len(coordinates),

                "min_lon": min_lon,

                "min_lat": min_lat,

                "max_lon": max_lon,

                "max_lat": max_lat,

                "within_check_bounds":
                    within_check_bounds

            })

    result = pd.DataFrame(records)

    print("Sample Row Groups:", len(row_groups))

    print("Sample Links:", len(result))

    print(
        "Samples Outside Check Bounds:",
        int((~result["within_check_bounds"]).sum())
    )

    print(
        "Geometry Types:",
        result["geometry_type"].value_counts().to_dict()
    )

    print(
        "Longitude Range:",
        result["min_lon"].min(),
        "~",
        result["max_lon"].max()
    )

    print(
        "Latitude Range:",
        result["min_lat"].min(),
        "~",
        result["max_lat"].max()
    )

    print("Sample Coordinate Transformation: PASS")

    return result


# ============================================================
# 9. 메인 실행
# ============================================================

def main():

    start = time.perf_counter()

    print(
        "===== National Service Compatibility Validation ====="
    )

    training_utils.print_memory(
        "Initial Memory"
    )

    output_paths = [
        SUMMARY_PATH,
        DIFFERENCE_PATH,
        NEW_LINKS_PATH,
        MISSING_LINKS_PATH,
        COORDINATE_PATH
    ]

    for path in output_paths:

        if path.exists():

            raise FileExistsError(path)

    tile_paths = inspect_reference_service()

    print("\nCalculating Source Hashes...")

    source_hashes = get_source_hashes(
        tile_paths
    )

    validate_new_schema()

    reference_links, tile_summary = (
        load_reference_links(tile_paths)
    )

    training_utils.print_memory(
        "After Reference Tile Loading"
    )

    (
        comparison,
        differences,
        additional,
        missing
    ) = compare_links(reference_links)

    del reference_links

    gc.collect()

    coordinates = validate_coordinate_transform()

    print("\n===== Source Integrity Verification =====")

    assert get_source_hashes(
        tile_paths
    ) == source_hashes

    print("Source Hash Verification: PASS")

    all_old_links_available = (
        comparison["missing_old_links"] == 0
    )

    all_lanes_match = (
        comparison["lane_differences"] == 0
    )

    all_intersections_match = (
        comparison["intersection_differences"] == 0
    )

    sample_coordinates_valid = bool(
        coordinates["within_check_bounds"].all()
    )

    summary = {

        "model_id": "T09",

        "reference_project":
            REFERENCE_ROOT.name,

        "reference_tile_count":
            len(tile_paths),

        "reference_tile_summary":
            dict(tile_summary),

        "link_comparison":
            dict(comparison),

        "all_old_links_available":
            all_old_links_available,

        "all_lanes_match":
            all_lanes_match,

        "all_intersections_match":
            all_intersections_match,

        "coordinate_sample_count":
            len(coordinates),

        "sample_coordinates_within_check_bounds":
            sample_coordinates_valid,

        "source_hash_verification":
            "PASS",

        "road_geometry_crs":
            "EPSG:5179",

        "service_geometry_crs":
            "EPSG:4326",

        "new_score_meaning":
            "historical_hotspot_membership_score",

        "new_score_range":
            [0, 1],

        "new_tiles_generated":
            False,

        "service_deployment":
            "not_modified",

        "limitations": [

            "Coordinate transformation checked "
            "on samples only.",

            "Old and new model scores have "
            "different meanings.",

            "Reference tile geometry equality "
            "was not evaluated.",

            "Duplicate tile attributes checked "
            "without comparing geometry.",

            "Live route matching not evaluated."

        ]

    }

    differences.to_csv(
        DIFFERENCE_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    additional.to_parquet(
        NEW_LINKS_PATH,
        index=False
    )

    missing.to_csv(
        MISSING_LINKS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    coordinates.to_csv(
        COORDINATE_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    with SUMMARY_PATH.open(
        "x", encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n===== Saved Results =====")

    for path in output_paths:

        print("Path:", path)

        print(
            "File Size:",
            f"{path.stat().st_size / 1024**2:.4f} MiB"
        )

    training_utils.print_memory(
        "Final Memory"
    )

    print("\n===== Processing Completed =====")

    print(
        "Elapsed Time:",
        f"{time.perf_counter() - start:.2f}s"
    )

    print("Reference Tiles:", len(tile_paths))

    print(
        "Common Links:",
        comparison["common_links"]
    )

    print(
        "Additional New Links:",
        comparison["additional_new_links"]
    )

    print(
        "Missing Old Links:",
        comparison["missing_old_links"]
    )

    print(
        "Lane Differences:",
        comparison["lane_differences"]
    )

    print(
        "Intersection Differences:",
        comparison["intersection_differences"]
    )

    print(
        "All Old Links Available:",
        all_old_links_available
    )

    print("All Lanes Match:", all_lanes_match)

    print(
        "All Intersections Match:",
        all_intersections_match
    )

    print(
        "Coordinate Sample Bounds:",
        "PASS" if sample_coordinates_valid else "REVIEW"
    )

    print("Source Hash Verification: PASS")

    print("New Tile Generation: Not Performed")

    print("Additional Model Training: Not Performed")

    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()