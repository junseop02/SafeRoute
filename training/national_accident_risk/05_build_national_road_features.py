from pathlib import Path
from collections import Counter
import time

import numpy as np
import pandas as pd
import shapefile
import shapely
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from shapely import STRtree
from pyproj import CRS, Transformer


# ============================================================
# 1. 경로 및 환경 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "legacy"
    / "raw"
    / "nodelink"
    / "[2026-09-14]NODELINKDATA"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

LINK_PATH = RAW_DIR / "MOCT_LINK.shp"

NODE_PATH = (
    PROCESSED_DIR / "national_node_features.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "national_road_features.parquet"
)

TEMP_PATH = OUTPUT_PATH.with_suffix(".tmp")

TARGET_CRS = "EPSG:5179"
BATCH_SIZE = 5000
INTERSECTION_RADIUS = 200

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 메모리 확인
# ============================================================

def print_memory(title):

    memory = psutil.virtual_memory()

    print(f"\n===== {title} =====")
    print(
        f"Available RAM: "
        f"{memory.available / 1024**3:.2f} GiB"
    )
    print(f"RAM Usage: {memory.percent}%")


# ============================================================
# 3. 전국 NODE 공간 인덱스 생성
# ============================================================

def prepare_node_indexes():

    nodes = pd.read_parquet(NODE_PATH)

    print("\n===== NODE Information =====")
    print("Total Nodes:", len(nodes))
    print("Duplicate IDs:", nodes["node_id"].duplicated().sum())

    assert not nodes["node_id"].duplicated().any()

    degree_nodes = nodes[
        nodes["is_degree_intersection"]
    ]

    type101_nodes = nodes[
        nodes["is_type101_intersection"]
    ]

    # 기존 방식의 교차로 후보
    degree_points = shapely.points(
        degree_nodes["x"].to_numpy(),
        degree_nodes["y"].to_numpy()
    )

    # 최신 방식의 교차로 후보
    type101_points = shapely.points(
        type101_nodes["x"].to_numpy(),
        type101_nodes["y"].to_numpy()
    )

    degree_tree = STRtree(degree_points)
    type101_tree = STRtree(type101_points)

    # 도착 노드 기준 분기 수 계산용
    outgoing_map = dict(
        zip(
            nodes["node_id"],
            nodes["outgoing_count"]
        )
    )

    print("Degree Intersections:", len(degree_points))
    print("Type 101 Intersections:", len(type101_points))
    print("Outgoing Map:", len(outgoing_map))

    assert len(degree_points) == 512346
    assert len(type101_points) == 888517

    return degree_tree, type101_tree, outgoing_map


# ============================================================
# 4. LINK Geometry 변환
# ============================================================

def create_geometry(shape):

    if shape.shapeType != shapefile.POLYLINE:
        raise ValueError("Unexpected geometry type.")

    coords = np.asarray(
        shape.points,
        dtype=np.float64
    )

    if len(coords) < 2:
        raise ValueError("Insufficient coordinates.")

    if not np.isfinite(coords).all():
        raise ValueError("Non-finite coordinates.")

    boundaries = list(shape.parts) + [len(coords)]

    pieces = []

    for i in range(len(boundaries) - 1):

        start = boundaries[i]
        end = boundaries[i + 1]

        part = coords[start:end]

        if len(part) < 2:
            raise ValueError("Invalid geometry part.")

        pieces.append(
            shapely.linestrings(part)
        )

    if len(pieces) == 1:
        return pieces[0]

    return shapely.multilinestrings(pieces)


# ============================================================
# 5. 숫자형 속성 변환
# ============================================================

def to_number(value):

    try:
        number = float(value)

        if np.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return np.nan


# ============================================================
# 6. Parquet 저장 스키마
# ============================================================

SCHEMA = pa.schema(
    [
        ("link_id", pa.string()),
        ("f_node", pa.string()),
        ("t_node", pa.string()),
        ("lanes", pa.float64()),
        ("max_spd", pa.float64()),
        ("road_rank", pa.float64()),
        ("road_type", pa.string()),
        ("length", pa.float64()),
        ("geometry_length_m", pa.float64()),
        ("n_branch", pa.int32()),
        ("n_intersect_200m", pa.int32()),
        ("intersection_nodes_200m", pa.int32()),
        ("center_x", pa.float64()),
        ("center_y", pa.float64()),
        ("geometry_wkb", pa.binary())
    ],
    metadata={
        b"geometry_crs": b"EPSG:5179",
        b"geometry_encoding": b"WKB"
    }
)


# ============================================================
# 7. 분할 도로 피처 계산
# ============================================================

def process_batch(
    batch,
    writer,
    transformer,
    degree_tree,
    type101_tree,
    outgoing_map,
    stats
):

    if not batch:
        return

    attributes = [item[0] for item in batch]

    native_geometries = np.asarray(
        [item[1] for item in batch],
        dtype=object
    )

    # 원본 좌표계를 EPSG:5179로 변환
    geometries = shapely.transform(
        native_geometries,
        transformer.transform,
        interleaved=False
    )

    geometry_lengths = shapely.length(geometries)

    valid_geometry = (
        shapely.is_valid(geometries)
        & ~shapely.is_empty(geometries)
        & np.isfinite(geometry_lengths)
        & (geometry_lengths > 0)
    )

    if not valid_geometry.all():

        invalid_ids = [
            attributes[i]["link_id"]
            for i in np.flatnonzero(~valid_geometry)
        ]

        raise RuntimeError(
            f"Invalid geometries: {invalid_ids[:10]}"
        )

    # 도로 전체 길이의 중간 지점
    centers = []

    for geometry in geometries:

        if geometry.geom_type == "LineString":

            center = shapely.line_interpolate_point(
                geometry,
                0.5,
                normalized=True
            )

        else:

            # MultiLineString 전체 길이 기준 중심점
            parts = list(geometry.geoms)

            remaining = geometry.length / 2

            center = None

            for part in parts:

                if remaining <= part.length:

                    center = part.interpolate(
                        remaining
                    )

                    break

                remaining -= part.length

            if center is None:
                center = parts[-1].interpolate(1, normalized=True)

        centers.append(center)

    centers = np.asarray(centers, dtype=object)

    # --------------------------------------------------------
    # 기존 교차로 피처
    # 중심점 Buffer 내부의 degree >= 3 노드 수
    # --------------------------------------------------------

    buffers = shapely.buffer(
        centers,
        INTERSECTION_RADIUS,
        quad_segs=16
    )

    degree_pairs = degree_tree.query(
        buffers,
        predicate="contains"
    )

    degree_counts = np.bincount(
        degree_pairs[0],
        minlength=len(batch)
    )

    # --------------------------------------------------------
    # 최신 교차로 피처
    # 도로 전체 200m 이내의 NODE_TYPE=101 노드 수
    # --------------------------------------------------------

    type101_pairs = type101_tree.query(
        geometries,
        predicate="dwithin",
        distance=INTERSECTION_RADIUS
    )

    type101_counts = np.bincount(
        type101_pairs[0],
        minlength=len(batch)
    )

    # --------------------------------------------------------
    # 기본 속성 및 분기 수 결합
    # --------------------------------------------------------

    frame = pd.DataFrame(attributes)

    frame["geometry_length_m"] = geometry_lengths

    frame["n_branch"] = (
        frame["t_node"]
        .map(outgoing_map)
    )

    if frame["n_branch"].isna().any():
        raise RuntimeError("Missing endpoint NODE.")

    frame["n_branch"] = (
        frame["n_branch"].astype("int32")
    )

    frame["n_intersect_200m"] = (
        degree_counts.astype("int32")
    )

    frame["intersection_nodes_200m"] = (
        type101_counts.astype("int32")
    )

    frame["center_x"] = shapely.get_x(centers)
    frame["center_y"] = shapely.get_y(centers)

    # Geometry를 WKB 바이너리로 저장
    frame["geometry_wkb"] = shapely.to_wkb(
        geometries
    )

    if not np.isfinite(
        frame[["center_x", "center_y"]].to_numpy()
    ).all():
        raise RuntimeError("Invalid center coordinates.")

    # 속성 이상값은 삭제하지 않고 집계
    for column in [
        "lanes",
        "max_spd",
        "road_rank",
        "length"
    ]:

        stats[f"missing_{column}"] += int(
            frame[column].isna().sum()
        )

    stats["invalid_lanes"] += int(
        (frame["lanes"] <= 0).sum()
    )

    stats["invalid_max_spd"] += int(
        (frame["max_spd"] <= 0).sum()
    )

    stats["invalid_length"] += int(
        (frame["length"] <= 0).sum()
    )

    stats["nonstandard_road_rank"] += int(
        (
            frame["road_rank"].notna()
            & ~frame["road_rank"].isin(
                range(101, 109)
            )
        ).sum()
    )

    stats["different_intersection_counts"] += int(
        (degree_counts != type101_counts).sum()
    )

    stats["degree_intersection_sum"] += int(
        degree_counts.sum()
    )

    stats["type101_intersection_sum"] += int(
        type101_counts.sum()
    )

    stats["saved_links"] += len(frame)

    # 지정된 스키마 순서대로 저장
    frame = frame[SCHEMA.names]

    table = pa.Table.from_pandas(
        frame,
        schema=SCHEMA,
        preserve_index=False
    )

    writer.write_table(table)


# ============================================================
# 8. 전국 도로 피처 생성
# ============================================================

def main():

    start_time = time.perf_counter()

    print("===== National Road Feature Builder =====")
    print("LINK:", LINK_PATH)
    print("NODE:", NODE_PATH)
    print("Batch Size:", BATCH_SIZE)

    print_memory("Initial Memory")

    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)

    if TEMP_PATH.exists():
        raise FileExistsError(TEMP_PATH)

    # 좌표 변환기
    source_crs = CRS.from_wkt(
        LINK_PATH.with_suffix(".prj").read_text(
            encoding="utf-8"
        )
    )

    transformer = Transformer.from_crs(
        source_crs,
        TARGET_CRS,
        always_xy=True
    )

    print("\n===== Prepare Spatial Indexes =====")

    (
        degree_tree,
        type101_tree,
        outgoing_map
    ) = prepare_node_indexes()

    print_memory("After Spatial Indexes")

    stats = Counter()
    seen_ids = set()
    batch = []

    fields = [
        "LINK_ID",
        "F_NODE",
        "T_NODE",
        "LANES",
        "MAX_SPD",
        "ROAD_RANK",
        "ROAD_TYPE",
        "LENGTH"
    ]

    print("\n===== Build Road Features =====")

    with pq.ParquetWriter(
        TEMP_PATH,
        SCHEMA,
        compression="zstd"
    ) as writer:

        with shapefile.Reader(
            str(LINK_PATH),
            encoding="cp949"
        ) as reader:

            expected_links = len(reader)

            for item in reader.iterShapeRecords(
                fields=fields
            ):

                stats["total_links"] += 1

                record = item.record

                link_id = str(
                    record["LINK_ID"]
                ).strip()

                f_node = str(
                    record["F_NODE"]
                ).strip()

                t_node = str(
                    record["T_NODE"]
                ).strip()

                if not link_id:
                    raise RuntimeError("Empty LINK_ID.")

                if link_id in seen_ids:
                    raise RuntimeError(
                        f"Duplicate LINK_ID: {link_id}"
                    )

                seen_ids.add(link_id)

                if (
                    f_node not in outgoing_map
                    or t_node not in outgoing_map
                ):
                    raise RuntimeError(
                        f"Missing NODE: {link_id}"
                    )

                geometry = create_geometry(
                    item.shape
                )

                stats[geometry.geom_type] += 1

                attributes = {
                    "link_id": link_id,
                    "f_node": f_node,
                    "t_node": t_node,
                    "lanes": to_number(record["LANES"]),
                    "max_spd": to_number(record["MAX_SPD"]),
                    "road_rank": to_number(record["ROAD_RANK"]),
                    "road_type": str(record["ROAD_TYPE"]),
                    "length": to_number(record["LENGTH"])
                }

                batch.append(
                    (attributes, geometry)
                )

                if len(batch) >= BATCH_SIZE:

                    process_batch(
                        batch,
                        writer,
                        transformer,
                        degree_tree,
                        type101_tree,
                        outgoing_map,
                        stats
                    )

                    batch.clear()

                if stats["total_links"] % 50000 == 0:

                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    memory = psutil.virtual_memory()

                    print(
                        f"Processed: "
                        f"{stats['total_links']:,} / "
                        f"{expected_links:,} | "
                        f"Elapsed: {elapsed:.1f}s | "
                        f"RAM Available: "
                        f"{memory.available / 1024**3:.2f} GiB",
                        flush=True
                    )

            if batch:

                process_batch(
                    batch,
                    writer,
                    transformer,
                    degree_tree,
                    type101_tree,
                    outgoing_map,
                    stats
                )

                batch.clear()

    # ========================================================
    # 9. 최종 검증
    # ========================================================

    print("\n===== Final Statistics =====")

    for key, value in sorted(stats.items()):
        print(f"{key}: {value:,}")

    assert stats["total_links"] == expected_links
    assert stats["saved_links"] == expected_links
    assert len(seen_ids) == expected_links

    parquet = pq.ParquetFile(TEMP_PATH)

    assert (
        parquet.metadata.num_rows
        == expected_links
    )

    assert parquet.schema_arrow.names == SCHEMA.names

    del parquet

    TEMP_PATH.replace(OUTPUT_PATH)

    # ========================================================
    # 10. 결과 출력
    # ========================================================

    print("\n===== Intersection Comparison =====")

    print(
        "Mean Degree Intersection Count:",
        stats["degree_intersection_sum"]
        / expected_links
    )

    print(
        "Mean Type101 Intersection Count:",
        stats["type101_intersection_sum"]
        / expected_links
    )

    print(
        "Different Counts:",
        stats["different_intersection_counts"]
    )

    print(
        "Different Count Ratio:",
        f"{stats['different_intersection_counts'] / expected_links * 100:.4f}%"
    )

    print("\n===== Saved Output =====")

    print("Path:", OUTPUT_PATH)
    print("Rows:", expected_links)
    print("Columns:", len(SCHEMA))
    print("CRS:", TARGET_CRS)

    print(
        f"File Size: "
        f"{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB"
    )

    print_memory("Final Memory")

    print("\n===== Processing Completed =====")

    print(
        f"Elapsed Time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    print("Original Files: Not Modified")
    print("Model Training: Not Started")
    print("Service Deployment: Not Modified")


if __name__ == "__main__":
    main()