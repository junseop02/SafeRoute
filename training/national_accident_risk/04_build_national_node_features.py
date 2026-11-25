from pathlib import Path
from collections import Counter
import time

import numpy as np
import pandas as pd
import shapefile
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "national_accident_risk"
    / "processed"
)

LINK_PATH = RAW_DIR / "MOCT_LINK.shp"
NODE_PATH = RAW_DIR / "MOCT_NODE.shp"

OUTPUT_PATH = (
    OUTPUT_DIR / "national_node_features.parquet"
)

TEMP_PATH = OUTPUT_PATH.with_suffix(".tmp")

BATCH_SIZE = 50000
TARGET_CRS = "EPSG:5179"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 메모리 확인 함수
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
# 3. 전국 LINK 연결 구조 계산
# ============================================================

def build_node_counts():

    degree = Counter()
    outgoing = Counter()

    total_links = 0
    invalid_links = 0

    with shapefile.Reader(
        str(LINK_PATH),
        encoding="cp949"
    ) as reader:

        expected_links = len(reader)

        for record in reader.iterRecords(
            fields=["F_NODE", "T_NODE"]
        ):

            f_node = str(record["F_NODE"]).strip()
            t_node = str(record["T_NODE"]).strip()

            total_links += 1

            if not f_node or not t_node:
                invalid_links += 1
                continue

            # 시작 및 도착 노드 연결 차수
            degree[f_node] += 1
            degree[t_node] += 1

            # 시작 노드에서 출발하는 링크 수
            outgoing[f_node] += 1

            if total_links % 200000 == 0:
                print(
                    f"LINK Processed: "
                    f"{total_links:,} / "
                    f"{expected_links:,}",
                    flush=True
                )

    if total_links != expected_links:
        raise RuntimeError("LINK count mismatch.")

    print("\n===== LINK Topology Summary =====")
    print("Total Links:", total_links)
    print("Invalid Links:", invalid_links)
    print("Connected Nodes:", len(degree))
    print("Outgoing Nodes:", len(outgoing))
    print("Degree Sum:", sum(degree.values()))
    print("Outgoing Sum:", sum(outgoing.values()))

    valid_links = total_links - invalid_links

    assert sum(degree.values()) == valid_links * 2
    assert sum(outgoing.values()) == valid_links

    if invalid_links:
        raise RuntimeError(
            "Invalid LINK endpoints detected."
        )

    return degree, outgoing


# ============================================================
# 4. 전국 NODE 피처 생성
# ============================================================

def build_node_features(degree, outgoing):

    source_crs = CRS.from_wkt(
        NODE_PATH.with_suffix(".prj").read_text(
            encoding="utf-8"
        )
    )

    transformer = Transformer.from_crs(
        source_crs,
        TARGET_CRS,
        always_xy=True
    )

    schema = pa.schema([
        ("node_id", pa.string()),
        ("node_type", pa.string()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("degree", pa.int32()),
        ("outgoing_count", pa.int32()),
        ("is_degree_intersection", pa.bool_()),
        ("is_type101_intersection", pa.bool_())
    ])

    total_nodes = 0
    saved_nodes = 0
    invalid_geometry = 0
    duplicate_nodes = 0

    seen_ids = set()

    degree_distribution = Counter()
    node_type_distribution = Counter()
    outgoing_distribution = Counter()

    degree_intersections = 0
    type101_intersections = 0
    both_intersections = 0

    batch = []

    def write_batch(writer, rows):

        frame = pd.DataFrame(
            rows,
            columns=[
                "node_id",
                "node_type",
                "x",
                "y",
                "degree",
                "outgoing_count"
            ]
        )

        x, y = transformer.transform(
            frame["x"].to_numpy(),
            frame["y"].to_numpy()
        )

        frame["x"] = x
        frame["y"] = y

        if not np.isfinite(
            frame[["x", "y"]].to_numpy()
        ).all():
            raise RuntimeError(
                "Invalid projected coordinates."
            )

        frame["is_degree_intersection"] = (
            frame["degree"] >= 3
        )

        frame["is_type101_intersection"] = (
            frame["node_type"] == "101"
        )

        table = pa.Table.from_pandas(
            frame,
            schema=schema,
            preserve_index=False
        )

        writer.write_table(table)

        return len(frame)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}"
        )

    if TEMP_PATH.exists():
        raise FileExistsError(
            f"Temporary file exists: {TEMP_PATH}"
        )

    with pq.ParquetWriter(
        TEMP_PATH,
        schema,
        compression="zstd"
    ) as writer:

        with shapefile.Reader(
            str(NODE_PATH),
            encoding="cp949"
        ) as reader:

            expected_nodes = len(reader)

            for item in reader.iterShapeRecords(
                fields=["NODE_ID", "NODE_TYPE"]
            ):

                total_nodes += 1

                node_id = str(
                    item.record["NODE_ID"]
                ).strip()

                node_type = str(
                    item.record["NODE_TYPE"]
                ).strip()

                if not node_id:
                    raise RuntimeError(
                        "Empty NODE_ID detected."
                    )

                if node_id in seen_ids:
                    duplicate_nodes += 1
                    continue

                seen_ids.add(node_id)

                if (
                    item.shape.shapeType
                    != shapefile.POINT
                    or len(item.shape.points) != 1
                ):
                    invalid_geometry += 1
                    continue

                x, y = item.shape.points[0]

                node_degree = degree.get(node_id, 0)
                node_outgoing = outgoing.get(node_id, 0)

                degree_distribution[node_degree] += 1
                node_type_distribution[node_type] += 1
                outgoing_distribution[node_outgoing] += 1

                is_degree = node_degree >= 3
                is_type101 = node_type == "101"

                degree_intersections += int(is_degree)
                type101_intersections += int(is_type101)

                both_intersections += int(
                    is_degree and is_type101
                )

                batch.append((
                    node_id,
                    node_type,
                    x,
                    y,
                    node_degree,
                    node_outgoing
                ))

                if len(batch) >= BATCH_SIZE:

                    saved_nodes += write_batch(
                        writer,
                        batch
                    )

                    batch.clear()

                if total_nodes % 200000 == 0:
                    print(
                        f"NODE Processed: "
                        f"{total_nodes:,} / "
                        f"{expected_nodes:,}",
                        flush=True
                    )

            if batch:
                saved_nodes += write_batch(
                    writer,
                    batch
                )

                batch.clear()

    # LINK가 참조하지만 존재하지 않는 NODE
    missing_endpoint_nodes = sum(
        node_id not in seen_ids
        for node_id in degree
    )

    print("\n===== NODE Summary =====")
    print("Total Nodes:", total_nodes)
    print("Saved Nodes:", saved_nodes)
    print("Duplicate Nodes:", duplicate_nodes)
    print("Invalid Geometry:", invalid_geometry)
    print(
        "Missing Endpoint Nodes:",
        missing_endpoint_nodes
    )

    print("\n===== NODE Type Distribution =====")

    for key, value in sorted(
        node_type_distribution.items()
    ):
        print(key, value)

    print("\n===== Degree Distribution =====")

    for key, value in sorted(
        degree_distribution.items()
    ):
        print(key, value)

    print("\n===== Outgoing Distribution =====")

    for key, value in sorted(
        outgoing_distribution.items()
    ):
        print(key, value)

    print("\n===== Intersection Comparison =====")

    print(
        "Degree >= 3:",
        degree_intersections
    )

    print(
        "NODE_TYPE = 101:",
        type101_intersections
    )

    print(
        "Both Conditions:",
        both_intersections
    )

    print(
        "Degree Only:",
        degree_intersections - both_intersections
    )

    print(
        "Type 101 Only:",
        type101_intersections - both_intersections
    )

    print(
        "Neither:",
        saved_nodes
        - degree_intersections
        - type101_intersections
        + both_intersections
    )

    # 최종 검증
    assert total_nodes == expected_nodes
    assert saved_nodes == total_nodes
    assert duplicate_nodes == 0
    assert invalid_geometry == 0
    assert missing_endpoint_nodes == 0

    parquet = pq.ParquetFile(TEMP_PATH)

    assert parquet.metadata.num_rows == saved_nodes

    del parquet

    # 검증 통과 후 최종 파일명으로 변경
    TEMP_PATH.replace(OUTPUT_PATH)

    print("\n===== Saved Output =====")
    print("Path:", OUTPUT_PATH)
    print("Rows:", saved_nodes)
    print("Columns:", len(schema))
    print("Coordinate CRS:", TARGET_CRS)

    print(
        f"File Size: "
        f"{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB"
    )


# ============================================================
# 5. 메인 실행
# ============================================================

def main():

    start = time.perf_counter()

    print("===== National Node Feature Builder =====")
    print("LINK:", LINK_PATH)
    print("NODE:", NODE_PATH)
    print("Batch Size:", BATCH_SIZE)

    print_memory("Initial Memory")

    print("\n===== Build LINK Topology =====")

    degree, outgoing = build_node_counts()

    print_memory("After LINK Topology")

    print("\n===== Build NODE Features =====")

    build_node_features(degree, outgoing)

    print_memory("Final Memory")

    elapsed = time.perf_counter() - start

    print("\n===== Processing Completed =====")
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    print("Original Files: Not Modified")


if __name__ == "__main__":
    main()