from pathlib import Path

import shapefile
import psutil
from pyproj import CRS


# 프로젝트 경로
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    ROOT
    / "data"
    / "legacy"
    / "raw"
    / "nodelink"
    / "[2026-09-14]NODELINKDATA"
)

LINK_PATH = RAW_DIR / "MOCT_LINK.shp"
NODE_PATH = RAW_DIR / "MOCT_NODE.shp"


# 메모리 사용량 확인
def print_memory():
    memory = psutil.virtual_memory()

    print(f"Total RAM: {memory.total / 1024**3:.2f} GiB")
    print(f"Available RAM: {memory.available / 1024**3:.2f} GiB")
    print(f"RAM Usage: {memory.percent}%")


# SHP 구성 파일 확인
def check_shapefile(path):
    print(f"\n===== File Check: {path.stem} =====")

    for extension in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        file_path = path.with_suffix(extension)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        size = file_path.stat().st_size / 1024**2

        print(f"{file_path.name}: {size:.4f} MB")


# 좌표계 확인
def inspect_crs(path):
    prj_path = path.with_suffix(".prj")

    wkt = prj_path.read_text(encoding="utf-8")
    crs = CRS.from_wkt(wkt)

    print("\n===== CRS =====")
    print("CRS Name:", crs.name)
    print("EPSG:", crs.to_epsg())
    print("Projected:", crs.is_projected)

    print("\nAxis Information:")

    for axis in crs.axis_info:
        print(axis.name, axis.unit_name)

    return crs


# SHP 메타데이터 및 샘플 확인
def inspect_shapefile(path, required_fields):
    encoding = path.with_suffix(".cpg").read_text(
        encoding="utf-8-sig"
    ).strip()

    print(f"\n===== Inspect: {path.stem} =====")
    print("Encoding:", encoding)

    with shapefile.Reader(
        str(path),
        encoding=encoding
    ) as reader:

        print("Record Count:", len(reader))
        print("Shape Type:", reader.shapeTypeName)
        print("Bounding Box:", reader.bbox)

        fields = [
            field[0]
            for field in reader.fields[1:]
        ]

        print("\n===== Fields =====")

        for field in reader.fields[1:]:
            print(field)

        missing_fields = [
            field
            for field in required_fields
            if field not in fields
        ]

        print("\n===== Required Fields =====")

        for field in required_fields:
            print(f"{field}: {field in fields}")

        if missing_fields:
            raise ValueError(
                f"Missing fields: {missing_fields}"
            )

        print("\n===== First 3 Records =====")

        for index, item in enumerate(
            reader.iterShapeRecords()
        ):
            if index >= 3:
                break

            record = item.record.as_dict()
            geometry = item.shape

            print(f"\nRecord {index + 1}")

            for field in required_fields:
                print(f"{field}: {record[field]}")

            print("Geometry Type:", geometry.shapeTypeName)
            print("Point Count:", len(geometry.points))
            print("First Points:", geometry.points[:3])

        return len(reader)


def main():
    print("===== National Road Data Inspection =====")
    print("Project Root:", ROOT)
    print("Raw Directory:", RAW_DIR)

    print("\n===== Initial Memory =====")
    print_memory()

    # 원본 구성 파일 검증
    check_shapefile(LINK_PATH)
    check_shapefile(NODE_PATH)

    # 원본 좌표계 검증
    link_crs = inspect_crs(LINK_PATH)
    node_crs = inspect_crs(NODE_PATH)

    print("\n===== CRS Comparison =====")
    print("Same CRS:", link_crs == node_crs)

    if link_crs != node_crs:
        raise ValueError("Link and node CRS mismatch.")

    # 도로 링크 필수 속성
    link_fields = [
        "LINK_ID",
        "F_NODE",
        "T_NODE",
        "LANES",
        "MAX_SPD",
        "ROAD_RANK",
        "ROAD_TYPE",
        "LENGTH"
    ]

    # 도로 노드 필수 속성
    node_fields = [
        "NODE_ID",
        "NODE_TYPE"
    ]

    link_count = inspect_shapefile(
        LINK_PATH,
        link_fields
    )

    node_count = inspect_shapefile(
        NODE_PATH,
        node_fields
    )

    print("\n===== Final Summary =====")
    print("National Links:", link_count)
    print("National Nodes:", node_count)
    print("Same CRS:", link_crs == node_crs)

    print("\n===== Final Memory =====")
    print_memory()

    print("\nInspection Completed.")
    print("Original Files: Not Modified")


if __name__ == "__main__":
    main()