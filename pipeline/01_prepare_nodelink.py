from pathlib import Path
import geopandas as gpd


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent

NODELINK_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nodelink"
    / "[2026-09-14]NODELINKDATA"
)

LINK_PATH = NODELINK_DIR / "MOCT_LINK.shp"
NODE_PATH = NODELINK_DIR / "MOCT_NODE.shp"


print("=== SafeRoute 국가표준노드링크 검사 ===")

# 1. 파일 존재 확인
print("\n[파일 확인]")
print("LINK:", LINK_PATH.exists(), LINK_PATH)
print("NODE:", NODE_PATH.exists(), NODE_PATH)

# 2. SHP 읽기
print("\n데이터를 읽는 중...")
links = gpd.read_file(LINK_PATH)
nodes = gpd.read_file(NODE_PATH)

# 3. 기본 정보
print("\n[LINK]")
print("행 개수:", len(links))
print("CRS:", links.crs)
print("컬럼:")
print(links.columns.tolist())

print("\n[NODE]")
print("행 개수:", len(nodes))
print("CRS:", nodes.crs)
print("컬럼:")
print(nodes.columns.tolist())

# 4. 주요 컬럼 일부 확인
link_columns = [
    "LINK_ID",
    "F_NODE",
    "T_NODE",
    "LANES",
    "ROAD_RANK",
    "MAX_SPD",
    "LENGTH",
]

node_columns = [
    "NODE_ID",
    "NODE_TYPE",
    "NODE_NAME",
]

print("\n[LINK 샘플]")
print(links[link_columns].head())

print("\n[NODE 샘플]")
print(nodes[node_columns].head())

# 5. 결측치 확인
print("\n[LINK 주요 컬럼 결측치]")
print(links[link_columns].isna().sum())

print("\n[NODE 주요 컬럼 결측치]")
print(nodes[node_columns].isna().sum())

# 6. 천안·아산 지역 코드 구조 확인
print("\n[천안·아산 LINK_ID 후보 확인]")

region_prefixes = ["44131", "44133", "44200"]

for code in region_prefixes:
    mask = links["LINK_ID"].astype(str).str.startswith(code)
    print(f"{code}: {mask.sum()}개")

print("\n[LINK_ID 앞 3자리 빈도 일부]")
print(
    links["LINK_ID"]
    .astype(str)
    .str[:3]
    .value_counts()
    .head(30)
)

print("\n[NODE_ID 앞 3자리 빈도 일부]")
print(
    nodes["NODE_ID"]
    .astype(str)
    .str[:3]
    .value_counts()
    .head(30)
)

print("\n=== 검사 완료 ===")