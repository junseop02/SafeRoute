from pathlib import Path
import geopandas as gpd


# ============================================================
# 1. 경로 설정
# ============================================================

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

BOUNDARY_ROOT = PROJECT_ROOT / "data" / "raw" / "boundary"

boundary_matches = list(
    BOUNDARY_ROOT.rglob("bnd_sigungu_00_2025_2Q.shp")
)

if len(boundary_matches) != 1:
    raise RuntimeError(
        f"시군구 SHP를 정확히 1개 찾아야 합니다. 현재: {len(boundary_matches)}개"
    )

BOUNDARY_PATH = boundary_matches[0]


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print("=== SafeRoute 천안·아산 노드링크 추출 ===")

print("\n[1] 데이터 불러오는 중...")

links = gpd.read_file(LINK_PATH)
nodes = gpd.read_file(NODE_PATH)
sigungu = gpd.read_file(BOUNDARY_PATH)

print("전국 LINK:", len(links))
print("전국 NODE:", len(nodes))
print("시군구 경계:", len(sigungu))


# ============================================================
# 3. CRS 통일
# ============================================================

print("\n[2] CRS를 EPSG:5179로 통일...")

links = links.to_crs(epsg=5179)
nodes = nodes.to_crs(epsg=5179)

if sigungu.crs.to_epsg() != 5179:
    sigungu = sigungu.to_crs(epsg=5179)

print("LINK CRS:", links.crs)
print("NODE CRS:", nodes.crs)
print("경계 CRS:", sigungu.crs)


# ============================================================
# 4. 천안·아산 경계 추출
# ============================================================

print("\n[3] 천안·아산 경계 추출...")

region_map = {
    "34011": 44131,   # 천안시 동남구
    "34012": 44133,   # 천안시 서북구
    "34040": 44200,   # 아산시
}

target_boundary = sigungu[
    sigungu["SIGUNGU_CD"].astype(str).isin(region_map.keys())
].copy()

target_boundary["region_code"] = (
    target_boundary["SIGUNGU_CD"]
    .astype(str)
    .map(region_map)
)

print(
    target_boundary[
        ["SIGUNGU_CD", "SIGUNGU_NM", "region_code"]
    ].to_string(index=False)
)

if len(target_boundary) != 3:
    raise RuntimeError(
        f"천안·아산 경계가 3개여야 합니다. 현재: {len(target_boundary)}개"
    )


# ============================================================
# 5. LINK 대표점 생성
# ============================================================

print("\n[4] LINK 대표점 생성...")

link_points = links[
    ["LINK_ID", "geometry"]
].copy()

link_points["geometry"] = (
    link_points.geometry.representative_point()
)


# ============================================================
# 6. 대표점과 행정경계 공간조인
# ============================================================

print("\n[5] 천안·아산 LINK 공간조인...")

joined = gpd.sjoin(
    link_points,
    target_boundary[
        ["SIGUNGU_NM", "region_code", "geometry"]
    ],
    how="inner",
    predicate="within",
)

print("공간조인된 LINK:", len(joined))


# ============================================================
# 7. 원본 LINK에서 천안·아산 LINK 추출
# ============================================================

print("\n[6] 천안·아산 LINK 추출...")

link_region = joined[
    ["LINK_ID", "SIGUNGU_NM", "region_code"]
].drop_duplicates(subset="LINK_ID")

cheonan_asan_links = links.merge(
    link_region,
    on="LINK_ID",
    how="inner",
)

print("추출 LINK:", len(cheonan_asan_links))

print("\n[지역별 LINK 개수]")
print(
    cheonan_asan_links[
        "SIGUNGU_NM"
    ].value_counts()
)


# ============================================================
# 8. LINK에 연결된 NODE 추출
# ============================================================

print("\n[7] 연결 NODE 추출...")

used_node_ids = set(
    cheonan_asan_links["F_NODE"].astype(str)
).union(
    set(cheonan_asan_links["T_NODE"].astype(str))
)

cheonan_asan_nodes = nodes[
    nodes["NODE_ID"].astype(str).isin(used_node_ids)
].copy()

print("사용 NODE ID:", len(used_node_ids))
print("추출 NODE:", len(cheonan_asan_nodes))


# ============================================================
# 9. 무결성 검사
# ============================================================

print("\n[8] 무결성 검사...")

missing_nodes = (
    used_node_ids
    - set(cheonan_asan_nodes["NODE_ID"].astype(str))
)

print("누락 NODE:", len(missing_nodes))

print(
    "LINK_ID 중복:",
    cheonan_asan_links["LINK_ID"].duplicated().sum()
)

print(
    "NODE_ID 중복:",
    cheonan_asan_nodes["NODE_ID"].duplicated().sum()
)

print(
    "region_code 결측:",
    cheonan_asan_links["region_code"].isna().sum()
)

print("\n[region_code별 LINK 개수]")
print(
    cheonan_asan_links[
        "region_code"
    ].value_counts().sort_index()
)

print("\n=== 추출 검사 완료 ===")

# ============================================================
# 10. 전처리 결과 저장
# ============================================================

print("\n[9] 전처리 결과 저장...")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LINK_OUTPUT = PROCESSED_DIR / "links_cheonan_asan.parquet"
NODE_OUTPUT = PROCESSED_DIR / "nodes_cheonan_asan.parquet"

cheonan_asan_links.to_parquet(
    LINK_OUTPUT,
    index=False
)

cheonan_asan_nodes.to_parquet(
    NODE_OUTPUT,
    index=False
)

print("LINK 저장:", LINK_OUTPUT)
print("NODE 저장:", NODE_OUTPUT)

print(
    "LINK 파일 존재:",
    LINK_OUTPUT.exists()
)

print(
    "NODE 파일 존재:",
    NODE_OUTPUT.exists()
)