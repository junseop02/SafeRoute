"""SafeRoute 전체 프로젝트 파일 무결성 목록 생성."""

import hashlib
import json
import os
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "bundle_manifest.json"

# 검사 대상에서 제외할 디렉터리
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "backups",
    ".pytest_cache",
    ".vscode",
    ".idea",
}

# 검사 대상에서 제외할 파일
EXCLUDED_FILES = {
    "bundle_manifest.json",
    "bundle_manifest.json.tmp",
    ".DS_Store",
    "Thumbs.db",
}


def should_exclude(path):
    name = path.name.lower()

    if path.name in EXCLUDED_FILES:
        return True

    if name.startswith(".env"):
        return True

    if any(word in name for word in (
        "secret",
        "credential",
        "api_key",
    )):
        return True

    if path.suffix.lower() in {
        ".pyc", ".pyo", ".tmp", ".log"
    }:
        return True

    return False


# 대용량 파일을 분할하여 SHA256 계산
def calculate_sha256(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    files = []

    for current, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(
            d for d in dirs
            if d not in EXCLUDED_DIRS
        )

        for name in sorted(names):
            path = Path(current) / name

            if should_exclude(path):
                continue

            if path.is_symlink() or not path.is_file():
                continue

            files.append({
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            })

            if len(files) % 200 == 0:
                print(f"Processed: {len(files)} files")

    files.sort(key=lambda item: item["path"])

    manifest = {
        "created_date": date.today().isoformat(),
        "excludes_self": True,
        "files": files,
    }

    # 생성 완료 후 기존 매니페스트 교체
    temporary = ROOT / "bundle_manifest.json.tmp"

    temporary.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(OUTPUT)

    total_size = sum(f["size_bytes"] for f in files)

    print("\n===== Bundle Manifest =====")
    print("Created date:", manifest["created_date"])
    print("Total files:", len(files))
    print(f"Total size: {total_size / 1024**3:.4f} GiB")
    print("Output:", OUTPUT.name)
    print("Manifest generation: PASS")


if __name__ == "__main__":
    main()