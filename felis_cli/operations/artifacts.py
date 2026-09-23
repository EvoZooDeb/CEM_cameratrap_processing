"""Create a verified, self-contained archive of pipeline artifacts."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from ..config import Config
from .paths import resolve_paths


ARCHIVE_NAME = "artifacts.zip"
MANIFEST_NAME = "manifest.json"


def _is_video_frame(path: Path) -> bool:
    return any(part.endswith("_frames") for part in path.parts)


def _file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _archive_member_digest(archive: zipfile.ZipFile, name: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(name) as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def package_artifacts(cfg: Config) -> Path:
    """Archive results, verify the archive, then remove the unpacked artifacts.

    Video frame directories are deliberately omitted. Cleanup happens only after
    the ZIP CRC and every manifest checksum have been validated.
    """
    analysis_dir = resolve_paths(cfg).per_file_root
    analysis_dir.mkdir(parents=True, exist_ok=True)
    archive_path = analysis_dir / ARCHIVE_NAME
    temporary_path = analysis_dir / f".{ARCHIVE_NAME}.tmp"
    detections_archive_path = analysis_dir / ".detections.zip.tmp"

    source_files = [
        path
        for path in sorted(analysis_dir.rglob("*"))
        if path.is_file()
        and path not in {archive_path, temporary_path, detections_archive_path}
        and path.name != ".felis_cancel"
        and not _is_video_frame(path.relative_to(analysis_dir))
    ]
    entries: list[dict[str, object]] = []

    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in source_files:
                relative = source.relative_to(analysis_dir).as_posix()
                size, digest = _file_digest(source)
                archive.write(source, relative)
                entries.append({"path": relative, "size": size, "sha256": digest})

            detections = [
                source for source in source_files
                if source.parent == analysis_dir / "results" / "detections"
                and source.suffix == ".json"
            ]
            if detections:
                with zipfile.ZipFile(
                    detections_archive_path, "w", compression=zipfile.ZIP_DEFLATED
                ) as nested:
                    for source in detections:
                        nested.write(source, arcname=f"detections/{source.name}")
                relative = "results/detections.zip"
                size, digest = _file_digest(detections_archive_path)
                archive.write(detections_archive_path, relative)
                entries.append({"path": relative, "size": size, "sha256": digest})

            manifest = {
                "schema_version": 1,
                "video_frames_included": False,
                "files": entries,
            }
            archive.writestr(
                MANIFEST_NAME,
                json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
            )

        with zipfile.ZipFile(temporary_path, "r") as archive:
            bad_file = archive.testzip()
            if bad_file is not None:
                raise RuntimeError(f"artifact archive CRC check failed: {bad_file}")
            manifest = json.loads(archive.read(MANIFEST_NAME))
            for entry in manifest["files"]:
                size, digest = _archive_member_digest(archive, entry["path"])
                if size != entry["size"] or digest != entry["sha256"]:
                    raise RuntimeError(f"artifact archive verification failed: {entry['path']}")

        temporary_path.replace(archive_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        detections_archive_path.unlink(missing_ok=True)

    for child in analysis_dir.iterdir():
        if child == archive_path:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return archive_path
