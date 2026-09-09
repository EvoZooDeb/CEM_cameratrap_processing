from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config


@dataclass
class PathsResolved:
    input_dir: Path
    results_root: Path
    per_file_root: Path
    results_dir: Path
    exif_csv: Path
    final_csv: Path
    per_image_csv: Path
    detections_dir: Path
    progress_json: Path
    two_stage_json: Path


def resolve_paths(cfg: Config) -> PathsResolved:
    input_dir = (
        cfg.paths.input_root
        / cfg.paths.username
        / "files"
        / "cameratrap"
        / cfg.paths.camera_id
        / cfg.paths.footage_date
    )
    results_root = cfg.paths.output_root / cfg.paths.username
    per_file_root = results_root / cfg.paths.camera_id / cfg.paths.footage_date
    results_dir = per_file_root / "results"
    exif_csv = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_exif.csv"
    )
    final_csv = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_results.csv"
    )
    per_image_csv = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_per_image.csv"
    )
    detections_dir = results_dir / "detections"
    progress_json = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_progress.json"
    )
    two_stage_json = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_two_stage.json"
    )
    return PathsResolved(
        input_dir=input_dir,
        results_root=results_root,
        per_file_root=per_file_root,
        results_dir=results_dir,
        exif_csv=exif_csv,
        final_csv=final_csv,
        per_image_csv=per_image_csv,
        detections_dir=detections_dir,
        progress_json=progress_json,
        two_stage_json=two_stage_json,
    )
