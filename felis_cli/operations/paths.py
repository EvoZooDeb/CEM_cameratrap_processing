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
    media_metadata_csv: Path
    event_results_csv: Path
    media_results_csv: Path
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
    media_metadata_csv = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_media_metadata.csv"
    )
    event_results_csv = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_event_results.csv"
    )
    media_results_csv = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_media_results.csv"
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
        media_metadata_csv=media_metadata_csv,
        event_results_csv=event_results_csv,
        media_results_csv=media_results_csv,
        detections_dir=detections_dir,
        progress_json=progress_json,
        two_stage_json=two_stage_json,
    )
