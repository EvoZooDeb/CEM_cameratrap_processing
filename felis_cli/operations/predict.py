from __future__ import annotations

import json
import os
from typing import Callable

from ultralytics import YOLO

from ..config import Config
from .paths import PathsResolved, resolve_paths


def _write_progress(paths: PathsResolved, completed_files: list[str], total_files: int) -> None:
    """Persist completed media names so an interrupted run can be finalized safely."""
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = paths.progress_json.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "completed_files": completed_files,
                "total_files": total_files,
            }
        ),
        encoding="utf-8",
    )
    temporary_path.replace(paths.progress_json)


def predict(
    cfg: Config,
    should_cancel: Callable[[], bool] | None = None,
) -> list[str]:
    p = resolve_paths(cfg)
    p.per_file_root.mkdir(parents=True, exist_ok=True)

    supported_files = [
        file
        for file in sorted(os.listdir(p.input_dir))
        if file.lower().endswith((".jpg", ".png", ".mp4", ".avi", ".mov"))
    ]
    completed_files: list[str] = []
    _write_progress(p, completed_files, len(supported_files))

    model_path = cfg.predict.model_path
    if cfg.two_stage.strategy == "two_stage":
        detector_weights = {
            "best_27": "best_27.pt",
            "mdv6": "md_v1000.0.0-redwood.pt",
            "deepfaune_1.4": "deepfaune_1.4.pt",
            "best_28": "best_28.pt",
        }[cfg.two_stage.detector]
        model_path = cfg.two_stage.models_dir / detector_weights
    if not model_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {model_path}")
    model = YOLO(str(model_path))

    for file in supported_files:
        if should_cancel and should_cancel():
            break
        stem = file.split(".")[0]
        dynamic_project_name = f"{cfg.paths.camera_id}/{cfg.paths.footage_date}/{stem}"
        full_path = str(p.input_dir / file)
        if file.lower().endswith((".jpg", ".png")):
            model.predict(
                full_path,
                save=False,
                save_txt=True,
                save_conf=True,
                imgsz=cfg.predict.imgsz,
                conf=cfg.predict.conf,
                iou=cfg.predict.iou,
                agnostic_nms=True,
                project=str(p.results_root),
                name=dynamic_project_name,
                device=cfg.predict.device,
            )
        elif file.lower().endswith((".mp4", ".avi", ".mov")):
            results = model.predict(
                full_path,
                save=False,
                # Classification needs the original video frames for its crops.
                save_frames=cfg.predict.save_frames or cfg.two_stage.strategy == "two_stage",
                save_txt=True,
                save_conf=True,
                show_labels=False,
                show_boxes=False,
                show_conf=False,
                imgsz=cfg.predict.imgsz,
                conf=cfg.predict.conf,
                iou=cfg.predict.iou,
                augment=True,
                agnostic_nms=True,
                stream=True,
                project=str(p.results_root),
                name=dynamic_project_name,
                device=cfg.predict.device,
            )
            # consume generator to execute
            for _ in results:
                if should_cancel and should_cancel():
                    break

            if should_cancel and should_cancel():
                break

        completed_files.append(file)
        _write_progress(p, completed_files, len(supported_files))

    return completed_files
