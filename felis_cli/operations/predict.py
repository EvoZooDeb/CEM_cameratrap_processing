from __future__ import annotations

import json
import logging
import os
from typing import Callable

import cv2
from ultralytics import YOLO

from ..config import Config
from ..verbosity import backend_verbose, detail, diagnostic, status, third_party_stdout
from .paths import PathsResolved, resolve_paths


LOG = logging.getLogger(__name__)


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

    status(LOG, "Prediction: %d media file(s) found.", len(supported_files))

    model_path = cfg.predict.model_path
    if cfg.two_stage.strategy == "two_stage":
        detector_weights = {
            "best_27": "best_27.pt",
            "mdv6": "MDV6-yolov10-c.pt",
            "deepfaune": "deepfaune-yolov8s_960.pt",
            "best_28": "best_28.pt",
        }[cfg.two_stage.detector]
        model_path = cfg.two_stage.models_dir / detector_weights
    if not model_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {model_path}")
    status(LOG, "Loading detector: %s", model_path)
    diagnostic(LOG, "Detector device=%s, image_size=%s", cfg.predict.device, cfg.predict.imgsz)
    with third_party_stdout():
        model = YOLO(str(model_path))

    for media_index, file in enumerate(supported_files, start=1):
        if should_cancel and should_cancel():
            break
        status(LOG, "Predicting [%d/%d]: %s", media_index, len(supported_files), file)
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
                verbose=backend_verbose(),
            )
        elif file.lower().endswith((".mp4", ".avi", ".mov")):
            save_video_frames = (
                cfg.predict.save_frames or cfg.two_stage.strategy == "two_stage"
            )
            frames_dir = p.per_file_root / stem / f"{stem}_frames"
            if save_video_frames:
                frames_dir.mkdir(parents=True, exist_ok=True)
            results = model.predict(
                full_path,
                save=False,
                save_txt=True,
                save_conf=True,
                show_labels=False,
                show_boxes=False,
                show_conf=False,
                imgsz=cfg.predict.imgsz,
                conf=cfg.predict.conf,
                iou=cfg.predict.iou,
                agnostic_nms=True,
                stream=True,
                project=str(p.results_root),
                name=dynamic_project_name,
                device=cfg.predict.device,
                verbose=backend_verbose(),
            )
            # consume generator to execute
            for frame_index, result in enumerate(results, start=1):
                if save_video_frames:
                    frame_path = frames_dir / f"{stem}_{frame_index}.jpg"
                    if not cv2.imwrite(str(frame_path), result.orig_img):
                        raise OSError(f"Could not save video frame: {frame_path}")
                detail(LOG, "Predicted frame %d: %s", frame_index, file)
                if should_cancel and should_cancel():
                    break

            if should_cancel and should_cancel():
                break

        completed_files.append(file)
        _write_progress(p, completed_files, len(supported_files))

    if len(completed_files) < len(supported_files):
        LOG.warning(
            "Prediction cancelled after %d/%d media file(s).",
            len(completed_files),
            len(supported_files),
        )
    else:
        status(LOG, "Prediction complete: %d media file(s).", len(completed_files))
    return completed_files
