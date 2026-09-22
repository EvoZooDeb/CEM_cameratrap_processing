from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from ..config import Config
from ..verbosity import detail, status
from .labels import NAMES
from .paths import resolve_paths


LOG = logging.getLogger(__name__)


MEDIA_COLUMNS = [
    "file_name",
    "label",
    "detection_count",
    "annotated_frame_count",
    "mean_group_size",
    "max_group_size",
    "mean_detector_confidence",
    "max_detector_confidence",
    "mean_classification_confidence",
    "max_classification_confidence",
]

EVENT_COLUMNS = [
    "camera_id",
    "analysis_name",
    "event_id",
    "event_start",
    "event_end",
    "event_duration_seconds",
    "media_files",
    "label",
    "media_count",
    "detection_count",
    "annotated_frame_count",
    "mean_group_size",
    "max_group_size",
    "mean_detector_confidence",
    "max_detector_confidence",
    "mean_classification_confidence",
    "max_classification_confidence",
    "comment",
]


@dataclass
class AggregateResult:
    events: pd.DataFrame
    media: pd.DataFrame
    event_csv: Path
    media_csv: Path


def _convert_to_datetime(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")


def _detection_file_path(detections_dir: Path, file_name: str) -> Path:
    """Return a filesystem-safe, stable detail-file path for one media file."""
    digest = hashlib.sha256(file_name.encode("utf-8")).hexdigest()
    return detections_dir / f"{digest}.json"


def _video_fps(video_path: Path) -> float | None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return fps if math.isfinite(fps) and fps > 0 else None


def _frame_index(frame_name: str) -> int | None:
    match = re.search(r"_(\d+)$", frame_name)
    return int(match.group(1)) if match else None


def _write_detection_file(
    detections_dir: Path,
    input_dir: Path,
    file_name: str,
    detections: list[dict[str, Any]],
) -> None:
    """Write final, per-object detections separately from compact CSV summaries."""
    path = _detection_file_path(detections_dir, file_name)
    if not detections:
        path.unlink(missing_ok=True)
        return

    detections_dir.mkdir(parents=True, exist_ok=True)
    is_video = Path(file_name).suffix.lower() in {".mp4", ".avi", ".mov"}
    document: dict[str, Any] = {
        "schema_version": 2,
        "media_name": file_name,
        "media_type": "video" if is_video else "image",
        "detections": detections,
    }

    if is_video:
        fps = _video_fps(input_dir / file_name)
        if fps is not None:
            document["frame_rate"] = fps
        for detection in detections:
            index = _frame_index(str(detection.get("frame", "")))
            if index is None:
                continue
            detection["frame_index"] = index
            if fps is not None:
                detection["timestamp_seconds"] = (index - 1) / fps

    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(document, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def _collect_detections(
    per_file_dir: Path,
    media_names: list[str],
    input_dir: Path,
    detections_dir: Path,
    classified_labels: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    detections_by_file: dict[str, list[dict[str, Any]]] = {}
    for media_index, file_name in enumerate(media_names, start=1):
        status(LOG, "Aggregating [%d/%d]: %s", media_index, len(media_names), file_name)
        stem = Path(file_name).stem
        labels_path = per_file_dir / stem / "labels"
        detections: list[dict[str, Any]] = []
        if labels_path.exists():
            for label_file in sorted(labels_path.glob("*.txt")):
                detail(LOG, "Reading detections: %s", label_file)
                lines = label_file.read_text(encoding="utf-8").splitlines()
                for line_number, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    class_id = int(float(parts[0]))
                    detector_confidence = float(parts[5])
                    if detector_confidence <= 0.25:
                        continue
                    prediction_key = f"{stem}/labels/{label_file.name}:{line_number}"
                    classified = (classified_labels or {}).get(prediction_key)
                    label = (
                        str(classified["label"])
                        if classified is not None
                        else NAMES.get(class_id, str(class_id))
                    )
                    detections.append(
                        {
                            "frame": label_file.stem,
                            "detector_class_id": class_id,
                            "label": label,
                            "detector_confidence": detector_confidence,
                            "classification_confidence": (
                                float(classified["confidence"])
                                if classified is not None
                                and classified.get("source") == "classifier"
                                and classified.get("confidence") is not None
                                else None
                            ),
                            "bbox": [float(value) for value in parts[1:5]],
                        }
                    )
        _write_detection_file(detections_dir, input_dir, file_name, detections)
        detections_by_file[file_name] = detections
    return detections_by_file


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _maximum(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _label_summary(
    detections: list[dict[str, Any]],
    *,
    frame_key,
) -> dict[str, int | float | None]:
    frame_counts = Counter(frame_key(detection) for detection in detections)
    detector_confidences = [float(item["detector_confidence"]) for item in detections]
    classification_confidences = [
        float(item["classification_confidence"])
        for item in detections
        if item.get("classification_confidence") is not None
    ]
    group_sizes = list(frame_counts.values())
    return {
        "detection_count": len(detections),
        "annotated_frame_count": len(frame_counts),
        "mean_group_size": _mean([float(value) for value in group_sizes]) or 0.0,
        "max_group_size": max(group_sizes, default=0),
        "mean_detector_confidence": _mean(detector_confidences),
        "max_detector_confidence": _maximum(detector_confidences),
        "mean_classification_confidence": _mean(classification_confidences),
        "max_classification_confidence": _maximum(classification_confidences),
    }


def _empty_summary() -> dict[str, int | float | None]:
    return {
        "detection_count": 0,
        "annotated_frame_count": 0,
        "mean_group_size": 0.0,
        "max_group_size": 0,
        "mean_detector_confidence": None,
        "max_detector_confidence": None,
        "mean_classification_confidence": None,
        "max_classification_confidence": None,
    }


def _build_media_results(
    media_names: list[str],
    detections_by_file: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for file_name in media_names:
        detections = detections_by_file.get(file_name, [])
        labels = sorted({str(item["label"]) for item in detections})
        if not labels:
            rows.append({"file_name": file_name, "label": "EMPTY", **_empty_summary()})
            continue
        for label in labels:
            label_detections = [item for item in detections if item["label"] == label]
            rows.append(
                {
                    "file_name": file_name,
                    "label": label,
                    **_label_summary(label_detections, frame_key=lambda item: item["frame"]),
                }
            )
    return pd.DataFrame(rows, columns=MEDIA_COLUMNS)


def _event_ranges(timestamps: list[dt.datetime]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(timestamps):
        start = index
        anchor = timestamps[index]
        end = index
        while end + 1 < len(timestamps) and (timestamps[end + 1] - anchor).total_seconds() <= 10:
            end += 1
        ranges.append((start, end))
        index = end + 1
    return ranges


def _build_event_results(
    cfg: Config,
    metadata: pd.DataFrame,
    detections_by_file: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ordered = metadata.copy()
    ordered["_timestamp"] = ordered["date_exif"].map(_convert_to_datetime)
    ordered = ordered.sort_values(["_timestamp", "file_name"]).reset_index(drop=True)
    ranges = _event_ranges(ordered["_timestamp"].tolist())
    rows: list[dict[str, Any]] = []

    for event_id, (start, end) in enumerate(ranges, start=1):
        event_media = ordered.iloc[start : end + 1]
        media_names = event_media["file_name"].astype(str).tolist()
        start_time = event_media["_timestamp"].min()
        end_candidates = [
            row["_timestamp"]
            + dt.timedelta(seconds=float(row.get("duration", 0.0) or 0.0))
            for _, row in event_media.iterrows()
        ]
        end_time = max(end_candidates, default=start_time)
        event_detections = [
            (file_name, detection)
            for file_name in media_names
            for detection in detections_by_file.get(file_name, [])
        ]
        labels = sorted({str(detection["label"]) for _, detection in event_detections})
        labels = labels or ["EMPTY"]

        for label in labels:
            selected = [
                {**detection, "_file_name": file_name}
                for file_name, detection in event_detections
                if detection["label"] == label
            ]
            if label == "EMPTY":
                summary = _empty_summary()
                media_count = len(media_names)
            else:
                summary = _label_summary(
                    selected,
                    frame_key=lambda item: (item["_file_name"], item["frame"]),
                )
                media_count = len({str(item["_file_name"]) for item in selected})
            rows.append(
                {
                    "camera_id": cfg.paths.camera_id,
                    "analysis_name": cfg.paths.footage_date,
                    "event_id": event_id,
                    "event_start": start_time.strftime("%Y:%m:%d %H:%M:%S"),
                    "event_end": end_time.strftime("%Y:%m:%d %H:%M:%S"),
                    "event_duration_seconds": (end_time - start_time).total_seconds(),
                    "media_files": json.dumps(
                        media_names, ensure_ascii=False, separators=(",", ":")
                    ),
                    "label": label,
                    "media_count": media_count,
                    **summary,
                    "comment": "",
                }
            )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def aggregate(cfg: Config, completed_files: list[str] | None = None) -> AggregateResult:
    paths = resolve_paths(cfg)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    status(LOG, "Aggregation started.")

    media_names = (
        list(completed_files)
        if completed_files is not None
        else sorted(
            file_name
            for file_name in os.listdir(paths.input_dir)
            if (paths.input_dir / file_name).is_file()
            and Path(file_name).suffix.lower() in {".jpg", ".png", ".mp4", ".avi", ".mov"}
        )
    )
    classified_labels: dict[str, dict[str, Any]] | None = None
    if cfg.two_stage.strategy == "two_stage":
        if not paths.two_stage_json.exists():
            raise FileNotFoundError(
                "Two-stage classifications not found. Run 'felis classify' before aggregation."
            )
        raw_predictions = json.loads(paths.two_stage_json.read_text(encoding="utf-8"))
        if not isinstance(raw_predictions, dict):
            raise ValueError(f"Invalid two-stage classification data: {paths.two_stage_json}")
        classified_labels = raw_predictions

    detections_by_file = _collect_detections(
        paths.per_file_root,
        media_names,
        paths.input_dir,
        paths.detections_dir,
        classified_labels,
    )
    media_df = _build_media_results(media_names, detections_by_file)
    media_df.to_csv(paths.media_results_csv, index=False)

    if not paths.media_metadata_csv.exists():
        raise FileNotFoundError(f"Media metadata CSV not found: {paths.media_metadata_csv}")
    metadata = pd.read_csv(paths.media_metadata_csv)
    metadata = metadata[metadata["file_name"].isin(media_names)]
    event_df = _build_event_results(cfg, metadata, detections_by_file)
    event_df.to_csv(paths.event_results_csv, index=False)

    status(
        LOG,
        "Aggregation complete: %d event-label row(s); wrote %s",
        len(event_df),
        paths.event_results_csv,
    )
    status(
        LOG,
        "Media results: %d media-label row(s); wrote %s",
        len(media_df),
        paths.media_results_csv,
    )
    return AggregateResult(
        events=event_df,
        media=media_df,
        event_csv=paths.event_results_csv,
        media_csv=paths.media_results_csv,
    )
