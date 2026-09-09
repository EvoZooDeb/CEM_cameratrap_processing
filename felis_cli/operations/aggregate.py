from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import cv2
import pandas as pd

from ..config import Config
from .labels import NAMES
from .paths import resolve_paths


@dataclass
class AggregateResult:
    sequences: pd.DataFrame
    per_image: pd.DataFrame
    sequence_csv: Path
    per_image_csv: Path
    per_image_written: bool


def _convert_to_datetime(string: str) -> dt.datetime:
    # expects format like YYYY:MM:DD HH:MM:SS
    return dt.datetime(
        int(string[:4]),
        int(string[5:7]),
        int(string[8:10]),
        int(string[11:13]),
        int(string[14:16]),
        int(string[17:19]),
    )


def _collect_file_summary(
    per_file_dir: Path,
    stem_to_name: dict[str, str],
    input_dir: Path,
    detections_dir: Path,
    classified_labels: dict[str, dict[str, Any]] | None = None,
    completed_files: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[tuple[str, str, float, int, int, float, int]] = []
    directories = (
        [Path(file_name).stem for file_name in completed_files]
        if completed_files is not None
        else os.listdir(per_file_dir)
    )
    for directory in directories:
        labels_path = per_file_dir / directory / "labels"
        if not labels_path.exists():
            continue
        resolved_name = stem_to_name.get(directory, directory)
        frame_labels: list[str] = []
        group_sizes: list[int] = []
        geometry_entries: list[dict[str, Any]] = []
        for label_file in os.listdir(labels_path):
            with open(labels_path / label_file) as f:
                group_size = 0
                for line_number, line in enumerate(f.readlines()):
                    parts = line.rstrip().split(" ")
                    if len(parts) < 6:
                        continue
                    cidx, x_center, y_center, width, height, conf = parts
                    confidence = float(conf)
                    if confidence > 0.25:
                        class_idx = int(cidx)
                        prediction_key = f"{directory}/labels/{label_file}:{line_number}"
                        classified = (classified_labels or {}).get(prediction_key)
                        label_name = (
                            str(classified["label"])
                            if classified is not None
                            else NAMES.get(class_idx, str(class_idx))
                        )
                        frame_labels.append(label_name)
                        group_size += 1
                        geometry_entries.append(
                            {
                                "frame": label_file.rsplit(".", 1)[0],
                                "class_id": class_idx,
                                "label": label_name,
                                "confidence": confidence,
                                "bbox": [
                                    float(x_center),
                                    float(y_center),
                                    float(width),
                                    float(height),
                                ],
                            }
                        )
                if group_size:
                    group_sizes.append(group_size)
        _write_detection_file(
            detections_dir=detections_dir,
            input_dir=input_dir,
            file_name=resolved_name,
            geometries=geometry_entries,
        )
        if frame_labels:
            s = pd.Series(frame_labels)
            top_label = s.value_counts().index.tolist()[0]
            top_count = int((s == top_label).sum())
            n_objects = int(s.count())
            n_frames = int(len(group_sizes))
            confidence = top_count / max(n_objects, 1)
            mean_group = float(sum(group_sizes) / max(len(group_sizes), 1)) if group_sizes else 0.0
            max_group = int(max(group_sizes)) if group_sizes else 0
            rows.append(
                (
                    resolved_name,
                    top_label,
                    confidence,
                    n_objects,
                    n_frames,
                    mean_group,
                    max_group,
                )
            )
        else:
            rows.append((resolved_name, "EMPTY", 0.0, 0, 0, 0.0, 0))

    df = pd.DataFrame(
        rows,
        columns=[
            "file_name",
            "label",
            "label_confidence",
            "n_annotated_objects",
            "n_annotated_frames",
            "mean_group_size",
            "max_group_size",
        ],
    )
    return df


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
    geometries: list[dict[str, Any]],
) -> None:
    """Write detailed detections separately from the compact per-media CSV."""
    path = _detection_file_path(detections_dir, file_name)
    if not geometries:
        path.unlink(missing_ok=True)
        return

    detections_dir.mkdir(parents=True, exist_ok=True)
    is_video = Path(file_name).suffix.lower() in {".mp4", ".avi", ".mov"}
    document: dict[str, Any] = {
        "schema_version": 1,
        "media_name": file_name,
        "media_type": "video" if is_video else "image",
        "detections": geometries,
    }

    if is_video:
        fps = _video_fps(input_dir / file_name)
        if fps is not None:
            document["frame_rate"] = fps
        for geometry in geometries:
            index = _frame_index(str(geometry.get("frame", "")))
            if index is None:
                continue
            geometry["frame_index"] = index
            if fps is not None:
                geometry["timestamp_seconds"] = (index - 1) / fps

    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(document, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def aggregate(
    cfg: Config,
    save_per_image: bool = False,
    completed_files: list[str] | None = None,
) -> AggregateResult:
    p = resolve_paths(cfg)
    p.results_dir.mkdir(parents=True, exist_ok=True)

    stem_to_name = {
        Path(file_name).stem: file_name
        for file_name in os.listdir(p.input_dir)
        if (p.input_dir / file_name).is_file()
    }
    classified_labels: dict[str, dict[str, Any]] | None = None
    if cfg.two_stage.strategy == "two_stage":
        if not p.two_stage_json.exists():
            raise FileNotFoundError(
                "Two-stage classifications not found. Run 'felis classify' before aggregation."
            )
        raw_predictions = json.loads(p.two_stage_json.read_text(encoding="utf-8"))
        if not isinstance(raw_predictions, dict):
            raise ValueError(f"Invalid two-stage classification data: {p.two_stage_json}")
        classified_labels = raw_predictions
    file_df = _collect_file_summary(
        p.per_file_root,
        stem_to_name,
        p.input_dir,
        p.detections_dir,
        classified_labels,
        completed_files,
    )
    file_df = file_df.sort_values("file_name")
    if save_per_image:
        file_df.to_csv(p.per_image_csv, index=False)

    # EXIF CSV must exist
    if not p.exif_csv.exists():
        raise FileNotFoundError(f"EXIF CSV not found: {p.exif_csv}")

    exif_df = pd.read_csv(p.exif_csv)
    exif_df["file_name"] = [i.split(".")[0] for i in exif_df["file_name"]]
    exif_sorted = exif_df.sort_values("date_exif")

    file_df_for_join = file_df.copy()
    file_df_for_join["file_name"] = file_df_for_join["file_name"].str.replace(
        r"\.[^.]+$", "", regex=True
    )

    # Group into sequences by <=10 seconds gaps
    timestamps = [_convert_to_datetime(x) for x in exif_sorted["date_exif"]]
    indices: list[Tuple[int, int]] = []
    i = 0
    while i < len(timestamps):
        start = i
        anchor = timestamps[i]
        j = i
        while j + 1 < len(timestamps) and (timestamps[j + 1] - anchor).total_seconds() <= 10:
            j += 1
        indices.append((start, j))
        i = j + 1

    sequence_rows: list[Tuple[str, str, str, float, str, str, float, int, int, float, int, str]] = []
    for start, end in indices:
        seq_df = exif_sorted.iloc[start : end + 1]
        joined_df = (
            pd.merge(seq_df, file_df_for_join, how="left", on=["file_name"])
            .sort_values("file_name")
        )

        if len(joined_df) == 1:
            joined_date = joined_df["date_exif"].iloc[0]
            duration = float(joined_df["duration"].iloc[0]) if "duration" in joined_df.columns else 1.0
            fname = joined_df["file_name"].iloc[0]
            label = joined_df["label"].iloc[0]
            n_objects = int(joined_df["n_annotated_objects"].iloc[0]) if "n_annotated_objects" in joined_df.columns else 0
            n_frames = int(joined_df["n_annotated_frames"].iloc[0]) if "n_annotated_frames" in joined_df.columns else 0
            confidence = float(joined_df["label_confidence"].iloc[0]) if "label_confidence" in joined_df.columns else 0.0
            mean_group = float(joined_df["mean_group_size"].iloc[0]) if "mean_group_size" in joined_df.columns else 0.0
            max_group = int(joined_df["max_group_size"].iloc[0]) if "max_group_size" in joined_df.columns else 0
        else:
            start_dt = joined_df["date_exif"].iloc[0]
            end_suffix = joined_df["date_exif"].iloc[-1][17:19]
            joined_date = f"{start_dt}-{end_suffix}"
            duration = float(len(joined_df))
            fname = f"{joined_df['file_name'].iloc[0]}-{joined_df['file_name'].iloc[-1][4:]}"
            non_empty = joined_df[joined_df["label"] != "EMPTY"]
            if len(non_empty) > 0:
                top_label = non_empty["label"].value_counts().index.tolist()[0]
                top_count = int((non_empty["label"] == top_label).sum())
                n_objects = int(non_empty["n_annotated_objects"].astype(int).sum())
                n_frames = int(len(non_empty))
                confidence = top_count / max(n_frames, 1)
                mean_group = float(sum(non_empty["max_group_size"]) / max(n_frames, 1))
                max_group = int(max(non_empty["max_group_size"]))
                label = top_label
            else:
                label = "EMPTY"
                n_objects = 0
                n_frames = 0
                confidence = 0.0
                mean_group = 0.0
                max_group = 0

        sequence_rows.append(
            (
                cfg.paths.camera_id,
                cfg.paths.footage_date,
                joined_date,
                duration,
                fname,
                label,
                confidence,
                n_objects,
                n_frames,
                mean_group,
                max_group,
                "",
            )
        )

    seq_df = pd.DataFrame(
        sequence_rows,
        columns=[
            "camera_id",
            "upload_date",
            "sequence_date",
            "sequence_duration",
            "sequence_file_names",
            "label",
            "label_confidence",
            "n_annotated_objects",
            "n_annotated_frames",
            "mean_group_size",
            "max_group_size",
            "comment",
        ],
    )
    seq_df.to_csv(p.final_csv, index=False)
    return AggregateResult(
        sequences=seq_df,
        per_image=file_df,
        sequence_csv=p.final_csv,
        per_image_csv=p.per_image_csv,
        per_image_written=save_per_image,
    )
