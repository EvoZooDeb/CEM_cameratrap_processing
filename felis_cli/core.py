from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple

import cv2
import exifread
import ffmpeg
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from .config import Config


@dataclass
class PathsResolved:
    input_dir: Path
    results_root: Path
    per_file_root: Path
    results_dir: Path
    exif_csv: Path
    final_csv: Path
    per_image_csv: Path
    progress_json: Path


@dataclass
class AggregateResult:
    sequences: pd.DataFrame
    per_image: pd.DataFrame
    sequence_csv: Path
    per_image_csv: Path
    per_image_written: bool


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
    progress_json = (
        results_dir
        / f"{cfg.paths.username}_{cfg.paths.camera_id}_{cfg.paths.footage_date}_progress.json"
    )
    return PathsResolved(
        input_dir=input_dir,
        results_root=results_root,
        per_file_root=per_file_root,
        results_dir=results_dir,
        exif_csv=exif_csv,
        final_csv=final_csv,
        per_image_csv=per_image_csv,
        progress_json=progress_json,
    )


# --------------------
# Predict
# --------------------
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

    model = YOLO(str(cfg.predict.model_path))

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
                save_frames=cfg.predict.save_frames,
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


# --------------------
# EXIF
# --------------------
def get_exif(
    cfg: Config,
    include_files: set[str] | None = None,
) -> Tuple[pd.DataFrame, int]:
    p = resolve_paths(cfg)
    p.results_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, str, float]] = []
    avi_count = 0
    for file in sorted(os.listdir(p.input_dir)):
        if include_files is not None and file not in include_files:
            continue
        full = p.input_dir / file
        try:
            if file.lower().endswith((".jpg", ".png")):
                with open(full, "rb") as f:
                    tags = exifread.process_file(f, details=False)
                dt_str = None
                for key in ("EXIF DateTimeOriginal", "Image DateTime"):
                    if key in tags:
                        dt_str = str(tags[key])
                        break
                if dt_str is None:
                    # fallback: file mtime
                    dt_str = dt.datetime.fromtimestamp(full.stat().st_mtime).strftime(
                        "%Y:%m:%d %H:%M:%S"
                    )
                duration = 1.0
                rows.append((file, dt_str, duration))
            elif file.lower().endswith((".mp4", ".mov")):
                probe = ffmpeg.probe(str(full))
                creation = probe.get("format", {}).get("tags", {}).get("creation_time")
                if creation:
                    # strip 'Z' and parse
                    dt_str = dt.datetime.fromisoformat(creation.rstrip("Z")).strftime(
                        "%Y:%m:%d %H:%M:%S"
                    )
                else:
                    dt_str = dt.datetime.fromtimestamp(full.stat().st_mtime).strftime(
                        "%Y:%m:%d %H:%M:%S"
                    )
                duration = float(probe.get("format", {}).get("duration", 0.0))
                rows.append((file, dt_str, duration))
            elif file.lower().endswith((".avi",)):
                avi_count += 1
                continue
        except Exception:
            # Skip problematic file but continue
            continue

    df = pd.DataFrame(rows, columns=["file_name", "date_exif", "duration"])
    df.to_csv(p.exif_csv, index=False)
    return df, avi_count


# --------------------
# Validate (visual)
# --------------------
NAMES = {
    0: "Canis lupus",
    1: "Person",
    2: "Cervus elaphus",
    3: "Capreolus capreolus",
    4: "Sus scrofa",
    5: "Vulpes vulpes",
    6: "Felis silvestris",
    7: "Meles meles",
    8: "Ovis orientalis",
    9: "Vehicle",
    10: "Dog",
    11: "Animal",
    12: "Dama dama",
    13: "Lepus europaeus",
    14: "Martes martes",
    15: "Buteo buteo",
    16: "Lutra lutra",
}


def _draw_annotation(image, labels_path: Path, label_file: str, txs: float = 5.0):
    h, w = image.shape[:2]
    ann = Annotator(image, font="Arial.ttf", pil=False)
    lines = 0
    boxes: list[Tuple[int, int, int, int, float, int]] = []
    with open(labels_path / label_file) as f:
        for line in f.readlines():
            parts = line.rstrip().split(" ")
            if len(parts) < 6:
                continue
            label, x_center, y_center, width, height, conf = parts
            x_min = int(w * max(float(x_center) - float(width) / 2, 0))
            x_max = int(w * min(float(x_center) + float(width) / 2, 1))
            y_min = int(h * max(float(y_center) - float(height) / 2, 0))
            y_max = int(h * min(float(y_center) + float(height) / 2, 1))
            if float(conf) > 0.25:
                boxes.append((x_min, y_min, x_max, y_max, float(conf), int(label)))
            lines += 1

    for x1, y1, x2, y2, conf, cidx in boxes:
        label = f"{NAMES.get(cidx, str(cidx))} {conf:.2f}"
        ann.box_label((x1, y1, x2, y2), label, color=colors(cidx, bgr=True))

    image = ann.result()
    if lines > 1:
        image = cv2.putText(image, str(lines), (round(w / 2), 200), cv2.FONT_HERSHEY_SIMPLEX, txs, (0, 0, 255), 20)
    return image


def validate(cfg: Config, show: bool = True, save_annotated: bool = False) -> None:
    p = resolve_paths(cfg)
    base_dir = p.per_file_root

    # Iterate input files and compose labels path exactly as in predict()
    for idx, file in enumerate(sorted(os.listdir(p.input_dir))):
        stem = file.rsplit(".", 1)[0]
        labels = base_dir / stem / "labels"
        # Skip when there are no label files (dir may exist but be empty)
        if not labels.exists() or not any(labels.glob("*.txt")):
            continue

        # If video frames were saved, use them; else handle still images
        frames_dir = base_dir / stem / f"{stem}_frames"
        if frames_dir.exists():
            for i, frame_name in enumerate(sorted(os.listdir(frames_dir))):
                ann_name = f"{stem}_{frame_name.split('.', 1)[0]}.txt"
                frame = cv2.imread(str(frames_dir / frame_name))
                if frame is None:
                    continue
                # If no annotation exists for this frame, skip gracefully
                if not (labels / ann_name).exists():
                    continue
                vis = _draw_annotation(frame, labels, ann_name)
                if save_annotated:
                    out_dir = base_dir / stem / "annotated"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_dir / frame_name), vis)
                if show:
                    cv2.imshow(f"{idx}_{frame_name}", vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            # Still image: load directly from input dir
            if file.lower().endswith((".jpg", ".png")):
                img_path = p.input_dir / file
                ann_name = f"{stem}.txt"
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                if not (labels / ann_name).exists():
                    continue
                vis = _draw_annotation(img, labels, ann_name)
                if save_annotated:
                    out_dir = base_dir / stem / "annotated"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_dir / file), vis)
                if show:
                    cv2.imshow(f"{idx}_{img_path.name}", vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                # Video without saved frames cannot be visualized here
                continue


# --------------------
# Aggregate to CSV
# --------------------
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
    completed_files: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[tuple[str, str, str, float, int, int, float, int, str]] = []
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
                for line in f.readlines():
                    parts = line.rstrip().split(" ")
                    if len(parts) < 6:
                        continue
                    cidx, x_center, y_center, width, height, conf = parts
                    confidence = float(conf)
                    if confidence > 0.25:
                        class_idx = int(cidx)
                        label_name = NAMES.get(class_idx, str(class_idx))
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
        geometry_json = json.dumps(geometry_entries, separators=(",", ":"))
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
                    geometry_json,
                )
            )
        else:
            rows.append((resolved_name, "EMPTY", 0.0, 0, 0, 0.0, 0, geometry_json))

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
            "frame_geometries",
        ],
    )
    return df


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
    file_df = _collect_file_summary(p.per_file_root, stem_to_name, completed_files)
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
