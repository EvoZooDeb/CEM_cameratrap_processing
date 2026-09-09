from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
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
    detections_dir: Path
    progress_json: Path
    two_stage_json: Path


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

    model_path = cfg.predict.model_path
    if cfg.two_stage.strategy == "two_stage":
        detector_weights = {
            "best_27": "best_27.pt",
            "mdv6": "MDV6-yolov10-c.pt",
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
            save_video_frames = (
                cfg.predict.save_frames or cfg.two_stage.strategy == "two_stage"
            )
            results = model.predict(
                full_path,
                # Ultralytics writes individual frames only through its normal save path.
                save=save_video_frames,
                save_frames=save_video_frames,
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
# Two-stage classification (internal intermediate data only)
# --------------------
_DETECTOR_CLASSES = {
    "best_27": {"animal": {0}, "passthrough": {1: "Person", 2: "Vehicle"}},
    "mdv6": {"animal": {0}, "passthrough": {1: "Person", 2: "Vehicle"}},
    "deepfaune_1.4": {"animal": {0}, "passthrough": {1: "Person", 2: "Vehicle"}},
    "best_28": {"animal": {0, 3}, "passthrough": {1: "Person", 2: "Vehicle"}},
}


def _classification_classes(path: Path) -> list[str]:
    """Read the fixed, output-index ordered class list beside a Keras model."""
    classes_path = path.with_suffix(".classes.txt")
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Classifier class list not found: {classes_path}. "
            "Create one line per output class, in model-output order."
        )
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
    classes = [name for name in classes if name]
    if not classes:
        raise ValueError(f"Classifier class list is empty: {classes_path}")
    return classes


def _load_classifier(cfg: Config) -> tuple[Callable[[Any], tuple[str, float]], str]:
    """Load the selected classifier lazily so single-stage installs stay lightweight."""
    classifier = cfg.two_stage.classifier
    if classifier == "deepfaune_classifier":
        try:
            from PytorchWildlife.models import classification as pw_classification
        except ImportError as exc:
            raise RuntimeError(
                "DeepFaune classification requires PytorchWildlife to be installed locally."
            ) from exc

        model = pw_classification.DeepfauneClassifier(device=cfg.predict.device)

        def classify(image: Any) -> tuple[str, float]:
            result = model.single_image_classification(cv2.resize(image, (224, 224)))
            label = str(result["prediction"])
            confidences = result.get("all_confidences", [])
            confidence = next((float(score) for name, score in confidences if name == label), 0.0)
            return _normalise_deepfaune_label(label), confidence

        return classify, "deepfaune"

    weights = cfg.two_stage.models_dir / f"{classifier}.keras"
    if not weights.exists():
        raise FileNotFoundError(f"Classifier weights not found: {weights}")
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "EfficientNet classification requires TensorFlow to be installed locally."
        ) from exc
    classes = _classification_classes(weights)
    model = tf.keras.models.load_model(weights)

    def classify(image: Any) -> tuple[str, float]:
        resized = cv2.resize(image, (380, 380))
        batch = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
        scores = model.predict(batch, verbose=0)[0]
        if len(scores) != len(classes):
            raise ValueError(
                f"Model output has {len(scores)} classes, but "
                f"{weights.with_suffix('.classes.txt')} "
                f"contains {len(classes)} names."
            )
        index = int(np.argmax(scores))
        return classes[index], float(scores[index])

    return classify, "efficientnet"


def _normalise_deepfaune_label(label: str) -> str:
    return {
        "badger": "Meles_meles",
        "red deer": "Cervus_elaphus",
        "cat": "Felis_silvestris",
        "roe deer": "Capreolus_capreolus",
        "dog": "Dog",
        "wolf": "Canis_lupus",
        "mouflon": "Ovis_orientalis",
        "fox": "Vulpes_vulpes",
        "wild boar": "Sus_scrofa",
    }.get(label, label.replace(" ", "_"))


def _square_crop(image: Any, bbox: list[float]) -> Any | None:
    """Crop the centred square used by the original two-stage analysis scripts."""
    height, width = image.shape[:2]
    x_center, y_center, box_width, box_height = bbox
    half_size = max(box_width * width, box_height * height) / 2
    x_center *= width
    y_center *= height
    x_min, x_max = int(x_center - half_size), int(x_center + half_size)
    y_min, y_max = int(y_center - half_size), int(y_center + half_size)
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if x_max > width:
        x_min -= x_max - width
        x_max = width
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if y_max > height:
        y_min -= y_max - height
        y_max = height
    x_min, y_min = max(0, x_min), max(0, y_min)
    crop = image[y_min:y_max, x_min:x_max]
    return crop if crop.size else None


def _source_for_label(p: PathsResolved, media_name: str, label_file: Path) -> Path | None:
    media_path = p.input_dir / media_name
    if media_path.suffix.lower() in {".jpg", ".png"}:
        return media_path
    stem = media_path.stem
    label_stem = label_file.stem
    frames_dir = p.per_file_root / stem / f"{stem}_frames"
    for extension in (".jpg", ".JPG", ".png", ".PNG"):
        candidate = frames_dir / f"{label_stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def classify(cfg: Config, completed_files: list[str] | None = None) -> None:
    """Classify saved detector boxes and persist only a private intermediate mapping."""
    if cfg.two_stage.strategy != "two_stage":
        return
    p = resolve_paths(cfg)
    classify_crop, _ = _load_classifier(cfg)
    detector_classes = _DETECTOR_CLASSES[cfg.two_stage.detector]
    media_names = (
        completed_files if completed_files is not None else sorted(os.listdir(p.input_dir))
    )
    predictions: dict[str, dict[str, Any]] = {}
    for media_name in media_names:
        stem = Path(media_name).stem
        labels_dir = p.per_file_root / stem / "labels"
        if not labels_dir.exists():
            continue
        for label_file in sorted(labels_dir.glob("*.txt")):
            source = _source_for_label(p, media_name, label_file)
            image = cv2.imread(str(source)) if source else None
            if image is None:
                continue
            for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines()):
                parts = line.split()
                if len(parts) < 6:
                    continue
                class_id = int(float(parts[0]))
                key = f"{stem}/labels/{label_file.name}:{line_number}"
                if class_id in detector_classes["passthrough"]:
                    predictions[key] = {
                        "label": detector_classes["passthrough"][class_id],
                        "confidence": float(parts[5]),
                    }
                elif class_id in detector_classes["animal"]:
                    crop = _square_crop(image, [float(value) for value in parts[1:5]])
                    if crop is not None:
                        label, confidence = classify_crop(crop)
                        predictions[key] = {"label": label, "confidence": confidence}
    p.results_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = p.two_stage_json.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(p.two_stage_json)


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
                ann_name = f"{Path(frame_name).stem}.txt"
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
