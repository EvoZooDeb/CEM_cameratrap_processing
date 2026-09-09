from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import cv2

from ..config import Config
from .paths import PathsResolved, resolve_paths

_DETECTOR_CLASSES = {
    "best_27": {"animal": {0}, "passthrough": {1: "Person", 2: "Vehicle"}},
    "mdv6": {"animal": {0}, "passthrough": {1: "Person", 2: "Vehicle"}},
    "deepfaune_1.4": {"animal": {0}, "passthrough": {1: "Person", 2: "Vehicle"}},
    "best_28": {"animal": {0, 3}, "passthrough": {1: "Person", 2: "Vehicle"}},
}


def _is_cpu_device(device: str) -> bool:
    """Return whether a configured inference device explicitly selects the CPU."""
    return device.strip().lower() in {"cpu", "cpu:0", "/cpu:0", "/device:cpu:0"}


def _prepare_backend_environment(device: str) -> None:
    """Disable CUDA discovery before importing an ML backend when CPU was requested."""
    if not _is_cpu_device(device):
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        current_log_level = int(os.environ.get("TF_CPP_MIN_LOG_LEVEL", "0"))
    except ValueError:
        current_log_level = 0
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(max(current_log_level, 1))


def _tensorflow_device(device: str) -> str:
    """Translate documented FELIS device names to TensorFlow device names."""
    normalized = device.strip().lower()
    if _is_cpu_device(normalized):
        return "/CPU:0"

    match = re.fullmatch(r"(?:cuda|gpu)(?::(\d+))?", normalized)
    if match:
        return f"/GPU:{match.group(1) or '0'}"

    raise ValueError(
        f"Unsupported TensorFlow device '{device}'. Use 'cpu' or 'cuda:<index>'."
    )


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
    _prepare_backend_environment(cfg.predict.device)
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
    tensorflow_device = _tensorflow_device(cfg.predict.device)
    if _is_cpu_device(cfg.predict.device):
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError(
                "TensorFlow was initialized before FELIS could disable GPU devices."
            ) from exc
    classes = _classification_classes(weights)
    with tf.device(tensorflow_device):
        model = tf.keras.models.load_model(weights)

    def classify(image: Any) -> tuple[str, float]:
        resized = cv2.resize(image, (380, 380))
        batch = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
        with tf.device(tensorflow_device):
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
    frame_stem = label_stem.removeprefix(f"{stem}_")
    frames_dir = p.per_file_root / stem / f"{stem}_frames"
    for extension in (".jpg", ".JPG", ".png", ".PNG"):
        candidate = frames_dir / f"{frame_stem}{extension}"
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
