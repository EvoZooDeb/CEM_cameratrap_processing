from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics.utils.plotting import Annotator, colors

from ..config import Config
from ..verbosity import detail, status
from .aggregate import _detection_file_path
from .paths import resolve_paths


LOG = logging.getLogger(__name__)


def _draw_annotation(image, detections: list[dict[str, Any]], txs: float = 5.0):
    h, w = image.shape[:2]
    ann = Annotator(image, font="Arial.ttf", pil=False)
    for detection in detections:
        x_center, y_center, width, height = detection["bbox"]
        x_min = int(w * max(float(x_center) - float(width) / 2, 0))
        x_max = int(w * min(float(x_center) + float(width) / 2, 1))
        y_min = int(h * max(float(y_center) - float(height) / 2, 0))
        y_max = int(h * min(float(y_center) + float(height) / 2, 1))
        detector_confidence = float(detection["detector_confidence"])
        classification_confidence = detection.get("classification_confidence")
        score = (
            f"cls {float(classification_confidence):.2f} det {detector_confidence:.2f}"
            if classification_confidence is not None
            else f"det {detector_confidence:.2f}"
        )
        class_id = int(detection["detector_class_id"])
        label = f"{detection['label']} {score}"
        ann.box_label((x_min, y_min, x_max, y_max), label, color=colors(class_id, bgr=True))

    image = ann.result()
    if len(detections) > 1:
        image = cv2.putText(
            image,
            str(len(detections)),
            (round(w / 2), 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            txs,
            (0, 0, 255),
            20,
        )
    return image


def validate(cfg: Config, show: bool = True, save_annotated: bool = False) -> None:
    p = resolve_paths(cfg)
    base_dir = p.per_file_root

    media_names = sorted(os.listdir(p.input_dir))
    annotated_count = 0
    status(LOG, "Validation: %d media file(s) to inspect.", len(media_names))

    # Iterate input files and compose labels path exactly as in predict()
    for idx, file in enumerate(media_names):
        status(LOG, "Validating [%d/%d]: %s", idx + 1, len(media_names), file)
        stem = file.rsplit(".", 1)[0]
        detection_path = _detection_file_path(p.detections_dir, file)
        if not detection_path.exists():
            continue
        document = json.loads(detection_path.read_text(encoding="utf-8"))
        detections = document.get("detections", [])

        # If video frames were saved, use them; else handle still images
        frames_dir = base_dir / stem / f"{stem}_frames"
        if frames_dir.exists():
            for i, frame_name in enumerate(sorted(os.listdir(frames_dir))):
                frame_stem = Path(frame_name).stem
                frame = cv2.imread(str(frames_dir / frame_name))
                if frame is None:
                    continue
                frame_detections = [item for item in detections if item.get("frame") == frame_stem]
                if not frame_detections:
                    continue
                vis = _draw_annotation(frame, frame_detections)
                annotated_count += 1
                detail(LOG, "Rendered annotated frame: %s", frame_name)
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
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                vis = _draw_annotation(img, detections)
                annotated_count += 1
                detail(LOG, "Rendered annotated image: %s", file)
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
                detail(LOG, "No saved frames available for video: %s", file)
                continue
    status(LOG, "Validation complete: %d annotated image(s)/frame(s).", annotated_count)


# --------------------
