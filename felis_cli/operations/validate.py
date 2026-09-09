from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import cv2
from ultralytics.utils.plotting import Annotator, colors

from ..config import Config
from .labels import NAMES
from .paths import resolve_paths


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
