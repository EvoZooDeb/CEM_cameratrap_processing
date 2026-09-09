from __future__ import annotations

import datetime as dt
import os
from typing import List, Tuple

import exifread
import ffmpeg
import pandas as pd

from ..config import Config
from .paths import resolve_paths


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
