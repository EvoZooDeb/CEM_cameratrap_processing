from __future__ import annotations

import datetime as dt
import logging
import os
from typing import List, Tuple

import exifread
import ffmpeg
import pandas as pd

from ..config import Config
from ..verbosity import detail, status
from .paths import resolve_paths


LOG = logging.getLogger(__name__)


def get_exif(
    cfg: Config,
    include_files: set[str] | None = None,
) -> Tuple[pd.DataFrame, int]:
    p = resolve_paths(cfg)
    p.results_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, str, float]] = []
    avi_count = 0
    media_names = [
        file
        for file in sorted(os.listdir(p.input_dir))
        if include_files is None or file in include_files
    ]
    status(LOG, "EXIF extraction: %d media file(s) to inspect.", len(media_names))
    for media_index, file in enumerate(media_names, start=1):
        status(LOG, "Extracting metadata [%d/%d]: %s", media_index, len(media_names), file)
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
                detail(LOG, "Skipping unsupported AVI metadata: %s", file)
                continue
        except Exception as exc:
            # Skip problematic file but continue
            LOG.warning("Could not extract metadata from %s: %s", file, exc)
            continue

    df = pd.DataFrame(rows, columns=["file_name", "date_exif", "duration"])
    df.to_csv(p.exif_csv, index=False)
    status(
        LOG,
        "EXIF extraction complete: %d record(s), %d AVI skipped; wrote %s",
        len(df),
        avi_count,
        p.exif_csv,
    )
    return df, avi_count

