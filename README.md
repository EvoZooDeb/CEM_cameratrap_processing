# CEM_cameratrap_processing

FELIS is a command-line pipeline for YOLO-based identification and counting of
Central European mammals in camera-trap media. It processes still images and
videos, extracts capture metadata, aggregates detections into camera-trap
sequences, and supports visual quality control.

## Install

```bash
pip install -e .
# or
pip install .
```

This installs the `felis` command.

## Configuration and input layout

Every command loads the same required configuration values:

- `input_root`: raw-data base directory.
- `output_root`: results base directory.
- `username`: survey or project name.
- `camera_id`: camera unit identifier.
- `footage_date`: date directory to process, commonly `YYYYMMDD`.
- `model_path`: YOLO weights path. It is required by the shared configuration,
  even for commands that do not load the model.

Media must be directly inside this directory:

```text
<input_root>/<username>/files/cameratrap/<camera_id>/<footage_date>/
```

Prediction recognizes `.jpg`, `.png`, `.mp4`, `.avi`, and `.mov` files.
Metadata extraction handles JPG, PNG, MP4, and MOV; it skips AVI files.

Values are resolved in this order, from lowest to highest precedence:

1. YAML configuration.
2. `FELIS_*` environment variables.
3. Command-line options.

Pass YAML with `--config`. Without it, FELIS checks `FELIS_CONFIG`, then
`./.felis.yml`, then `~/.config/felis/config.yml`.

```yaml
# .felis.yml
input_root: /work/raw
output_root: /work/results
username: admin
camera_id: test
footage_date: "20250518"
model_path: /work/best_26.pt
device: cuda:0
imgsz: 1280
conf: 0.25
iou: 0.45
save_frames: false
```

The environment variable equivalents are `FELIS_INPUT_ROOT`,
`FELIS_OUTPUT_ROOT`, `FELIS_USERNAME`, `FELIS_CAMERA_ID`,
`FELIS_FOOTAGE_DATE`, `FELIS_MODEL_PATH`, `FELIS_DEVICE`, `FELIS_IMGSZ`,
`FELIS_CONF`, `FELIS_IOU`, and `FELIS_SAVE_FRAMES`.

## Commands

Use `felis <command> --help` for the complete option reference.

### `felis predict`

Runs the configured YOLO model for all supported media in the selected input
directory. It writes YOLO-format labels, including confidence values, for each
detected image or video frame. Use `--device`, `--imgsz`, `--conf`, and `--iou`
to control inference. `--save-frames` saves video frames and is required for
later video validation.

```bash
felis predict --config .felis.yml
felis predict --config .felis.yml --device cpu --conf 0.35
```

Labels are written below:

```text
<output_root>/<username>/<camera_id>/<footage_date>/<media_stem>/labels/
```

### `felis exif`

Creates the EXIF CSV for the selected input. JPG and PNG timestamps come from
`EXIF DateTimeOriginal` or `Image DateTime`, with file modification time as a
fallback; still-image duration is `1.0`. MP4 and MOV creation time and duration
come from FFmpeg metadata, with the same timestamp fallback. AVI files and
unreadable files are skipped.

```bash
felis exif --config .felis.yml
```

### `felis aggregate`

Requires prediction labels and the EXIF CSV. It summarizes detections per media
file, groups captures that occur within 10 seconds of the first capture in a
sequence, and writes a sequence-level CSV. The selected sequence label is the
most frequent non-empty label; count fields use the built-in `0.25` confidence
cutoff. It also writes one detection-detail JSON file per media item.

```bash
felis aggregate --config .felis.yml
felis aggregate --config .felis.yml --save-per-image
```

`--save-per-image` additionally writes the per-media summary CSV.

### `felis validate`

Draws saved YOLO detections above `0.25` confidence on input images or saved
video frames. By default, each overlay opens in an OpenCV window; press a key
to continue. Use `--no-show` in headless environments and `--save-annotated`
to write overlays. Video validation only works when `predict` used
`--save-frames`.

```bash
felis validate --config .felis.yml --no-show --save-annotated
```

Saved overlays go to:

```text
<output_root>/<username>/<camera_id>/<footage_date>/<media_stem>/annotated/
```

### `felis run`

Runs `predict`, `exif`, and `aggregate` in that order. Add `--validate` to run
visual validation after aggregation. The command also accepts all prediction
options plus `--save-per-image`, `--no-show`, and `--save-annotated`.

```bash
felis run --config .felis.yml --save-per-image
felis run --config .felis.yml --validate --no-show --save-annotated
```

On `SIGTERM`, or when the path in `FELIS_CANCEL_FILE` exists, FELIS stops after
the current processing boundary and writes EXIF and per-media results only for
media that completed prediction. This partial finalization always writes the
per-media CSV.

## Outputs

All paths below are relative to:

```text
<output_root>/<username>/<camera_id>/<footage_date>/
```

- `<media_stem>/labels/*.txt`: YOLO prediction labels.
- `<media_stem>/<media_stem>_frames/`: saved video frames, when enabled.
- `<media_stem>/annotated/`: validation overlays, when enabled.
- `results/<username>_<camera_id>_<footage_date>_exif.csv`: timestamps and durations.
- `results/<username>_<camera_id>_<footage_date>_results.csv`: sequence summary.
- `results/<username>_<camera_id>_<footage_date>_per_image.csv`: optional per-media summary.
- `results/detections/<sha256-media-name>.json`: detailed detection geometry,
  confidence, and video timing data when available.
- `results/<username>_<camera_id>_<footage_date>_progress.json`: completed
  media names for the current or most recent prediction run.

## Notes

- `predict` and `run` require valid Ultralytics-compatible YOLO weights. A GPU
  is recommended, but `--device cpu` is supported.
- `aggregate` must follow a successful `exif` run because the EXIF CSV is required.
- `validate` opens GUI windows unless `--no-show` is supplied.

## Docker

```bash
docker build -t felis:local .

docker run --rm \
  -v ./.felis.yml:/etc/felis/config.yml:ro \
  -v nc_data:/work/raw:ro \
  -v ./results:/work/results \
  -v ./models/best_26.pt:/work/best_26.pt \
  felis:local run --config /etc/felis/config.yml --no-show
```

The container remaps its internal `appuser` to match the owner of `/work`
mounts. Override this with `-e FELIS_UID=<uid> -e FELIS_GID=<gid>` when needed.
For raw-data mounts readable only by root, use `-e FELIS_FORCE_ROOT=1`; the
entrypoint restores ownership of `/work/results` after the run.
