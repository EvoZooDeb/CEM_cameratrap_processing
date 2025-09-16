# CEM_cameratrap_processing

AI based mammal identification, counting in Central Europe

A small CLI to run a YOLO-based camera trap pipeline: predict detections, extract EXIF, visually validate annotations, and aggregate per-sequence results to CSV.

## Install

```
pip install -e .
# or: pip install .
```

This installs the console command `felis`.

## Quick Start

```
# Use a config file in the repo root
felis run --config .felis.yml --validate --no-show
# Validate and save annotated outputs
felis run --config .felis.yml --validate --save-annotated --no-show

# Or override via CLI flags (no config file needed)
felis predict \
  --input-root /path/to/raw_data \
  --output-root /path/to/results \
  --dir-name Egererdo_Felsotarkany \
  --camera-id FELIS-8 \
  --footage-date 20250518 \
  --model-path /path/to/model.pt \
  --device cuda:0

# Run steps separately
felis exif --config .felis.yml
felis aggregate --config .felis.yml
felis validate --config .felis.yml --no-show
```

## Commands

- `felis predict`: Run YOLO, write YOLO-format labels per file.
- `felis exif`: Extract capture timestamps/durations to CSV.
- `felis aggregate`: Merge labels + EXIF, group image bursts into sequences, export CSV.
- `felis validate`: Visualize annotations with OpenCV windows (for QC).
- `felis validate --save-annotated`: Save overlays to per-file `annotated/` folders under results (works with or without `--no-show`).
- `felis run`: Chain predict → exif → aggregate; optional `--validate`.
  - Options: `--no-show` to suppress windows, `--save-annotated` to write overlays.

Common options (apply to most commands):
- `--input-root`: Base raw data root.
- `--output-root`: Base results root.
- `--dir-name`: Project/survey name.
- `--camera-id`: Camera unit ID.
- `--footage-date`: Date string (YYYYMMDD).
- `--model-path`: Path to YOLO weights (.pt). Used by predict.
- Predict params: `--device` (e.g., `cuda:0`), `--imgsz`, `--conf`, `--iou`, `--save-frames`.

## Configuration

Three sources are supported; later entries override earlier ones:
1) YAML file (pass `--config` or use defaults below)
2) Environment variables
3) CLI flags (highest precedence)

Default config locations:
- `./.felis.yml`
- `~/.config/felis/config.yml`
- Or set `FELIS_CONFIG=/path/to/config.yml`

### Example `.felis.yml`

```
input_root: /home/golah/FELIS_monitoring/raw_data
output_root: /home/golah/FELIS_monitoring/results
username: bela
camera_id: FELIS-8
footage_date: "20250518"
model_path: /home/golah/wolf_camtrap/scripts/runs/detect/train21/weights/best.pt
# Optional predict params
device: cuda:0
imgsz: 1280
conf: 0.25
iou: 0.45
save_frames: false
```

### Environment variables

```
export FELIS_INPUT_ROOT=...
export FELIS_OUTPUT_ROOT=...
export FELIS_USERNAME=...
export FELIS_CAMERA_ID=...
export FELIS_FOOTAGE_DATE=...
export FELIS_MODEL_PATH=...
# Optional
export FELIS_DEVICE=cuda:0
export FELIS_IMGSZ=1280
export FELIS_CONF=0.25
export FELIS_IOU=0.45
export FELIS_SAVE_FRAMES=false
```

## Outputs & Layout

- Per-file labels: `<output_root>/<username>/<camera_id>/<footage_date>/<file_stem>/labels/*.txt`
- EXIF CSV: `<output_root>/<username>/<camera_id>/results/<dir>_<cam>_<date>_exif.csv`
- Final CSV: `<output_root>/<username>/<camera_id>/results/<dir>_<cam>_<date>_results.csv`

## Notes

- Requires `ultralytics` and a valid model weights file. GPU is recommended (`--device cuda:0`).
- `validate` opens GUI windows; use `--no-show` in headless environments.
- AVI metadata is currently skipped in EXIF; MP4/MOV supported via ffmpeg.



```
docker run --rm -u 0 -v ./.felis.yml:/etc/felis/config.yml:ro -v nc_data:/work/raw:ro -v ./results:/work/results -v ./models/best_26.pt:/work/best_26.pt felis:local exif --config /etc/felis/config.yml --username bela --camera-id Q1 --footage-date 16 
docker run --rm -u 0 -v ./.felis.yml:/etc/felis/config.yml:ro -v nc_data:/work/raw:ro -v ./results:/work/results -v ./models/best_26.pt:/work/best_26.pt felis:local predict --config /etc/felis/config.yml --username bela --camera-id Q1 --footage-date 16 
docker run --rm -u 0 -v ./.felis.yml:/etc/felis/config.yml:ro -v nc_data:/work/raw:ro -v ./results:/work/results -v ./models/best_26.pt:/work/best_26.pt felis:local aggregate --config /etc/felis/config.yml --username bela --camera-id Q1 --footage-date 16
```
