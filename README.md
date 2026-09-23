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

For two-stage classifiers, install the local runtime dependencies too:

```bash
pip install -e '.[two-stage]'
pip uninstall -y opencv-python opencv-python-headless
pip install --no-deps opencv-python==5.0.0.93
python scripts/check_runtime_dependencies.py --opencv-variant gui
```

The explicit OpenCV selection is required because PytorchWildlife dependencies
request the headless wheel while FELIS and Ultralytics request the desktop
wheel. Only the desktop variant is retained for direct CLI use, so
`felis validate` can open native windows. The Docker build performs the inverse
selection and retains only `opencv-python-headless`. This installs the `felis`
command.

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
strategy: two_stage
detector: best_27
classifier: deepfaune_classifier
models_dir: /work/models
device: cuda:0
imgsz: 1280
conf: 0.25
iou: 0.45
save_frames: false
```

The environment variable equivalents are `FELIS_INPUT_ROOT`,
`FELIS_OUTPUT_ROOT`, `FELIS_USERNAME`, `FELIS_CAMERA_ID`,
`FELIS_FOOTAGE_DATE`, `FELIS_MODEL_PATH`, `FELIS_DEVICE`, `FELIS_IMGSZ`,
`FELIS_CONF`, `FELIS_IOU`, and `FELIS_SAVE_FRAMES`. Two-stage settings are
`FELIS_STRATEGY`, `FELIS_DETECTOR`, `FELIS_CLASSIFIER`, and `FELIS_MODELS_DIR`.

## Two-stage models

`two_stage` is the default strategy. It first detects broad categories and then
classifies animal crops to species level. Supported detector/classifier pairs are
`best_27`, `mdv6`, or `deepfaune` with `deepfaune_classifier` or `4_camtrap`,
and `best_28` with `2_artiodactyla` or `2_carnivora`.

All weights must be present locally in `models_dir`: `best_27.pt`,
`md_v1000.0.0-redwood.pt`, `deepfaune-yolov8s_960.pt`, `best_28.pt`, and the
applicable `.keras` classifier file.
Each Keras classifier additionally needs a same-named `.classes.txt` file with
one output class per line in model-output order. The DeepFaune classifier weight
must be stored at:

```text
models/checkpoints/deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt
```

Download and verify the PytorchWildlife 1.3 checkpoint once on the host:

```bash
mkdir -p models/checkpoints
curl -fL --retry 3 \
  -o models/checkpoints/deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt \
  https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.3/deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt
echo "b1d31940067fe7e7e973b093ad787d5463a7c88ad81a8341859667e61a2e4391  models/checkpoints/deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt" \
  | sha256sum --check
```

The Docker `models` volume exposes this file at `/work/models/checkpoints`.
FELIS refuses to run the DeepFaune classifier when it is absent instead of
downloading a temporary copy inside the container.

## Commands

Use `felis <command> --help` for the complete option reference.

### Output verbosity

Verbosity is handled consistently by every command. With no verbosity flag,
FELIS reports pipeline stages, progress for each media file, and final summaries.
Use the global flags before the subcommand:

```bash
felis -q classify --config .felis.yml       # warnings and errors only
felis -v classify --config .felis.yml       # per-frame/label/crop details
felis -vv classify --config .felis.yml      # backend and configuration diagnostics
felis --log-file felis.log run --config .felis.yml
```

Normal and quiet modes suppress Ultralytics' frame-by-frame output. `-v` enables
that backend progress, while `-vv` also exposes otherwise noisy third-party
diagnostics. Messages written to a `--log-file` include FELIS detail records even
when the console is quiet.

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

### `felis classify`

Runs only the second stage against previously written detector labels. It writes
private intermediate data used by `aggregate`; public result CSV and detection
JSON formats are unchanged.

```bash
felis classify --config .felis.yml
felis classify --config .felis.yml --device cpu
```

The command honors the YAML `device` value and accepts `--device` as a CLI
override. When CPU is selected, CUDA devices are hidden before the classifier
backend is loaded.

### `felis exif`

Creates the media metadata CSV for the selected input. JPG and PNG timestamps
come from `EXIF DateTimeOriginal` or `Image DateTime`, with file modification
time as a fallback; still-image duration is `1.0`. MP4 and MOV creation time and
duration come from FFmpeg metadata, with the same timestamp fallback. AVI files
and unreadable files are skipped.

```bash
felis exif --config .felis.yml
```

### `felis aggregate`

Requires prediction labels and the media metadata CSV. It summarizes detections
per media and species, groups captures that occur within 10 seconds of the first
capture in an event, and writes one result row per event and species. Count
fields use the built-in `0.25` detector-confidence cutoff. It also writes one
detection-detail JSON file per non-empty media item.

```bash
felis aggregate --config .felis.yml
```

The per-media result CSV is always written because it is also the media search
index.

### `felis validate`

Draws final schema-v2 detections on input images or saved video frames. By
default, each overlay opens in an OpenCV window; press a key to continue. Use
`--no-show` in headless environments and `--save-annotated` to write overlays.
Video validation only works when `predict` used `--save-frames`.

```bash
felis validate --config .felis.yml --no-show --save-annotated
```

Saved overlays go to:

```text
<output_root>/<username>/<camera_id>/<footage_date>/<media_stem>/annotated/
```

### `felis run`

Runs `predict`, `exif`, `classify`, and `aggregate` in that order for the default
two-stage strategy; `single_stage` omits `classify`. Add `--validate` to run
visual validation after aggregation. The command also accepts all prediction
options plus `--no-show` and `--save-annotated`.

```bash
felis run --config .felis.yml
felis run --config .felis.yml --validate --no-show --save-annotated
felis run --config .felis.yml --package-artifacts
```

`--package-artifacts` creates and verifies `artifacts.zip`, omits video frame
directories, adds a checksum manifest and `results/detections.zip`, then removes
the unpacked output. If packaging or verification fails, the unpacked files are
left in place and the command exits with an error.

On `SIGTERM`, or when the path in `FELIS_CANCEL_FILE` exists, FELIS stops after
the current processing boundary and writes metadata, media results, and event
results only for media that completed prediction.

## Outputs

All paths below are relative to:

```text
<output_root>/<username>/<camera_id>/<footage_date>/
```

- `<media_stem>/labels/*.txt`: YOLO prediction labels.
- `<media_stem>/<media_stem>_frames/`: saved video frames, when enabled.
- `<media_stem>/annotated/`: validation overlays, when enabled.
- `results/<username>_<camera_id>_<footage_date>_media_metadata.csv`: timestamps and durations.
- `results/<username>_<camera_id>_<footage_date>_event_results.csv`: one row per event and species.
- `results/<username>_<camera_id>_<footage_date>_media_results.csv`: one row per media and species.
- `results/detections/<sha256-media-name>.json`: schema-v2 detection geometry,
  separate detector/classifier confidence, and video timing data when available.
- `results/<username>_<camera_id>_<footage_date>_progress.json`: completed
  media names for the current or most recent prediction run.

The published CSV files and detection JSON schema do not depend on the selected
strategy. Two-stage intermediate classifications are deliberately private.

## Notes

- `predict` and `run` require valid Ultralytics-compatible YOLO weights. A GPU
  is recommended, but `--device cpu` is supported.
- `aggregate` must follow a successful `exif` run because the media metadata CSV is required.
- `validate` opens GUI windows unless `--no-show` is supplied.

## Docker

The published Docker image targets Linux `amd64` and uses CPU-only PyTorch.
Model weights are not baked into the image; they remain on the host and are
mounted read-only at runtime.

```bash
docker buildx build --platform linux/amd64 --load -t felis:local .
```

With nextcloud data

```
docker run --rm \
  -v ./.felis.docker.yml:/etc/felis/config.yml:ro \
  -v nc_data:/work/raw:ro \
  -v ./results:/work/results \
  -v ./models:/work/models:ro \
  felis:local run --config /etc/felis/config.yml --no-show
```

with local data

```
docker run --rm \
  -v ./.felis.docker.yml:/etc/felis/config.yml:ro \
  -v ./raw:/work/raw:ro \
  -v ./results:/work/results \
  -v ./models:/work/models:ro \
  felis:local run --config /etc/felis/config.yml --no-show
```

The first clean build downloads the full machine-learning runtime. Subsequent
builds reuse that dependency layer as long as `requirements*.txt` is unchanged,
so changes limited to `felis.py` or `felis_cli/` do not reinstall PyTorch,
TensorFlow, or PytorchWildlife.

PytorchWildlife 1.3.0 omits some imports and checkpoint-deserialization
requirements from its package metadata. They are pinned explicitly by this
project, including `librosa`, `soundfile`, and `dill`.

The container remaps its internal `appuser` to match the owner of `/work`
mounts. Override this with `-e FELIS_UID=<uid> -e FELIS_GID=<gid>` when needed.
For raw-data mounts readable only by root, use `-e FELIS_FORCE_ROOT=1`; the
entrypoint restores ownership of `/work/results` after the run.
