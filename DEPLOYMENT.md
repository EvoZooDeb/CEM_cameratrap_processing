# Production Deployment (Cron + PHP)

This guide shows how to install and run the CLI in production, called from a PHP script that is triggered by cron.

## Install

1) Create a dedicated virtualenv and install the package:

```
sudo mkdir -p /opt/camtrap
sudo chown $(whoami) /opt/camtrap
python3 -m venv /opt/camtrap/.venv
/opt/camtrap/.venv/bin/pip install --upgrade pip
# From repo root
/opt/camtrap/.venv/bin/pip install '.[two-stage]'
/opt/camtrap/.venv/bin/pip check
```

The package pins the tested runtime versions, including the TensorFlow/Keras pair and
the setuptools version required by PytorchWildlife's yolov5 dependency. Upgrade these
pins together only after testing every supported detector/classifier pair.

2) System packages:

```
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1
# Optional GPU: install NVIDIA driver + CUDA if using --device cuda:0
```

3) Configuration:

```
sudo mkdir -p /etc/camtrap
sudo tee /etc/camtrap/config.yml > /dev/null <<'YAML'
input_root: /home/USER/FELIS_monitoring/raw_data
output_root: /home/USER/FELIS_monitoring/results
username: bela
camera_id: FELIS-8
footage_date: "20250518"
model_path: /home/USER/wolf_camtrap/scripts/runs/detect/train21/weights/best.pt
strategy: single_stage
# Optional predict params
device: cpu  # or cuda:0 if GPU
imgsz: 1280
conf: 0.25
iou: 0.45
save_frames: false
YAML
```

The example uses `single_stage`, so `model_path` is the detector that FELIS
loads. For the default `two_stage` strategy, also set `detector`, `classifier`,
and `models_dir`; see the model matrix and required filenames in `README.md`.

## Paths

- CLI binary: `/opt/camtrap/.venv/bin/felis`
- Config file: `/etc/camtrap/config.yml`
- Use absolute paths in cron and PHP.

## Wrapper Script (recommended)

Create a safe wrapper to handle virtualenv activation, locking, and logging.

```
sudo tee /usr/local/bin/camtrap-run > /dev/null <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
LOCK=/var/lock/camtrap.lock
LOG_DIR=/var/log/camtrap
mkdir -p "$LOG_DIR"
exec {fd}>$LOCK || exit 1
if ! flock -n "$fd"; then
  echo "$(date -Is) Another run is in progress" >> "$LOG_DIR/runner.log"
  exit 0
fi
VENV=/opt/camtrap/.venv
CLI=$VENV/bin/felis
CONFIG=/etc/camtrap/config.yml
DEVICE=${DEVICE:-cpu}   # override with env if needed
START=$(date -Is)
echo "$START Starting pipeline" >> "$LOG_DIR/pipeline.log"
set +e
"$CLI" run --config "$CONFIG" --device "$DEVICE" --no-show >> "$LOG_DIR/pipeline.log" 2>&1
RC=$?
END=$(date -Is)
echo "$END Finished with code $RC" >> "$LOG_DIR/pipeline.log"
exit $RC
BASH
sudo chmod +x /usr/local/bin/camtrap-run
```

Notes:
- For GPU: add `export CUDA_VISIBLE_DEVICES=0` before calling the CLI.
- Consider adding PATH/LD_LIBRARY_PATH if CUDA libs aren’t found for cron user.

## PHP Integration

Minimal PHP wrapper that invokes the shell script and captures output.

```php
<?php
$cmd = '/usr/local/bin/camtrap-run';
$descriptorSpec = [
  0 => ["pipe", "r"],
  1 => ["pipe", "w"],
  2 => ["pipe", "w"],
];
$proc = proc_open($cmd, $descriptorSpec, $pipes);
if (!is_resource($proc)) {
    error_log("camtrap: failed to start");
    exit(1);
}
fclose($pipes[0]);
$stdout = stream_get_contents($pipes[1]); fclose($pipes[1]);
$stderr = stream_get_contents($pipes[2]); fclose($pipes[2]);
$status = proc_close($proc);
file_put_contents('/var/log/camtrap/php-trigger.log',
    date('c')." status=$status\nSTDOUT:\n$stdout\nSTDERR:\n$stderr\n",
    FILE_APPEND
);
if ($status !== 0) {
    // handle alerting here (email, webhook, etc.)
    exit($status);
}
```

You can also use a one-liner:

```php
exec('/usr/local/bin/camtrap-run 2>&1', $out, $code);
```

## Cron Setup

Use the PHP CLI or call the wrapper directly. Always use absolute paths.

```
# Run daily at 02:00 via PHP
0 2 * * * /usr/bin/php /opt/camtrap/run_camtrap.php

# Or call the wrapper directly (no PHP intermediary)
0 2 * * * /usr/local/bin/camtrap-run
```

## Environment & Permissions

- Ensure the cron user can read input directories, model weights, and config; and write to results and logs.
- For GPU: `nvidia-smi` should work for the cron user; export CUDA env vars in the wrapper if needed.
- Headless servers: pass `--no-show` (already in wrapper) or avoid `--validate`.

## Operational Tips

- Logging: logs under `/var/log/camtrap/`; set up logrotate if needed.
- Concurrency: wrapper lockfile avoids overlapping runs.
- Timeouts: wrap call with `timeout 6h /usr/local/bin/camtrap-run` if desired.
- Health checks: verify final CSV exists/has size; alert otherwise.

## Docker Deployment

The project publishes a Linux `amd64`, CPU-only image to GitHub Container
Registry. Model weights and camera-trap data are not included in the image.
Docker Engine is sufficient; no Python installation is needed on the host.
Building locally also requires the Docker Buildx plugin.

### Get the image

Pull the image built from the default branch:

```bash
docker pull ghcr.io/evozoodeb/cem_cameratrap_processing:latest
```

If the package is private, authenticate first with a GitHub token that has
`read:packages` permission:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u GITHUB_USER --password-stdin
```

Alternatively, build the current checkout locally:

```bash
docker buildx build --platform linux/amd64 --load -t felis:local .
```

Use `felis:local` instead of the GHCR image name in the commands below when
running a local build.

### Container configuration

Paths in the YAML file must be the paths visible **inside the container**, not
host paths. Save this as `/etc/camtrap/config.yml` (or another host path):

```yaml
input_root: /work/raw
output_root: /work/results
username: bela
camera_id: FELIS-8
footage_date: "20250518"
model_path: /work/models/best.pt
strategy: single_stage
models_dir: /work/models
device: cpu
imgsz: 1280
conf: 0.25
iou: 0.45
save_frames: false
```

For `two_stage`, change `strategy` and configure a supported pair, for example:

```yaml
strategy: two_stage
detector: best_27
classifier: deepfaune_classifier
models_dir: /work/models
```

Place every required weight and `.classes.txt` file under the mounted host
models directory. The DeepFaune classifier checkpoint has the special path
`models/checkpoints/deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt`; download
and checksum it as documented in `README.md`.

### Run with host directories

Create the writable output directory before starting the container:

```bash
mkdir -p /home/USER/FELIS_monitoring/results

docker run --rm --init --stop-timeout 120 \
  -v /etc/camtrap/config.yml:/etc/felis/config.yml:ro \
  -v /home/USER/FELIS_monitoring/raw_data:/work/raw:ro \
  -v /home/USER/FELIS_monitoring/results:/work/results \
  -v /home/USER/FELIS_monitoring/models:/work/models:ro \
  ghcr.io/evozoodeb/cem_cameratrap_processing:latest \
  run --config /etc/felis/config.yml --no-show
```

The entrypoint accepts the shorthand `run`; spelling out `felis run` also works.
It normally changes the container user to match the owner of `/work/results`,
so generated files retain useful host ownership. Override detection when needed
with `-e FELIS_UID=$(id -u) -e FELIS_GID=$(id -g)`. If the raw-data mount is
readable only by root, add `-e FELIS_FORCE_ROOT=1`; after the run, the entrypoint
restores ownership below `/work/results`.

For a Docker-managed Nextcloud data volume, replace the raw-data mount with:

```bash
-v nc_data:/work/raw:ro
```

The configured `input_root` remains `/work/raw`. Its content must still follow
FELIS's expected layout:
`<input_root>/<username>/files/cameratrap/<camera_id>/<footage_date>/`.

### Run from cron

Create `/usr/local/bin/camtrap-docker-run`:

```bash
#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE=ghcr.io/evozoodeb/cem_cameratrap_processing:latest
readonly CONFIG=/etc/camtrap/config.yml
readonly RAW=/home/USER/FELIS_monitoring/raw_data
readonly RESULTS=/home/USER/FELIS_monitoring/results
readonly MODELS=/home/USER/FELIS_monitoring/models
readonly LOG_DIR=/var/log/camtrap
readonly LOCK=/var/lock/camtrap-docker.lock

mkdir -p "$LOG_DIR" "$RESULTS"
exec {lock_fd}>"$LOCK"
if ! flock -n "$lock_fd"; then
  echo "$(date -Is) Another run is in progress" >> "$LOG_DIR/runner.log"
  exit 0
fi

docker run --rm --init --stop-timeout 120 \
  -v "$CONFIG:/etc/felis/config.yml:ro" \
  -v "$RAW:/work/raw:ro" \
  -v "$RESULTS:/work/results" \
  -v "$MODELS:/work/models:ro" \
  "$IMAGE" run --config /etc/felis/config.yml --no-show \
  >> "$LOG_DIR/pipeline.log" 2>&1
```

Install it and schedule it just like the native wrapper:

```bash
sudo chmod +x /usr/local/bin/camtrap-docker-run
```

```cron
0 2 * * * /usr/local/bin/camtrap-docker-run
```

The cron user must be allowed to access the Docker daemon and all mounted host
paths. Membership in the `docker` group is effectively root-level access; use a
root-owned wrapper and root cron entry if that better fits the host's security
policy. Do not run `docker pull` automatically in the processing job: update the
image separately, verify it, then let the next scheduled run use it.

### Operations and limitations

- The published image cannot use NVIDIA/CUDA; use the native installation or
  build and test a separate CUDA image for GPU inference.
- `SIGTERM` triggers FELIS partial finalization. The generous stop timeout gives
  the current processing boundary time to finish before Docker sends `SIGKILL`.
- Use `docker logs` only for a named, non-`--rm` container. The cron example
  deliberately redirects application output to `/var/log/camtrap/pipeline.log`.
- The repository currently has no `compose.yml`; the explicit `docker run`
  command above is the supported deployment definition.
