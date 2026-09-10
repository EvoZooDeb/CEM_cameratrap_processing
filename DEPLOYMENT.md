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
# Optional predict params
device: cpu  # or cuda:0 if GPU
imgsz: 1280
conf: 0.25
iou: 0.45
save_frames: false
YAML
```

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

## Docker (optional)

Containerize the CLI for isolation and portability; call from PHP/cron:

```
docker run --rm \
  -v /home/USER/FELIS_monitoring/raw_data:/data/raw:ro \
  -v /home/USER/FELIS_monitoring/results:/data/out \
  -v /etc/camtrap/config.yml:/etc/camtrap/config.yml:ro \
  --gpus all \  # if using NVIDIA GPUs
  your-image felis run --config /etc/camtrap/config.yml --no-show
```

Replace `USER` and paths to match your environment.
