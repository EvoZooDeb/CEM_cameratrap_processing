# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE} AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# System dependencies for OpenCV, ffmpeg, and scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      gosu \
      libgl1 \
      libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

# Create virtual environment. The cache mount speeds up pip upgrades without
# leaving downloaded wheels in the final image.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv "$VENV_PATH" && \
    "$VENV_PATH"/bin/pip install --upgrade pip

WORKDIR /app

# Install the expensive CPU-only ML runtime before copying application source.
# This layer remains cached when only felis.py or felis_cli changes.
COPY requirements.txt requirements-two-stage.txt requirements-torch-cpu.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: \
      --index-url https://download.pytorch.org/whl/cpu \
      -r requirements-torch-cpu.txt && \
    pip install --only-binary=soundfile \
      -r requirements.txt \
      -r requirements-two-stage.txt && \
    TF_CPP_MIN_LOG_LEVEL=2 python -c \
      "import keras, librosa, soundfile, tensorflow; import torch, torchaudio, torchvision; from PytorchWildlife.models import classification; assert torch.version.cuda is None; assert not tensorflow.config.list_physical_devices('GPU'); assert torch.__version__.startswith('2.11.0'); assert torchvision.__version__.startswith('0.26.0'); assert torchaudio.__version__.startswith('2.11.0')"

# PytorchWildlife has runtime-only requirements that are not declared in its
# package metadata. Keep this small layer independent from the ML runtime.
COPY requirements-pytorchwildlife-runtime.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-pytorchwildlife-runtime.txt && \
    python -c "import dill"

# Copy project files and install the CLI package
COPY pyproject.toml README.md ./
COPY felis.py ./
COPY felis_cli ./felis_cli
COPY scripts/check_dependency_sync.py ./scripts/check_dependency_sync.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python scripts/check_dependency_sync.py && \
    pip install --no-deps . && \
    pip check && \
    python -c \
      "from importlib.metadata import distributions; packages = sorted({dist.metadata['Name'] for dist in distributions() if dist.metadata['Name'] and dist.metadata['Name'].lower().startswith('nvidia-') and dist.metadata['Name'].lower() != 'nvidia-ml-py'}); assert not packages, f'Unexpected NVIDIA CUDA packages: {packages}'"

# Non-root user for runtime
RUN useradd -m appuser && chown -R appuser:appuser /app

COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER root

# Default command shows CLI help; override with args in `docker run`
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["felis", "--help"]
