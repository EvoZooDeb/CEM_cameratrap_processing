# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE} AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# System dependencies for OpenCV, ffmpeg, and scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv "$VENV_PATH" && \
    "$VENV_PATH"/bin/pip install --upgrade pip

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy project files and install the CLI package
COPY pyproject.toml README.md ./
COPY felis.py ./
COPY felis_cli ./felis_cli
RUN pip install .

# Non-root user for runtime
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command shows CLI help; override with args in `docker run`
ENTRYPOINT ["felis"]
CMD ["--help"]

