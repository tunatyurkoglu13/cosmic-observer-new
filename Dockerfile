# syntax=docker/dockerfile:1
#
# Cosmic Observer — production image.
#
# Multi-stage: the builder stage installs Python dependencies into a venv
# (needs build tools for anything without a prebuilt wheel); the runtime
# stage copies only the finished venv + app code, so compilers and pip's
# download cache never end up in the shipped image.
#
# torch/torchvision are installed from PyPI's CPU-only wheel index — this
# app has no CUDA use case (Apple Silicon dev machines use MPS; a Linux
# container has no GPU here), and the default PyPI wheels otherwise bundle
# several GB of unused CUDA runtime libraries.
#
# Trained model weights (models/) and the SQLite caches/alert store
# (data/cache/) are intentionally NOT baked into this image — they're
# runtime state, meant to be bind/volume-mounted (see docker-compose.yml).
# The app already degrades gracefully when they're absent (e.g.
# cv/streamer.py's ANOMALY_WEIGHTS_PATH / SPACE_FINE_TUNED_WEIGHTS_PATH
# existence checks) — a freshly-built image with no mounted models simply
# reports those features as "not trained" instead of erroring.

FROM python:3.14-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu


FROM python:3.14-slim AS runtime

# libgl1/libglib2.0-0: opencv-python-headless still dlopen()s libGL.so.1 on
# some Debian base images despite being the "headless" build — a common,
# well-documented Docker gotcha for any cv2 import. ffmpeg: backs the real
# NASA ISS live-video ingestion path (cv/iss_live.py + cv2.VideoCapture).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
# PUBLIC_DEMO=1: any deployed image is a read-only showcase by default
# (see app.py's public-deployment hardening). docker-compose.yml turns it
# off for local use.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PUBLIC_DEMO=1

WORKDIR /app
COPY app.py .
COPY core/ core/
COPY cv/ cv/
COPY stm/ stm/
COPY viz/ viz/
COPY reports/ reports/
COPY data/ data/
COPY static/ static/

RUN mkdir -p models data/cache data/cv_uploads data/cv_training && \
    chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
