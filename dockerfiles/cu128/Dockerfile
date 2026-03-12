# ══════════════════════════════════════════════════════════════════════════════
# NIND Denoise Manager — single-stage runtime image
#
# Layer order is deliberate — heaviest / slowest-changing layers are first so
# Docker cache reuse is maximised on every rebuild:
#
#   1. System deps           (changes: almost never)
#   2. PyTorch + torchvision (changes: on torch upgrade only)       [cu128]
#   3. NumPy                 (changes: on numpy upgrade only)
#   4. App Python deps       (changes: on dep version bump)
#   5. nind-denoise source   (changes: on upstream update or local edit)
#   6. App source            (changes: on every code edit — fastest layer)
#
# Git is installed only for the clone, then immediately purged within the same
# RUN layer so it never appears in the final image.
#
# GPU requirements:
#   Linux  — NVIDIA Container Toolkit must be installed on the host.
#             https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#   WSL2   — toolkit ships with the CUDA driver; no extra install needed.
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

# Bake BuildKit cache metadata into the image so the layer cache survives
# Docker daemon restarts, docker system prune --volumes, and WSL2 reboots.
# Without this, BuildKit stores cache only in memory and loses it on restart.
ARG BUILDKIT_INLINE_CACHE=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONWARNINGS="ignore::FutureWarning:torch"

WORKDIR /app

# ── Layer 1: system runtime dependencies ─────────────────────────────────────
# libgl1 + libglib2.0-0     — required by opencv-python-headless
# libimage-exiftool-perl    — called by denoise_image.py to copy EXIF to output
# NOTE: libsm6 / libxext6 / libxrender-dev are NOT needed — those are pulled
#       only by the full opencv-python (GUI) build, not the headless variant.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: PyTorch + torchvision (CUDA 12.8) ───────────────────────────────
# Largest download (~2.5 GB); pinned for reproducibility.
# Placed first so this expensive layer is never invalidated by code changes.
#
# CUDA 12.8 (cu128) is chosen over cu118 and cu126 for widest GPU coverage
# on Linux:
#   cu118 → sm_50–sm_90  but PyTorch 2.7.x+cu118 raises minimum to sm_70
#   cu126 → sm_50–sm_90  full range preserved (Maxwell through Ada/Hopper) ✓
#   cu128 → sm_75–sm_120 drops everything pre-Turing on Linux
#
# Minimum host driver required: 525.60 (Linux) / 526.x (Windows/WSL2)
# See https://pytorch.org/get-started/previous-versions/ for the cu128 wheel set.
RUN pip install --no-cache-dir \
        torch==2.7.1+cu128 \
        torchvision==0.22.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# ── Layer 3: NumPy (must be pinned before OpenCV) ────────────────────────────
# opencv-python-headless 4.9.x was compiled against NumPy 1.x; NumPy 2.x
# breaks the cv2 binary extension at import time (_ARRAY_API mismatch).
# The nind-denoise codebase also predates NumPy 2.x.  1.26.4 is the latest
# stable 1.x release.  Separate layer so the torch layer above is unaffected
# by a numpy-only bump.
RUN pip install --no-cache-dir \
        numpy==1.26.4

# ── Layer 4: remaining Python dependencies ───────────────────────────────────
# opencv-python-headless: GUI-free build; ~50 MB smaller than opencv-python,
#   no Qt / X11 dependency chain — correct for all server / Docker envs.
# nvidia-ml-py: official NVIDIA package (replaces deprecated pynvml);
#   import name in Python is still pynvml — no code changes required.
# All versions pinned for fully reproducible builds.
RUN pip install --no-cache-dir \
        ConfigArgParse==1.7 \
        opencv-python-headless==4.9.0.80 \
        PyYAML==6.0.1 \
        piqa==1.3.2 \
        flask==3.0.3 \
        flask-socketio==5.3.6 \
        nvidia-ml-py==13.590.48

# ── Layer 5: nind-denoise source ─────────────────────────────────────────────
# Cloned AFTER all pip layers so updating nind-denoise source (pull latest or
# modify locally) only invalidates this layer and Layer 6 — never the heavy
# pip layers above.
#
# git is installed, used, and fully purged within a single RUN instruction so
# no git binary, apt metadata, or .git history appears in the final image.
#
# Sparse clone — only src/ is fetched; model weights (.pt / .pth) are
# bind-mounted from the host at runtime and are never baked into the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && git clone --depth 1 --filter=blob:none --no-checkout \
           https://github.com/trougnouf/nind-denoise.git /app/nind-denoise \
    && cd /app/nind-denoise \
    && git sparse-checkout init --cone \
    && git sparse-checkout set src \
    && git checkout \
    && apt-get purge -y --auto-remove git ca-certificates \
    && rm -rf /var/lib/apt/lists/* /app/nind-denoise/.git

# ── Layer 6: application source ──────────────────────────────────────────────
# Last and fastest layer — only this invalidates on every code edit.
COPY app/ /app/app/

# ── Runtime directories ───────────────────────────────────────────────────────
# /app/state — no leading dot so it is visible on the host bind mount for
#              easy debug inspection (settings.json, lock.json, etc.)
RUN mkdir -p \
        /data/input \
        /data/output \
        /app/state \
        /app/nind-denoise/models/nind_denoise

# ── Non-root user ─────────────────────────────────────────────────────────────
# UID/GID 1000 matches the default Linux desktop and WSL2 user so bind-mounted
# host volumes have correct read/write permissions without extra configuration.
RUN groupadd --gid 1000 nind \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /sbin/nologin nind \
    && chown -R nind:nind /app /data

USER nind

EXPOSE 10010

CMD ["python3", "-m", "app.main"]