# ── Base image ────────────────────────────────────────────────────────────────
# python:3.10-slim saves ~400 MB vs python:3.10 on HuggingFace Spaces free tier
FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────────────────
# libgl1-mesa-glx + libglib2.0-0 are required by OpenCV (cv2).
# They are missing from the slim image and cause a silent ImportError at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches pip install between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ────────────────────────────────────────────────────────
COPY . .

# Model weights are copied from the repo root into /app/ by COPY . .
# The defaults in app.py already point to these paths:
#   /app/road_model_best.pth
#   /app/landcover_best.pth
#   /app/building_model_best.pth
# Override via Space Secrets / env vars if needed:
#   ROAD_WEIGHTS, LANDCOVER_WEIGHTS, BUILDING_WEIGHTS
ENV RESULTS_DIR=/tmp/results

# ── Port ──────────────────────────────────────────────────────────────────────
# HuggingFace Spaces requires port 7860 (not 8000)
EXPOSE 7860

# ── Launch ────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
