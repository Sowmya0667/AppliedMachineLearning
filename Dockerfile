# ──────────────────────────────────────────────
# Dockerfile — Spam Classifier Flask App
# Assignment 4: Containerization & CI
# ──────────────────────────────────────────────

# Base image: slim Python 3.11
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# ── Install system dependencies ───────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────
# Copy requirements first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source files ─────────────
COPY app.py .
COPY score.py .

# ── Copy the trained MLflow model artifacts ───
# Place your model folder (the artifacts directory) at ./model
# before building the image, e.g.:
#   cp -r <your_mlruns_artifacts_path> ./model
COPY model/ ./model/

# ── Environment variable for model path ───────
# score.py will read MLFLOW_MODEL_PATH if set;
# this replaces the hard-coded Windows path.
ENV MLFLOW_MODEL_PATH=/app/model

# ── Expose Flask port ─────────────────────────
EXPOSE 5000

# ── Launch the Flask app ──────────────────────
ENTRYPOINT ["python", "app.py"]
