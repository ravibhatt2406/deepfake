# ── Dockerfile ────────────────────────────────────────────────────────────────
# Multi-stage build: builder → runtime
# Stage 1: builder installs Python deps and compiles extensions
# Stage 2: lean runtime image with only production artifacts

# ╔══════════════════════════════════════════════════╗
# ║  BUILD STAGE                                      ║
# ╚══════════════════════════════════════════════════╝
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 \
    libgl1-mesa-glx libglib2.0-0 \
    tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ╔══════════════════════════════════════════════════╗
# ║  RUNTIME STAGE                                    ║
# ╚══════════════════════════════════════════════════╝
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Runtime system deps (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    tesseract-ocr tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# App code
WORKDIR /app
COPY . .

# Create non-root user
RUN useradd -r -u 1001 appuser && \
    mkdir -p /app/data/chroma_db /app/models && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info", "--access-log"]

