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

---

# ── docker-compose.yml ─────────────────────────────────────────────────────────
version: "3.9"

services:
  # ── TruthScan API ────────────────────────────────────────────────────────────
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    image: truthscan-backend:2.0
    container_name: truthscan_api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - GOOGLE_FACT_CHECK_API_KEY=${GOOGLE_FACT_CHECK_API_KEY}
      - VECTOR_DB_PROVIDER=chroma
      - CHROMA_PERSIST_DIR=/app/data/chroma_db
      - EFFICIENTNET_MODEL_PATH=/app/models/efficientnet_b4_ai_detector.onnx
      - XCEPTION_MODEL_PATH=/app/models/xception_deepfake.onnx
      - ENV=production
    volumes:
      - model_data:/app/models
      - chroma_data:/app/data/chroma_db
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: "4"
    networks:
      - truthscan_net

  # ── Redis Cache ───────────────────────────────────────────────────────────────
  redis:
    image: redis:7-alpine
    container_name: truthscan_redis
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru --save 60 1
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - truthscan_net

  # ── Nginx Reverse Proxy ───────────────────────────────────────────────────────
  nginx:
    image: nginx:1.25-alpine
    container_name: truthscan_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - truthscan_net

volumes:
  model_data:
  chroma_data:
  redis_data:

networks:
  truthscan_net:
    driver: bridge

---

# ── nginx.conf ────────────────────────────────────────────────────────────────
events { worker_connections 1024; }

http {
    upstream truthscan {
        least_conn;
        server api:8000;
        keepalive 64;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/m;

    server {
        listen 80;
        server_name _;

        # Redirect HTTP → HTTPS in production
        # return 301 https://$host$request_uri;

        location /api/ {
            limit_req zone=api_limit burst=10 nodelay;
            proxy_pass         http://truthscan;
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;
            proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 60s;
            proxy_send_timeout 60s;

            # CORS
            add_header Access-Control-Allow-Origin  *;
            add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
            add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        }

        location /health {
            proxy_pass http://truthscan;
        }

        # Serve frontend static files (if bundled)
        location / {
            root /var/www/frontend;
            try_files $uri $uri/ /index.html;
        }
    }
}
