"""
TruthScan Backend — Production-Grade API
=========================================
Unified FastAPI application serving:
  • /api/v1/ai-detection   — AI/Deepfake content detection
  • /api/v1/fact-check     — Fake news & claim verification
  • /api/v1/image-forensics — Image originality & tampering

Run: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import time
import os

from ai_detection.router import router as ai_router
from fake_news.router import router as fn_router
from shared.cache import Cache
from shared.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("truthscan")
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("🚀 TruthScan backend starting up…")
    # Warm up model caches
    from ai_detection.ensemble import EnsembleDetector
    from fake_news.pipeline import FactCheckPipeline
    app.state.ai_detector = EnsembleDetector()
    app.state.fact_checker = FactCheckPipeline()
    app.state.cache = Cache(settings.REDIS_URL)
    await app.state.ai_detector.warmup()
    await app.state.fact_checker.warmup()
    logger.info("✅ All models warmed up and ready")
    yield
    logger.info("🛑 Shutting down TruthScan backend…")
    await app.state.cache.close()


app = FastAPI(
    title="TruthScan API",
    version="2.0.0",
    description="Production-grade AI detection & fact-checking backend",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ai_router,  prefix="/api/v1/ai-detection",  tags=["AI Detection"])
app.include_router(fn_router,  prefix="/api/v1/fact-check",    tags=["Fact Check"])


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=4)
