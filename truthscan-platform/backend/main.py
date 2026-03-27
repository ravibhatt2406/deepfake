from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import time
import os
import httpx
from pydantic import BaseModel

from ai_detection.router import router as ai_router
from fake_news.router import router as fn_router
from image_forensics.router import router as img_router
from audio_forensics.router import router as audio_router
from shared.cache import Cache
from shared.config import Settings
from audio_forensics.analyzer import AudioAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("truthscan")
settings = Settings()

# Initialize audio analyzer
audio_analyzer = AudioAnalyzer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("🚀 TruthScan backend starting up…")
    from ai_detection.ensemble import EnsembleDetector
    from fake_news.pipeline import FactCheckPipeline
    app.state.ai_detector = EnsembleDetector()
    app.state.fact_checker = FactCheckPipeline()
    app.state.audio_analyzer = audio_analyzer
    app.state.cache = Cache(settings.REDIS_URL)
    await app.state.ai_detector.warmup()
    await app.state.fact_checker.warmup()
    logger.info("✅ All models warmed up and ready")
    yield
    logger.info("🛑 Shutting down TruthScan backend…")
    await app.state.cache.close()

app = FastAPI(
    title="TruthScan API",
    version="2.1.0",
    description="Production-grade AI detection & forensics backend",
    lifespan=lifespan,
)

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

# Routers
app.include_router(ai_router,  prefix="/api/v1/ai-detection",  tags=["AI Detection"])
app.include_router(fn_router,  prefix="/api/v1/fact-check",    tags=["Fact Check"])
app.include_router(img_router, prefix="/api/v1/image-forensics", tags=["Image Forensics"])
app.include_router(audio_router, prefix="/api/v1/audio-forensics", tags=["Audio Forensics"])

class ClaudeProxyRequest(BaseModel):
    system: str = ""
    messages: list
    max_tokens: int = 3000
    model: str = "claude-3-5-sonnet-20240620"
    tools: list = []

@app.post("/api/v1/proxy/claude", tags=["Proxy"])
async def proxy_claude(req: ClaudeProxyRequest):
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")
    headers = {
        "x-api-key": settings.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = req.model_dump(exclude_unset=True)
    if not payload.get("tools"):
        payload.pop("tools", None)
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=60.0)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.1.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=4)
