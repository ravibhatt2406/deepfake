"""
fake_news/router.py — FastAPI router for Fake News detection.

POST /api/v1/fact-check/analyze
  - form field: input_type (text|url|image)
  - form field: text_content
  - form field: url
  - file: image (optional)
  - form field: check_sensational (bool)
  - form field: check_bias (bool)
  - form field: check_sources (bool)

Response: FactCheckResponse (matches existing frontend shape)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from shared.schemas import FactCheckResponse
from shared.config import Settings
from .pipeline import FactCheckPipeline

logger = logging.getLogger("truthscan.fn_router")
settings = Settings()
router = APIRouter()


def get_pipeline(request: Request) -> FactCheckPipeline:
    return request.app.state.fact_checker

def get_cache(request: Request):
    return request.app.state.cache


@router.post("/analyze", response_model=FactCheckResponse)
async def analyze(
    request: Request,
    input_type: str = Form(..., description="text|url|image"),
    text_content: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    check_sensational: bool = Form(True),
    check_bias: bool = Form(True),
    check_sources: bool = Form(True),
    pipeline: FactCheckPipeline = Depends(get_pipeline),
    cache = Depends(get_cache),
):
    t_start = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────────
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")

    content_text = ""
    if input_type == "text":
        if not text_content or not text_content.strip():
            raise HTTPException(400, "text_content required")
        content_text = text_content.strip()[:settings.MAX_TEXT_CHARS]
    elif input_type == "url":
        if not url:
            raise HTTPException(400, "url field required")
        from ai_detection.preprocessor import TextPreprocessor
        content_text = await TextPreprocessor.fetch_url(url)
    elif input_type == "image":
        if not file:
            raise HTTPException(400, "image file required")
        # OCR — use pytesseract if available, else use filename as hint
        try:
            import pytesseract
            from PIL import Image
            import io
            file_bytes = await file.read()
            img = Image.open(io.BytesIO(file_bytes))
            content_text = pytesseract.image_to_string(img)
            if not content_text.strip():
                raise ValueError("No text in image")
        except Exception as e:
            logger.warning("OCR failed: %s", e)
            content_text = f"[Image uploaded: {file.filename}. OCR text extraction unavailable. Analysing filename and context.]"
    else:
        raise HTTPException(400, f"Unknown input_type: {input_type}")

    if not content_text or len(content_text.strip()) < 20:
        raise HTTPException(422, "Content too short to analyse (minimum 20 characters)")

    # ── Cache lookup ──────────────────────────────────────────────────────────
    import hashlib
    content_hash = hashlib.sha256(content_text.encode()).hexdigest()
    cache_key = cache.make_key("fn", input_type, content_hash,
                               str(check_sensational), str(check_bias), str(check_sources))

    cached = await cache.get(cache_key)
    if cached:
        logger.info("Fact-check cache hit")
        cached["cacheHit"] = True
        return FactCheckResponse(**cached)

    # ── Pipeline execution ────────────────────────────────────────────────────
    check_opts = {
        "sensational": check_sensational,
        "bias": check_bias,
        "sources": check_sources,
    }

    try:
        result = await pipeline.run(content_text, input_type, check_opts)
    except Exception as e:
        logger.error("Fact-check pipeline error: %s", e, exc_info=True)
        raise HTTPException(500, f"Fact-check pipeline error: {e}")

    result.processingMs = round((time.perf_counter() - t_start) * 1000, 1)

    # ── Cache result ──────────────────────────────────────────────────────────
    await cache.set(cache_key, result.model_dump(), ttl=settings.CACHE_TTL_FACT_CHECK)

    return result
