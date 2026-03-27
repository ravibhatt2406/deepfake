"""
shared/config.py — Centralized settings via Pydantic BaseSettings
All values can be overridden by environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    ENV: str = "production"
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Redis Cache ───────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600          # 1 hour default
    CACHE_TTL_FACT_CHECK: int = 21600      # 6 hours for fact-checks
    CACHE_TTL_AI_DETECT: int = 7200        # 2 hours for AI detection

    # ── AI Detection Models ───────────────────────────────────────────────────
    # EfficientNet-B4 for general image AI detection
    EFFICIENTNET_MODEL_PATH: str = "models/efficientnet_b4_ai_detector.onnx"
    # XceptionNet for deepfake/face detection
    XCEPTION_MODEL_PATH: str = "models/xception_deepfake.onnx"
    # CLIP ViT for semantic AI detection
    CLIP_MODEL_NAME: str = "openai/clip-vit-large-patch14"
    # Grad-CAM explainability model
    GRADCAM_ENABLED: bool = True

    # Ensemble weights (must sum to 1.0)
    ENSEMBLE_WEIGHT_EFFICIENTNET: float = 0.35
    ENSEMBLE_WEIGHT_XCEPTION: float = 0.30
    ENSEMBLE_WEIGHT_CLIP: float = 0.20
    ENSEMBLE_WEIGHT_FREQUENCY: float = 0.15

    # Confidence thresholds
    AI_THRESHOLD_HIGH: float = 0.75        # > 75% → AI-Generated
    AI_THRESHOLD_MID: float = 0.40         # > 40% → Partially AI
    AI_CONFIDENCE_MIN: float = 0.55        # below → flag as uncertain

    # ── Fake News Models ──────────────────────────────────────────────────────
    LLM_PROVIDER: str = "anthropic"        # anthropic | openai | local
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # Claude model for claim verification
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS: int = 4096

    # Embedding model for semantic search
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768

    # Vector DB (Chroma or Pinecone)
    VECTOR_DB_PROVIDER: str = "chroma"    # chroma | pinecone | weaviate
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "truthscan-facts"

    # Retrieval settings
    RAG_TOP_K: int = 8                     # top-K chunks to retrieve
    RAG_SIMILARITY_THRESHOLD: float = 0.72  # minimum semantic match

    # ── External APIs ─────────────────────────────────────────────────────────
    GOOGLE_FACT_CHECK_API_KEY: Optional[str] = None
    NEWSAPI_KEY: Optional[str] = None
    GDELT_API_URL: str = "https://api.gdeltproject.org/api/v2"
    SERPAPI_KEY: Optional[str] = None

    # Trusted domain scores (higher = more trusted)
    TRUSTED_DOMAINS: dict = {
        "reuters.com": 1.0, "apnews.com": 1.0, "bbc.com": 0.95, "bbc.co.uk": 0.95,
        "who.int": 1.0, "cdc.gov": 1.0, "nih.gov": 1.0, "un.org": 1.0,
        "snopes.com": 0.95, "politifact.com": 0.95, "factcheck.org": 0.95,
        "altnews.in": 0.90, "pib.gov.in": 0.90, "data.gov": 0.90,
        "npr.org": 0.88, "theguardian.com": 0.85, "nytimes.com": 0.85,
        "washingtonpost.com": 0.85, "economist.com": 0.87,
    }

    # ── Performance ───────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int = 20
    MAX_TEXT_CHARS: int = 50_000
    INFERENCE_TIMEOUT_SECONDS: int = 30
    BATCH_SIZE: int = 8                    # for batch inference
    USE_ONNX_RUNTIME: bool = True          # faster CPU inference
    USE_HALF_PRECISION: bool = True        # FP16 where GPU available

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
