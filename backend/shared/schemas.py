"""
shared/schemas.py — All Pydantic request/response models.
These define the API contract. Frontend expects these exact shapes.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class AIVerdict(str, Enum):
    AI_GENERATED = "AI-GENERATED"
    PARTIALLY_AI = "PARTIALLY-AI"
    LIKELY_HUMAN = "LIKELY-HUMAN"

class NewsVerdict(str, Enum):
    TRUE = "TRUE"
    FAKE = "FAKE"
    MISLEADING = "MISLEADING"
    UNVERIFIED = "UNVERIFIED"

class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"

class Relevance(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PARTIAL = "PARTIAL"
    CONTEXT = "CONTEXT"

class ClaimVerdict(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    UNVERIFIED = "UNVERIFIED"

class ForgeryVerdict(str, Enum):
    ORIGINAL = "ORIGINAL"
    MANIPULATED = "MANIPULATED"
    LIKELY_MANIPULATED = "LIKELY_MANIPULATED"
    INCONCLUSIVE = "INCONCLUSIVE"


# ─────────────────────────────────────────────────────────────────────────────
# AI DETECTION SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class MetricItem(BaseModel):
    name: str
    val: int = Field(ge=0, le=100)

class BreakdownItem(BaseModel):
    icon: str
    label: str
    val: int = Field(ge=0, le=100)

class ModelScore(BaseModel):
    """Individual model output before ensemble fusion."""
    model_name: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    weight: float
    latency_ms: float

class FrequencyFeatures(BaseModel):
    """DCT/FFT frequency domain analysis results."""
    dct_energy_ratio: float
    high_freq_anomaly: float
    periodicity_score: float
    spectral_flatness: float

class AIDetectionResponse(BaseModel):
    """
    Response shape expected by the existing TruthScan frontend.
    Fields: aiScore, verdict, title, desc, metrics, breakdowns,
            plus new accuracy fields hidden from UI.
    """
    # ── Core (frontend-visible) ───────────────────────────────────────────────
    aiScore: int = Field(ge=0, le=100, description="AI involvement percentage")
    verdict: AIVerdict
    title: str
    desc: str
    metrics: List[MetricItem]
    breakdowns: List[BreakdownItem]

    # ── Extended (may be used by advanced UI / logging) ───────────────────────
    confidence: float = Field(ge=0.0, le=1.0)
    ensembleScores: List[ModelScore]
    frequencyFeatures: Optional[FrequencyFeatures] = None
    processingMs: float
    cacheHit: bool = False
    modelVersion: str = "2.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# FAKE NEWS SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ClaimItem(BaseModel):
    id: int
    text: str
    verdict: ClaimVerdict
    confidence: int = Field(ge=0, le=100)
    explanation: str
    flagged: bool
    sources: List[str] = []             # URLs that verify/refute this claim
    similarityScore: Optional[float] = None  # semantic sim to matching source

class SignalItem(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)
    description: str
    level: Literal["HIGH", "MEDIUM", "LOW"]

class LanguageFlag(BaseModel):
    text: str
    type: Literal["SENSATIONAL", "BIAS", "UNVERIFIED_CLAIM",
                  "MISSING_SOURCE", "SATIRE", "AI_GENERATED"]
    severity: Literal["HIGH", "MEDIUM", "LOW"]

class EvidenceItem(BaseModel):
    source: str
    type: Literal["FACT-CHECK", "NEWS", "OFFICIAL", "ACADEMIC", "GOVERNMENT"]
    title: str
    summary: str
    url: str
    relevance: Relevance
    credibility: Literal["HIGH", "MEDIUM"]
    publishedAt: Optional[str] = None
    domainTrustScore: Optional[float] = None   # 0–1 from trusted domains map
    semanticSimilarity: Optional[float] = None # cosine sim to claim

class HighlightSegment(BaseModel):
    text: str
    type: Literal["FAKE", "MISLEADING", "OK"]
    reason: str

class FactCheckResponse(BaseModel):
    """
    Response shape matching existing frontend contract.
    """
    # ── Core (frontend-visible) ───────────────────────────────────────────────
    verdict: NewsVerdict
    confidence: int = Field(ge=0, le=100)
    summary: str
    verdictReason: str
    inputSummary: str
    claims: List[ClaimItem]
    signals: List[SignalItem]
    languageFlags: List[LanguageFlag]
    evidence: List[EvidenceItem]
    highlightedSegments: List[HighlightSegment]
    overallRisk: RiskLevel
    recommendedAction: str

    # ── Accuracy metadata (not shown in UI but available) ────────────────────
    retrievedDocuments: int = 0
    ragQueriesRun: int = 0
    sourcesSearched: int = 0
    antiHallucinationPassed: bool = True
    processingMs: float = 0.0
    cacheHit: bool = False
    modelVersion: str = "2.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE FORENSICS SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ForensicFinding(BaseModel):
    title: str
    severity: Literal["HIGH", "MEDIUM", "LOW", "INFO"]
    description: str
    type: Literal["TAMPERING", "METADATA", "COMPRESSION",
                  "LIGHTING", "NOISE", "COPY_MOVE", "GAN", "INFO"]
    region: Optional[Dict[str, int]] = None   # {x, y, w, h} pixel coords

class ForensicSignal(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)
    description: str
    flagged: bool

class EXIFAnalysis(BaseModel):
    softwareDetected: Optional[str]
    timestampConsistency: Literal["CONSISTENT", "SUSPICIOUS", "MISSING"]
    metadataStripped: bool
    suspiciousFields: List[str]
    notes: str
    rawFields: Dict[str, Any] = {}

class ImageForensicsResponse(BaseModel):
    verdict: ForgeryVerdict
    originalityScore: int = Field(ge=0, le=100)
    confidenceScore: int = Field(ge=0, le=100)
    summary: str
    verdictReason: str
    findings: List[ForensicFinding]
    signals: List[ForensicSignal]
    exifAnalysis: EXIFAnalysis
    lightingAnalysis: Dict[str, Any]
    noiseAnalysis: Dict[str, Any]
    ganDetection: Dict[str, Any]
    recommendations: str
    processingMs: float = 0.0
    cacheHit: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ERROR SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    details: Optional[List[ErrorDetail]] = None
    requestId: Optional[str] = None
