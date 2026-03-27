from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ImageVerdict(str, Enum):
    ORIGINAL = "ORIGINAL"
    MANIPULATED = "MANIPULATED"
    LIKELY_MANIPULATED = "LIKELY_MANIPULATED"
    AI_GENERATED = "AI_GENERATED"
    INCONCLUSIVE = "INCONCLUSIVE"

class AudioVerdict(str, Enum):
    ORIGINAL = "ORIGINAL"
    AI_GENERATED = "AI_GENERATED"
    EDITED = "EDITED"
    AI_AND_EDITED = "AI_AND_EDITED"
    INCONCLUSIVE = "INCONCLUSIVE"

class AIVerdict(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    PARTIALLY_AI = "PARTIALLY_AI"
    LIKELY_HUMAN = "LIKELY_HUMAN"

class ClaimVerdict(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    UNVERIFIED = "UNVERIFIED"

class NewsVerdict(str, Enum):
    TRUE = "TRUE"
    FAKE = "FAKE"
    MISLEADING = "MISLEADING"
    UNVERIFIED = "UNVERIFIED"

class Relevance(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CONTEXT = "CONTEXT"

class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"

class MetricItem(BaseModel):
    name: str
    val: int

class BreakdownItem(BaseModel):
    icon: str
    label: str
    val: int

class ModelScore(BaseModel):
    model_name: str
    score: float
    confidence: float
    weight: float
    latency_ms: float

class FrequencyFeatures(BaseModel):
    dct_energy_ratio: float
    high_freq_anomaly: float
    periodicity_score: float
    spectral_flatness: float

class AIAnalysis(BaseModel):
    aiLikelihood: float
    syntheticProsodyScore: float
    phonemeRepetitionScore: float
    detectedModel: str
    findings: List[str]

class TamperAnalysis(BaseModel):
    editingLikelihood: float
    spectralDiscontinuity: float
    phaseInconsistency: float
    backgroundNoiseMismatched: bool
    findings: List[str]

class SuspiciousSegment(BaseModel):
    startTime: float
    endTime: float
    type: str
    description: str

class AudioForensicsResponse(BaseModel):
    verdict: AudioVerdict
    confidenceScore: float
    summary: str
    verdictReason: str
    aiAnalysis: AIAnalysis
    tamperAnalysis: TamperAnalysis
    suspiciousSegments: List[SuspiciousSegment]
    processingMs: float

class ImageForensicsResponse(BaseModel):
    verdict: ImageVerdict
    confidence_score: float
    detailed_analysis: Dict[str, Any]

class AIDetectionResponse(BaseModel):
    aiScore: int
    verdict: AIVerdict
    title: str
    desc: str
    metrics: List[MetricItem]
    breakdowns: List[BreakdownItem]
    confidence: float
    ensembleScores: List[ModelScore]
    frequencyFeatures: Optional[Any] = None
    processingMs: float
    cacheHit: bool = False

class SignalItem(BaseModel):
    name: str
    score: int
    description: str
    level: str

class LanguageFlag(BaseModel):
    text: str
    type: str
    severity: str

class EvidenceItem(BaseModel):
    source: str
    type: str
    title: str
    summary: str
    url: str
    relevance: Relevance
    credibility: str
    publishedAt: str
    domainTrustScore: float
    semanticSimilarity: float

class HighlightSegment(BaseModel):
    text: str
    type: str
    reason: str

class ClaimItem(BaseModel):
    id: int
    text: str
    verdict: ClaimVerdict
    confidence: int
    explanation: str
    flagged: bool
    sources: List[str]

class FactCheckResponse(BaseModel):
    verdict: NewsVerdict
    confidence: int
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
    retrievedDocuments: int
    ragQueriesRun: int
    sourcesSearched: int
    antiHallucinationPassed: bool
    processingMs: float
    cacheHit: bool = False
