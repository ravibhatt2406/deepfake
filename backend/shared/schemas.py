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
    is_ai: bool
    confidence: float
    explanation: str

class FactCheckResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    sources: List[str]
    explanation: str
