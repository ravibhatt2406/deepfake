import logging
import time
from typing import Dict, Any, List
import numpy as np

from shared.schemas import (
     AudioForensicsResponse, AudioVerdict, 
     AIAnalysis, TamperAnalysis, SuspiciousSegment
)
from .preprocessor import AudioPreprocessor
from .ai_voice_detector import AIVoiceDetector
from .tampering_detector import TamperingDetector

logger = logging.getLogger("truthscan.audio.analyzer")

class AudioAnalyzer:
    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.ai_detector = AIVoiceDetector()
        self.tamper_detector = TamperingDetector()

    async def analyze(self, audio_bytes: bytes, filename: str) -> AudioForensicsResponse:
        start_time = time.time()
        y, sr = await self.preprocessor.load_audio(audio_bytes)
        features = self.preprocessor.extract_features(y, sr)
        ai_result = await self.ai_detector.detect(y, sr, features)
        tamper_result = await self.tamper_detector.detect(y, sr, features)
        verdict = AudioVerdict.ORIGINAL
        confidence = 0
        ai_prob = ai_result['aiLikelihood']
        edit_prob = tamper_result['editingLikelihood']
        if ai_prob > 70 and edit_prob > 60:
            verdict = AudioVerdict.AI_AND_EDITED
            confidence = max(ai_prob, edit_prob)
        elif ai_prob > 50:
            verdict = AudioVerdict.AI_GENERATED
            confidence = ai_prob
        elif edit_prob > 50:
            verdict = AudioVerdict.EDITED
            confidence = edit_prob
        else:
            verdict = AudioVerdict.ORIGINAL
            confidence = 100 - max(ai_prob, edit_prob)
        summary = self._generate_summary(verdict, ai_result, tamper_result)
        segments = tamper_result.get('suspiciousSegments', [])
        if ai_prob > 80:
            segments.insert(0, SuspiciousSegment(
                startTime=0.0,
                endTime=float(features['duration']),
                type="AI_SIGNATURE",
                description="Global synthetic voice signature detected throughout recording."
            ))
        return AudioForensicsResponse(
            verdict=verdict,
            confidenceScore=float(confidence),
            summary=summary,
            verdictReason=f"Analysis found {len(ai_result['findings'])} AI signals and {len(tamper_result['findings'])} tampering signals.",
            aiAnalysis=AIAnalysis(**ai_result),
            tamperAnalysis=TamperAnalysis(**tamper_result),
            suspiciousSegments=segments,
            processingMs=(time.time() - start_time) * 1000
        )

    def _generate_summary(self, verdict: AudioVerdict, ai: dict, tamper: dict) -> str:
        if verdict == AudioVerdict.ORIGINAL:
            return "The audio appears to be an authentic human recording with no significant signs of AI generation or editing."
        elif verdict == AudioVerdict.AI_GENERATED:
            return f"The audio is likely AI-generated. Detected {ai.get('detectedModel')} synthetic vocal patterns."
        elif verdict == AudioVerdict.EDITED:
            return "The audio shows clear signs of manual editing, splicing, or noise floor inconsistencies."
        elif verdict == AudioVerdict.AI_AND_EDITED:
            return "High-Risk Alert: This audio is both AI-generated and manually manipulated."
        return "Analysis inconclusive."
