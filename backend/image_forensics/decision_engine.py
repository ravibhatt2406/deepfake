import logging
from typing import Dict, Any, List
from shared.schemas import ImageVerdict, AudioVerdict # Import AudioVerdict too for completeness if needed elsewhere

logger = logging.getLogger("truthscan.image.engine")

class ImageDecisionEngine:
    def __init__(self):
        # Thresholds for final decision
        self.ai_threshold = 70.0        # High confidence AI
        self.tamper_threshold = 60.0    # Significant tampering
        self.maybe_threshold = 40.0     # Suspicious but not certain

    def finalize_verdict(self, ai_result: Dict[str, Any], tamper_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Logic for merging multi-model signals into a single, non-contradictory verdict.
        """
        ai_score = ai_result.get("ai_score", 0.0)
        tamper_score = tamper_result.get("tamper_score", 0.0)
        
        # 1. AI Content prioritized if extremely high
        if ai_score >= self.ai_threshold:
            verdict = ImageVerdict.AI_GENERATED
            explanation = "High-confidence AI generation signals detected in pixel distribution and frequency domain."
        
        # 2. Tampering/Manipulation detection
        elif tamper_score >= self.tamper_threshold:
            verdict = ImageVerdict.MANIPULATED
            explanation = "Significant digital tampering detected (cloning, ELA anomalies, or metadata inconsistencies)."
            
        # 3. Probable/Likely manipulation
        elif tamper_score >= self.maybe_threshold or ai_score >= self.maybe_threshold:
            verdict = ImageVerdict.LIKELY_MANIPULATED
            explanation = "Suspicious artifacts detected. While not conclusive, the image shows signs of potential editing."
            
        # 4. Default Original
        else:
            verdict = ImageVerdict.ORIGINAL
            explanation = "No significant signs of AI generation or digital tampering detected."
            
        return {
            "verdict": verdict,
            "explanation": explanation,
            "confidence": max(ai_score, tamper_score) if verdict != ImageVerdict.ORIGINAL else (100 - max(ai_score, tamper_score))
        }
