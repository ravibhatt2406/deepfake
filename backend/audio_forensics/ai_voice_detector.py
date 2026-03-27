import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger("truthscan.audio.ai_detector")

class AIVoiceDetector:
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "pitch_variance": 5.0,
            "spectral_flatness": 0.02,
            "continuity_score": 0.85
        }

    async def detect(self, y: np.ndarray, sr: int, features: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        reasons = []
        pitches = features['pitches']
        active_pitches = pitches[pitches > 0]
        if len(active_pitches) > 10:
            pitch_std = np.std(active_pitches)
            if pitch_std < self.thresholds['pitch_variance']:
                score += 25
                reasons.append("Low pitch variance (synthetic prosody)")
        flatness = np.mean(features['spectral_flatness'])
        if flatness < self.thresholds['spectral_flatness']:
            score += 30
            reasons.append("Unnatural spectral smoothness (vocoder signature)")
        mfcc_delta = np.diff(features['mfcc'], axis=1)
        mean_delta = np.mean(np.abs(mfcc_delta))
        if mean_delta < 0.5:
            score += 20
            reasons.append("Over-smoothed phoneme transitions")
        final_score = min(score, 100)
        return {
            "aiLikelihood": float(final_score),
            "syntheticProsodyScore": float(1.0 - (min(pitch_std, 50)/50.0)) if 'active_pitches' in locals() else 0.5,
            "phonemeRepetitionScore": float(1.0 - (min(mean_delta, 2.0)/2.0)) if 'mean_delta' in locals() else 0.5,
            "detectedModel": "High-Confidence Synthetic" if final_score > 70 else "None",
            "findings": reasons
        }
