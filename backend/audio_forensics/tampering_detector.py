import numpy as np
import librosa
import logging
from typing import Dict, Any, List

logger = logging.getLogger("truthscan.audio.tampering")

class TamperingDetector:
    def __init__(self):
        pass

    async def detect(self, y: np.ndarray, sr: int, features: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        findings = []
        segments = []
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_diff = np.diff(contrast, axis=1)
        spikes = np.where(np.abs(contrast_diff) > np.std(contrast_diff) * 5)
        if len(spikes[0]) > 0:
            frame_indices = np.unique(spikes[1])
            for idx in frame_indices:
                time = librosa.frames_to_time(idx, sr=sr)
                score += 10
                segments.append({
                    "startTime": float(time),
                    "endTime": float(time + 0.1),
                    "type": "SPECTRAL_JUMP",
                    "description": "Abrupt spectral discontinuity detected"
                })
        centroid = features['spectral_centroid'][0]
        centroid_delta = np.abs(np.diff(centroid))
        centroid_spikes = np.where(centroid_delta > np.mean(centroid_delta) * 8)[0]
        if len(centroid_spikes) > 0:
            score += 15
            findings.append("Phase/Frequency alignment artifacts")
        rms = librosa.feature.rms(y=y)[0]
        silent_frames = np.where(rms < np.mean(rms) * 0.1)[0]
        if len(silent_frames) > 10:
             noise_levels = rms[silent_frames]
             if np.std(noise_levels) > np.mean(noise_levels) * 0.5:
                 score += 20
                 findings.append("Inconsistent background noise floor")
        final_score = min(score, 100)
        return {
            "editingLikelihood": float(final_score),
            "spectralDiscontinuity": float(np.max(contrast_diff)) if contrast_diff.size > 0 else 0.0,
            "phaseInconsistency": float(np.max(centroid_delta)) if centroid_delta.size > 0 else 0.0,
            "backgroundNoiseMismatched": True if "Inconsistent background noise floor" in findings else False,
            "suspiciousSegments": segments,
            "findings": findings
        }
