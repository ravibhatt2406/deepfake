import librosa
import numpy as np
import io
import soundfile as sf
import logging

logger = logging.getLogger("truthscan.audio.preprocessor")

class AudioPreprocessor:
    @staticmethod
    async def load_audio(audio_bytes: bytes):
        """Loads audio from bytes and returns (y, sr)"""
        try:
            audio_file = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_file, sr=None)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise ValueError(f"Could not decode audio: {e}")

    @staticmethod
    def extract_features(y, sr):
        """Extracts key spectral and temporal features for analysis"""
        features = {}
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        features['mel_spectrogram'] = librosa.power_to_db(S, ref=np.max)
        features['mfcc'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_flatness'] = librosa.feature.spectral_flatness(y=y)
        features['zcr'] = librosa.feature.zero_crossing_rate(y)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        features['pitches'] = pitches
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        return features

    @staticmethod
    def detect_segments(y, sr, frame_length=2048, hop_length=512):
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        threshold = np.max(rms) * 0.05
        is_silent = rms < threshold
        return {"times": times, "rms": rms, "silence_mask": is_silent}
