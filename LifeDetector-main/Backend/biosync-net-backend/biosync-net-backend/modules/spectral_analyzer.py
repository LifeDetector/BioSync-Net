import os
import librosa
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

def analyze_spectral(audio_path: str) -> Dict[str, Any]:
    """
    MODULE 2: Spectral Gap Analysis
    Detects unnatural voice frequencies and silence gaps.
    Real human voice has continuous harmonics.
    Deepfakes often have unnatural "gaps" or flat spectral regions.
    """
    try:
        # Load audio at 16kHz (standard for voice analysis)
        if not audio_path or not os.path.exists(audio_path):
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "explanation": "No audio file available for analysis"
            }

        y, sr = librosa.load(audio_path, sr=16000)

        if len(y) < 16000:  # less than 1 second
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "explanation": "Audio too short for spectral analysis"
            }

        # Compute STFT (Short-Time Fourier Transform)
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)

        # Spectral flatness (high = more noise-like / synthetic)
        flatness = librosa.feature.spectral_flatness(S=magnitude).mean()

        # Spectral rolloff (where most energy is concentrated)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()

        # Detect silence gaps using librosa
        intervals = librosa.effects.split(y, top_db=20, frame_length=2048, hop_length=512)
        total_duration = len(y) / sr
        voice_duration = sum((end - start) for start, end in intervals) / sr
        gap_ratio = 1 - (voice_duration / total_duration) if total_duration > 0 else 1.0

        # Combine anomalies
        anomaly_score = (gap_ratio + flatness) / 2.0
        score = max(0.0, min(1.0, 1.0 - anomaly_score))

        if score > 0.65:
            label = "Voice Frequencies Normal"
            explanation = "Natural continuous voice harmonics detected"
        else:
            label = "Unnatural Spectral Gaps Detected"
            explanation = "Possible deepfake - unnatural silence gaps or flat voice spectrum"

        return {
            "score": round(float(score), 2),
            "label": label,
            "explanation": explanation
        }

    except Exception as e:
        # Graceful failure
        return {
            "score": 0.5,
            "label": "Inconclusive",
            "explanation": f"Spectral analysis failed: {str(e)[:100]}"
        }

# Quick test
if __name__ == "__main__":
    print("✅ Spectral Analyzer loaded successfully!")
