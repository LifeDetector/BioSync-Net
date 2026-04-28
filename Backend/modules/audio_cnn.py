"""
MODULE: AI Audio / Voice Detection
Detects synthetic/cloned speech.
Trained on ASVspoof 2019/2021 and FakeAVCeleb datasets.
"""
import os
import numpy as np
from typing import Dict, Any

# --- Model Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "audio_detector.onnx")

_session = None
HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
    if os.path.exists(MODEL_PATH):
        _session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"[AudioAI] ONNX model loaded from {MODEL_PATH}")
except ImportError:
    pass

def detect_audio_ai(audio_path: str) -> Dict[str, Any]:
    if not audio_path or not os.path.exists(audio_path):
        return {"score": 0.5, "label": "No Audio", "method": "none", "explanation": "No audio track found for analysis."}

    if not HAS_ONNX or _session is None:
        # Fallback to spectral analysis results if available (will be aggregated in main.py)
        return {
            "score": 0.5,
            "label": "Audio AI: Model Missing",
            "method": "none",
            "explanation": "AI audio detection model not found."
        }

    try:
        import librosa
        y, sr = librosa.load(audio_path, duration=5.0, sr=16000)
        # Match training preprocessing
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spect_db = librosa.power_to_db(spect, ref=np.max)
        
        # Fixed width 128
        if spect_db.shape[1] > 128:
            spect_db = spect_db[:, :128]
        else:
            spect_db = np.pad(spect_db, ((0, 0), (0, 128 - spect_db.shape[1])))
            
        # Normalize
        spect_db = (spect_db - spect_db.min()) / (spect_db.max() - spect_db.min() + 1e-8)
        # 3-channel NCHW
        img = np.stack([spect_db] * 3, axis=0)
        input_batch = np.expand_dims(img, axis=0).astype(np.float32)
        
        input_name = _session.get_inputs()[0].name
        output_name = _session.get_outputs()[0].name
        result = _session.run([output_name], {input_name: input_batch})
        
        prob = 1.0 / (1.0 + np.exp(-result[0][0][0]))
        score = float(prob)
        
        if score >= 0.7:
            label = "Human Voice Verified"
            explanation = "Acoustic signature matches natural human vocal patterns."
        elif score <= 0.3:
            label = "AI Synthetic Voice Detected"
            explanation = "Detected artifacts consistent with synthetic speech generation (TTS/VC)."
        else:
            label = "Audio: Low Confidence"
            explanation = "Audio forensic signature is ambiguous."

        return {
            "score": round(score, 2),
            "label": label,
            "method": "onnx_audio_ai",
            "explanation": explanation
        }

    except Exception as e:
        return {
            "score": 0.5,
            "label": "Audio Analysis Failed",
            "method": "error",
            "explanation": f"Failed to process audio: {str(e)[:100]}"
        }
