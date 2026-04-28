import cv2
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Try to initialize MediaPipe FaceMesh, fallback if not available
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    HAS_MEDIAPIPE = True
except (AttributeError, ImportError, ModuleNotFoundError):
    HAS_MEDIAPIPE = False
    face_mesh = None

# Dummy function for lite mode (no heavy model)
def get_emotion_pipeline():
    """Lite version - no real model loaded"""
    return None

def check_emotion_consistency(frames: List[np.ndarray], audio_path: str) -> Dict[str, Any]:
    """
    LITE VERSION - No big model needed
    Returns simple dummy result so the backend works on slow internet.
    """
    try:
        if len(frames) < 10:
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "face_emotions": [],
                "voice_emotions": [],
                "agreement_ratio": 0.0,
                "explanation": "Lite mode - emotion check skipped"
            }

        return {
            "score": 0.65,
            "label": "Emotion Check Skipped (Lite Mode)",
            "face_emotions": ["neutral"],
            "voice_emotions": ["neutral"],
            "agreement_ratio": 0.65,
            "explanation": "Emotion module disabled for slow internet"
        }

    except Exception as e:
        return {
            "score": 0.5,
            "label": "Inconclusive",
            "face_emotions": [],
            "voice_emotions": [],
            "agreement_ratio": 0.0,
            "explanation": "Emotion analysis skipped"
        }

# Quick test
if __name__ == "__main__":
    print("✅ Lite Emotion Checker loaded successfully!")
