import cv2
import numpy as np
import mediapipe as mp
import librosa
from scipy.signal import correlate
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

def check_av_sync(frames: List[np.ndarray], audio_path: str) -> Dict[str, Any]:
    """
    MODULE 3: Audio-Visual Cross Sync
    Checks if lip movements match the audio timing.
    Real humans have natural lip-sync. Deepfakes often have a lag or mismatch.
    """
    try:
        if len(frames) < 10 or not audio_path:
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "lag_frames": 0,
                "explanation": "Not enough data for lip-sync check"
            }

        # 1. Extract mouth movement from video
        mouth_open_ratios = []
        fps = 30  # approximate

        for frame in frames:
            if not HAS_MEDIAPIPE:
                mouth_open_ratios.append(0.0)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                mouth_open_ratios.append(0.0)
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            # Lip landmarks (13 = upper lip center, 14 = lower lip center)
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]

            # Simple mouth open ratio
            mouth_dist = abs(upper_lip.y - lower_lip.y)
            face_height = abs(landmarks[10].y - landmarks[152].y)  # top to bottom of face
            ratio = mouth_dist / face_height if face_height > 0 else 0.0
            mouth_open_ratios.append(ratio)

        mouth_signal = np.array(mouth_open_ratios)

        # 2. Extract audio activity
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            # Detect onsets (where sound starts)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
            # Convert to time series at video FPS
            audio_active = np.zeros(len(mouth_signal))
            for onset in onset_frames:
                time_sec = onset * 512 / sr
                frame_idx = int(time_sec * fps)
                if frame_idx < len(audio_active):
                    audio_active[max(0, frame_idx-2):min(len(audio_active), frame_idx+3)] = 1.0
        except:
            audio_active = np.zeros(len(mouth_signal))

        # 3. Compute cross-correlation to find lag
        if len(mouth_signal) < 5 or len(audio_active) < 5:
            return {"score": 0.5, "label": "Inconclusive", "lag_frames": 0,
                    "explanation": "Insufficient data for sync analysis"}

        # Normalize signals
        mouth_norm = (mouth_signal - np.mean(mouth_signal)) / (np.std(mouth_signal) + 1e-8)
        audio_norm = (audio_active - np.mean(audio_active)) / (np.std(audio_active) + 1e-8)

        # Cross correlation
        corr = correlate(mouth_norm, audio_norm, mode='full')
        lags = np.arange(-len(mouth_norm) + 1, len(mouth_norm))
        peak_idx = np.argmax(np.abs(corr))
        peak_lag = lags[peak_idx]
        peak_value = corr[peak_idx]

        # Score: how strong the correlation is
        score = min(1.0, max(0.0, (peak_value + 1) / 2.0))  # normalize from -1..1 to 0..1

        if abs(peak_lag) <= 3:  # within ~100ms
            label = "Lip Sync Consistent"
            explanation = f"Perfect lip-audio sync (lag: {peak_lag} frames)"
        else:
            label = "Lip Sync Mismatch Detected"
            explanation = f"Noticeable delay between lips and voice (lag: {peak_lag} frames)"

        return {
            "score": round(float(score), 2),
            "label": label,
            "lag_frames": int(peak_lag),
            "explanation": explanation
        }

    except Exception as e:
        return {
            "score": 0.5,
            "label": "Inconclusive",
            "lag_frames": 0,
            "explanation": f"AV sync analysis failed: {str(e)[:100]}"
        }

# Quick test
if __name__ == "__main__":
    print("✅ AV Sync Checker loaded successfully!")
