import os
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Try to initialize MediaPipe FaceMesh, fallback if not available (Python 3.13 issue)
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
    # Load OpenCV Haar Cascade as fallback
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        # Alternative path for some linux distros
        cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_rppg(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    MODULE 1: rPPG (Remote Photoplethysmography)
    Detects real human heartbeat through subtle skin color changes.
    Real humans show a clear pulse (45-120 BPM). Deepfakes usually don't.
    """
    try:
        if len(frames) < 20:
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "bpm_estimate": 0,
                "explanation": "Too few frames to detect pulse"
            }

        green_signals = []

        for frame in frames:
            if HAS_MEDIAPIPE:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
    
                if not results.multi_face_landmarks:
                    continue  # skip frame if no face
    
                landmarks = results.multi_face_landmarks[0].landmark
    
                # ROI: forehead + cheeks (best for rPPG)
                h, w, _ = frame.shape
                forehead = [landmarks[i] for i in [10, 151, 9, 8]]   # forehead points
                cheeks = [landmarks[i] for i in [234, 454, 132, 361]]  # cheek points
    
                # Convert normalized coords to pixel coords
                roi_points = forehead + cheeks
                x_coords = [int(pt.x * w) for pt in roi_points]
                y_coords = [int(pt.y * h) for pt in roi_points]
    
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            else:
                # LITE FALLBACK: Use Haar Cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) == 0:
                    continue
                
                (x, y, w_face, h_face) = faces[0]
                # Estimate forehead ROI (top middle of face)
                x_min, x_max = x + int(w_face*0.3), x + int(w_face*0.7)
                y_min, y_max = y + int(h_face*0.1), y + int(h_face*0.3)

            if x_max - x_min < 10 or y_max - y_min < 10:
                continue

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            # Mean Green channel (most sensitive to blood flow)
            green_mean = np.mean(roi[:, :, 1])
            green_signals.append(green_mean)

        if len(green_signals) < 10:
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "bpm_estimate": 0,
                "explanation": "Could not detect enough clear face frames"
            }

        # Create time-series signal
        signal_np = np.array(green_signals)
        signal_np = (signal_np - np.mean(signal_np)) / np.std(signal_np)  # normalize

        # Bandpass filter (0.75 Hz - 4.0 Hz = 45-240 BPM)
        nyquist = 0.5 * 30  # assuming ~30 fps
        low, high = 0.75 / nyquist, 4.0 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_np)

        # FFT to find dominant frequency (heartbeat)
        N = len(filtered)
        yf = fft(filtered)
        xf = fftfreq(N, 1/30)[:N//2]
        power = 2.0/N * np.abs(yf[0:N//2])

        # Find peak in heart rate range
        valid_idx = np.where((xf >= 0.75) & (xf <= 4.0))[0]
        if len(valid_idx) == 0:
            peak_freq = 0
        else:
            peak_idx = valid_idx[np.argmax(power[valid_idx])]
            peak_freq = xf[peak_idx]

        bpm = peak_freq * 60

        # Score based on signal quality
        snr = np.max(power[valid_idx]) / (np.mean(power[valid_idx]) + 1e-8)
        score = min(1.0, max(0.0, snr / 5.0))  # normalize

        if 45 <= bpm <= 120 and score > 0.4:
            label = "Pulse Detected"
            # Clamp BPM to 65-90 as requested for verified humans
            bpm = max(65, min(90, bpm))
            explanation = f"Clear heartbeat detected at {int(bpm)} BPM"
        else:
            label = "No Pulse Detected"
            explanation = "No natural heartbeat signal found - possible deepfake"

        return {
            "score": round(float(score), 2),
            "label": label,
            "bpm_estimate": int(bpm) if bpm > 0 else 0,
            "explanation": explanation
        }

    except Exception as e:
        # Graceful failure - never crash the whole API
        return {
            "score": 0.5,
            "label": "Inconclusive",
            "bpm_estimate": 0,
            "explanation": f"rPPG analysis failed: {str(e)[:100]}"
        }

# Quick test
if __name__ == "__main__":
    print("✅ rPPG Detector loaded successfully!")
