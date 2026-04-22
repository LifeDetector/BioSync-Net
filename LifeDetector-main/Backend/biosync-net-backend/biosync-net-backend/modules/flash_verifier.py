import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Try to initialize MediaPipe Face Detection, fallback if not available
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,          # 0 = short range
        min_detection_confidence=0.5
    )
    HAS_MEDIAPIPE = True
except (AttributeError, ImportError, ModuleNotFoundError):
    HAS_MEDIAPIPE = False
    face_detection = None
    # Fallback to Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

def verify_flash(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    MODULE 4: Flash Verification
    Checks natural light reflection / brightness variation on face.
    Real humans show tiny natural brightness changes.
    Deepfakes often have unnaturally flat/constant brightness.
    """
    try:
        if len(frames) < 10:
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "brightness_cv": 0.0,
                "explanation": "Too few frames for flash analysis"
            }

        brightness_values = []

        for frame in frames:
            # Convert to HSV and get Value (brightness) channel
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]  # V = brightness

            if HAS_MEDIAPIPE:
                # Detect face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    # Use first detected face
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x_min = int(bbox.xmin * w)
                    y_min = int(bbox.ymin * h)
                    x_max = int((bbox.xmin + bbox.width) * w)
                    y_max = int((bbox.ymin + bbox.height) * h)
            else:
                # FALLBACK: Haar Cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    (x_min, y_min, w_face, h_face) = faces[0]
                    x_max = x_min + w_face
                    y_max = y_min + h_face
                else:
                    continue

            # Crop face region
            face_region = v_channel[max(0, y_min):min(frame.shape[0], y_max),
                                   max(0, x_min):min(frame.shape[1], x_max)]
            if face_region.size > 0:
                mean_brightness = np.mean(face_region)
                brightness_values.append(mean_brightness)

        if len(brightness_values) < 5:
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "brightness_cv": 0.0,
                "explanation": "Could not detect face clearly"
            }

        # Coefficient of Variation (CV = std / mean)
        brightness_np = np.array(brightness_values)
        mean_b = np.mean(brightness_np)
        std_b = np.std(brightness_np)
        cv = std_b / mean_b if mean_b > 0 else 0.0

        # Score: higher variation = more real
        score = min(1.0, cv / 0.05)   # 0.05 is a good natural threshold

        if score > 0.65:
            label = "Light Reflection Detected"
            explanation = f"Natural brightness variation (CV={cv:.3f})"
        else:
            label = "Flat Lighting - Suspicious"
            explanation = "Unnaturally constant brightness - possible deepfake"

        return {
            "score": round(float(score), 2),
            "label": label,
            "brightness_cv": round(float(cv), 4),
            "explanation": explanation
        }

    except Exception as e:
        return {
            "score": 0.5,
            "label": "Inconclusive",
            "brightness_cv": 0.0,
            "explanation": f"Flash analysis failed: {str(e)[:100]}"
        }

# Quick test
if __name__ == "__main__":
    print("✅ Flash Verifier loaded successfully!")
