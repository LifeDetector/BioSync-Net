"""
MODULE 6: Deep CNN Deepfake Detector
Uses a trained lightweight CNN to classify face frames as REAL or FAKE.
Trained on features derived from FaceForensics++, Celeb-DF, and DFDC datasets.
Uses ONNX Runtime for fast, lightweight inference on Render.
Falls back to frequency-domain analysis if model is unavailable.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# ─── Model Configuration ───────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_detector.onnx")
INPUT_SIZE = (224, 224)

# Try to load ONNX Runtime
_session = None
HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
    if os.path.exists(MODEL_PATH):
        _session = ort.InferenceSession(
            MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        print(f"[DeepfakeCNN] ONNX model loaded from {MODEL_PATH}")
    else:
        print(f"[DeepfakeCNN] No ONNX model at {MODEL_PATH}, using frequency analysis fallback")
except ImportError:
    print("[DeepfakeCNN] ONNX Runtime not available, using frequency analysis fallback")


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess a single frame for CNN inference."""
    # Resize to model input size
    resized = cv2.resize(frame, INPUT_SIZE)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1] then apply ImageNet normalization
    img = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # NCHW format for ONNX
    img = np.transpose(img, (2, 0, 1))
    return img


def _detect_face_roi(frame: np.ndarray) -> np.ndarray:
    """Extract face region from frame using Haar Cascade."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        # Get largest face
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]
        # Add margin
        margin = int(max(w, h) * 0.2)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        return frame[y1:y2, x1:x2]
    return frame  # Return full frame if no face found


def _frequency_domain_analysis(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Fallback: Frequency domain analysis for deepfake detection.
    Real faces have natural high-frequency textures (pores, micro-expressions).
    Deepfakes often lack these and show GAN artifacts in frequency domain.
    """
    scores = []

    for frame in frames[:30]:  # Analyze up to 30 frames
        face = _detect_face_roi(frame)
        if face is None or face.size == 0:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))

        # 1. DCT (Discrete Cosine Transform) analysis
        dct = cv2.dct(np.float32(gray))
        # GAN-generated images have less energy in high frequencies
        h, w = dct.shape
        # High-frequency region (bottom-right quadrant)
        hf_energy = np.sum(np.abs(dct[h//2:, w//2:])) / (h * w / 4)
        # Low-frequency region (top-left quadrant)
        lf_energy = np.sum(np.abs(dct[:h//4, :w//4])) / (h * w / 16)
        hf_ratio = hf_energy / (lf_energy + 1e-8)

        # 2. Laplacian variance (edge sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 3. Color channel consistency
        if len(face.shape) == 3 and face.shape[2] == 3:
            b, g, r = cv2.split(cv2.resize(face, (128, 128)))
            # Cross-channel correlation (deepfakes often have inconsistencies)
            rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
            rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
            color_consistency = (abs(rg_corr) + abs(rb_corr)) / 2
        else:
            color_consistency = 0.5

        # 4. JPEG artifact detection (GAN artifacts)
        # Block artifacts are common in deepfakes
        block_diff = 0
        for i in range(0, gray.shape[0] - 8, 8):
            for j in range(0, gray.shape[1] - 8, 8):
                block = gray[i:i+8, j:j+8]
                if i + 8 < gray.shape[0]:
                    next_block = gray[i+8:i+16, j:j+8]
                    if next_block.shape == block.shape:
                        block_diff += np.mean(np.abs(block.astype(float) - next_block.astype(float)))
        block_diff /= max(1, (gray.shape[0] // 8) * (gray.shape[1] // 8))

        # 5. Noise pattern analysis
        # Apply bilateral filter to remove texture, keep edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        noise = gray.astype(float) - denoised.astype(float)
        noise_std = np.std(noise)
        # Real images have natural noise; GAN images have structured noise
        noise_kurtosis = _kurtosis(noise.flatten())

        # Combine features into a realness score
        # Higher = more likely real
        feature_score = 0.0

        # High frequency ratio: real faces have more (0.001-0.01 range)
        feature_score += min(1.0, hf_ratio / 0.005) * 0.25

        # Laplacian variance: real faces are sharper
        feature_score += min(1.0, laplacian_var / 500) * 0.20

        # Color consistency: real faces have high natural correlation
        feature_score += color_consistency * 0.15

        # Block artifacts: lower is better (less GAN artifacts)
        feature_score += max(0, 1.0 - block_diff / 20) * 0.15

        # Noise: natural noise (moderate std, low kurtosis)
        noise_score = 1.0 if (1.5 < noise_std < 8.0 and abs(noise_kurtosis) < 5) else 0.4
        feature_score += noise_score * 0.25

        scores.append(min(1.0, max(0.0, feature_score)))

    if not scores:
        return {
            "score": 0.5,
            "label": "Inconclusive",
            "method": "frequency_analysis",
            "explanation": "No faces detected for frequency analysis"
        }

    avg_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    # Consistency bonus: real videos have consistent scores across frames
    consistency_penalty = min(0.1, std_score * 0.5)
    final_score = max(0.0, min(1.0, avg_score - consistency_penalty))

    if final_score >= 0.65:
        label = "Natural Face Textures Detected"
        explanation = f"Frequency analysis shows natural micro-textures (score: {final_score:.2f})"
    elif final_score <= 0.35:
        label = "Synthetic Artifacts Detected"
        explanation = f"GAN/deepfake frequency artifacts found (score: {final_score:.2f})"
    else:
        label = "Ambiguous Frequency Signature"
        explanation = f"Frequency analysis inconclusive (score: {final_score:.2f})"

    return {
        "score": round(final_score, 2),
        "label": label,
        "method": "frequency_analysis",
        "frames_analyzed": len(scores),
        "explanation": explanation
    }


def _kurtosis(data: np.ndarray) -> float:
    """Compute kurtosis of a 1D array."""
    n = len(data)
    if n < 4:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-8:
        return 0.0
    return float(np.mean(((data - mean) / std) ** 4) - 3.0)


def detect_deepfake_cnn(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Main entry point: Detect deepfakes using CNN model or frequency analysis fallback.

    Args:
        frames: List of video frames (BGR numpy arrays)

    Returns:
        Dictionary with score (0-1, higher = more real), label, and explanation
    """
    try:
        if len(frames) < 1:
            return {
                "score": 0.5,
                "label": "Inconclusive",
                "method": "none",
                "explanation": "Too few frames for CNN analysis"
            }

        # Try ONNX model first
        if _session is not None:
            return _run_onnx_inference(frames)

        # Fallback to frequency domain analysis
        return _frequency_domain_analysis(frames)

    except Exception as e:
        return {
            "score": 0.5,
            "label": "Inconclusive",
            "method": "error",
            "explanation": f"CNN analysis failed: {str(e)[:100]}"
        }


def _run_onnx_inference(frames: List[np.ndarray]) -> Dict[str, Any]:
    """Run ONNX model inference on frames."""
    predictions = []

    for frame in frames[:30]:  # Max 30 frames for speed
        face = _detect_face_roi(frame)
        if face is None or face.size == 0:
            continue

        preprocessed = _preprocess_frame(face)
        input_batch = np.expand_dims(preprocessed, axis=0)

        input_name = _session.get_inputs()[0].name
        output_name = _session.get_outputs()[0].name
        result = _session.run([output_name], {input_name: input_batch})

        # Sigmoid to get probability
        prob = 1.0 / (1.0 + np.exp(-result[0][0][0]))
        predictions.append(float(prob))

    if not predictions:
        return _frequency_domain_analysis(frames)

    avg_prob = float(np.mean(predictions))
    std_prob = float(np.std(predictions))

    # Score: higher = more real
    score = max(0.0, min(1.0, avg_prob))

    if score >= 0.70:
        label = "CNN: Real Face Confirmed"
        explanation = f"Neural network confirms real human face (confidence: {score:.2f})"
    elif score <= 0.30:
        label = "CNN: Deepfake Detected"
        explanation = f"Neural network detects synthetic face generation (confidence: {1-score:.2f})"
    else:
        label = "CNN: Low Confidence"
        explanation = f"Model confidence is low (score: {score:.2f})"

    return {
        "score": round(score, 2),
        "label": label,
        "method": "onnx_cnn",
        "frames_analyzed": len(predictions),
        "prediction_std": round(std_prob, 4),
        "explanation": explanation
    }


# Quick test
if __name__ == "__main__":
    print("✅ DeepfakeCNN module loaded successfully!")
    print(f"   ONNX Runtime: {'Available' if HAS_ONNX else 'Not installed'}")
    print(f"   Model: {'Loaded' if _session else 'Using frequency fallback'}")
