import cv2
from moviepy import VideoFileClip
import numpy as np
from typing import List
import tempfile
import os

def extract_frames(video_path: str, max_frames: int = 150) -> List[np.ndarray]:
    """
    Extract frames from video for biometric analysis.
    Caps at 150 frames (~5 seconds at 30fps) for speed.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback
    
    frame_count = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Sample every frame (you can adjust skip for longer videos)
        frames.append(frame)
        frame_count += 1
        if frame_count >= max_frames:
            break
    
    cap.release()
    return frames

def extract_audio(video_path: str) -> str:
    """
    Extract audio from video as .wav file (required by librosa & wav2vec).
    Returns path to the temporary .wav file.
    """
    # Create temp wav file
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, f"biosync_audio_{os.getpid()}.wav")
    
    try:
        video = VideoFileClip(video_path)
        if video.audio is not None:
            video.audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
        else:
            print("No audio stream found in video.")
        return audio_path
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        # Return empty audio path as fallback
        return audio_path  # file may not exist, modules will handle gracefully

# Test function (you can ignore)
if __name__ == "__main__":
    print("Video utils loaded successfully!")
