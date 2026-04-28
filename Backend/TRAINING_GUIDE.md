# BioSync-Net: Dataset Training Guide

To achieve professional-grade accuracy (>95%), you must train the models using standard forensic datasets instead of synthetic data.

## 1. Recommended Datasets

### A. Deepfake Detection (Task: `deepfake`)
*   **FaceForensics++**: [GitHub](https://github.com/ondyari/FaceForensics) - The industry standard.
*   **Celeb-DF (v2)**: [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) - High-quality deepfakes.
*   **DFDC (Deepfake Detection Challenge)**: [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) - Large-scale dataset from Facebook.

### B. Anti-Spoofing (Task: `antispoof`)
*   **CASIA-SURF**: [Link](https://sites.google.com/view/face-anti-spoofing-challenge/home) - Large scale multi-modal.
*   **SiW (Spoof in the Wild)**: [Link](http://cvlab.cse.msu.edu/project-as.html).
*   **Replay-Attack**: [Link](https://www.idiap.ch/dataset/replayattack).

### C. AI Audio / Voice (Task: `audio`)
*   **ASVspoof 2019/2021**: [Link](https://www.asvspoof.org/) - The definitive voice spoofing dataset.
*   **FakeAVCeleb**: [GitHub](https://github.com/DASH-Lab/FakeAVCeleb).

## 2. Training Procedure

1.  **Download and Extract**: Extract your chosen dataset.
2.  **Organize Folders**:
    ```text
    data/
    ├── deepfake/
    │   ├── train/
    │   │   ├── real/ (place real face images here)
    │   │   └── fake/ (place deepfake face images here)
    │   └── val/
    │       ├── real/
    │       └── fake/
    ```
3.  **Run Training**:
    ```bash
    python train_model.py --task deepfake --epochs 50 --num_samples 10000
    ```
4.  **Automatic Deployment**: The script will automatically export the `.onnx` model to `models/` for the backend to use.

## 3. Deployment on Render

1.  Commit your trained `.onnx` models to your repository.
2.  The `render.yaml` and `Procfile` are already configured to serve these models.
3.  Ensure `onnxruntime` is installed in your environment.
