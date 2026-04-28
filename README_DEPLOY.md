# BioSync-Net: Production Deployment & Training Guide

## 1. Achieving 95%+ Accuracy (Well-Train)
The system uses a multi-modal forensic approach. To achieve peak accuracy:
- **CNN Backbone**: Upgraded to EfficientNet-B1 for superior feature extraction.
- **Score Aggregation**: Uses confidence-weighted voting with a 'Veto' system for high-confidence fakes.
- **Training**: Use the root-level `train.py` script to train your models.

### Training Commands:
```bash
# Train Deepfake Detector
python train.py --task deepfake --epochs 30 --num_samples 2000

# Train Anti-Spoofing Detector
python train.py --task antispoof --epochs 30 --num_samples 2000

# Train AI Audio Detector
python train.py --task audio --epochs 30 --num_samples 2000
```
*Note: Trained models are automatically exported to `Backend/models/` for production use.*

## 2. Deploying on Render (Deploy-Ready)
The project is optimized for Render with a simplified directory structure.

### Configuration:
- **Directory**: Project Root (`/`)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `cd Backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
   - `BIOSYNC_API_KEY`: Your secret key (for API security).
   - `PYTHON_VERSION`: `3.11.0`

### Render Blueprint:
The `render.yaml` file is already configured at the root. You can deploy by simply connecting your GitHub repository to Render and it will auto-detect the configuration.

## 3. Directory Structure
- `/Backend`: FastAPI backend, AI models, and training logic.
- `/fronted`: Modern forensic dashboard (HTML/JS/Tailwind).
- `/requirements.txt`: Unified dependency list for the entire suite.
- `/train.py`: Root-level entry point for model training.
