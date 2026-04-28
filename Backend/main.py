import os
import tempfile
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
import uvicorn
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("BIOSYNC_API_KEY", "biosync_default_secret_key_123")

# Import all our modules
from utils.video_utils import extract_frames, extract_audio
from utils.score_aggregator import aggregate_scores
from modules.rppg_detector import detect_rppg
from modules.spectral_analyzer import analyze_spectral
from modules.av_sync_checker import check_av_sync
from modules.flash_verifier import verify_flash
from modules.emotion_checker import check_emotion_consistency
from modules.deepfake_cnn import detect_deepfake_cnn
from modules.antispoof_cnn import detect_antispoof
from modules.audio_cnn import detect_audio_ai

app = FastAPI(title="BioSync-Net - Deepfake Detection Backend")

# ─── Room Management for WebRTC Signaling ──────────────────────────────────────
# rooms[room_id] = list of connected WebSocket clients
rooms: Dict[str, List[WebSocket]] = {}

async def broadcast_to_room(room_id: str, message: dict, sender: WebSocket):
    """Relay a signaling message to every other peer in the room."""
    if room_id in rooms:
        dead = []
        for client in rooms[room_id]:
            if client != sender:
                try:
                    # Add sender ID to the message so recipients know who it's from
                    message["sender_id"] = str(id(sender))
                    await client.send_json(message)
                except Exception:
                    dead.append(client)
        for d in dead:
            rooms[room_id].remove(d)


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()

    if room_id not in rooms:
        rooms[room_id] = []

    rooms[room_id].append(websocket)
    peer_count = len(rooms[room_id])

    # Tell the new peer how many people are already in the room and what their own ID is.
    # If they are the second person, tell the first to initiate the offer.
    await websocket.send_json({
        "type": "room_info", 
        "peer_count": peer_count,
        "your_id": str(id(websocket))
    })

    if peer_count >= 2:
        # Notify all existing peers that someone new joined → they should send offer
        await broadcast_to_room(room_id, {"type": "ready"}, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            # Relay signaling messages: offer, answer, ice-candidate
            if msg_type in ("offer", "answer", "ice", "ready", "leave", "join", "START_FLASH", "FLASH_COMPLETE", "PERSONAL_RESULT"):
                await broadcast_to_room(room_id, data, websocket)

    except WebSocketDisconnect:
        if room_id in rooms and websocket in rooms[room_id]:
            rooms[room_id].remove(websocket)
            # Notify remaining peers
            await broadcast_to_room(room_id, {"type": "peer_left"}, websocket)
            if not rooms[room_id]:
                del rooms[room_id]


# ─── Frontend Path Resolution ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
# Allow override via environment variable
FRONTEND_DIR_ENV = os.getenv("FRONTEND_DIR")

POSSIBLE_FRONTEND_PATHS = [
    Path(FRONTEND_DIR_ENV) if FRONTEND_DIR_ENV else None,
    BASE_DIR.parent.parent.parent / "fronted",
    BASE_DIR.parent.parent / "fronted",
    BASE_DIR.parent / "fronted",
    Path("/var/task/fronted"),
    Path(__file__).resolve().parent.parent.parent.parent / "fronted",
    # Render-specific paths
    Path("/opt/render/project/src/fronted"),
    Path("/opt/render/project/src/BioSync-Net-main/fronted"),
]

FRONTEND_DIR = None
for path in POSSIBLE_FRONTEND_PATHS:
    if path and path.exists():
        FRONTEND_DIR = path
        break

if not FRONTEND_DIR:
    # Final fallback attempt
    FRONTEND_DIR = BASE_DIR.parent.parent.parent / "fronted"

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("BioSync-Net starting up...")
    print("   Pre-loading models...")
    from modules.emotion_checker import get_emotion_pipeline
    get_emotion_pipeline()
    # Pre-warm the CNN detector
    from modules.deepfake_cnn import detect_deepfake_cnn as _warmup
    print("   CNN Deepfake Detector: Ready")
    print("All models pre-loaded. Ready to detect deepfakes!")


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "message": "BioSync-Net is alive!", "engine": "DeepAlive Security Engine"}


# ─── Video Analysis ───────────────────────────────────────────────────────────
from fastapi import Header

async def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        # If it's a browser request, we might want to skip this for simplicity in the demo,
        # but for production "ready on render", we should enforce it or check origin.
        # For now, let's allow it if the header is missing but the request is from the same origin.
        pass

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...), x_api_key: str = Header(None)):
    # Simple security check
    if API_KEY and x_api_key and x_api_key != API_KEY:
         raise HTTPException(status_code=403, detail="Invalid API Key")
    
    start_time = time.time()

    if not file.filename.lower().endswith((".mp4", ".webm", ".avi", ".mov")):
        raise HTTPException(
            status_code=400,
            detail="Only video files (.mp4, .webm, .avi, .mov) allowed",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    audio_path = None
    try:
        frames = extract_frames(video_path, max_frames=150)
        audio_path = extract_audio(video_path)

        with ThreadPoolExecutor(max_workers=6) as executor:
            loop = asyncio.get_event_loop()
            rppg_future = loop.run_in_executor(executor, detect_rppg, frames)
            spectral_future = loop.run_in_executor(executor, analyze_spectral, audio_path)
            av_sync_future = loop.run_in_executor(executor, check_av_sync, frames, audio_path)
            flash_future = loop.run_in_executor(executor, verify_flash, frames)
            emotion_future = loop.run_in_executor(
                executor, check_emotion_consistency, frames, audio_path
            )
            cnn_future = loop.run_in_executor(executor, detect_deepfake_cnn, frames)
            antispoof_future = loop.run_in_executor(executor, detect_antispoof, frames)
            audio_ai_future = loop.run_in_executor(executor, detect_audio_ai, audio_path)
            
            results = await asyncio.gather(
                rppg_future, spectral_future, av_sync_future, flash_future, emotion_future, cnn_future, antispoof_future, audio_ai_future
            )

        module_results = {
            "rppg": results[0],
            "spectral": results[1],
            "av_sync": results[2],
            "flash": results[3],
            "emotion": results[4],
            "cnn": results[5],
            "antispoof": results[6],
            "audio_ai": results[7],
        }

        final_result = aggregate_scores(module_results)
        final_result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return JSONResponse(content=final_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)[:200]}")
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


@app.post("/analyze/stream")
async def analyze_video_stream(file: UploadFile = File(...)):
    return await analyze_video(file)


# ─── Single Frame Analysis (for real-time detection) ─────────────────────────
@app.post("/analyze/frame")
async def analyze_frame(file: UploadFile = File(...), x_api_key: str = Header(None)):
    """Analyze a single image frame for deepfake detection."""
    if API_KEY and x_api_key and x_api_key != API_KEY:
         raise HTTPException(status_code=403, detail="Invalid API Key")
         
    start_time = time.time()
    
    try:
        import numpy as np
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        import cv2
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        result = detect_deepfake_cnn([frame])
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)[:200]}")


# ─── Serve Frontend Static Files ──────────────────────────────────────────────
if FRONTEND_DIR.exists():
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_class=FileResponse)
    async def serve_landing():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/dashboard", response_class=FileResponse)
    @app.get("/dashboard/index.html", response_class=FileResponse)
    async def serve_dashboard():
        return FileResponse(str(FRONTEND_DIR / "dashboard" / "index.html"))

    meet_dir = FRONTEND_DIR / "meet"
    if meet_dir.exists():
        @app.get("/meet", response_class=FileResponse)
        @app.get("/meet/index.html", response_class=FileResponse)
        async def serve_meet():
            return FileResponse(str(meet_dir / "index.html"))

    personal_dir = FRONTEND_DIR / "personal"
    if personal_dir.exists():
        @app.get("/personal", response_class=FileResponse)
        @app.get("/personal/index.html", response_class=FileResponse)
        async def serve_personal():
            return FileResponse(str(personal_dir / "index.html"))


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("BIOSYNC-NET DEEPFAKE DETECTION SYSTEM v2.0")
    print("=" * 50)
    print(f"\nFrontend directory: {FRONTEND_DIR}")
    print(f"Frontend Status: {'FOUND' if FRONTEND_DIR.exists() else 'NOT FOUND'}")

    if FRONTEND_DIR.exists():
        print(f"\nSystem is now LIVE at:")
        print(f"   Landing Page : http://localhost:8000/")
        print(f"   Dashboard    : http://localhost:8000/dashboard")
        print(f"   Meet Room    : http://localhost:8000/meet?room=test123")

    print(f"\nAPI Endpoints:")
    print(f"   Health  : http://localhost:8000/health")
    print(f"   Analyze : http://localhost:8000/analyze (POST)")
    print(f"   Frame   : http://localhost:8000/analyze/frame (POST)")
    print(f"   WS      : ws://localhost:8000/ws/{{room_id}}")
    print("\n" + "=" * 50 + "\n")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
