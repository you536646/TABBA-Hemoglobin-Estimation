"""
=============================================================================
TABBA Web: FastAPI Server
=============================================================================
Real-time hemoglobin estimation via WebSocket.
- Serves the frontend SPA
- WebSocket endpoint for camera frame processing
- Per-client session management with 15s chemistry-correct protocol
=============================================================================
"""

import os
import uuid
import json
import base64
import asyncio
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
from typing import List
import re

# Import inference engine
from inference_engine import InferenceEngine


# =============================================================================
# APP LIFECYCLE
# =============================================================================
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("=" * 60)
    print("  TABBA Web: Initializing...")
    print("=" * 60)
    engine = InferenceEngine()
    print("=" * 60)
    print("  TABBA Web: Ready! Open http://localhost:8000")
    print("=" * 60)
    yield
    print("  TABBA Web: Shutting down...")


app = FastAPI(
    title="TABBA Web - Hemoglobin Estimation",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.get("/api/status")
async def status():
    return {
        "status": "ready",
        "model_loaded": engine is not None,
        "yolo": "best.pt",
        "ml_model": "tabba_brain_v66_kaggle.pkl (XGBoost)",
        "features": 142,
        "session_duration": 15,
    }


@app.post("/api/upload_offline")
async def upload_offline(files: List[UploadFile] = File(...)):
    """
    Handle offline video or image folder uploads.
    Extracts the last 5 seconds of frames and passes them to the InferenceEngine.
    """
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_uploads')
    os.makedirs(tmp_dir, exist_ok=True)
    
    frames_rgb = []
    
    try:
        # Check if single video file
        if len(files) == 1 and files[0].filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
            tmp_path = os.path.join(tmp_dir, files[0].filename)
            with open(tmp_path, "wb") as buffer:
                buffer.write(await files[0].read())
                
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0: fps = 24.0
            duration = total_frames / fps
            
            # Start at last 5 seconds (or start of video if it's shorter)
            start_sec = max(0.0, duration - 5.0)
            start_frame = int(start_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Set extraction interval to match precisely 3 FPS (kinetic protocol)
                frame_count += 1
                frame_interval = int(round(fps / 3.0)) if fps >= 3 else 1
                if frame_count % frame_interval != 0:
                    continue
                        
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(img_rgb)
                
                if len(frames_rgb) >= 16:  # 5 seconds at 3 FPS = 15-16 frames
                    break
                    
            cap.release()
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            
        else:
            # It's a list of images (e.g. folder upload)
            all_imgs = []
            files_sorted = sorted(files, key=lambda f: f.filename)
            for f in files_sorted:
                if not f.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_bytes = await f.read()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    all_imgs.append((f.filename, img_rgb))
            
            if not all_imgs:
                return {'status': 'error', 'message': 'No valid image files found'}
                
            # If filenames have 'time_XXX.Xs' we can extract time
            frames_with_time = []
            for fname, img in all_imgs:
                m = re.search(r'time_(\d+\.\d+)s', fname)
                if m:
                    frames_with_time.append((float(m.group(1)), img))
                
            if len(frames_with_time) == len(all_imgs):
                # We have timestamps for all of them
                max_time = max(t for t, i in frames_with_time)
                # Take last 5 seconds
                frames_rgb = [img for t, img in frames_with_time if t >= max_time - 5.0]
            else:
                # Assume chronological, take the last third of the images or max 40
                num_take = min(40, max(1, len(all_imgs) // 3))
                frames_rgb = [img for f, img in all_imgs[-num_take:]]

        if not frames_rgb:
            return {'status': 'error', 'message': 'Could not extract frames'}
            
        print(f"[*] Offline upload: Processing {len(frames_rgb)} final frames.")
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, engine.process_offline_frames, frames_rgb
        )
        
        return sanitize_for_json(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}


# =============================================================================
# WEBSOCKET (Real-Time Frame Processing)
# =============================================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = str(uuid.uuid4())[:8]
    print(f"[WS] Client {client_id} connected")

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            action = msg.get('action', 'frame')

            if action == 'start_session':
                result = engine.start_session(client_id)
                await ws.send_json(result)

            elif action == 'frame':
                # Decode base64 JPEG → numpy RGB
                frame_data = msg.get('frame', '')
                if ',' in frame_data:
                    frame_data = frame_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix

                try:
                    img_bytes = base64.b64decode(frame_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        await ws.send_json({'status': 'decode_error'})
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    await ws.send_json({'status': f'decode_error: {str(e)[:50]}'})
                    continue

                # Process frame through inference engine
                result = await asyncio.get_event_loop().run_in_executor(
                    None, engine.process_frame, client_id, img_rgb
                )

                # Remove non-serializable data (numpy arrays etc)
                clean_result = sanitize_for_json(result)
                await ws.send_json(clean_result)

            elif action == 'finalize':
                # Client signals that 15s have elapsed — compute final verdict
                result = await asyncio.get_event_loop().run_in_executor(
                    None, engine.finalize_session, client_id
                )
                clean_result = sanitize_for_json(result)
                await ws.send_json(clean_result)

            elif action == 'end_session':
                engine.end_session(client_id)
                await ws.send_json({'status': 'session_ended'})

    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} disconnected")
        engine.end_session(client_id)
    except Exception as e:
        print(f"[WS] Error for {client_id}: {e}")
        engine.end_session(client_id)


def sanitize_for_json(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
