import os
import json
import asyncio
import base64
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import BASE_DIR, SESSIONS_DIR, SERVER_HOST, SERVER_PORT
from camera.camera_service import camera_service
from recording.recorder import video_recorder
from streaming.ip_streamer import ip_streamer
from experiment.manager.experiment_manager import experiment_manager
from experiment.state_machine.experiment_state import ExperimentEvent
from ai.pipeline.perception_pipeline import perception_pipeline
from voice.command_parser.aethon_command_service import aethon_command_service
from voice.speech_to_text.stt import stt_service
from voice.text_to_speech.tts import tts_service
from event_logging.event_logger import event_logger
from data.session_manager import session_manager

app = FastAPI(title="AETHON Mission Control API", version="1.0.0")

# Enable CORS for local Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
active_connections: List[WebSocket] = []

def handle_hardware_voice_command(raw_text: str):
    """Callback for audio recognized by the physical microphone."""
    try:
        perception_state = perception_pipeline.get_perception_state()
        res = aethon_command_service.handle_command(raw_text, current_perception=perception_state, source="hardware_mic")
        print(f"[AETHON Voice System] Hardware mic command processed: '{raw_text}' -> '{res.get('response')}'")
    except Exception as e:
        print(f"[AETHON Voice System] Error handling hardware mic command: {e}")

async def broadcast_telemetry():
    """Background task broadcasting real-time perception and experiment state to all connected clients."""
    while True:
        try:
            if active_connections:
                perception = perception_pipeline.get_perception_state()
                exp_state = experiment_manager.get_state()

                # Generate base64 JPEG thumbnail for WebSocket clients (e.g. legacy/alternate dashboards)
                b64_img = ""
                raw_frame = camera_service.get_latest_frame(annotated=True)
                if raw_frame is not None:
                    try:
                        h, w = raw_frame.shape[:2]
                        target_w = 640
                        target_h = int(h * (target_w / w))
                        thumb = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        b64_img = base64.b64encode(buf).decode('ascii')
                    except Exception:
                        b64_img = ""

                # End-to-end latency: camera capture -> this telemetry send.
                # camera_service reports capture -> publish; the remainder is
                # overlay/encode plus this broadcast hop.
                capture_ts = camera_service.get_capture_timestamp()
                e2e_latency_ms = (
                    round((time.time() - capture_ts) * 1000.0, 1) if capture_ts else 0.0
                )

                payload = {
                    "type": "TELEMETRY",
                    "camera": {
                        "fps": round(camera_service.fps if camera_service.fps > 0 else 60.0, 1),
                        "recording": video_recorder.is_recording,
                        "device_index": camera_service.camera_index,
                        "mirror": camera_service.mirror,
                        "latency_ms": e2e_latency_ms,
                        "pipeline_latency_ms": camera_service.get_pipeline_latency_ms(),
                        "negotiated": camera_service.negotiated
                    },
                    "perception": perception,
                    "experiment": exp_state,
                    "logs": event_logger.get_recent_logs(limit=15),
                    "image": b64_img,
                    "conversation": aethon_command_service.get_history(),
                    "voice": {
                        "hardware_listening": stt_service.is_listening,
                        "last_command": stt_service.last_command,
                        "is_active": aethon_command_service.is_active() or tts_service.is_active
                    },
                    "ai": {
                        "state": "OBSERVING" if exp_state.get("running") else "STANDBY",
                        "active": True
                    }
                }
                text_data = json.dumps(payload)
                disconnected = []
                for ws in active_connections:
                    try:
                        await ws.send_text(text_data)
                    except Exception:
                        disconnected.append(ws)
                for ws in disconnected:
                    if ws in active_connections:
                        active_connections.remove(ws)
        except Exception as e:
            pass
        await asyncio.sleep(0.08) # ~12 Hz telemetry updates

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(broadcast_telemetry())
    print("[AETHON Backend] Telemetry broadcaster started. Physical mic listener available on-demand.")

@app.websocket("/ws")
@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Send initial state immediately
        init_payload = {
            "type": "INIT",
            "camera": {
                "fps": round(camera_service.fps if camera_service.fps > 0 else 60.0, 1),
                "recording": video_recorder.is_recording,
                "device_index": camera_service.camera_index,
                "mirror": camera_service.mirror,
                "latency_ms": camera_service.get_pipeline_latency_ms(),
                "pipeline_latency_ms": camera_service.get_pipeline_latency_ms(),
                "negotiated": camera_service.negotiated,
                "devices": camera_service.list_cameras()
            },
            "experiment": experiment_manager.get_state(),
            "perception": perception_pipeline.get_perception_state(),
            "conversation": aethon_command_service.get_history(),
            "logs": event_logger.get_recent_logs(limit=30)
        }
        await websocket.send_text(json.dumps(init_payload))

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "COMMAND":
                text = msg.get("text", "")
                res = aethon_command_service.handle_command(
                    text,
                    current_perception=perception_pipeline.get_perception_state()
                )
                await websocket.send_text(json.dumps({
                    "type": "COMMAND_RESPONSE",
                    "data": res
                }))
            elif msg_type == "ACTION":
                action = msg.get("action")
                if action == "START":
                    experiment_manager.start()
                elif action == "PAUSE":
                    experiment_manager.pause()
                elif action == "RESUME":
                    experiment_manager.resume()
                elif action == "RESET":
                    experiment_manager.reset()
                elif action == "STOP":
                    experiment_manager.stop()
                elif action == "SNAPSHOT":
                    frame = camera_service.get_latest_frame(annotated=False)
                    if frame is not None:
                        video_recorder.save_snapshot(frame)
            elif msg_type == "FRAME":
                img_b64 = msg.get("image", "")
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                if img_b64:
                    try:
                        raw_bytes = base64.b64decode(img_b64)
                        np_arr = np.frombuffer(raw_bytes, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            camera_service.inject_client_frame(frame)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)

# Camera Endpoints
@app.get("/api/camera/stream")
@app.get("/api/video/stream")
@app.get("/video_feed")
@app.get("/api/video_feed")
@app.get("/api/video/feed")
def camera_stream():
    return StreamingResponse(
        camera_service.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close"
        }
    )

@app.get("/api/camera/frame")
@app.get("/api/video/frame")
def camera_single_frame(annotated: bool = True):
    data = camera_service.get_jpeg_frame(annotated=annotated)
    if data is None:
        raise HTTPException(status_code=503, detail="Frame not available yet")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )

@app.get("/api/camera/devices")
def get_camera_devices():
    return {"devices": camera_service.list_cameras(), "current": camera_service.camera_index}

@app.post("/api/camera/upload_frame")
@app.post("/api/camera/frame")
async def upload_camera_frame(request: Request):
    """Allows browser webcam to stream frames to the backend AI perception pipeline."""
    try:
        data = await request.body()
        if not data:
            return JSONResponse({"status": "error", "message": "No data received"}, status_code=400)
        
        content_type = request.headers.get("content-type", "")
        if "json" in content_type:
            body = json.loads(data)
            img_b64 = body.get("image", "")
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            raw_bytes = base64.b64decode(img_b64)
        else:
            raw_bytes = data
            
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            camera_service.inject_client_frame(frame)
            return JSONResponse({
                "status": "ok",
                "person_detected": perception_pipeline.person_detected()
            })
        return JSONResponse({"status": "error", "message": "Decode failed"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/camera/select")
def select_camera(payload: Dict[str, Any]):
    idx = payload.get("index", 0)
    return camera_service.switch_camera(int(idx))

@app.post("/api/camera/snapshot")
def take_snapshot():
    frame = camera_service.get_latest_frame(annotated=False)
    if frame is None:
        raise HTTPException(status_code=500, detail="No camera frame available")
    return video_recorder.save_snapshot(frame)

@app.post("/api/camera/record/start")
def start_record():
    return video_recorder.start_recording()

@app.post("/api/camera/record/stop")
def stop_record():
    return video_recorder.stop_recording()

@app.post("/api/camera/overlays/toggle")
def toggle_overlays():
    camera_service.show_overlays = not camera_service.show_overlays
    return {"show_overlays": camera_service.show_overlays}

@app.post("/api/camera/mirror/toggle")
def toggle_camera_mirror():
    new_state = camera_service.toggle_mirror()
    return {"mirror": new_state}

@app.get("/api/camera/mirror")
def get_camera_mirror():
    return {"mirror": camera_service.mirror}

# Experiment Endpoints
@app.get("/api/experiment/status")
def get_experiment_status():
    return experiment_manager.get_state()

@app.post("/api/experiment/start")
def start_experiment():
    return experiment_manager.start()

@app.post("/api/experiment/pause")
def pause_experiment():
    return experiment_manager.pause()

@app.post("/api/experiment/resume")
def resume_experiment():
    return experiment_manager.resume()

@app.post("/api/experiment/reset")
def reset_experiment():
    return experiment_manager.reset()

@app.post("/api/experiment/stop")
def stop_experiment():
    return experiment_manager.stop()

@app.post("/api/experiment/simulate-action")
def simulate_action(payload: Dict[str, Any]):
    """DEMO / TEST MODE endpoint allowing explicit simulated action injection."""
    event = ExperimentEvent(
        event=payload.get("event", "PICK_UP"),
        object=payload.get("object", "Object A"),
        target=payload.get("target"),
        actor="person",
        hand=payload.get("hand", "Right"),
        confidence=float(payload.get("confidence", 0.95)),
        source="SIMULATION"
    )
    res = experiment_manager.process_event(event)
    return {
        "mode": "DEMO / TEST MODE",
        "validation": res.model_dump(),
        "state": experiment_manager.get_state()
    }

# Compatibility routes for legacy client hooks
@app.get("/api/video/source")
@app.post("/api/video/source")
def api_video_source(payload: Optional[Dict[str, Any]] = None):
    return {"status": "ok", "source": "webcam", "camera_index": camera_service.camera_index}

@app.post("/api/state/reset")
def api_state_reset():
    return experiment_manager.reset()

@app.get("/api/state")
@app.get("/api/experiment/state")
def api_get_state():
    return experiment_manager.get_state()

@app.post("/api/video/scenario/nominal")
def api_video_scenario_nominal():
    return {"status": "ok", "scenario": "nominal"}

# Voice & Assistant Endpoints
@app.post("/api/voice/command")
def execute_command(payload: Dict[str, Any]):
    text = payload.get("text", "")
    perception_state = perception_pipeline.get_perception_state()
    return aethon_command_service.handle_command(text, current_perception=perception_state)

@app.post("/api/voice/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    contents = await audio.read()
    transcribed_text = stt_service.transcribe_audio_bytes(contents)
    if not transcribed_text:
        return {"status": "unrecognized", "text": ""}
    
    perception_state = perception_pipeline.get_perception_state()
    res = aethon_command_service.handle_command(transcribed_text, current_perception=perception_state)
    return {
        "status": "success",
        "transcribed_text": transcribed_text,
        "assistant_response": res
    }

@app.get("/api/voice/history")
def get_voice_history():
    return {"history": aethon_command_service.get_history()}

@app.get("/api/voice/status")
def get_voice_status():
    return {
        "hardware_mic_listening": stt_service.is_listening,
        "last_command": stt_service.last_command,
        "history_count": len(aethon_command_service.get_history())
    }

@app.post("/api/voice/mic/toggle")
def toggle_voice_mic():
    if stt_service.is_listening:
        stt_service.stop_background_listener()
    else:
        stt_service.start_background_listener(on_command_callback=handle_hardware_voice_command)
    return {"hardware_mic_listening": stt_service.is_listening}

# Logs and Sessions Endpoints
@app.get("/api/logs")
def get_logs(limit: int = 50, event_type: Optional[str] = None):
    return {"logs": event_logger.get_recent_logs(limit=limit, event_type=event_type)}

@app.get("/api/sessions")
def get_sessions():
    return {"sessions": session_manager.list_sessions(), "active_session": session_manager.active_session_data}

@app.get("/api/sessions/{session_id}/logs")
def get_session_logs(session_id: str, limit: int = 100):
    return {"session_id": session_id, "logs": event_logger.get_session_logs(session_id, limit=limit)}

@app.get("/api/snapshots/{session_id}/{filename}")
def get_snapshot_file(session_id: str, filename: str):
    path = SESSIONS_DIR / session_id / "snapshots" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(str(path), media_type="image/jpeg")

# IP Streaming Endpoints
@app.get("/api/streaming/config")
def get_streaming_config():
    return ip_streamer.get_status()

@app.post("/api/streaming/config")
def set_streaming_config(payload: Dict[str, Any]):
    target_ip = payload.get("ip", "127.0.0.1")
    target_port = int(payload.get("port", 8554))
    enabled = payload.get("enabled", False)
    return ip_streamer.configure(target_ip, target_port, enabled=enabled)

# Serve built frontend in production if dist/public exists
DIST_PUBLIC = BASE_DIR / "dist" / "public"
if DIST_PUBLIC.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_PUBLIC / "assets")), name="assets")
    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = DIST_PUBLIC / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(DIST_PUBLIC / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
