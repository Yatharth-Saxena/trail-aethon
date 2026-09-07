import cv2
import time
import os
import queue
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from data.session_manager import session_manager
from event_logging.event_logger import event_logger
from backend.config import SESSIONS_DIR

class VideoRecorder:
    def __init__(self):
        self.is_recording = False
        self.video_writer = None
        self.frame_queue = queue.Queue(maxsize=120)
        self.worker_thread = None
        self.recording_path: Optional[Path] = None
        self.recording_start_time: Optional[float] = None
        self.width = 1280
        self.height = 720
        self.fps = 60.0

    def start_recording(self, session_id: Optional[str] = None, width: int = 1280, height: int = 720, fps: float = 60.0) -> Dict[str, Any]:
        if self.is_recording:
            return {"status": "already_recording", "file": str(self.recording_path)}

        session_id = session_id or session_manager.active_session_id
        session_dir = SESSIONS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self.recording_path = session_dir / "recording.mp4"

        self.width = width
        self.height = height
        self.fps = max(15.0, fps)

        # FourCC codec - use mp4v for high Windows compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.recording_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        self.is_recording = True
        self.recording_start_time = time.time()
        self.worker_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.worker_thread.start()

        session_manager.update_session(has_recording=True)
        event_logger.log("RECORDING_STARTED", f"Video recording started: {self.recording_path.name}")
        return {"status": "recording_started", "file": str(self.recording_path)}

    def add_frame(self, frame: np.ndarray):
        if not self.is_recording or self.video_writer is None:
            return
        try:
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame.copy())
        except Exception:
            pass

    def _writer_worker(self):
        while self.is_recording or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.2)
                if frame is not None and self.video_writer:
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    self.video_writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Recorder Error] Worker exception: {e}")

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def stop_recording(self) -> Dict[str, Any]:
        if not self.is_recording:
            return {"status": "not_recording"}

        self.is_recording = False
        duration = round(time.time() - (self.recording_start_time or time.time()), 1)
        event_logger.log("RECORDING_STOPPED", f"Video recording stopped ({duration}s)", {"duration": duration})
        return {
            "status": "recording_stopped",
            "file": str(self.recording_path),
            "duration": duration
        }

    def save_snapshot(self, frame: np.ndarray, session_id: Optional[str] = None) -> Dict[str, Any]:
        session_id = session_id or session_manager.active_session_id
        session_dir = SESSIONS_DIR / session_id
        snapshot_dir = session_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        filename = datetime.now().strftime("%H-%M-%S") + ".jpg"
        filepath = snapshot_dir / filename

        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        event_logger.log("SNAPSHOT_CAPTURED", f"Snapshot captured: {filename}", {"path": str(filepath)})

        # Update session count
        cur_count = session_manager.active_session_data.get("snapshots_count", 0)
        session_manager.update_session(snapshots_count=cur_count + 1)

        return {
            "status": "snapshot_saved",
            "filename": filename,
            "path": str(filepath),
            "url": f"/api/snapshots/{session_id}/{filename}"
        }

video_recorder = VideoRecorder()
