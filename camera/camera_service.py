import cv2
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Generator
from camera.overlays import draw_perception_overlays
from recording.recorder import video_recorder
from streaming.ip_streamer import ip_streamer
from backend.config import DEFAULT_CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

class CameraService:
    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.raw_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()
        
        self.show_overlays = True
        self.current_detections: List[Dict[str, Any]] = []
        self.current_hands: List[Dict[str, Any]] = []
        self.current_pose: Optional[Dict[str, Any]] = None
        self.current_action: Optional[Dict[str, Any]] = None

        self.start()

    def list_cameras(self, max_test: int = 2) -> List[Dict[str, Any]]:
        return [
            {"index": 0, "name": "Camera 0 (Integrated/USB)", "active": (self.camera_index == 0)},
            {"index": 1, "name": "Camera 1 (External USB)", "active": (self.camera_index == 1)}
        ]

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._init_camera()
        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker_thread.start()

    def _init_camera(self):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            try:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                else:
                    # Fallback standard backend
                    self.cap = cv2.VideoCapture(self.camera_index)
            except Exception as e:
                print(f"[CameraService Error] Failed to open camera {self.camera_index}: {e}")

    def switch_camera(self, new_index: int) -> Dict[str, Any]:
        self.camera_index = int(new_index)
        self._init_camera()
        return {
            "status": "switched",
            "camera_index": self.camera_index,
            "cameras": self.list_cameras()
        }

    def _generate_standby_frame(self) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Gradient background
        for y in range(720):
            frame[y, :] = int(10 + (y / 720.0) * 15)
        
        # Grid lines
        for x in range(0, 1280, 80):
            cv2.line(frame, (x, 0), (x, 720), (25, 30, 35), 1)
        for y in range(0, 720, 60):
            cv2.line(frame, (0, y), (1280, y), (25, 30, 35), 1)

        cv2.putText(frame, "AETHON SENSOR FEED - CAMERA STANDBY", (380, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 230, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Connecting to Camera {self.camera_index}...", (470, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 180, 200), 1, cv2.LINE_AA)
        return frame

    def _capture_loop(self):
        prev_time = time.time()
        frames_in_second = 0

        while self.is_running:
            frame = None
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    frame = None

            if frame is None:
                frame = self._generate_standby_frame()
                time.sleep(0.033)

            # Update FPS
            now = time.time()
            frames_in_second += 1
            if now - self.fps_timer >= 1.0:
                self.fps = round(frames_in_second / (now - self.fps_timer), 1)
                frames_in_second = 0
                self.fps_timer = now

            with self.lock:
                self.raw_frame = frame
                # Feed frame into recorder & IP streamer
                video_recorder.add_frame(frame)
                ip_streamer.send_frame(frame)

                # Render overlays if enabled
                if self.show_overlays and (self.current_detections or self.current_hands or self.current_action or self.current_pose):
                    self.annotated_frame = draw_perception_overlays(
                        frame,
                        self.current_detections,
                        self.current_hands,
                        self.current_pose,
                        self.current_action
                    )
                else:
                    self.annotated_frame = frame.copy()

            time.sleep(0.005) # Tiny yield

    def get_latest_frame(self, annotated: bool = True) -> Optional[np.ndarray]:
        with self.lock:
            if annotated and self.annotated_frame is not None:
                return self.annotated_frame.copy()
            if self.raw_frame is not None:
                return self.raw_frame.copy()
            return None

    def update_perception_state(
        self,
        detections: List[Dict[str, Any]],
        hands: List[Dict[str, Any]],
        pose: Optional[Dict[str, Any]],
        action: Optional[Dict[str, Any]]
    ):
        with self.lock:
            self.current_detections = detections
            self.current_hands = hands
            self.current_pose = pose
            self.current_action = action

    def get_jpeg_frame(self, annotated: bool = True) -> Optional[bytes]:
        frame = self.get_latest_frame(annotated=annotated)
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                return jpeg.tobytes()
        return None

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        while self.is_running:
            frame = self.get_latest_frame(annotated=self.show_overlays)
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    b_data = jpeg.tobytes()
                    header = (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(b_data)).encode('ascii') + b'\r\n\r\n'
                    )
                    yield header + b_data + b'\r\n'
            time.sleep(0.033) # ~30 FPS

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

camera_service = CameraService()
