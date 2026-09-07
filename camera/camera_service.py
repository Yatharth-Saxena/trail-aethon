import cv2
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Generator
from camera.overlays import draw_perception_overlays
from recording.recorder import video_recorder
from streaming.ip_streamer import ip_streamer
from backend.config import DEFAULT_CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_MIRROR

class CameraService:
    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.mirror = bool(CAMERA_MIRROR)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.hw_thread: Optional[threading.Thread] = None
        self.stream_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.raw_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self._latest_hw_frame: Optional[np.ndarray] = None
        self._hw_timestamp = 0.0
        
        self.target_fps = CAMERA_FPS
        self.fps = float(CAMERA_FPS)
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
        self.hw_thread = threading.Thread(target=self._hw_capture_loop, daemon=True)
        self.hw_thread.start()
        self.stream_thread = threading.Thread(target=self._stream_pipeline_loop, daemon=True)
        self.stream_thread.start()

    def _init_camera(self):
        with self.lock:
            old_cap = self.cap
            self.cap = None
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass
            try:
                # Try DirectShow first
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, _ = cap.read()
                    if not ret:
                        # Fallback to default backend if DSHOW frame read fails
                        cap.release()
                        cap = cv2.VideoCapture(self.camera_index)
                else:
                    cap = cv2.VideoCapture(self.camera_index)
                
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
            except Exception as e:
                print(f"[CameraService Error] Failed to open camera {self.camera_index}: {e}")

    def inject_client_frame(self, frame: np.ndarray):
        """Receives a frame sent by browser webcam and injects it into the perception pipeline."""
        if self.mirror and frame is not None:
            frame = cv2.flip(frame, 1)
        with self.lock:
            self._latest_hw_frame = frame
            self._hw_timestamp = time.time()
            self._last_client_frame_time = time.time()

    def toggle_mirror(self) -> bool:
        """Toggles horizontal mirror/inversion on camera feed."""
        self.mirror = not self.mirror
        return self.mirror

    def _hw_capture_loop(self):
        """Continuously reads from physical webcam at hardware rate so driver buffer never stalls."""
        failed_count = 0
        while self.is_running:
            with self.lock:
                cap = self.cap
            if cap is not None and cap.isOpened():
                try:
                    ret, frame = cap.read()
                except Exception:
                    ret, frame = False, None

                if ret and frame is not None:
                    failed_count = 0
                    if self.mirror:
                        frame = cv2.flip(frame, 1)
                    with self.lock:
                        # Only use physical camera if browser webcam isn't actively providing frames
                        if (time.time() - getattr(self, "_last_client_frame_time", 0.0)) > 2.0:
                            self._latest_hw_frame = frame
                            self._hw_timestamp = time.time()
                else:
                    failed_count += 1
                    if failed_count > 40:
                        # Reinitialize camera if read fails repeatedly
                        failed_count = 0
                        self._init_camera()
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

    def switch_camera(self, new_index: int) -> Dict[str, Any]:
        self.camera_index = int(new_index)
        self._init_camera()
        return {
            "status": "switched",
            "camera_index": self.camera_index,
            "mirror": self.mirror,
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

    def _stream_pipeline_loop(self):
        """Paces the camera output at a solid 60 FPS, overlaying annotations and pushing to stream/recorder."""
        target_interval = 1.0 / self.target_fps
        frames_in_second = 0
        self.fps_timer = time.time()

        while self.is_running:
            t0 = time.time()
            now = time.time()

            # Retrieve latest frame (or standby if no camera or stale)
            frame = None
            with self.lock:
                if self._latest_hw_frame is not None and (now - self._hw_timestamp) < 2.5:
                    frame = self._latest_hw_frame.copy()

            if frame is None:
                frame = self._generate_standby_frame()

            # Process overlays and assign
            with self.lock:
                self.raw_frame = frame
                video_recorder.add_frame(frame)
                ip_streamer.send_frame(frame)

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

            frames_in_second += 1
            if now - self.fps_timer >= 1.0:
                self.fps = round(frames_in_second / (now - self.fps_timer), 1)
                frames_in_second = 0
                self.fps_timer = now

            elapsed = time.time() - t0
            sleep_needed = target_interval - elapsed
            if sleep_needed > 0.001:
                time.sleep(sleep_needed)
            else:
                time.sleep(0.0005)

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
        target_interval = 1.0 / self.target_fps
        while self.is_running:
            t0 = time.time()
            frame = self.get_latest_frame(annotated=self.show_overlays)
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    b_data = jpeg.tobytes()
                    header = (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(b_data)).encode('ascii') + b'\r\n\r\n'
                    )
                    yield header + b_data + b'\r\n'
            elapsed = time.time() - t0
            sleep_needed = target_interval - elapsed
            if sleep_needed > 0.001:
                time.sleep(sleep_needed)
            else:
                time.sleep(0.001)

    def stop(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

camera_service = CameraService()
