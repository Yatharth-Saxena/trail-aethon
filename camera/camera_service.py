import cv2
import time
import threading
import platform
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Tuple
from camera.overlays import draw_perception_overlays
from recording.recorder import video_recorder
from streaming.ip_streamer import ip_streamer
from backend.config import (
    DEFAULT_CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    CAMERA_MIRROR,
    STREAM_TARGET_FPS,
    MJPEG_QUALITY,
    SNAPSHOT_JPEG_QUALITY,
    LATENCY_WINDOW_FRAMES,
    CAMERA_PREFER_MJPG,
    CAMERA_FPS_PROBE_SECONDS,
    CAMERA_FPS_PROBE_FRAMES,
    CAMERA_MIN_ACCEPTABLE_FPS,
)

if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


class CameraService:
    """
    Optical acquisition and frame distribution.

    Locking is deliberately split so the three consumers never serialise on
    one mutex:

    * `_device_lock` guards the VideoCapture handle only (open/read/release).
    * `_frame_lock` guards the latest raw/annotated frame slots.
    * `_perception_lock` guards the overlay state written by the AI pipeline.
    * `_jpeg_lock` guards the single shared JPEG encode.

    The MJPEG frame is encoded exactly once per frame by the stream loop and
    fanned out to every `/video_feed` consumer, rather than re-encoded per
    client per frame.
    """

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.mirror = bool(CAMERA_MIRROR)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.hw_thread: Optional[threading.Thread] = None
        self.stream_thread: Optional[threading.Thread] = None

        self._device_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._perception_lock = threading.Lock()
        self._jpeg_lock = threading.Condition()
        self._new_frame_event = threading.Event()

        self.raw_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self._latest_hw_frame: Optional[np.ndarray] = None
        self._hw_timestamp = 0.0
        self._last_client_frame_time = 0.0

        # Timestamp of the capture that produced the frame currently published.
        self._published_capture_ts = 0.0

        # Shared MJPEG buffer: encoded once, read by all stream consumers.
        # `_jpeg_seq` lets a consumer block until a genuinely new frame exists.
        self._jpeg_bytes: Optional[bytes] = None
        self._jpeg_seq: int = 0

        self.target_fps = STREAM_TARGET_FPS
        self.fps = float(CAMERA_FPS)
        self.frame_count = 0
        self.fps_timer = time.time()

        # Rolling capture -> publish latency samples (milliseconds).
        self._latency_samples: List[float] = []
        self.latency_ms = 0.0

        # Actual values negotiated with the device, filled in at open time.
        self.negotiated: Dict[str, Any] = {}

        self.show_overlays = True
        self.current_detections: List[Dict[str, Any]] = []
        self.current_hands: List[Dict[str, Any]] = []
        self.current_action: Optional[Dict[str, Any]] = None

        self.start()

    def list_cameras(self, max_test: int = 2) -> List[Dict[str, Any]]:
        return [
            {"index": 0, "name": "Camera 0 (Integrated Webcam)", "active": (self.camera_index == 0)},
            {"index": 1, "name": "Camera 1 (External 60 FPS USB)", "active": (self.camera_index == 1)}
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

    @staticmethod
    def _fourcc_to_str(value: float) -> str:
        v = int(value)
        if v <= 0:
            return "?"
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip()

    @staticmethod
    def _measure_delivered_fps(cap: cv2.VideoCapture) -> float:
        """
        Time real `cap.read()` calls instead of trusting CAP_PROP_FPS.

        Drivers routinely report the requested rate whether or not they can
        deliver it, so the property alone cannot detect a slow camera. The
        budget keeps startup short even when the device is very slow.
        """
        deadline = time.time() + CAMERA_FPS_PROBE_SECONDS
        frames = 0
        t0 = time.perf_counter()
        while frames < CAMERA_FPS_PROBE_FRAMES and time.time() < deadline:
            ok, _ = cap.read()
            if not ok:
                break
            frames += 1
        elapsed = time.perf_counter() - t0
        if frames == 0 or elapsed <= 0:
            return 0.0
        return frames / elapsed

    def _log_negotiated_settings(self, cap: cv2.VideoCapture):
        """
        Report what the device actually agreed to, and what it actually
        delivers, then warn on any shortfall against backend/config.py.
        """
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fourcc = self._fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        measured_fps = self._measure_delivered_fps(cap)

        self.negotiated = {
            "width": actual_w,
            "height": actual_h,
            "fourcc": fourcc,
            "reported_fps": round(reported_fps, 1),
            "measured_fps": round(measured_fps, 1),
            "requested_width": CAMERA_WIDTH,
            "requested_height": CAMERA_HEIGHT,
            "requested_fps": CAMERA_FPS,
        }

        print(
            f"[CameraService] Device {self.camera_index}: {actual_w}x{actual_h} "
            f"{fourcc}, driver reports {reported_fps:.1f} FPS, "
            f"measured {measured_fps:.1f} FPS "
            f"(requested {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS)"
        )

    def _try_open_cap(self, index: int):
        """Open VideoCapture with the best available backend for this platform."""
        import platform
        system = platform.system()
        cap = None
        if system == "Darwin":
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        elif system == "Windows":
            # Media Foundation (MSMF) provides true hardware 60 FPS on Windows USB webcams
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        elif system == "Linux":
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap is None or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(index)
        return cap

    def _configure_cap(self, cap, width: int, height: int, fps: int, try_mjpg: bool = True) -> bool:
        """
        Apply resolution / FPS / format hints, drain initial buffer frames,
        and return True when a valid frame is readable.
        """
        target_fps = 30 if self.camera_index == 0 else fps
        if try_mjpg:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Drain initial driver buffer frames so newest frame is always delivered
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret and frame is not None:
            return True

        # Fallback if MJPG wasn't supported: try native pixel format
        if try_mjpg:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, 0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, target_fps)
                for _ in range(3):
                    cap.read()
                ret2, frame2 = cap.read()
                if ret2 and frame2 is not None:
                    return True
            except Exception:
                pass

        return False

    def _init_camera(self):
        """
        Open the physical camera with 60 FPS configuration and minimal latency.
        """
        with self._device_lock:
            old_cap = self.cap
            self.cap = None
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass

            try:
                cap = self._try_open_cap(self.camera_index)
                if cap is None or not cap.isOpened():
                    # If requested index fails, try fallback index
                    fallback_idx = 0 if self.camera_index != 0 else 1
                    cap = self._try_open_cap(fallback_idx)
                    if cap is not None and cap.isOpened():
                        print(f"[CameraService] Camera index {self.camera_index} unavailable, fell back to index {fallback_idx}.")
                        self.camera_index = fallback_idx

                if cap is None or not cap.isOpened():
                    if cap:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    self.cap = None
                    print(f"[CameraService] No physical camera found at index {self.camera_index}. Standby & browser webcam stream active.")
                    return

                opened = self._configure_cap(
                    cap,
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT,
                    CAMERA_FPS,
                    try_mjpg=CAMERA_PREFER_MJPG
                )

                if opened:
                    self.cap = cap
                    print(f"[CameraService] Hardware camera {self.camera_index} initialized successfully.")
                    self._log_negotiated_settings(cap)
                else:
                    cap.release()
                    self.cap = None
                    print(f"[CameraService] Camera {self.camera_index} opened but could not read frame. Using standby/browser stream.")

            except Exception as e:
                print(f"[CameraService Warning] Camera {self.camera_index} probe: {e}")
                self.cap = None

    def inject_client_frame(self, frame: np.ndarray):
        """Receives a frame sent by browser webcam and injects it into the perception pipeline."""
        if self.mirror and frame is not None:
            frame = cv2.flip(frame, 1)
        now = time.time()
        with self._frame_lock:
            self._latest_hw_frame = frame
            self._hw_timestamp = now
            self._last_client_frame_time = now
            self._new_frame_event.set()

    def toggle_mirror(self) -> bool:
        """Toggles horizontal mirror/inversion on camera feed."""
        self.mirror = not self.mirror
        return self.mirror

    def _hw_capture_loop(self):
        """Continuously reads from physical webcam at hardware rate so driver buffer never stalls."""
        failed_count = 0
        last_retry = 0.0
        while self.is_running:
            with self._device_lock:
                cap = self.cap
                if cap is not None and cap.isOpened():
                    try:
                        ret, frame = cap.read()
                    except Exception:
                        ret, frame = False, None
                else:
                    ret, frame = False, None

            if ret and frame is not None:
                failed_count = 0
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                now = time.time()
                with self._frame_lock:
                    # Only use physical camera if browser webcam isn't actively providing frames
                    if (now - self._last_client_frame_time) > 2.0:
                        self._latest_hw_frame = frame
                        self._hw_timestamp = now
                        self._new_frame_event.set()
            else:
                failed_count += 1
                if failed_count > 30:
                    failed_count = 0
                    now = time.time()
                    if now - last_retry > 10.0:
                        last_retry = now
                        self._init_camera()
                time.sleep(0.005)

    def switch_camera(self, new_index: int) -> Dict[str, Any]:
        self.camera_index = int(new_index)
        with self._frame_lock:
            self._latest_hw_frame = None
            self._hw_timestamp = 0.0
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

    def _publish_jpeg(self, frame: np.ndarray):
        """Encode the display frame once and wake every waiting stream consumer."""
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ok:
            return
        data = jpeg.tobytes()
        with self._jpeg_lock:
            self._jpeg_bytes = data
            self._jpeg_seq += 1
            self._jpeg_lock.notify_all()

    def _stream_pipeline_loop(self):
        """
        Paces camera output at `target_fps` (60 FPS), overlaying the most recent
        perception state and immediately pushing to the shared MJPEG buffer with
        minimal latency.
        """
        target_interval = 1.0 / self.target_fps
        frames_in_second = 0
        self.fps_timer = time.time()

        while self.is_running:
            # Wake immediately when a new hardware frame arrives, or timeout at 16.6ms to maintain 60 FPS
            self._new_frame_event.wait(timeout=target_interval)
            self._new_frame_event.clear()

            t0 = time.time()

            # Retrieve latest frame (or standby if no camera or stale)
            frame = None
            capture_ts = 0.0
            with self._frame_lock:
                if self._latest_hw_frame is not None and (t0 - self._hw_timestamp) < 2.5:
                    frame = self._latest_hw_frame
                    capture_ts = self._hw_timestamp

            if frame is None:
                frame = self._generate_standby_frame()
                capture_ts = t0

            # Snapshot the perception state under its own short-lived lock so
            # overlay drawing never blocks the capture thread.
            with self._perception_lock:
                detections = self.current_detections
                hands = self.current_hands
                action = self.current_action
                overlays_on = self.show_overlays

            video_recorder.add_frame(frame)
            ip_streamer.send_frame(frame)

            if overlays_on and (detections or hands or action):
                annotated = draw_perception_overlays(frame, detections, hands, action)
            else:
                annotated = frame

            with self._frame_lock:
                self.raw_frame = frame
                self.annotated_frame = annotated
                self._published_capture_ts = capture_ts

            self._publish_jpeg(annotated if overlays_on else frame)

            # Capture -> publish latency, smoothed over a rolling window.
            latency = (time.time() - capture_ts) * 1000.0
            self._latency_samples.append(latency)
            if len(self._latency_samples) > LATENCY_WINDOW_FRAMES:
                del self._latency_samples[:-LATENCY_WINDOW_FRAMES]
            self.latency_ms = round(sum(self._latency_samples) / len(self._latency_samples), 1)

            frames_in_second += 1
            now = time.time()
            if now - self.fps_timer >= 1.0:
                self.fps = round(frames_in_second / (now - self.fps_timer), 1)
                frames_in_second = 0
                self.fps_timer = now

    def get_latest_frame(self, annotated: bool = True) -> Optional[np.ndarray]:
        with self._frame_lock:
            if annotated and self.annotated_frame is not None:
                return self.annotated_frame.copy()
            if self.raw_frame is not None:
                return self.raw_frame.copy()
            return None

    def get_capture_timestamp(self) -> float:
        """Capture time of the frame currently published, for latency metrics."""
        with self._frame_lock:
            return self._published_capture_ts

    def get_pipeline_latency_ms(self) -> float:
        return self.latency_ms

    def update_perception_state(
        self,
        detections: List[Dict[str, Any]],
        hands: List[Dict[str, Any]],
        action: Optional[Dict[str, Any]]
    ):
        with self._perception_lock:
            self.current_detections = detections
            self.current_hands = hands
            self.current_action = action

    def get_jpeg_frame(self, annotated: bool = True) -> Optional[bytes]:
        frame = self.get_latest_frame(annotated=annotated)
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_JPEG_QUALITY])
            if ret:
                return jpeg.tobytes()
        return None

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Serve the shared JPEG buffer. Consumers block until the stream loop
        publishes a new frame, so N clients cost one encode per frame rather
        than N, and no client can slow down capture.
        """
        last_seq = -1
        while self.is_running:
            with self._jpeg_lock:
                if self._jpeg_seq == last_seq:
                    # Timeout keeps the loop responsive to shutdown.
                    self._jpeg_lock.wait(timeout=1.0)
                if self._jpeg_bytes is None or self._jpeg_seq == last_seq:
                    continue
                data = self._jpeg_bytes
                last_seq = self._jpeg_seq

            header = (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(data)).encode('ascii') + b'\r\n\r\n'
            )
            yield header + data + b'\r\n'

    def stop(self):
        self.is_running = False
        self._new_frame_event.set()
        with self._jpeg_lock:
            self._jpeg_lock.notify_all()
        with self._device_lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
        if platform.system() == "Windows":
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass


camera_service = CameraService()
