import cv2
import time
import threading
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
)


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

        if measured_fps and measured_fps < CAMERA_FPS * 0.5:
            print(
                f"[CameraService WARNING] Camera delivers only {measured_fps:.1f} FPS, "
                f"far below the configured {CAMERA_FPS}. The video path will pace at "
                f"the device rate and camera.latency_ms will read high because frames "
                f"are reused between captures. Likely causes: an uncompressed "
                f"({fourcc}) pixel format saturating USB bandwidth, a virtual/stub "
                f"camera on index {self.camera_index}, or a driver limit. Try another "
                f"index via /api/camera/select, or lower CAMERA_WIDTH/CAMERA_HEIGHT/"
                f"CAMERA_FPS in backend/config.py."
            )
        elif reported_fps > 0 and measured_fps and measured_fps < reported_fps * 0.6:
            print(
                f"[CameraService WARNING] Driver claims {reported_fps:.1f} FPS but only "
                f"{measured_fps:.1f} FPS is delivered — CAP_PROP_FPS is not honoured "
                f"on this device."
            )

        if (actual_w and actual_w != CAMERA_WIDTH) or (actual_h and actual_h != CAMERA_HEIGHT):
            print(
                f"[CameraService WARNING] Resolution mismatch: got {actual_w}x{actual_h}, "
                f"requested {CAMERA_WIDTH}x{CAMERA_HEIGHT}."
            )

    def _init_camera(self):
        with self._device_lock:
            old_cap = self.cap
            self.cap = None
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass
            try:
                import platform
                system = platform.system()
                cap = None

                # Try platform-optimized backend first
                if system == "Darwin":
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
                elif system == "Windows":
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                elif system == "Linux":
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

                # Fallback to default backend if platform-specific backend didn't open
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    cap = cv2.VideoCapture(self.camera_index)

                if cap is not None and cap.isOpened():
                    # Request an on-camera-compressed format first: at 720p+
                    # an uncompressed stream (YUY2) exceeds USB 2.0 bandwidth
                    # and the driver silently drops to a few FPS. Must be set
                    # before the resolution for DirectShow to honour it.
                    # Cameras that don't support it simply keep their format.
                    if CAMERA_PREFER_MJPG:
                        try:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                        except Exception:
                            pass
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    # Minimal driver buffering: always hand us the newest
                    # frame rather than a queued backlog.
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
                        print(f"[CameraService] Hardware camera {self.camera_index} initialized successfully.")
                        self._log_negotiated_settings(cap)
                    else:
                        cap.release()
                        self.cap = None
                        print(f"[CameraService] Camera {self.camera_index} opened but could not read frame. Using standby/browser stream.")
                else:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    self.cap = None
                    print(f"[CameraService] No physical camera found at index {self.camera_index}. Standby & browser webcam stream active.")
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
                else:
                    failed_count += 1
                    if failed_count > 30:
                        failed_count = 0
                        now = time.time()
                        if now - last_retry > 10.0:
                            last_retry = now
                            self._init_camera()
                    time.sleep(0.01)
            else:
                time.sleep(0.2)

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
        Paces camera output at `target_fps`, overlaying the most recent
        perception state and pushing to the recorder, IP streamer and the
        shared MJPEG buffer.

        AI inference runs on its own thread at its own (lower) rate; this loop
        simply reuses the last known detections for intermediate frames, so
        video stays smooth even when inference cannot keep up.
        """
        target_interval = 1.0 / self.target_fps
        frames_in_second = 0
        self.fps_timer = time.time()

        while self.is_running:
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

            elapsed = time.time() - t0
            sleep_needed = target_interval - elapsed
            if sleep_needed > 0.001:
                time.sleep(sleep_needed)
            else:
                time.sleep(0.0005)

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
        with self._jpeg_lock:
            self._jpeg_lock.notify_all()
        with self._device_lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass


camera_service = CameraService()
