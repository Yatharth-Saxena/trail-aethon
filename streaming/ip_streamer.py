import cv2
import socket
import threading
import time
import numpy as np
from typing import Dict, Any, Optional

class IPStreamer:
    def __init__(self, target_ip: str = "127.0.0.1", target_port: int = 8554):
        self.target_ip = target_ip
        self.target_port = target_port
        self.is_streaming = False
        self.sock = None
        self.lock = threading.Lock()
        self.last_frame_sent = 0
        self.fps_limit = 60 # High-speed 60 FPS streaming

    def configure(self, target_ip: str, target_port: int, enabled: Optional[bool] = None) -> Dict[str, Any]:
        with self.lock:
            self.target_ip = target_ip
            self.target_port = int(target_port)
            if enabled is not None:
                if enabled and not self.is_streaming:
                    self.start()
                elif not enabled and self.is_streaming:
                    self.stop()
        return self.get_status()

    def start(self) -> Dict[str, Any]:
        with self.lock:
            if not self.is_streaming:
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self.is_streaming = True
                except Exception as e:
                    print(f"[IPStreamer Error] Failed to open UDP socket: {e}")
                    self.is_streaming = False
        return self.get_status()

    def stop(self) -> Dict[str, Any]:
        with self.lock:
            self.is_streaming = False
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
        return self.get_status()

    def send_frame(self, frame: np.ndarray):
        if not self.is_streaming or self.sock is None:
            return

        now = time.time()
        if now - self.last_frame_sent < (1.0 / self.fps_limit):
            return
        self.last_frame_sent = now

        try:
            # Downscale frame for fast UDP transmission
            h, w = frame.shape[:2]
            scale = 480 / max(h, 1)
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ret, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                data = jpeg.tobytes()
                # UDP packets must be < 65507 bytes
                if len(data) < 65000:
                    self.sock.sendto(data, (self.target_ip, self.target_port))
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "target_ip": self.target_ip,
            "target_port": self.target_port,
            "is_streaming": self.is_streaming,
            "protocol": "UDP/MJPEG"
        }

ip_streamer = IPStreamer()
