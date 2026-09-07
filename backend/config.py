import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
CONFIGS_DIR = BASE_DIR / "experiment" / "configs"
DATASET_DIR = BASE_DIR / "dataset"

# Ensure directories exist
for p in [DATA_DIR, SESSIONS_DIR, LOGS_DIR, MODELS_DIR, CONFIGS_DIR, DATASET_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Camera Configuration
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60
CAMERA_MIRROR = True

# Ask the camera for MJPG (compressed on-device). At 720p and above an
# uncompressed YUY2 stream exceeds USB 2.0 bandwidth and the driver silently
# collapses to a few FPS. Cameras without MJPG support keep their own format.
CAMERA_PREFER_MJPG = True
# Startup probe that times real cap.read() calls, because CAP_PROP_FPS often
# reports the requested rate regardless of what the device can deliver.
CAMERA_FPS_PROBE_FRAMES = 8
CAMERA_FPS_PROBE_SECONDS = 1.2

# Frame pacing for the video path. The capture/stream loop targets this rate
# independently of AI inference so the video stays smooth even when the
# perception pipeline is slower.
STREAM_TARGET_FPS = CAMERA_FPS

# AI inference rate. Deliberately lower than STREAM_TARGET_FPS: YOLOv8 plus
# MediaPipe cannot sustain 60 Hz on CPU, and the stream loop carries the last
# known detections forward for the frames in between.
#
# Measured cost per inference pass on CPU is ~115 ms (~8.8 FPS), so on this
# class of hardware the throttle is a ceiling rather than a brake. It matters
# on a GPU/Jetson build, where uncapped inference would otherwise starve the
# 60 FPS video path.
AI_INFERENCE_FPS = 15
# Floor on the inference thread's per-cycle sleep. Without this the loop spins
# with a ~2 ms yield, competing with the capture and stream threads for the
# GIL and dragging the video path below its target FPS.
AI_MIN_IDLE_SECONDS = 0.008

# JPEG quality for the shared MJPEG encode (encoded once per frame, fanned out
# to every /video_feed consumer).
MJPEG_QUALITY = 75
SNAPSHOT_JPEG_QUALITY = 85
# Rolling window used for the capture -> telemetry-send latency metric.
LATENCY_WINDOW_FRAMES = 60

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
# Global floor. Per-class values below override this and may go lower for
# small/thin components that the nano model detects weakly.
CONFIDENCE_THRESHOLD_OBJECT = 0.45

# Per-class minimum confidence, keyed by COCO class id -> (label, min_conf).
# Small or thin items get a lower bar so they are not missed; large,
# high-contrast items keep a higher bar to suppress false positives.
OBJECT_CLASS_CONFIDENCE = {
    0:  ("Person", 0.45),
    24: ("Backpack", 0.55),
    26: ("Handbag", 0.55),
    28: ("Suitcase", 0.55),
    32: ("Sports Ball", 0.40),
    39: ("Bottle", 0.42),
    40: ("Wine Glass", 0.45),
    41: ("Cup", 0.42),
    # Cutlery and stationery are thin, low-contrast and easily missed.
    42: ("Fork", 0.32),
    43: ("Knife", 0.32),
    44: ("Spoon", 0.32),
    45: ("Bowl", 0.45),
    46: ("Banana", 0.45),
    47: ("Apple", 0.45),
    49: ("Orange", 0.45),
    63: ("Laptop", 0.50),
    64: ("Mouse", 0.38),
    65: ("Remote", 0.35),
    66: ("Keyboard", 0.45),
    67: ("Cell Phone", 0.38),
    73: ("Book", 0.45),
    74: ("Clock", 0.45),
    75: ("Vase", 0.50),
    76: ("Scissors", 0.35),
    77: ("Teddy Bear", 0.55),
}

# YOLO inference resolution, matched to AI_FRAME_WIDTH and to YOLOv8's own
# training resolution so the frame is neither up- nor downsampled.
#
# Measured on this CPU with a 640x360 source (median of 10 runs, YOLO stage
# only / total with MediaPipe Hands):
#     imgsz 480 ->  57 ms /  90 ms (11.2 FPS)   fewer boxes
#     imgsz 640 ->  77 ms / 114 ms ( 8.8 FPS)   best recall
#     imgsz 768 ->  88 ms / 120 ms ( 8.3 FPS)   worse recall
#     imgsz 960 -> 125 ms / 157 ms ( 6.4 FPS)   worse still
#
# Raising it past 640 costs FPS *and* loses detections, because upsampling a
# 640-wide frame adds no detail while pushing objects outside the scale
# distribution the model was trained on. Only increase this together with
# AI_FRAME_WIDTH, and only with a genuinely high-resolution camera source.
YOLO_INFERENCE_IMGSZ = 640
YOLO_NMS_IOU = 0.55
YOLO_MAX_DETECTIONS = 24

# Fine-tuned weights. Any of these paths (checked in order, relative to
# MODELS_DIR unless absolute) is preferred over the stock yolov8n.pt, so a
# model trained by scripts/train_detector.py is picked up automatically.
CUSTOM_DETECTOR_WEIGHTS = [
    "object_detector/weights.pt",
    "object_detector/best.pt",
]
# When true, any *.pt in models/ is considered as a fallback custom weight.
AUTODISCOVER_MODEL_WEIGHTS = True
FALLBACK_DETECTOR_WEIGHTS = "yolov8n.pt"

# ---------------------------------------------------------------------------
# Temporal stability
# ---------------------------------------------------------------------------
ACTION_DEBOUNCE_SECONDS = 3.0
TEMPORAL_BUFFER_SIZE = 30

# Detection commit voting: a track is reported once it is seen in
# DETECTION_VOTE_MIN_HITS of the last DETECTION_VOTE_WINDOW frames, and is
# dropped only after DETECTION_MAX_MISSES consecutive misses. This stops
# single-frame misses on small objects from flickering the UI/state machine.
DETECTION_VOTE_WINDOW = 6
DETECTION_VOTE_MIN_HITS = 3
DETECTION_MAX_MISSES = 8
# Minimum IoU to associate a detection with an existing track.
TRACK_IOU_MATCH = 0.28
# Frames of colour/label history voted on per track.
TRACK_VOTE_WINDOW = 15

# ---------------------------------------------------------------------------
# Object motion detection (independent object movement, not hand-driven)
# ---------------------------------------------------------------------------
# Speed is measured as bbox-centroid travel per second, expressed as a
# fraction of frame width so it is resolution independent.
OBJECT_MOVING_SPEED = 0.045
# Hysteresis: separate enter/exit thresholds prevent flapping at the boundary.
OBJECT_STILL_SPEED = 0.020
# Consecutive frames above/below threshold before the flag flips. Guards
# against camera shake and detector jitter being read as movement.
OBJECT_MOTION_CONFIRM_FRAMES = 4
OBJECT_MOTION_RELEASE_FRAMES = 6
# Centroid samples used to smooth the velocity estimate.
OBJECT_MOTION_HISTORY = 8
# An object flagged as moving while no tracked hand is in contact is reported
# as displaced (moved by something other than the operator's hands).
OBJECT_DISPLACED_DEBOUNCE_SECONDS = ACTION_DEBOUNCE_SECONDS

# ---------------------------------------------------------------------------
# MediaPipe tracking (tunable per lighting condition)
# ---------------------------------------------------------------------------
# Body-pose stage is disabled (hands-only pipeline). Kept so lighting can be
# retuned from config if a pose model is re-enabled later.
CONFIDENCE_THRESHOLD_POSE = 0.50
# Lower these in dim cabin lighting; raise them under bright even light to
# reject spurious hand candidates.
CONFIDENCE_THRESHOLD_HAND = 0.55
HAND_TRACKING_CONFIDENCE = 0.55
# 0 = lite model, 1 = full model (better fingertip accuracy).
HAND_MODEL_COMPLEXITY = 1
MAX_TRACKED_HANDS = 2
# Landmark smoothing factor (higher = more responsive, less smooth).
HAND_SMOOTHING_ALPHA = 0.6

# IP Streaming Configuration
DEFAULT_STREAM_HOST = "0.0.0.0"
DEFAULT_STREAM_PORT = 8554
STREAMING_ENABLED_BY_DEFAULT = False

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
