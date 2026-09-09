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
# -----------------------------------------------------------------------
# 640×480 @ 30 FPS is the universal "safe" mode: every USB 2.0 webcam can
# deliver this in YUY2 without saturating the bus.  1280×720 in uncompressed
# YUY2 requires ~148 MB/s — more than USB 2.0's 60 MB/s practical throughput —
# so the driver silently throttles to ~8 FPS.  MJPG fixes that, but not all
# cameras honour the fourcc hint after the device is already open, so we use
# a safe default and let camera_service auto-upgrade when MJPG is confirmed.
DEFAULT_CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60
CAMERA_MIRROR = True

# Ask the camera for MJPG (compressed on-device) for high-bandwidth 60 FPS capture.
CAMERA_PREFER_MJPG = True
CAMERA_FPS_PROBE_FRAMES = 8
CAMERA_FPS_PROBE_SECONDS = 1.0
CAMERA_MIN_ACCEPTABLE_FPS = 15

# Frame pacing for the video path. The capture/stream loop targets this rate
# independently of AI inference so the video stays smooth even when the
# perception pipeline is slower.
STREAM_TARGET_FPS = CAMERA_FPS

# AI inference rate. Deliberately lower than STREAM_TARGET_FPS: YOLOv8 plus
# MediaPipe cannot sustain 60 Hz on CPU, and the stream loop carries the last
# known detections forward for the frames in between.
#
# Measured cost per inference pass on CPU:
#   MediaPipe Hands only : ~12-18 ms  → up to ~55-80 FPS headroom (optimized)
#   YOLO + MediaPipe Hands: ~80-90 ms  → ~11-12 FPS effective
#
# Hand tracking and YOLO now run on separate threads so the hand skeleton
# overlay updates at ~45-50 Hz (matching perceived hand motion) while the heavier
# YOLO detection runs at ~10 Hz and its results are carried forward by the
# stream loop for the frames in between.
AI_HAND_TRACKING_FPS = 48   # MediaPipe Hands & Pose — fast, drives overlay sync
AI_INFERENCE_FPS = 25       # YOLOv8 object detection — GPU accelerated on Apple Silicon M4
# Floor on each inference thread's per-cycle sleep so neither loop spins and
# starves the capture / stream threads for the GIL.
AI_MIN_IDLE_SECONDS = 0.001

# JPEG quality for the shared MJPEG encode (encoded once per frame, fanned out
# to every /video_feed consumer).
MJPEG_QUALITY = 75
SNAPSHOT_JPEG_QUALITY = 85
# Rolling window used for the capture -> telemetry-send latency metric.
LATENCY_WINDOW_FRAMES = 15

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
# Global floor. Per-class values below override this and may go lower for
# small/thin components that the nano model detects weakly.
CONFIDENCE_THRESHOLD_OBJECT = 0.28

# Per-class minimum confidence, keyed by COCO class id -> (label, min_conf).
# Low floors ensure small/thin/angled objects are detected reliably.
OBJECT_CLASS_CONFIDENCE = {
    0:  ("Person", 0.40),
    24: ("Backpack", 0.35),
    25: ("Umbrella", 0.35),
    26: ("Handbag", 0.35),
    27: ("Tie", 0.30),
    28: ("Suitcase", 0.35),
    32: ("Sports Ball", 0.30),
    39: ("Bottle", 0.28),
    40: ("Wine Glass", 0.28),
    41: ("Cup", 0.28),
    # Cutlery, tools and instruments
    42: ("Fork", 0.25),
    43: ("Knife", 0.25),
    44: ("Spoon", 0.25),
    45: ("Bowl", 0.30),
    46: ("Banana", 0.30),
    47: ("Apple", 0.30),
    48: ("Sandwich", 0.30),
    49: ("Orange", 0.30),
    50: ("Broccoli", 0.30),
    51: ("Carrot", 0.30),
    52: ("Hot Dog", 0.30),
    53: ("Pizza", 0.30),
    54: ("Donut", 0.30),
    55: ("Cake", 0.30),
    56: ("Chair", 0.40),
    57: ("Couch", 0.40),
    58: ("Potted Plant", 0.35),
    60: ("Dining Table", 0.40),
    62: ("TV", 0.35),
    63: ("Laptop", 0.32),
    64: ("Mouse", 0.26),
    65: ("Remote", 0.26),
    66: ("Keyboard", 0.30),
    67: ("Cell Phone", 0.26),
    68: ("Microwave", 0.35),
    69: ("Oven", 0.35),
    70: ("Toaster", 0.30),
    71: ("Sink", 0.35),
    72: ("Refrigerator", 0.35),
    73: ("Book", 0.30),
    74: ("Clock", 0.30),
    75: ("Vase", 0.35),
    76: ("Scissors", 0.25),
    77: ("Teddy Bear", 0.35),
    78: ("Hair Drier", 0.30),
    79: ("Toothbrush", 0.25),
}

# ---------------------------------------------------------------------------
# Domain Object Taxonomy & Desired Mission Labels
# Maps recognized physical items to spaceflight/payload mission taxonomy.
# ---------------------------------------------------------------------------
OBJECT_DOMAIN_TAXONOMY = {
    # Tools & Cutlery
    "scissors": {
        "category": "TOOL",
        "domain_name": "Payload Shears",
        "category_badge": "TOOL",
        "color_hex": "#f59e0b",
    },
    "knife": {
        "category": "TOOL",
        "domain_name": "Precision Cutter / Scalpel",
        "category_badge": "TOOL",
        "color_hex": "#ef4444",
    },
    "fork": {
        "category": "TOOL",
        "domain_name": "Sample Probe",
        "category_badge": "TOOL",
        "color_hex": "#f59e0b",
    },
    "spoon": {
        "category": "TOOL",
        "domain_name": "Reagent Scoop",
        "category_badge": "TOOL",
        "color_hex": "#f59e0b",
    },
    "toothbrush": {
        "category": "TOOL",
        "domain_name": "Micro Sampling Brush",
        "category_badge": "TOOL",
        "color_hex": "#10b981",
    },
    # Specimen Containers & Laboratory Glassware
    "bottle": {
        "category": "SPECIMEN",
        "domain_name": "Fluid Sample Flask",
        "category_badge": "SAMPLE",
        "color_hex": "#06b6d4",
    },
    "cup": {
        "category": "SPECIMEN",
        "domain_name": "Specimen Beaker",
        "category_badge": "SAMPLE",
        "color_hex": "#06b6d4",
    },
    "wine glass": {
        "category": "SPECIMEN",
        "domain_name": "Laboratory Glassware",
        "category_badge": "SAMPLE",
        "color_hex": "#8b5cf6",
    },
    "bowl": {
        "category": "SPECIMEN",
        "domain_name": "Specimen Vessel",
        "category_badge": "SAMPLE",
        "color_hex": "#06b6d4",
    },
    "vase": {
        "category": "CONTAINER",
        "domain_name": "Specimen Receptacle",
        "category_badge": "CONTAINER",
        "color_hex": "#a855f7",
    },
    # Flight Avionics & Electronics
    "laptop": {
        "category": "TECH",
        "domain_name": "Flight Terminal Console",
        "category_badge": "TECH",
        "color_hex": "#3b82f6",
    },
    "cell phone": {
        "category": "TECH",
        "domain_name": "Mobile Comm Terminal",
        "category_badge": "COMM",
        "color_hex": "#38bdf8",
    },
    "keyboard": {
        "category": "TECH",
        "domain_name": "Input Terminal Keyboard",
        "category_badge": "TECH",
        "color_hex": "#3b82f6",
    },
    "mouse": {
        "category": "TECH",
        "domain_name": "Console Pointer Device",
        "category_badge": "TECH",
        "color_hex": "#3b82f6",
    },
    "remote": {
        "category": "TECH",
        "domain_name": "Telemetry Remote",
        "category_badge": "TECH",
        "color_hex": "#3b82f6",
    },
    "clock": {
        "category": "TECH",
        "domain_name": "Mission Chronometer",
        "category_badge": "CHRONO",
        "color_hex": "#f59e0b",
    },
    "tv": {
        "category": "TECH",
        "domain_name": "Telemetry Display Monitor",
        "category_badge": "DISPLAY",
        "color_hex": "#38bdf8",
    },
    # Flight Gear & Storage
    "backpack": {
        "category": "CARGO",
        "domain_name": "EVA Gear Pack",
        "category_badge": "CARGO",
        "color_hex": "#10b981",
    },
    "handbag": {
        "category": "CARGO",
        "domain_name": "Utility Equipment Pouch",
        "category_badge": "CARGO",
        "color_hex": "#10b981",
    },
    "suitcase": {
        "category": "CARGO",
        "domain_name": "Avionics Cargo Case",
        "category_badge": "CARGO",
        "color_hex": "#10b981",
    },
    "umbrella": {
        "category": "CARGO",
        "domain_name": "Deployable Shield",
        "category_badge": "CARGO",
        "color_hex": "#64748b",
    },
    # Experiment Items & Props
    "book": {
        "category": "PAYLOAD",
        "domain_name": "Flight Operations Log",
        "category_badge": "MANUAL",
        "color_hex": "#a855f7",
    },
    "sports ball": {
        "category": "PAYLOAD",
        "domain_name": "Spherical Test Mass",
        "category_badge": "PAYLOAD",
        "color_hex": "#ec4899",
    },
    "apple": {
        "category": "SPECIMEN",
        "domain_name": "Organic Specimen (Apple)",
        "category_badge": "SPECIMEN",
        "color_hex": "#22c55e",
    },
    "banana": {
        "category": "SPECIMEN",
        "domain_name": "Organic Specimen (Banana)",
        "category_badge": "SPECIMEN",
        "color_hex": "#eab308",
    },
    "orange": {
        "category": "SPECIMEN",
        "domain_name": "Organic Specimen (Citrus)",
        "category_badge": "SPECIMEN",
        "color_hex": "#f97316",
    },
    "chair": {
        "category": "FURNITURE",
        "domain_name": "Flight Deck Seat",
        "category_badge": "CABIN",
        "color_hex": "#64748b",
    },
    "couch": {
        "category": "FURNITURE",
        "domain_name": "Crew Rest Berth",
        "category_badge": "CABIN",
        "color_hex": "#64748b",
    },
    "dining table": {
        "category": "FURNITURE",
        "domain_name": "Workstation Workbench",
        "category_badge": "CABIN",
        "color_hex": "#64748b",
    },
}

# ---------------------------------------------------------------------------
# Multi-Factor Astronaut Validation (Zero False Positive Guard)
# ---------------------------------------------------------------------------
PERSON_MIN_ASPECT_RATIO = 0.35       # min height/width (reject flat/wide false boxes)
PERSON_MAX_ASPECT_RATIO = 4.5        # max height/width (reject thin vertical false strips)
PERSON_MIN_AREA_FRAC = 0.015        # min fraction of frame area (reject tiny false positives)
PERSON_MAX_AREA_FRAC = 0.85         # max fraction of frame area (reject near-full-frame boxes)
PERSON_CEILING_ZONE_FRAC = 0.12     # top 12% of frame treated as ceiling zone — suppress detections there

# YOLO inference resolution optimized for ultra-low latency (<15ms on Apple M4 MPS)
YOLO_INFERENCE_IMGSZ = 384
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
# Temporal stability & Occlusion Tracking
# ---------------------------------------------------------------------------
ACTION_DEBOUNCE_SECONDS = 3.0
TEMPORAL_BUFFER_SIZE = 30

# Detection commit voting: a track is reported once it is seen in
# DETECTION_VOTE_MIN_HITS of the last DETECTION_VOTE_WINDOW frames.
DETECTION_VOTE_WINDOW = 8
DETECTION_VOTE_MIN_HITS = 5
# Hysteresis: frames above threshold before a track becomes confirmed.
DETECTION_CONFIRM_FRAMES_ON = 3
# Hysteresis: frames below threshold before a confirmed track loses confirmation.
# Must be > DETECTION_CONFIRM_FRAMES_ON so confirmation is sticky.
DETECTION_CONFIRM_FRAMES_OFF = 7
# Maximum frames a confirmed track is predicted (bbox extrapolated) during occlusion
# before it is deleted. Predicted frames are marked track_status="PREDICTED".
DETECTION_OCCLUSION_PREDICT_FRAMES = 8
# EMA alpha for per-track confidence smoothing (lower = smoother, higher = more reactive).
CONFIDENCE_EMA_ALPHA = 0.30
# Two-tiered miss limits for tracking:
DETECTION_MAX_MISSES_CONFIRMED = 15  # Confirmed tracks survive temporary occlusion
DETECTION_MAX_MISSES_CANDIDATE = 2   # Candidate unconfirmed tracks are pruned immediately
DETECTION_MAX_MISSES = 15
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
# Landmark smoothing factors (higher = more responsive, minimal latency).
HAND_SMOOTHING_ALPHA = 0.85
POSE_SMOOTHING_ALPHA = 0.88

# IP Streaming Configuration
DEFAULT_STREAM_HOST = "0.0.0.0"
DEFAULT_STREAM_PORT = 8554
STREAMING_ENABLED_BY_DEFAULT = False

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
