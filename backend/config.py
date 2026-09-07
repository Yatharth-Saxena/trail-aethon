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
AI_TARGET_FPS = 60
CAMERA_MIRROR = True

# Model & Thresholds
CONFIDENCE_THRESHOLD_OBJECT = 0.50
CONFIDENCE_THRESHOLD_POSE = 0.50
CONFIDENCE_THRESHOLD_HAND = 0.50
ACTION_DEBOUNCE_SECONDS = 3.0
TEMPORAL_BUFFER_SIZE = 30

# IP Streaming Configuration
DEFAULT_STREAM_HOST = "0.0.0.0"
DEFAULT_STREAM_PORT = 8554
STREAMING_ENABLED_BY_DEFAULT = False

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
