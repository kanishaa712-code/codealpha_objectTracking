"""Central settings for VisionTrack AI.

Everything you might want to change lives here, so the rest of the code
does not contain "magic numbers".
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# ---- Uploads -------------------------------------------------------------
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = int(os.environ.get("VT_MAX_UPLOAD_MB", "500"))
# Frames wider than this are shrunk before detection (the output video is
# shrunk too). This keeps big 4K files from taking forever on a CPU.
MAX_PROCESS_WIDTH = int(os.environ.get("VT_MAX_WIDTH", "1280"))
# Uploaded/processed files older than this are deleted automatically.
FILE_MAX_AGE_HOURS = 24
# How many videos may be processed at the same time (CPU/GPU is the limit).
MAX_CONCURRENT_JOBS = 1

# ---- Models --------------------------------------------------------------
# Pretrained COCO models from Ultralytics. "n" = nano (fastest), "s" = small
# (a bit more accurate, slower). Weights are downloaded into ./models/.
AVAILABLE_MODELS = {
    "yolo11n.pt": "YOLO11 nano (fastest, recommended)",
    "yolo11s.pt": "YOLO11 small (more accurate, slower)",
    "yolov8n.pt": "YOLOv8 nano (older, very widely used)",
}
DEFAULT_MODEL = "yolo11n.pt"

# ---- Trackers (both ship with Ultralytics) -------------------------------
TRACKERS = {
    "bytetrack.yaml": "ByteTrack (fast, default)",
    "botsort.yaml": "BoT-SORT (handles camera motion, a bit slower)",
}
DEFAULT_TRACKER = "bytetrack.yaml"

# ---- Which objects to look for (COCO class ids) --------------------------
# None means "all 80 COCO classes".
CLASS_PRESETS = {
    "all": ("All 80 COCO objects", None),
    "people": ("People only", [0]),
    "vehicles": ("Vehicles (bicycle, car, motorbike, bus, truck)", [1, 2, 3, 5, 7]),
    "people_vehicles": ("People and vehicles", [0, 1, 2, 3, 5, 7]),
    "animals": ("Animals (bird, cat, dog, horse, sheep, cow, ...)", list(range(14, 24))),
}
DEFAULT_PRESET = "all"

# ---- Confidence slider ---------------------------------------------------
MIN_CONF = 0.10
MAX_CONF = 0.90
DEFAULT_CONF = 0.35

# ---- Webcam --------------------------------------------------------------
# 0 is the first camera. Change with:  VT_CAMERA_INDEX=1 python app.py
CAMERA_INDEX = int(os.environ.get("VT_CAMERA_INDEX", "0"))
TRAIL_LENGTH = 30  # how many past positions a movement trail remembers
