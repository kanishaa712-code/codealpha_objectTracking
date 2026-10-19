"""Detection + tracking with Ultralytics YOLO.

Beginner-friendly overview
--------------------------
* ``load_model``      loads a pretrained YOLO model (downloads it the first time).
* ``TrackingSession`` owns ONE model and ONE tracker. You feed it frames one
  at a time with ``process(frame)``. It returns the frame with boxes drawn on
  it plus a list of detections (class, confidence, box, track ID).

Why one session per video/webcam run?
  A tracker remembers objects from earlier frames. Giving every run its own
  session means a new video never inherits IDs from the previous one.
"""
import threading
from collections import deque

import numpy as np

import config
from .drawing import color_for, draw_box, draw_trail

_load_lock = threading.Lock()  # stops two threads downloading weights at once


class ModelLoadError(Exception):
    """Raised when the YOLO model cannot be imported, downloaded or loaded."""


def load_model(model_name: str):
    """Load a pretrained YOLO model by file name (e.g. ``yolo11n.pt``).

    If ``models/<name>`` does not exist, Ultralytics tries to download it.
    """
    if model_name not in config.AVAILABLE_MODELS:
        raise ModelLoadError(f"Unknown model '{model_name}'.")
    try:
        from ultralytics import YOLO  # imported here so the app starts even if it is missing
    except ImportError as exc:
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    weights = config.MODELS_DIR / model_name
    with _load_lock:
        try:
            return YOLO(str(weights))
        except Exception as exc:  # network errors, corrupt files, etc.
            raise ModelLoadError(
                f"Could not load or download '{model_name}'. If you are offline, "
                f"download it manually and place it in the 'models/' folder "
                f"(see README, section 'Model setup'). Details: {exc}"
            ) from exc


def _to_numpy(value):
    """Convert a PyTorch tensor (or array-like) to a NumPy array."""
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


class TrackingSession:
    """Runs detection (and optionally tracking) frame by frame."""

    def __init__(self, model_name, conf, class_ids, tracker, enable_tracking,
                 show_trails, model=None):
        self.model = model if model is not None else load_model(model_name)
        self.conf = conf                    # can be changed live (webcam slider)
        self.class_ids = class_ids          # None = all classes
        self.tracker = tracker              # "bytetrack.yaml" or "botsort.yaml"
        self.enable_tracking = enable_tracking
        self.show_trails = show_trails and enable_tracking
        self.frame_index = 0
        self._trails = {}                   # track_id -> deque of (x, y)
        self._last_seen = {}                # track_id -> frame_index

    # -- public ------------------------------------------------------------
    def process(self, frame):
        """Detect/track objects in one frame and draw the results on it.

        Returns ``(frame, detections)`` where each detection is a dict:
        ``track_id`` (int or None), ``class_id``, ``class_name``,
        ``confidence`` and ``box`` (x1, y1, x2, y2).
        """
        self.frame_index += 1
        results = self._run_model(frame)
        detections = self._parse(results)

        if self.show_trails:
            self._update_trails(detections)
            for track_id, points in self._trails.items():
                draw_trail(frame, points, color_for(track_id))

        for det in detections:
            if det["track_id"] is not None:
                label = f"#{det['track_id']} {det['class_name']} {det['confidence']:.2f}"
                color = color_for(det["track_id"])
            else:
                label = f"{det['class_name']} {det['confidence']:.2f}"
                color = color_for(det["class_id"])
            draw_box(frame, det["box"], label, color)
        return frame, detections

    # -- internals ---------------------------------------------------------
    def _run_model(self, frame):
        common = dict(conf=self.conf, classes=self.class_ids, verbose=False)
        if self.enable_tracking:
            # persist=True keeps the tracker's memory between frames.
            return self.model.track(frame, persist=True, tracker=self.tracker, **common)
        return self.model.predict(frame, **common)

    def _class_name(self, class_id):
        names = self.model.names
        if isinstance(names, (list, tuple)):
            return names[class_id] if class_id < len(names) else str(class_id)
        return names.get(class_id, str(class_id))

    def _parse(self, results):
        """Turn Ultralytics results into plain Python dicts."""
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = _to_numpy(boxes.xyxy).astype(int)
        confs = _to_numpy(boxes.conf)
        class_ids = _to_numpy(boxes.cls).astype(int)
        # boxes.id is None when the tracker has not assigned IDs (e.g. detect-only mode).
        ids = [None] * len(xyxy) if boxes.id is None else _to_numpy(boxes.id).astype(int)

        detections = []
        for box, conf, cid, tid in zip(xyxy, confs, class_ids, ids):
            detections.append({
                "track_id": None if tid is None else int(tid),
                "class_id": int(cid),
                "class_name": self._class_name(int(cid)),
                "confidence": float(conf),
                "box": tuple(int(v) for v in box),
            })
        return detections

    def _update_trails(self, detections):
        for det in detections:
            tid = det["track_id"]
            if tid is None:
                continue
            x1, y1, x2, y2 = det["box"]
            trail = self._trails.setdefault(tid, deque(maxlen=config.TRAIL_LENGTH))
            trail.append(((x1 + x2) // 2, (y1 + y2) // 2))
            self._last_seen[tid] = self.frame_index
        # Forget objects that have not been seen for 2 seconds-ish (60 frames).
        for tid in [t for t, seen in self._last_seen.items() if self.frame_index - seen > 60]:
            self._trails.pop(tid, None)
            self._last_seen.pop(tid, None)
