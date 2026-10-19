"""Live webcam detection, streamed to the browser as MJPEG.

Important: the camera is opened by the *Python server* (cv2.VideoCapture),
so webcam mode works when the app runs on the same computer as the camera.
It does not use the visitor's own browser camera.
"""
import logging
import threading
import time

import cv2

import config
from .detector import ModelLoadError, TrackingSession
from .drawing import draw_hud

log = logging.getLogger("visiontrack")


class CameraError(Exception):
    """The camera is missing, busy, blocked or stopped sending frames."""


class WebcamManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cap = None
        self._session = None
        self._active = False
        self._gen = 0                 # identifies the current run
        self._error = None
        self._fps = 0.0
        self._objects_now = 0
        self._ids_seen = set()

    # -- control -----------------------------------------------------------
    def start(self, settings):
        """Open the camera and load the model. Raises CameraError/ModelLoadError."""
        with self._lock:
            if self._active:
                raise CameraError("The webcam is already running.")
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            try:
                if not cap.isOpened():
                    raise CameraError(
                        "No camera could be opened. Check that a webcam is connected, "
                        "that no other app is using it, and that your operating system "
                        "allows camera access for Python/your terminal."
                    )
                ok, _ = cap.read()
                if not ok:
                    raise CameraError("The camera opened but is not sending video frames.")
                session = TrackingSession(
                    model_name=settings["model"], conf=settings["conf"],
                    class_ids=settings["class_ids"], tracker=settings["tracker"],
                    enable_tracking=settings["tracking"], show_trails=settings["trails"],
                )
            except Exception:
                cap.release()
                raise
            self._cap, self._session = cap, session
            self._error, self._fps, self._objects_now = None, 0.0, 0
            self._ids_seen = set()
            self._gen += 1
            self._active = True
            return self._gen

    def stop(self, gen=None):
        """Stop the webcam. If ``gen`` is given, only stop that specific run."""
        with self._lock:
            if gen is not None and gen != self._gen:
                return
            self._active = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._session = None

    def set_confidence(self, value):
        session = self._session
        if session is not None:
            session.conf = value  # takes effect on the next frame

    def status(self):
        return {
            "running": self._active, "error": self._error,
            "fps": round(self._fps, 1), "objects_now": self._objects_now,
            "unique_ids": len(self._ids_seen),
        }

    @property
    def active(self):
        return self._active

    # -- streaming ---------------------------------------------------------
    def frames(self):
        """Generator yielding multipart JPEG chunks for a <img> tag."""
        gen = self._gen
        last = time.perf_counter()
        try:
            while self._active and gen == self._gen:
                cap, session = self._cap, self._session
                if cap is None or session is None:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._error = "The camera stopped sending frames (was it unplugged?)."
                    break
                annotated, detections = session.process(frame)

                now = time.perf_counter()
                instant = 1.0 / max(now - last, 1e-6)
                last = now
                self._fps = instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant
                self._objects_now = len(detections)
                for det in detections:
                    if det["track_id"] is not None:
                        self._ids_seen.add(det["track_id"])

                draw_hud(annotated, [f"{self._fps:.1f} FPS", f"{len(detections)} objects"])
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        except Exception as exc:
            log.exception("Webcam stream failed")
            self._error = f"Webcam processing failed: {exc}"
        finally:
            self.stop(gen)  # release the camera when the stream ends


webcam = WebcamManager()
