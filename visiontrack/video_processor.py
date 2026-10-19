"""Process uploaded video files in a background thread.

Beginner-friendly overview
--------------------------
1. ``probe_video``   quickly checks the file is a readable video.
2. ``JobManager``    creates a ``Job`` and runs ``run_job`` in a thread, so the
                     web page stays responsive while frames are processed.
3. ``run_job``       reads frames -> detects/tracks -> writes the output video.
4. The page polls the job's status (progress, live preview, errors).
"""
import logging
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import cv2

import config
from .detector import ModelLoadError, TrackingSession

log = logging.getLogger("visiontrack")
_job_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_JOBS)


class VideoError(Exception):
    """The uploaded file is not a readable video."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def probe_video(path) -> dict:
    """Open the file and read one frame. Raises ``VideoError`` if that fails."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("This file could not be opened as a video. It may be corrupted or use an unsupported codec.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("The video opened but no frames could be read from it.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0 or fps > 240:  # 0, NaN or nonsense
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        height, width = frame.shape[:2]
        return {"width": width, "height": height, "fps": float(fps), "frames": max(total, 0)}
    finally:
        cap.release()


def _target_size(width, height):
    """Shrink very wide videos and keep both sides even (needed by H.264)."""
    if width > config.MAX_PROCESS_WIDTH:
        height = int(height * config.MAX_PROCESS_WIDTH / width)
        width = config.MAX_PROCESS_WIDTH
    return max(2, width - width % 2), max(2, height - height % 2)


def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    return _find_ffmpeg() is not None


def _make_browser_playable(raw: Path, final: Path) -> bool:
    """Re-encode to H.264 so every browser can play it. Returns True on success.

    OpenCV's default 'mp4v' codec is fine for download but many browsers will
    not play it inline. If ffmpeg is missing we keep the raw file instead.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                 "-movflags", "+faststart", str(final)],
                check=True, timeout=1800, capture_output=True,
            )
            raw.unlink(missing_ok=True)
            return True
        except Exception as exc:
            log.warning("ffmpeg re-encode failed, keeping raw file: %s", exc)
    raw.replace(final)
    return False


def _encode_preview(frame):
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, int(h * 960 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def cleanup_old_files():
    """Delete uploads/outputs older than FILE_MAX_AGE_HOURS."""
    cutoff = time.time() - config.FILE_MAX_AGE_HOURS * 3600
    for folder in (config.UPLOAD_DIR, config.OUTPUT_DIR):
        for item in folder.glob("*"):
            if item.name == ".gitkeep" or not item.is_file():
                continue
            try:
                if item.stat().st_mtime < cutoff:
                    item.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------
# Job object and manager
# --------------------------------------------------------------------------
class Job:
    def __init__(self, input_path: Path, original_name: str, settings: dict):
        self.id = uuid.uuid4().hex[:12]
        self.input_path = input_path
        self.original_name = original_name
        self.output_path = config.OUTPUT_DIR / f"{self.id}.mp4"
        self.settings = settings
        self.status = "queued"   # queued | loading_model | processing | encoding | done | error | cancelled
        self.message = "Waiting to start…"
        self.error = None
        self.frames_done = 0
        self.total_frames = 0
        self.progress = 0.0
        self.processing_fps = 0.0
        self.playable = True     # False if we could not make a browser-friendly MP4
        self.summary = None
        self.preview = None      # latest annotated frame as JPEG bytes
        self.cancel_event = threading.Event()

    def fail(self, message):
        self.status = "error"
        self.error = message
        self.message = message

    def to_dict(self):
        return {
            "id": self.id, "status": self.status, "message": self.message,
            "error": self.error, "progress": round(self.progress, 4),
            "frames_done": self.frames_done, "total_frames": self.total_frames,
            "processing_fps": round(self.processing_fps, 1),
            "playable": self.playable, "summary": self.summary,
            "download_name": f"visiontrack_{Path(self.original_name).stem}.mp4",
        }


def _default_factory(settings):
    return TrackingSession(
        model_name=settings["model"], conf=settings["conf"],
        class_ids=settings["class_ids"], tracker=settings["tracker"],
        enable_tracking=settings["tracking"], show_trails=settings["trails"],
    )


class JobManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create(self, input_path, original_name, settings) -> Job:
        job = Job(Path(input_path), original_name, settings)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job, session_factory=None):
        thread = threading.Thread(target=run_job, args=(job, session_factory), daemon=True)
        thread.start()
        return thread


JOBS = JobManager()


# --------------------------------------------------------------------------
# The main loop
# --------------------------------------------------------------------------
def run_job(job: Job, session_factory=None):
    """Read -> detect/track -> draw -> write, for every frame of the video."""
    cap = writer = None
    raw_path = config.OUTPUT_DIR / f"{job.id}_raw.mp4"
    try:
        info = probe_video(job.input_path)
        job.total_frames = info["frames"]
        out_w, out_h = _target_size(info["width"], info["height"])

        job.message = "Waiting for another video to finish…"
        with _job_slots:
            job.status = "loading_model"
            job.message = "Loading model (the first run may download it)…"
            session = (session_factory or _default_factory)(job.settings)

            cap = cv2.VideoCapture(str(job.input_path))
            writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     info["fps"], (out_w, out_h))
            if not writer.isOpened():
                raise RuntimeError("Could not create the output video file.")

            job.status = "processing"
            job.message = "Detecting and tracking…"
            ids_by_class = defaultdict(set)   # class -> set of track IDs
            peak_per_frame = Counter()        # class -> max objects in one frame
            started = time.time()

            while True:
                if job.cancel_event.is_set():
                    job.status, job.message = "cancelled", "Processing cancelled."
                    return
                ok, frame = cap.read()
                if not ok or frame is None:
                    break  # end of video (or an unreadable tail)
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h))

                annotated, detections = session.process(frame)
                writer.write(annotated)

                counts = Counter()
                for det in detections:
                    counts[det["class_name"]] += 1
                    if det["track_id"] is not None:
                        ids_by_class[det["class_name"]].add(det["track_id"])
                for name, n in counts.items():
                    peak_per_frame[name] = max(peak_per_frame[name], n)

                job.frames_done += 1
                elapsed = max(time.time() - started, 1e-6)
                job.processing_fps = job.frames_done / elapsed
                if job.total_frames:
                    job.progress = min(job.frames_done / job.total_frames, 0.99)
                if job.frames_done % 3 == 1:
                    job.preview = _encode_preview(annotated)

            cap.release(); cap = None
            writer.release(); writer = None

        if job.frames_done == 0:
            raise VideoError("No frames could be processed.")

        job.status, job.message = "encoding", "Preparing a browser-friendly video…"
        job.playable = _make_browser_playable(raw_path, job.output_path)
        job.summary = {
            "tracking": job.settings["tracking"],
            "frames": job.frames_done,
            "classes": [
                {"name": name, "unique_ids": len(ids_by_class[name]) if job.settings["tracking"] else None,
                 "peak_in_frame": peak_per_frame[name]}
                for name in sorted(peak_per_frame, key=lambda n: -peak_per_frame[n])
            ],
        }
        job.progress, job.status, job.message = 1.0, "done", "Finished."

    except (VideoError, ModelLoadError) as exc:
        job.fail(str(exc))
    except Exception as exc:  # anything unexpected: show a clear message, log details
        log.exception("Video processing failed")
        job.fail(f"Processing failed: {exc}")
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        job.input_path.unlink(missing_ok=True)       # no need to keep the upload
        if job.status != "done":
            raw_path.unlink(missing_ok=True)
            job.output_path.unlink(missing_ok=True)
