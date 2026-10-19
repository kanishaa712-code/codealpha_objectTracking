"""VisionTrack AI - Flask web server.

Run with:  python app.py   then open http://127.0.0.1:5000
"""
import logging
import os
import uuid

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import config
from visiontrack.detector import ModelLoadError
from visiontrack.video_processor import (JOBS, VideoError, cleanup_old_files,
                                         ffmpeg_available, probe_video)
from visiontrack.webcam import CameraError, webcam

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024


# --------------------------------------------------------------------------
# Settings validation (shared by video upload and webcam start)
# --------------------------------------------------------------------------
class SettingsError(ValueError):
    pass


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_settings(data):
    """Validate user-supplied settings and return a clean dict."""
    def get(key, default):
        value = data.get(key)
        return default if value in (None, "") else value

    try:
        conf = float(get("conf", config.DEFAULT_CONF))
    except (TypeError, ValueError):
        raise SettingsError("Confidence threshold must be a number.")
    if not config.MIN_CONF <= conf <= config.MAX_CONF:
        raise SettingsError(
            f"Confidence threshold must be between {config.MIN_CONF} and {config.MAX_CONF}.")

    model = get("model", config.DEFAULT_MODEL)
    if model not in config.AVAILABLE_MODELS:
        raise SettingsError("Unsupported model selected.")
    tracker = get("tracker", config.DEFAULT_TRACKER)
    if tracker not in config.TRACKERS:
        raise SettingsError("Unsupported tracker selected.")
    preset = get("preset", config.DEFAULT_PRESET)
    if preset not in config.CLASS_PRESETS:
        raise SettingsError("Unsupported object selection.")

    return {
        "conf": conf, "model": model, "tracker": tracker,
        "class_ids": config.CLASS_PRESETS[preset][1],
        "tracking": _as_bool(get("tracking", True)),
        "trails": _as_bool(get("trails", True)),
    }


def error(message, status):
    return jsonify({"error": message}), status


# --------------------------------------------------------------------------
# Pages and system info
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


def _device_label():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return "CPU (no CUDA GPU found)"
    except Exception:
        return "Unknown (PyTorch not installed yet)"


@app.get("/api/system")
def system_info():
    return jsonify({
        "device": _device_label(),
        "ffmpeg": ffmpeg_available(),
        "models": config.AVAILABLE_MODELS,
        "default_model": config.DEFAULT_MODEL,
        "trackers": config.TRACKERS,
        "default_tracker": config.DEFAULT_TRACKER,
        "presets": {k: v[0] for k, v in config.CLASS_PRESETS.items()},
        "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "conf": {"min": config.MIN_CONF, "max": config.MAX_CONF, "default": config.DEFAULT_CONF},
    })


# --------------------------------------------------------------------------
# Video upload -> background job
# --------------------------------------------------------------------------
@app.post("/api/upload")
def upload():
    file = request.files.get("video")
    if file is None or not file.filename:
        return error("No video file was uploaded.", 400)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        return error(f"Unsupported file type '{ext or 'none'}'. Please upload one of: {allowed}.", 400)

    try:
        settings = parse_settings(request.form)
    except SettingsError as exc:
        return error(str(exc), 400)

    config.UPLOAD_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    cleanup_old_files()

    saved = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    file.save(saved)
    try:
        probe_video(saved)  # reject broken files immediately
    except VideoError as exc:
        saved.unlink(missing_ok=True)
        return error(str(exc), 422)

    job = JOBS.create(saved, secure_filename(file.filename) or f"video{ext}", settings)
    JOBS.start(job)
    return jsonify({"job_id": job.id}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return error("Unknown job. It may have expired; please upload the video again.", 404)
    return jsonify(job.to_dict())


@app.get("/api/jobs/<job_id>/preview")
def job_preview(job_id):
    job = JOBS.get(job_id)
    if job is None or job.preview is None:
        return error("No preview available yet.", 404)
    response = Response(job.preview, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/jobs/<job_id>/cancel")
def job_cancel(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return error("Unknown job.", 404)
    job.cancel_event.set()
    return jsonify({"ok": True})


@app.get("/api/jobs/<job_id>/video")
def job_video(job_id):
    job = JOBS.get(job_id)
    if job is None or job.status != "done" or not job.output_path.exists():
        return error("The processed video is not available.", 404)
    as_download = request.args.get("download") == "1"
    return send_file(job.output_path, mimetype="video/mp4", conditional=True,
                     as_attachment=as_download,
                     download_name=job.to_dict()["download_name"])


# --------------------------------------------------------------------------
# Webcam
# --------------------------------------------------------------------------
@app.post("/api/webcam/start")
def webcam_start():
    try:
        settings = parse_settings(request.get_json(silent=True) or {})
        webcam.start(settings)
    except SettingsError as exc:
        return error(str(exc), 400)
    except CameraError as exc:
        return error(str(exc), 503)
    except ModelLoadError as exc:
        return error(str(exc), 500)
    return jsonify({"ok": True})


@app.get("/api/webcam/feed")
def webcam_feed():
    if not webcam.active:
        return error("The webcam is not running. Press Start webcam first.", 409)
    return Response(webcam.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/webcam/stop")
def webcam_stop():
    webcam.stop()
    return jsonify({"ok": True})


@app.post("/api/webcam/confidence")
def webcam_confidence():
    try:
        conf = parse_settings(request.get_json(silent=True) or {})["conf"]
    except SettingsError as exc:
        return error(str(exc), 400)
    webcam.set_confidence(conf)
    return jsonify({"ok": True})


@app.get("/api/webcam/status")
def webcam_status():
    return jsonify(webcam.status())


# --------------------------------------------------------------------------
# Errors -> JSON so the page can show a readable message
# --------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_):
    return error(f"File too large. The limit is {config.MAX_UPLOAD_MB} MB.", 413)


@app.errorhandler(Exception)
def unexpected(exc):
    if isinstance(exc, HTTPException):
        return error(exc.description, exc.code)
    app.logger.exception("Unhandled error")
    return error("Something went wrong on the server. Check the terminal for details.", 500)


if __name__ == "__main__":
    for folder in (config.UPLOAD_DIR, config.OUTPUT_DIR, config.MODELS_DIR):
        folder.mkdir(exist_ok=True)
    # 127.0.0.1 = only this computer can reach the app (recommended: webcam mode
    # streams from your camera, so do not expose it to a network).
    app.run(host=os.environ.get("VT_HOST", "127.0.0.1"),
            port=int(os.environ.get("VT_PORT", "5000")),
            threaded=True, debug=False)
