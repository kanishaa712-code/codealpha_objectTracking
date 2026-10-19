"""Automated tests.

These tests use a FAKE model that returns results in the same structure as
Ultralytics (boxes.xyxy / conf / cls / id). That lets us test the whole
pipeline - upload, frame loop, drawing, video writing, download, webcam
stream, error handling - without PyTorch or downloaded weights.

They do NOT test real YOLO accuracy. See README for manual tests with the
real model.

Run:  pytest -v
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import app as app_module  # noqa: E402
from visiontrack import video_processor, webcam as webcam_module  # noqa: E402
from visiontrack.detector import TrackingSession  # noqa: E402


# ---------------------------------------------------------------- fakes ----
class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids):
        self.xyxy, self.conf, self.cls, self.id = xyxy, conf, cls, ids

    def __len__(self):
        return len(self.xyxy)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Pretends to find one 'person' moving to the right; tracker ID is always 1."""
    names = {0: "person"}

    def __init__(self):
        self.calls = 0

    def _result(self, with_id):
        self.calls += 1
        x = 10 + self.calls * 3
        ids = np.array([1.0]) if with_id else None
        return [FakeResult(FakeBoxes(np.array([[x, 20, x + 40, 80]], dtype=float),
                                     np.array([0.9]), np.array([0.0]), ids))]

    def track(self, frame, persist, tracker, conf, classes, verbose):
        return self._result(True)

    def predict(self, frame, conf, classes, verbose):
        return self._result(False)


def fake_session(settings):
    return TrackingSession(
        model_name=settings["model"], conf=settings["conf"], class_ids=settings["class_ids"],
        tracker=settings["tracker"], enable_tracking=settings["tracking"],
        show_trails=settings["trails"], model=FakeModel())


def make_video(path, frames=12, size=(160, 120), fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        cv2.rectangle(img, (10 + i * 3, 20), (50 + i * 3, 80), (200, 200, 200), -1)
        writer.write(img)
    writer.release()


# ------------------------------------------------------------- fixtures ----
@pytest.fixture()
def client(tmp_path, monkeypatch):
    for name in ("UPLOAD_DIR", "OUTPUT_DIR"):
        folder = tmp_path / name
        folder.mkdir()
        monkeypatch.setattr(config, name, folder)
    monkeypatch.setattr(video_processor, "_default_factory", fake_session)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


# ---------------------------------------------------------------- tests ----
def test_index_and_system_info(client):
    assert client.get("/").status_code == 200
    data = client.get("/api/system").get_json()
    assert config.DEFAULT_MODEL in data["models"]
    assert "bytetrack.yaml" in data["trackers"]


def test_rejects_missing_file(client):
    assert client.post("/api/upload", data={}).status_code == 400


def test_rejects_unsupported_extension(client):
    from io import BytesIO
    r = client.post("/api/upload", data={"video": (BytesIO(b"hello"), "notes.txt")})
    assert r.status_code == 400
    assert "Unsupported file type" in r.get_json()["error"]


def test_rejects_invalid_video(client):
    from io import BytesIO
    r = client.post("/api/upload", data={"video": (BytesIO(b"not really a video"), "fake.mp4")})
    assert r.status_code == 422
    assert "video" in r.get_json()["error"].lower()


def test_rejects_bad_confidence(client, tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    with open(video, "rb") as fh:
        r = client.post("/api/upload", data={"video": (fh, "v.mp4"), "conf": "5"})
    assert r.status_code == 400


def test_full_video_flow_with_tracking(client, tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video, frames=12)
    with open(video, "rb") as fh:
        r = client.post("/api/upload", data={"video": (fh, "clip.mp4"), "tracking": "true", "conf": "0.4"})
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]

    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert job["status"] == "done", job
    assert job["frames_done"] == 12
    person = job["summary"]["classes"][0]
    assert person["name"] == "person" and person["unique_ids"] == 1

    assert client.get(f"/api/jobs/{job_id}/preview").status_code == 200
    video_resp = client.get(f"/api/jobs/{job_id}/video")
    assert video_resp.status_code == 200
    dl = client.get(f"/api/jobs/{job_id}/video?download=1")
    assert "attachment" in dl.headers["Content-Disposition"]

    # The output must be a readable video of the same length.
    out = tmp_path / "out.mp4"
    out.write_bytes(video_resp.data)
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 12


def test_detect_only_has_no_ids(client, tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video, frames=6)
    with open(video, "rb") as fh:
        r = client.post("/api/upload", data={"video": (fh, "clip.mp4"), "tracking": "false"})
    job_id = r.get_json()["job_id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert job["status"] == "done"
    assert job["summary"]["classes"][0]["unique_ids"] is None


def test_drawing_changes_pixels():
    frame = np.full((120, 160, 3), 40, dtype=np.uint8)
    before = frame.copy()
    session = fake_session({"model": "x", "conf": .4, "class_ids": None, "tracker": "bytetrack.yaml",
                            "tracking": True, "trails": True})
    out, dets = session.process(frame)
    assert not np.array_equal(out, before)
    assert dets[0]["track_id"] == 1 and dets[0]["class_name"] == "person"


def test_model_failure_is_reported(client, tmp_path, monkeypatch):
    from visiontrack.detector import ModelLoadError

    def broken(_settings):
        raise ModelLoadError("Could not load or download 'yolo11n.pt'.")
    monkeypatch.setattr(video_processor, "_default_factory", broken)
    video = tmp_path / "clip.mp4"
    make_video(video, frames=4)
    with open(video, "rb") as fh:
        job_id = client.post("/api/upload", data={"video": (fh, "clip.mp4")}).get_json()["job_id"]
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job["status"] == "error" and "download" in job["error"]


# -------------------------------------------------------------- webcam ----
class NoCamera:
    def __init__(self, *_):
        pass

    def isOpened(self):
        return False

    def release(self):
        pass


class FakeCamera:
    def __init__(self, *_):
        self.open = True

    def isOpened(self):
        return self.open

    def read(self):
        return True, np.full((120, 160, 3), 40, dtype=np.uint8)

    def release(self):
        self.open = False


def test_webcam_missing_camera(client, monkeypatch):
    monkeypatch.setattr(webcam_module.cv2, "VideoCapture", NoCamera)
    r = client.post("/api/webcam/start", json={})
    assert r.status_code == 503
    assert "camera" in r.get_json()["error"].lower()
    assert client.get("/api/webcam/feed").status_code == 409


def test_webcam_stream_start_stop(client, monkeypatch):
    monkeypatch.setattr(webcam_module.cv2, "VideoCapture", FakeCamera)
    # Use the fake model instead of loading real YOLO weights.
    monkeypatch.setattr(webcam_module, "TrackingSession",
                        lambda **kw: TrackingSession(model=FakeModel(), **kw))
    assert client.post("/api/webcam/start", json={"tracking": True}).status_code == 200
    gen = webcam_module.webcam.frames()
    chunk = next(gen)
    assert chunk.startswith(b"--frame") and b"image/jpeg" in chunk
    assert client.post("/api/webcam/confidence", json={"conf": 0.6}).status_code == 200
    next(gen)
    status = client.get("/api/webcam/status").get_json()
    assert status["running"] and status["unique_ids"] == 1
    client.post("/api/webcam/stop")
    assert client.get("/api/webcam/status").get_json()["running"] is False
    gen.close()
