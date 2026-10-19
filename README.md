# VisionTrack AI

**Object detection and tracking in video, with a web interface.**
Built for the CodeAlpha Artificial Intelligence Internship (Task: Object Detection and Tracking).

VisionTrack AI uses a pretrained **YOLO** model (via [Ultralytics](https://docs.ultralytics.com/)) to find objects in each video frame, and the built-in **ByteTrack** (or **BoT-SORT**) tracker to give every object an ID number that is reused from frame to frame. A Flask web page lets you upload a video, watch a live preview while it is processed, play the result and download it. It can also run on a webcam.

---

## Features

- Detect objects with a pretrained YOLO11 / YOLOv8 model (80 COCO classes: people, cars, dogs, bottles, ...)
- Bounding boxes with **class label, confidence score and tracking ID** (for example `#3 person 0.87`)
- **ByteTrack** (default) or **BoT-SORT** tracking, plus a "detect only" mode
- Upload a video (`.mp4 .avi .mov .mkv .webm`), see a live preview while processing, play the result, **download** it
- Webcam detection with **Start / Stop** buttons and live FPS / object counts
- Controls: model, tracker, object filter (all / people / vehicles / animals ...), confidence threshold, movement trails
- Clear error messages for: no camera, unsupported files, broken videos, model download problems, processing errors
- Responsive layout that follows your light/dark system theme

---

## Folder structure

```
CodeAlpha_ObjectDetectionTracking/
├── app.py                    # Flask server: web pages + API routes
├── config.py                 # All settings in one place (models, limits, presets)
├── download_model.py         # Optional: download model weights ahead of time
├── requirements.txt
├── README.md
├── .gitignore
├── visiontrack/
│   ├── __init__.py
│   ├── detector.py           # Loads YOLO, runs detection + tracking per frame
│   ├── drawing.py            # Draws boxes, labels, trails, FPS text with OpenCV
│   ├── video_processor.py    # Background job that processes an uploaded video
│   └── webcam.py             # Opens the camera and streams processed frames
├── templates/
│   └── index.html            # The web page
├── static/
│   ├── css/style.css
│   └── js/app.js             # Page behaviour (upload, progress, webcam controls)
├── tests/
│   └── test_app.py           # Automated tests (use a stand-in model)
├── models/                   # Model weights go here (not committed to Git)
├── uploads/                  # Temporary uploaded videos (deleted after processing)
└── outputs/                  # Processed videos (auto-deleted after 24 hours)
```

---

## How it works (beginner-friendly)

1. **You upload a video.** `app.py` checks the file type, checks the settings, and checks that OpenCV can really read a frame from it. If not, you get an error straight away.
2. **A background job starts** (`video_processor.run_job`), so the page does not freeze. It reads the video one frame at a time.
3. **Each frame goes to `TrackingSession.process`** (`detector.py`):
   - In tracking mode it calls `model.track(frame, persist=True, tracker="bytetrack.yaml")`. YOLO finds objects; the tracker matches them with objects from earlier frames and returns an ID for each. `persist=True` is what tells the tracker to remember the previous frames.
   - In detect-only mode it calls `model.predict(frame)`, which has no IDs.
4. **`drawing.py` draws** the box, label (`#ID class confidence`) and optional movement trail on the frame. The ID decides the colour, so an object keeps its colour while its ID is kept.
5. The frame is **written to the output video**. Every few frames a small JPEG is saved as the **live preview**.
6. When finished, the video is **re-encoded to H.264 with ffmpeg** (bundled via `imageio-ffmpeg`) so browsers can play it, and the page shows the video, a **Download** button and a summary table.

For the webcam, `webcam.py` opens the camera with `cv2.VideoCapture`, runs the same `process` function on each frame and sends the frames to the page as an MJPEG stream (a sequence of JPEG images the browser shows in an `<img>` tag).

### Main functions

| Function / class | File | What it does |
|---|---|---|
| `load_model(name)` | `detector.py` | Loads `models/<name>`; Ultralytics downloads it if it is missing. Raises a readable `ModelLoadError` on failure. |
| `TrackingSession.process(frame)` | `detector.py` | Detect/track one frame, draw on it, return `(frame, detections)`. |
| `draw_box`, `draw_trail`, `draw_hud` | `drawing.py` | OpenCV drawing helpers. |
| `probe_video(path)` | `video_processor.py` | Opens a file and reads one frame to prove it is a valid video. |
| `run_job(job)` | `video_processor.py` | The frame loop for an uploaded video. |
| `WebcamManager.start / frames / stop` | `webcam.py` | Open the camera, stream processed frames, release the camera. |
| `parse_settings(data)` | `app.py` | Validates confidence, model, tracker and object selection. |

---

## Requirements

### Software
- **Python 3.9 – 3.12** (a recent version is best)
- The packages in `requirements.txt` (Flask, OpenCV, NumPy, Ultralytics + PyTorch, `lap`, `imageio-ffmpeg`)
- An internet connection **once**, to install packages and to download the model weights

### Hardware
| | Minimum | Recommended |
|---|---|---|
| CPU | Any modern 64-bit CPU | 4+ cores |
| RAM | 4 GB | 8 GB or more |
| GPU | Not required | NVIDIA GPU with CUDA for much faster processing |
| Disk | ~3 GB (mostly PyTorch) | 5 GB+ |
| Webcam | Only for webcam mode | Any USB / built-in camera |

### Performance limits (please read)
- Speed depends heavily on your hardware, the model and the video resolution. As a rough guide (not measured on your machine), the **nano** model on a laptop CPU runs at single-digit to low-teens frames per second, so processing a 1-minute 30 FPS video can take several minutes. A CUDA GPU is usually many times faster. Your own FPS is shown in the page while processing.
- Videos wider than **1280 px** are shrunk before processing (change `VT_MAX_WIDTH` or `MAX_PROCESS_WIDTH` in `config.py`). The output video has the reduced size.
- Only **one video is processed at a time**; extra uploads wait.
- Upload limit: **500 MB** (`VT_MAX_UPLOAD_MB`).
- The pretrained model knows the 80 COCO classes only. It will not recognise things outside that list (for example a specific product or a license plate number).
- Small, distant, blurry or dark objects are detected less reliably.

---

## Setup

```bash
# 1. Get the code
git clone https://github.com/<your-username>/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (cmd):          .venv\Scripts\activate.bat
# macOS / Linux:          source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**NVIDIA GPU (optional):** `pip install` gives you a PyTorch build that may be CPU-only. For GPU support, install PyTorch from the selector at <https://pytorch.org/get-started/locally/> *before* step 3, then verify with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
The page header shows whether a GPU was found.

**If `lap` fails to install** (some platforms), run `pip install lapx` instead. It is a drop-in replacement.

---

## Model setup

The app uses pretrained weights from Ultralytics. They are **not stored in this repository**.

| File | Notes |
|---|---|
| `yolo11n.pt` (default) | Nano: smallest and fastest. A few MB. |
| `yolo11s.pt` | Small: more accurate, slower. |
| `yolov8n.pt` | Older nano model, widely used. |

**Automatic download:** the first time you use a model, Ultralytics downloads it. This needs internet, so the first run is slower.

**Download ahead of time (recommended, so you can see any problem early):**
```bash
python download_model.py            # yolo11n.pt
python download_model.py yolo11s.pt # another model
```

**Offline / manual download:** if the download fails (no internet, firewall), get the file from <https://github.com/ultralytics/assets/releases> on another connection and save it as `models/yolo11n.pt` (same file name). The app uses any file found in `models/`.

**Trackers:** `bytetrack.yaml` and `botsort.yaml` ship inside the Ultralytics package, so nothing extra needs downloading.

**Licence note:** Ultralytics YOLO is released under **AGPL-3.0** (with a commercial option). That is fine for learning and an internship project, but read the licence before using it in a product.

---

## Run the app

```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser. Stop the server with `Ctrl+C`.

Optional environment variables: `VT_PORT`, `VT_HOST`, `VT_CAMERA_INDEX` (which camera, default `0`), `VT_MAX_UPLOAD_MB`, `VT_MAX_WIDTH`.
Example: `VT_CAMERA_INDEX=1 python app.py` (on Windows PowerShell: `$env:VT_CAMERA_INDEX=1; python app.py`).

### Process a video file
1. Open the **Video file** tab.
2. Choose settings on the left (the defaults are fine: YOLO11 nano, Detect and track, ByteTrack, all objects, confidence 0.35).
3. Choose or drag in a video, then press **Process video**.
4. Watch the progress bar and live preview. When it finishes, the video plays on the page, a summary table appears, and **Download result** saves the processed MP4.

### Use the webcam
1. Open the **Webcam** tab and press **Start webcam**.
2. The first start loads the model (and may download it), so it can take a while.
3. Drag the **confidence slider** while it runs to change the threshold live. Model, mode, tracker and object selection apply the next time you start.
4. Press **Stop webcam** to release the camera.

**Webcam notes**
- The camera is opened by the **Python server**, so the app must run on the computer that has the camera. It does not use the camera of a remote visitor's browser.
- The server only listens on `127.0.0.1` by default, so only your own computer can see the camera stream. Do not expose it to a network without adding authentication.
- If the camera is not found, is busy in another app, or blocked by the operating system, the page shows an error. On macOS and Windows you may need to allow camera access for your terminal / Python in the system privacy settings.

---

## Understanding tracking IDs (limitations)

A tracker **does not** know what an object really is; it matches boxes between frames using position, motion and size. This is useful but imperfect. **IDs are not guaranteed to stay the same in every situation.** Common problems:

- **Occlusion:** an object hidden behind another one for long enough can return with a **new ID**.
- **Crossing paths / crowds:** similar objects that overlap can **swap IDs**.
- **Missed detections:** if the detector misses an object for several frames (blur, low light, small size), the track can break.
- **Fast motion or camera movement:** BoT-SORT handles camera motion somewhat better than ByteTrack, but neither is perfect.
- **Video cuts:** the tracker has no concept of scene changes.
- **Leaving and re-entering the frame:** normally a new ID.
- **Confidence threshold and tracking:** ByteTrack and BoT-SORT also use their own thresholds from their YAML files (inside the Ultralytics package). In tracking mode, setting a very low slider value may therefore have less effect than in detect-only mode.
- **Counts:** the summary's "Track IDs assigned" can be **higher than the real number of objects**. It is not an exact people/vehicle counter.

---

## Troubleshooting

| Message / symptom | What to do |
|---|---|
| "The 'ultralytics' package is not installed" | Activate your virtual environment and run `pip install -r requirements.txt`. |
| "Could not load or download 'yolo11n.pt'" | Check your internet connection, or download the file manually into `models/` (see Model setup). |
| "Unsupported file type" | Use `.mp4`, `.avi`, `.mov`, `.mkv` or `.webm`. |
| "This file could not be opened as a video" | The file is corrupted or uses a codec OpenCV cannot read. Convert it to a normal H.264 MP4 and try again. |
| "File too large" | Trim the video or raise `VT_MAX_UPLOAD_MB`. |
| "No camera could be opened" | Plug in / enable the camera, close other apps using it, allow camera access, or try `VT_CAMERA_INDEX=1`. |
| Processing is very slow | Use `yolo11n.pt`, a smaller video, or an NVIDIA GPU. |
| Result will not play in the browser | Make sure `imageio-ffmpeg` installed correctly (`pip install imageio-ffmpeg`). You can still download the file. |

---

## Testing

### Automated tests
```bash
pytest -v
```
The tests use a **fake model** that returns results in the same format as Ultralytics, so they run without PyTorch or downloaded weights. They check: page and API responses, rejection of missing / wrong-type / corrupted files, bad settings, the full upload → process → preview → play → download flow (with and without IDs), the output video frame count, drawing, model-load errors, webcam "no camera" handling, and the webcam stream start / confidence update / stop.
They do **not** measure real YOLO accuracy or speed.

### Manual tests with the real model
Do these once after setup.

| # | Test | Steps | Expected result |
|---|---|---|---|
| 1 | Server starts | `python app.py` | Terminal shows `Running on http://127.0.0.1:5000`; the page loads and the header shows `CPU ...` or `GPU: ...`. |
| 2 | Video, tracking | Upload a short clip with people or cars (10–30 s), defaults | Progress bar and live preview advance; boxes labelled like `#2 person 0.83`; result plays; Download saves an MP4; summary lists classes with "Track IDs assigned" and "Most seen in one frame". |
| 3 | IDs persist (mostly) | Watch an unobstructed person walk across the clip | The ID stays the same for most of the walk. After a long occlusion or crossing another person it may change. |
| 4 | Detect only | Choose "Detect only" and process again | Labels like `person 0.83` with no `#`; no trails; summary has no ID column. |
| 5 | Object filter | Choose "People only" | Only people are boxed. |
| 6 | Confidence | Process once at 0.2 and once at 0.7 | More (and less certain) detections at 0.2; fewer at 0.7. |
| 7 | Wrong file type | Upload a `.txt` or `.pdf` | Red message: "Unsupported file type ...". |
| 8 | Broken video | Rename a text file to `broken.mp4` and upload | Red message: "This file could not be opened as a video ...". |
| 9 | Webcam | Webcam tab → Start | Live feed with boxes, IDs, FPS text; stats line updates. Stop releases the camera. |
| 10 | No camera | Start webcam with no camera / camera in use | Red message "No camera could be opened ...". |
| 11 | Offline model | Disconnect internet, delete `models/*.pt`, process a video | Red message about being unable to load or download the model. |

**Example of a successful summary (illustrative layout, not captured output; numbers will differ):**

| Object | Track IDs assigned | Most seen in one frame |
|---|---|---|
| person | 6 | 4 |
| car | 3 | 2 |

### What has and has not been tested
Tested in a development sandbox without PyTorch, a camera or internet: the 11 automated tests above passed using the fake model; the Flask server started and served the page and static files; the "no camera" and "ultralytics not installed" error messages appeared as designed; the JavaScript passed a syntax check.
**Not tested by the author:** real YOLO inference and tracking, model download, a physical webcam, GPU use, and visual checking of the web page in a browser. Please run the manual tests above on your own machine and report any problems.

---

## Upload to GitHub

1. Create an **empty** repository named `CodeAlpha_ObjectDetectionTracking` on <https://github.com/new> (no README / .gitignore / licence, since they already exist here).
2. In the project folder:

```bash
git init
git add .
git status              # check: no .pt model files, no videos, no .venv listed
git commit -m "VisionTrack AI: YOLO object detection and tracking web app"
git branch -M main
git remote add origin https://github.com/<your-username>/CodeAlpha_ObjectDetectionTracking.git
git push -u origin main
```

The `.gitignore` keeps model weights, videos and the virtual environment out of the repository. If GitHub asks for a password, use a [personal access token](https://github.com/settings/tokens) or sign in with the GitHub CLI (`gh auth login`).

Suggested extras before submitting: add a screenshot of your running app to the README, and record a short demo video.

---

## Customising

- **Add a model:** put the `.pt` file in `models/` and add its name to `AVAILABLE_MODELS` in `config.py`. Object presets assume COCO class numbers, so custom-trained models need their own `CLASS_PRESETS`.
- **Change the object groups:** edit `CLASS_PRESETS` in `config.py` (COCO class IDs: <https://docs.ultralytics.com/datasets/detect/coco/>).
- **Tune the tracker:** copy `bytetrack.yaml` from the Ultralytics package, edit values such as `track_buffer` (how long a lost object is remembered), and pass its path as the `tracker` value.
