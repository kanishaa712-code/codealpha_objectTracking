"""Download pretrained YOLO weights into the ./models folder.

Usage:
    python download_model.py              # downloads yolo11n.pt (default)
    python download_model.py yolo11s.pt   # downloads another supported model

You do not have to run this: the app downloads the model on first use.
Running it first just lets you see any network problem before starting the app.
"""
import os
import sys

import config


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_MODEL
    if name not in config.AVAILABLE_MODELS:
        print(f"Unknown model '{name}'. Choose one of: {', '.join(config.AVAILABLE_MODELS)}")
        sys.exit(1)

    config.MODELS_DIR.mkdir(exist_ok=True)
    target = config.MODELS_DIR / name
    if target.exists():
        print(f"Already downloaded: {target}")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Downloading {name} ... (needs an internet connection)")
    # Ultralytics downloads a bare model name into the current folder,
    # so we step into models/ first. All paths in config.py are absolute.
    os.chdir(config.MODELS_DIR)
    try:
        YOLO(name)
    except Exception as exc:
        print(f"\nDownload failed: {exc}\n")
        print("Manual option: download the file from the Ultralytics assets releases page")
        print("  https://github.com/ultralytics/assets/releases")
        print(f"and save it as: {target}")
        sys.exit(1)

    if target.exists():
        print(f"Saved to {target}")
    else:
        print("Finished, but the file was not found in models/. Check the Ultralytics output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
