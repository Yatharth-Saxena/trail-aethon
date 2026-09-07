"""
AETHON - Custom Detector Fine-Tuning Utility

Fine-tunes a lightweight YOLO model on the annotated frames in dataset/ and
writes the result to models/object_detector/weights.pt.

That output path is the first entry in CUSTOM_DETECTOR_WEIGHTS
(backend/config.py), so the perception pipeline picks the fine-tuned model up
automatically on the next backend start and falls back to the stock
yolov8n.pt only when no custom weights exist.

Workflow:
  1. python scripts/collect_dataset.py   # capture + label frames
  2. python scripts/train_detector.py    # fine-tune
  3. restart the backend                 # weights are auto-detected

Usage:
  python scripts/train_detector.py [--epochs 25] [--imgsz 640] [--base yolov8n.pt]
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import (  # noqa: E402  (path setup must precede import)
    DATASET_DIR,
    MODELS_DIR,
    CUSTOM_DETECTOR_WEIGHTS,
    FALLBACK_DETECTOR_WEIGHTS,
    YOLO_INFERENCE_IMGSZ,
)

DETECTOR_DIR = MODELS_DIR / "object_detector"

# Must match the class order written by scripts/collect_dataset.py.
CLASS_NAMES = {
    0: "person",
    1: "object_a",
    2: "object_b",
    3: "tray",
    4: "complete_button",
}


def prepare_data_yaml() -> Path:
    DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    data_yaml = DATASET_DIR / "data.yaml"
    content = {
        "path": str(DATASET_DIR.as_posix()),
        "train": "images",
        "val": "images",
        "names": CLASS_NAMES,
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(content, f)
    return data_yaml


def check_dataset() -> bool:
    """Fail early and clearly rather than deep inside Ultralytics."""
    images = DATASET_DIR / "images"
    labels = DATASET_DIR / "labels"
    image_files = sorted(images.glob("*.jpg")) if images.exists() else []
    label_files = sorted(labels.glob("*.txt")) if labels.exists() else []

    if not image_files:
        print(f"[Train Error] No images found in {images}.")
        print("              Capture frames first: python scripts/collect_dataset.py")
        return False
    if not label_files:
        print(f"[Train Error] Found {len(image_files)} images but no labels in {labels}.")
        print("              Annotate the captured frames before training.")
        return False

    unlabelled = len(image_files) - len(label_files)
    if unlabelled > 0:
        print(f"[Train Warning] {unlabelled} image(s) have no matching label file and will be ignored.")
    print(f"[Train] Dataset: {len(image_files)} images, {len(label_files)} labels.")
    return True


def train(epochs: int = 25, imgsz: int = None, base_model: str = None):
    imgsz = imgsz or YOLO_INFERENCE_IMGSZ
    base_model = base_model or FALLBACK_DETECTOR_WEIGHTS

    if not check_dataset():
        return False

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Train Error] ultralytics is not installed. Run: pip install -r requirements.txt")
        return False

    try:
        data_yaml = prepare_data_yaml()
        print(f"[Train] Fine-tuning {base_model} on {data_yaml} for {epochs} epochs at imgsz={imgsz}...")
        model = YOLO(base_model)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            project=str(DETECTOR_DIR),
            name="train_run",
            exist_ok=True,
        )

        best_pt = DETECTOR_DIR / "train_run" / "weights" / "best.pt"
        if not best_pt.exists():
            print(f"[Train Error] Training finished but {best_pt} was not produced.")
            return False

        # First configured custom-weights path, so the detector finds it.
        target_pt = MODELS_DIR / CUSTOM_DETECTOR_WEIGHTS[0]
        target_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, target_pt)
        print(f"[Train Success] Weights saved to {target_pt}")
        print("[Train] Restart the backend — the detector loads these automatically.")
        return True
    except Exception as e:
        print(f"[Train Error] {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the AETHON payload-component detector.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs (default: 25)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help=f"Training image size (default: YOLO_INFERENCE_IMGSZ={YOLO_INFERENCE_IMGSZ})")
    parser.add_argument("--base", type=str, default=None,
                        help=f"Base model to fine-tune (default: {FALLBACK_DETECTOR_WEIGHTS})")
    args = parser.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz, base_model=args.base)
