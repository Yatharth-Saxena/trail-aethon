"""
AETHON - Custom Detector Fine-Tuning Utility
Trains/Fine-tunes a lightweight YOLO model (yolov8n) on the custom dataset in dataset/
Outputs weights to models/object_detector/weights.pt
"""
import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models" / "object_detector"

def prepare_data_yaml():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data_yaml = DATASET_DIR / "data.yaml"
    content = {
        "path": str(DATASET_DIR.as_posix()),
        "train": "images",
        "val": "images",
        "names": {
            0: "person",
            1: "object_a",
            2: "object_b",
            3: "tray",
            4: "complete_button"
        }
    }
    with open(data_yaml, "w") as f:
        yaml.dump(content, f)
    return data_yaml

def train(epochs=25, imgsz=640):
    try:
        from ultralytics import YOLO
        data_yaml = prepare_data_yaml()
        print(f"[Train] Starting YOLO fine-tuning on {data_yaml} for {epochs} epochs...")
        model = YOLO("yolov8n.pt") # Base lightweight model
        results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(MODELS_DIR), name="train_run")
        
        # Copy best weights to models/object_detector/weights.pt
        best_pt = MODELS_DIR / "train_run" / "weights" / "best.pt"
        if best_pt.exists():
            target_pt = MODELS_DIR / "weights.pt"
            import shutil
            shutil.copy(best_pt, target_pt)
            print(f"[Train Success] Trained model weights saved to {target_pt}")
    except Exception as e:
        print(f"[Train Error] {e}")

if __name__ == "__main__":
    train()
