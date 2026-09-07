"""
AETHON - Dataset Collection Utility
Usage: python scripts/collect_dataset.py
Captures images from webcam for custom detector fine-tuning.
Hotkeys:
  [a] - Capture Object A (Red block)
  [b] - Capture Object B (Wooden block)
  [t] - Capture Tray
  [c] - Capture Complete Button
  [p] - Capture Person
  [s] - General Scene snapshot
  [q] - Quit
"""
import cv2
import time
import os
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
CLASSES_FILE = DATASET_DIR / "classes.txt"

CLASSES = ["person", "object_a", "object_b", "tray", "complete_button"]

def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASSES_FILE, "w") as f:
        f.write("\n".join(CLASSES) + "\n")

    print(f"[Dataset Collector] Initializing camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open camera!")
        return

    print("=" * 60)
    print("AETHON DATASET COLLECTION TOOL")
    print("Keys: [a] Object A | [b] Object B | [t] Tray | [c] Button | [p] Person | [s] Scene | [q] Quit")
    print("=" * 60)

    count = len(list(IMAGES_DIR.glob("*.jpg")))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        display = frame.copy()
        cv2.putText(display, f"Captured: {count} images | Keys: [a,b,t,c,p,s] | [q] Quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow("AETHON Dataset Collector", display)
        key = cv2.waitKey(1) & 0xFF

        target_class = None
        if key == ord('q'):
            break
        elif key == ord('a'):
            target_class = "object_a"
        elif key == ord('b'):
            target_class = "object_b"
        elif key == ord('t'):
            target_class = "tray"
        elif key == ord('c'):
            target_class = "complete_button"
        elif key == ord('p'):
            target_class = "person"
        elif key == ord('s'):
            target_class = "scene"

        if target_class:
            filename = f"{target_class}_{int(time.time()*1000)}.jpg"
            img_path = IMAGES_DIR / filename
            cv2.imwrite(str(img_path), frame)
            count += 1
            print(f"Saved: {filename} (Total: {count})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[Done] Total dataset images: {count} in {IMAGES_DIR}")

if __name__ == "__main__":
    main()
