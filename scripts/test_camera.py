"""
AETHON - Camera Diagnostics
Tests OpenCV camera capture, DirectShow, and measures actual FPS.
"""
import cv2
import time

def test():
    print("[Camera Test] Probing Camera 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[FAIL] Could not open camera 0!")
        return

    print("[PASS] Camera opened successfully.")
    start = time.time()
    frames = 0
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames += 1

    elapsed = time.time() - start
    fps = round(frames / elapsed, 1)
    print(f"[PASS] Captured {frames} frames in {elapsed:.2f}s (~{fps} FPS). Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()

if __name__ == "__main__":
    test()
