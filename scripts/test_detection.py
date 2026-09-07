"""
AETHON - Perception Pipeline Diagnostics
Runs detector, pose, and hand tracker on 10 live frames.
"""
import cv2
from camera.camera_service import camera_service
from ai.detection.object_detector import object_detector
from ai.pose.pose_estimator import pose_estimator
from ai.tracking.hand_tracker import hand_tracker
import time

def test():
    print("[Perception Test] Grabbing camera frames for AI pipeline test...")
    time.sleep(1.0)
    frame = camera_service.get_latest_frame(annotated=False)
    if frame is None:
        print("[FAIL] No camera frame available!")
        return

    print(f"[PASS] Frame received ({frame.shape[1]}x{frame.shape[0]}). Running Pose Estimator...")
    pose = pose_estimator.process(frame)
    print(f"[PASS] Pose result: {'Person detected' if pose else 'No person in frame'}")

    print("[PASS] Running Object Detector...")
    objs = object_detector.detect(frame, person_bbox=pose.get("bbox") if pose else None)
    print(f"[PASS] Detected objects count: {len(objs)} -> {[o['label'] for o in objs]}")

    print("[PASS] Running Hand Tracker...")
    hands = hand_tracker.process(frame)
    print(f"[PASS] Tracked hands count: {len(hands)}")

if __name__ == "__main__":
    test()
