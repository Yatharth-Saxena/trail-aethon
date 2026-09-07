"""
AETHON - Perception Pipeline Diagnostics
Runs the hand tracker and object detector on live camera frames.
"""
import time

from camera.camera_service import camera_service
from ai.detection.object_detector import object_detector
from ai.tracking.hand_tracker import hand_tracker

# Detections are only reported once a track has been confirmed across several
# frames, so the diagnostic feeds the pipeline a short burst of frames.
WARMUP_FRAMES = 8


def test():
    print("[Perception Test] Grabbing camera frames for AI pipeline test...")
    time.sleep(1.0)
    frame = camera_service.get_latest_frame(annotated=False)
    if frame is None:
        print("[FAIL] No camera frame available!")
        return

    print(f"[PASS] Frame received ({frame.shape[1]}x{frame.shape[0]}). Running Hand Tracker...")
    hands = hand_tracker.process(frame)
    print(f"[PASS] Tracked hands count: {len(hands)}")
    for hand in hands:
        grip = "grasping" if hand.get("is_grasping") else "open"
        print(f"        {hand['side']} hand -> {grip}, pinch ratio {hand['pinch_ratio']}")

    print(f"[PASS] Running Object Detector over {WARMUP_FRAMES} frames to confirm tracks...")
    objs = []
    for _ in range(WARMUP_FRAMES):
        live = camera_service.get_latest_frame(annotated=False)
        if live is None:
            continue
        hands = hand_tracker.process(live)
        objs = object_detector.detect(live, hands=hands)
        time.sleep(0.05)

    print(f"[PASS] Confirmed detections: {len(objs)}")
    for obj in objs:
        print(f"        {obj['display_name']} (class={obj['class_label']}, conf={obj['confidence']})")

    people = [o for o in objs if o["raw_label"] == "Person"]
    print(f"[PASS] Human detection: {'Person detected' if people else 'No person in frame'}")


if __name__ == "__main__":
    test()
