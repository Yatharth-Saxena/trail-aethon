import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Dict, Any, Optional

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )

    def process(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        tracked_hands = []
        if not results.multi_hand_landmarks:
            return tracked_hands

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            side = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                side = results.multi_handedness[idx].classification[0].label # "Left" or "Right"

            landmarks = []
            xs, ys = [], []

            for lm in hand_lms.landmark:
                xs.append(lm.x * w)
                ys.append(lm.y * h)
                landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})

            # Hand bounding box
            x1 = max(0, int(min(xs) - 10))
            y1 = max(0, int(min(ys) - 10))
            x2 = min(w - 1, int(max(xs) + 10))
            y2 = min(h - 1, int(max(ys) + 10))

            # Key points: Wrist (0), Thumb Tip (4), Index Tip (8), Palm Center approx (9)
            wrist = {"x": hand_lms.landmark[0].x, "y": hand_lms.landmark[0].y}
            thumb_tip = {"x": hand_lms.landmark[4].x, "y": hand_lms.landmark[4].y}
            index_tip = {"x": hand_lms.landmark[8].x, "y": hand_lms.landmark[8].y}
            palm_center = {"x": hand_lms.landmark[9].x, "y": hand_lms.landmark[9].y}

            # Pinch distance (normalized Euclidean distance between thumb and index tip)
            pinch_dist = math.sqrt(
                (thumb_tip["x"] - index_tip["x"]) ** 2 +
                (thumb_tip["y"] - index_tip["y"]) ** 2
            )

            tracked_hands.append({
                "side": side,
                "bbox": [x1, y1, x2, y2],
                "wrist": wrist,
                "thumb_tip": thumb_tip,
                "index_tip": index_tip,
                "palm_center": palm_center,
                "pinch_distance": pinch_dist,
                "is_pinching": pinch_dist < 0.08,
                "landmarks": landmarks
            })

        return tracked_hands

hand_tracker = HandTracker()
