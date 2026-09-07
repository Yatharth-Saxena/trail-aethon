import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.30,
            min_tracking_confidence=0.30
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = []
        xs, ys = [], []

        for lm in results.pose_landmarks.landmark:
            xs.append(lm.x * w)
            ys.append(lm.y * h)
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        # Derive person bounding box from visible landmarks
        x1 = max(0, int(min(xs) - 20))
        y1 = max(0, int(min(ys) - 30))
        x2 = min(w - 1, int(max(xs) + 20))
        y2 = min(h - 1, int(max(ys) + 20))

        lms = results.pose_landmarks.landmark
        shoulders_mid_y = (lms[11].y + lms[12].y) / 2.0
        wrists_avg_y = (lms[15].y + lms[16].y) / 2.0

        return {
            "bbox": [x1, y1, x2, y2],
            "landmarks": landmarks,
            "landmarks_count": len(landmarks),
            "shoulders_mid_y": shoulders_mid_y,
            "wrists_avg_y": wrists_avg_y,
            "head": {"x": round(lms[0].x, 3), "y": round(lms[0].y, 3)},
            "left_wrist": {"x": round(lms[15].x, 3), "y": round(lms[15].y, 3)},
            "right_wrist": {"x": round(lms[16].x, 3), "y": round(lms[16].y, 3)}
        }

pose_estimator = PoseEstimator()
