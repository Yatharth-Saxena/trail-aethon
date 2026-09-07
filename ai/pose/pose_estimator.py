import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self):
        self.mp_pose = None
        self.pose = None
        self.yolo_pose = None
        self.use_yolo = False
        
        # Try MediaPipe first
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.30,
                    min_tracking_confidence=0.30
                )
            except Exception as e:
                print(f"[PoseEstimator Warning] Could not init mediapipe pose: {e}")
                self.pose = None
        
        # Fallback to YOLO Pose
        if self.pose is None:
            print("[PoseEstimator Info] MediaPipe not available, falling back to YOLOv8-pose.")
            try:
                self.yolo_pose = YOLO("yolov8n-pose.pt")
                self.use_yolo = True
            except Exception as e:
                print(f"[PoseEstimator Warning] YOLO fallback failed: {e}")

    def process(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.use_yolo and self.yolo_pose is not None:
            return self._process_yolo(frame_bgr)
        elif self.pose is not None:
            return self._process_mediapipe(frame_bgr)
        return None

    def _process_yolo(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        res = self.yolo_pose(frame_bgr, verbose=False)
        if not res or len(res[0].boxes) == 0:
            return None
            
        confs = res[0].boxes.conf.cpu().numpy()
        best_idx = np.argmax(confs)
        
        if res[0].keypoints is None or res[0].keypoints.xyn is None:
            return None
            
        xyn = res[0].keypoints.xyn[best_idx].cpu().numpy()
        kpt_conf = res[0].keypoints.conf[best_idx].cpu().numpy()
        
        yolo_to_mp = {
            0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
            5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
            11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28
        }
        
        landmarks = [{"x": 0, "y": 0, "z": 0, "visibility": 0.0} for _ in range(33)]
        xs, ys = [], []
        
        for y_idx, mp_idx in yolo_to_mp.items():
            cx, cy = xyn[y_idx]
            cf = kpt_conf[y_idx]
            landmarks[mp_idx] = {"x": float(cx), "y": float(cy), "z": 0.0, "visibility": float(cf)}
            if cf > 0.3:
                xs.append(cx * w)
                ys.append(cy * h)
                
        if not xs:
            return None
            
        x1 = max(0, int(min(xs) - 20))
        y1 = max(0, int(min(ys) - 30))
        x2 = min(w - 1, int(max(xs) + 20))
        y2 = min(h - 1, int(max(ys) + 20))
        
        shoulders_mid_y = (landmarks[11]["y"] + landmarks[12]["y"]) / 2.0
        wrists_avg_y = (landmarks[15]["y"] + landmarks[16]["y"]) / 2.0
        
        return {
            "bbox": [x1, y1, x2, y2],
            "landmarks": landmarks,
            "landmarks_count": 33,
            "shoulders_mid_y": shoulders_mid_y,
            "wrists_avg_y": wrists_avg_y,
            "head": {"x": landmarks[0]["x"], "y": landmarks[0]["y"]},
            "left_wrist": {"x": landmarks[15]["x"], "y": landmarks[15]["y"]},
            "right_wrist": {"x": landmarks[16]["x"], "y": landmarks[16]["y"]}
        }

    def _process_mediapipe(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
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
