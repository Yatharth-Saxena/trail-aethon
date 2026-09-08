import os
import math
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Try MediaPipe Tasks Vision first (ultra-fast ~13ms CPU inference)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# Fallback YOLO pose (MPS/CPU ~19ms-65ms)
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


try:
    from backend.config import POSE_SMOOTHING_ALPHA
except Exception:
    POSE_SMOOTHING_ALPHA = 0.88


class PoseTracker:
    """
    High-performance, minimal-latency Body Pose and Gesture tracker.
    Uses MediaPipe PoseLandmarker (13ms) with seamless YOLOv8-pose fallback.
    Outputs 33 normalized landmarks (MediaPipe format) and classifies body gestures in O(1) time.
    """

    def __init__(self, model_path: str = "ai/models/pose_landmarker_lite.task"):
        self.mp_landmarker = None
        self.yolo_model = None
        self.device = "cpu"

        # Initialize MediaPipe PoseLandmarker
        if MP_AVAILABLE and os.path.exists(model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    num_poses=4,
                    min_pose_detection_confidence=0.45,
                    min_pose_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                self.mp_landmarker = vision.PoseLandmarker.create_from_options(options)
                print("[PoseTracker] MediaPipe Multi-Pose Landmarker initialized successfully (~14ms latency, up to 4 poses).")
            except Exception as e:
                print(f"[PoseTracker Warning] Could not init MediaPipe PoseLandmarker: {e}")
                self.mp_landmarker = None

        # Fallback to YOLOv8-pose if MediaPipe not available
        if self.mp_landmarker is None and YOLO_AVAILABLE and os.path.exists("yolov8n-pose.pt"):
            try:
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                self.yolo_model = YOLO("yolov8n-pose.pt")
                print(f"[PoseTracker] YOLOv8-Pose fallback initialized on device '{self.device}'.")
            except Exception as e:
                print(f"[PoseTracker Warning] Could not init YOLO pose: {e}")

        # Temporal history for dynamic body gestures & landmark smoothing
        self._history = []
        self._last_gesture = "STATIONARY"
        self._smoothed_pts: Optional[np.ndarray] = None
        self._missed_frames: int = 0

    def _smooth_landmarks(self, raw_pts: np.ndarray, alpha: float = POSE_SMOOTHING_ALPHA) -> np.ndarray:
        if self._smoothed_pts is None or self._smoothed_pts.shape != raw_pts.shape:
            self._smoothed_pts = raw_pts
            return raw_pts
        self._smoothed_pts = self._smoothed_pts * (1.0 - alpha) + raw_pts * alpha
        return self._smoothed_pts

    def process(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process frame and return 33 body landmarks, bounding box, posture, and body gestures.
        Supports tracking multiple people in the camera frame (up to 4 poses).
        """
        if frame_bgr is None:
            return None

        h, w = frame_bgr.shape[:2]

        # 1. Primary: MediaPipe PoseLandmarker (Multi-Pose)
        if self.mp_landmarker is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = self.mp_landmarker.detect(mp_image)

                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    all_poses: List[Dict[str, Any]] = []

                    for p_idx, lms_raw in enumerate(results.pose_landmarks):
                        vis_arr = [float(getattr(lm, "visibility", 1.0) or 1.0) for lm in lms_raw]
                        valid_vis_count = sum(1 for v in vis_arr if v >= 0.28)
                        has_shoulder = (vis_arr[11] >= 0.32 or vis_arr[12] >= 0.32)
                        has_head = (vis_arr[0] >= 0.28 or vis_arr[2] >= 0.28 or vis_arr[5] >= 0.28)
                        has_hip = (vis_arr[23] >= 0.28 or vis_arr[24] >= 0.28)

                        if valid_vis_count < 6 or not (has_shoulder or has_head or has_hip):
                            continue

                        raw_arr = np.array(
                            [[lm.x, lm.y, getattr(lm, "z", 0.0), getattr(lm, "visibility", 1.0) or 1.0] for lm in lms_raw],
                            dtype=np.float32,
                        )
                        if p_idx == 0:
                            smoothed_arr = self._smooth_landmarks(raw_arr)
                        else:
                            smoothed_arr = raw_arr

                        landmarks = [
                            {
                                "x": float(smoothed_arr[i, 0]),
                                "y": float(smoothed_arr[i, 1]),
                                "z": float(smoothed_arr[i, 2]),
                                "visibility": float(smoothed_arr[i, 3]),
                            }
                            for i in range(len(lms_raw))
                        ]

                        # Calculate bounding box & count visible landmarks
                        valid_xs = [lm["x"] * w for lm in landmarks if lm["visibility"] > 0.22]
                        valid_ys = [lm["y"] * h for lm in landmarks if lm["visibility"] > 0.22]
                        visible_count = len(valid_xs)
                        if valid_xs and valid_ys:
                            x1 = max(0, int(min(valid_xs) - 15))
                            y1 = max(0, int(min(valid_ys) - 20))
                            x2 = min(w - 1, int(max(xs for xs in valid_xs) + 15))
                            y2 = min(h - 1, int(max(ys for ys in valid_ys) + 15))
                        else:
                            x1, y1, x2, y2 = 0, 0, w - 1, h - 1

                        world_landmarks = []
                        if (
                            hasattr(results, "pose_world_landmarks") and
                            results.pose_world_landmarks and
                            p_idx < len(results.pose_world_landmarks)
                        ):
                            world_lms_raw = results.pose_world_landmarks[p_idx]
                            world_landmarks = [
                                {
                                    "x": float(lm.x),
                                    "y": float(lm.y),
                                    "z": float(getattr(lm, "z", 0.0)),
                                    "visibility": float(getattr(lm, "visibility", 1.0) or 1.0)
                                }
                                for lm in world_lms_raw
                            ]

                        p_dict = self._build_pose_dict(landmarks, [x1, y1, x2, y2])
                        p_dict["visible_landmarks_count"] = visible_count
                        p_dict["world_landmarks"] = world_landmarks
                        p_dict["pose_index"] = p_idx
                        all_poses.append(p_dict)

                    if all_poses:
                        self._missed_frames = 0
                        # Sort by area descending so primary foreground person is index 0
                        all_poses.sort(
                            key=lambda pd: (pd["bbox"][2] - pd["bbox"][0]) * (pd["bbox"][3] - pd["bbox"][1]),
                            reverse=True,
                        )
                        primary_pose = dict(all_poses[0])
                        primary_pose["all_poses"] = all_poses
                        return primary_pose

            except Exception as e:
                print('POSE TRACKER ERROR:', e)

        # 2. Fallback: YOLOv8-pose
        if self.yolo_model is not None:
            try:
                res = self.yolo_model(frame_bgr, imgsz=256, device=self.device, verbose=False)
                if res and len(res[0].boxes) > 0 and res[0].keypoints is not None and res[0].keypoints.xyn is not None:
                    confs = res[0].boxes.conf.cpu().numpy()
                    order = np.argsort(-confs)[:4]

                    # Map YOLO 17 keypoints to MediaPipe 33 keypoints
                    yolo_to_mp = {
                        0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
                        5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
                        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28
                    }

                    all_poses = []
                    for p_rank, idx in enumerate(order):
                        xyn = res[0].keypoints.xyn[idx].cpu().numpy()
                        kpt_conf = res[0].keypoints.conf[idx].cpu().numpy()

                        if sum(1 for cf in kpt_conf if cf >= 0.30) < 5:
                            continue

                        landmarks = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for _ in range(33)]
                        xs, ys = [], []

                        for y_idx, mp_idx in yolo_to_mp.items():
                            cx, cy = float(xyn[y_idx][0]), float(xyn[y_idx][1])
                            cf = float(kpt_conf[y_idx])
                            landmarks[mp_idx] = {"x": cx, "y": cy, "z": 0.0, "visibility": cf}
                            if cf > 0.25:
                                xs.append(cx * w)
                                ys.append(cy * h)

                        if xs and ys:
                            x1 = max(0, int(min(xs) - 15))
                            y1 = max(0, int(min(ys) - 20))
                            x2 = min(w - 1, int(max(xs) + 15))
                            y2 = min(h - 1, int(max(ys) + 15))
                        else:
                            x1, y1, x2, y2 = 0, 0, w - 1, h - 1

                        p_dict = self._build_pose_dict(landmarks, [x1, y1, x2, y2])
                        p_dict["visible_landmarks_count"] = len(xs)
                        p_dict["pose_index"] = p_rank
                        all_poses.append(p_dict)

                    if all_poses:
                        self._missed_frames = 0
                        all_poses.sort(
                            key=lambda pd: (pd["bbox"][2] - pd["bbox"][0]) * (pd["bbox"][3] - pd["bbox"][1]),
                            reverse=True,
                        )
                        primary_pose = dict(all_poses[0])
                        primary_pose["all_poses"] = all_poses
                        return primary_pose

            except Exception as e:
                print('POSE TRACKER ERROR:', e)

        self._missed_frames += 1
        if self._missed_frames > 3:
            self._smoothed_pts = None
        return None

    def _build_pose_dict(self, landmarks: List[Dict[str, float]], bbox: List[int]) -> Dict[str, Any]:
        """Classify posture & body gestures in O(1) time (<0.1ms)."""
        # Key landmark positions
        nose = landmarks[0]
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        l_el = landmarks[13]
        r_el = landmarks[14]
        l_wr = landmarks[15]
        r_wr = landmarks[16]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        l_knee = landmarks[25]
        r_knee = landmarks[26]

        sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
        sh_mid_x = (l_sh["x"] + r_sh["x"]) / 2.0
        hip_mid_y = (l_hip["y"] + r_hip["y"]) / 2.0
        wrists_avg_y = (l_wr["y"] + r_wr["y"]) / 2.0

        # Classify Posture (Standing, Seated, Crouching)
        torso_len = max(0.01, abs(hip_mid_y - sh_mid_y))
        knee_mid_y = (l_knee["y"] + r_knee["y"]) / 2.0 if l_knee["visibility"] > 0.3 else 0.0
        if knee_mid_y > 0 and (knee_mid_y - hip_mid_y) < torso_len * 0.6:
            posture = "Seated"
        elif bbox[3] - bbox[1] < bbox[2] - bbox[0]:
            posture = "Crouching"
        else:
            posture = "Standing"

        # Classify Body Gestures with precise anatomical thresholds
        body_gestures = []

        # 1. Both Hands Raised (Emergency, Attention, Celebration)
        if l_wr["y"] < sh_mid_y - 0.08 and r_wr["y"] < sh_mid_y - 0.08:
            body_gestures.append({
                "gesture": "BOTH_HANDS_RAISED",
                "confidence": 0.96,
                "description": "Both hands raised above head"
            })
        # 2. Right Hand Raised
        elif r_wr["y"] < sh_mid_y - 0.08 and (l_wr["y"] > r_wr["y"] + 0.08 or l_wr["y"] >= sh_mid_y):
            body_gestures.append({
                "gesture": "RIGHT_HAND_RAISED",
                "confidence": 0.94,
                "description": "Right hand raised"
            })
        # 3. Left Hand Raised
        elif l_wr["y"] < sh_mid_y - 0.08 and (r_wr["y"] > l_wr["y"] + 0.08 or r_wr["y"] >= sh_mid_y):
            body_gestures.append({
                "gesture": "LEFT_HAND_RAISED",
                "confidence": 0.94,
                "description": "Left hand raised"
            })

        # 4. Salute (Hand brought to temple/forehead)
        # MediaPipe pose landmarks: 7 = left ear, 8 = right ear
        ear_l = landmarks[7] if len(landmarks) > 7 else nose
        ear_r = landmarks[8] if len(landmarks) > 8 else nose
        d_r_salute = math.hypot(r_wr["x"] - ear_r["x"], r_wr["y"] - ear_r["y"])
        d_l_salute = math.hypot(l_wr["x"] - ear_l["x"], l_wr["y"] - ear_l["y"])
        if d_r_salute < 0.14 and r_el["y"] > r_wr["y"] - 0.06:
            body_gestures.append({
                "gesture": "SALUTE_RIGHT",
                "confidence": 0.93,
                "description": "Saluting with right hand"
            })
        elif d_l_salute < 0.14 and l_el["y"] > l_wr["y"] - 0.06:
            body_gestures.append({
                "gesture": "SALUTE_LEFT",
                "confidence": 0.93,
                "description": "Saluting with left hand"
            })

        # 5. Arms Crossed over chest
        sh_span = max(0.06, abs(r_sh["x"] - l_sh["x"]))
        if abs(l_wr["y"] - sh_mid_y) < 0.20 and abs(r_wr["y"] - sh_mid_y) < 0.20:
            # Wrists crossed across body centerline
            if abs(r_wr["x"] - l_wr["x"]) < sh_span * 0.40 and min(r_wr["y"], l_wr["y"]) > nose["y"]:
                body_gestures.append({
                    "gesture": "ARMS_CROSSED",
                    "confidence": 0.90,
                    "description": "Arms crossed over chest"
                })

        # 6. T-Pose (Horizontal arms spread)
        if abs(l_wr["y"] - sh_mid_y) < 0.10 and abs(r_wr["y"] - sh_mid_y) < 0.10:
            arm_span = abs(r_wr["x"] - l_wr["x"])
            if arm_span > sh_span * 2.0:
                body_gestures.append({
                    "gesture": "T_POSE",
                    "confidence": 0.88,
                    "description": "Arms extended horizontally"
                })

        # 7. Hands on Hips
        if abs(l_wr["y"] - hip_mid_y) < 0.14 and abs(r_wr["y"] - hip_mid_y) < 0.14:
            elbow_span = abs(r_el["x"] - l_el["x"])
            if elbow_span > sh_span * 1.5:
                body_gestures.append({
                    "gesture": "HANDS_ON_HIPS",
                    "confidence": 0.86,
                    "description": "Hands resting on hips"
                })

        # Primary gesture
        primary = body_gestures[0]["gesture"] if body_gestures else "STATIONARY"

        return {
            "bbox": bbox,
            "landmarks": landmarks,
            "landmarks_count": 33,
            "shoulders_mid_y": sh_mid_y,
            "wrists_avg_y": wrists_avg_y,
            "head": {"x": nose["x"], "y": nose["y"]},
            "left_wrist": {"x": l_wr["x"], "y": l_wr["y"]},
            "right_wrist": {"x": r_wr["x"], "y": r_wr["y"]},
            "posture": posture,
            "gestures": body_gestures,
            "primary_gesture": primary,
        }


pose_tracker = PoseTracker()
