import os
import math
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Try MediaPipe Tasks Vision first (ultra-fast ~6.9ms inference)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_TASKS_AVAILABLE = True
except Exception:
    MP_TASKS_AVAILABLE = False

from backend.config import (
    CONFIDENCE_THRESHOLD_HAND,
    HAND_TRACKING_CONFIDENCE,
    MAX_TRACKED_HANDS,
    HAND_SMOOTHING_ALPHA,
)

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Hands)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
MIDDLE_MCP = 9

FINGER_JOINTS: List[Tuple[int, int, int]] = [
    # (tip, pip, mcp) for index, middle, ring, pinky
    (8, 6, 5),
    (12, 10, 9),
    (16, 14, 13),
    (20, 18, 17),
]

# Pinch & Grasp thresholds
PINCH_RATIO_THRESHOLD = 0.55
GRASP_CURLED_FINGERS = 3
HANDEDNESS_MIN_SCORE = 0.50


class HandTracker:
    """
    Ultra-low-latency Hand Tracking & Gesture Recognition engine (~6.9ms).
    Uses MediaPipe Tasks GestureRecognizer with geometric heuristic fallback.
    Outputs all 21 3D landmarks, pinch/grasp kinematics, and recognized gestures.
    """

    def __init__(self, model_path: str = "ai/models/gesture_recognizer.task"):
        self.recognizer = None
        self._smoothed: Dict[str, np.ndarray] = {}

        if MP_TASKS_AVAILABLE and os.path.exists(model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.GestureRecognizerOptions(
                    base_options=base_options,
                    num_hands=MAX_TRACKED_HANDS,
                    min_hand_detection_confidence=CONFIDENCE_THRESHOLD_HAND,
                    min_hand_presence_confidence=CONFIDENCE_THRESHOLD_HAND,
                    min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
                )
                self.recognizer = vision.GestureRecognizer.create_from_options(options)
                print("[HandTracker] MediaPipe GestureRecognizer initialized successfully (~6.9ms latency).")
            except Exception as e:
                print(f"[HandTracker Warning] Could not init GestureRecognizer: {e}")
                self.recognizer = None
        else:
            # Check legacy mediapipe solutions
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                try:
                    self.recognizer = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=MAX_TRACKED_HANDS,
                        min_detection_confidence=CONFIDENCE_THRESHOLD_HAND,
                        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
                    )
                    print("[HandTracker] Legacy MediaPipe solutions.hands initialized.")
                except Exception:
                    self.recognizer = None

    def process(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self.recognizer is None or frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. MediaPipe Tasks Vision GestureRecognizer path
        if hasattr(self.recognizer, "recognize"):
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = self.recognizer.recognize(mp_img)

                if not results.hand_landmarks:
                    self._smoothed.clear()
                    return []

                tracked_hands: List[Dict[str, Any]] = []
                seen_sides: List[str] = []

                for idx, hand_lms in enumerate(results.hand_landmarks):
                    # Handedness
                    side, side_score = "Right", 0.8
                    if results.handedness and idx < len(results.handedness):
                        categories = results.handedness[idx]
                        if categories:
                            side = categories[0].category_name
                            side_score = categories[0].score

                    if side in seen_sides:
                        side = "Left" if seen_sides[0] == "Right" else "Right"
                    seen_sides.append(side)

                    # ML Gesture prediction
                    ml_gesture = "NONE"
                    ml_gesture_score = 0.0
                    if results.gestures and idx < len(results.gestures):
                        g_cats = results.gestures[idx]
                        if g_cats:
                            ml_gesture = g_cats[0].category_name
                            ml_gesture_score = float(g_cats[0].score)

                    pts = np.array(
                        [[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in hand_lms],
                        dtype=np.float32,
                    )
                    pts = self._smooth(side, pts)

                    world_landmarks = []
                    if hasattr(results, "hand_world_landmarks") and results.hand_world_landmarks and idx < len(results.hand_world_landmarks):
                        w_lms = results.hand_world_landmarks[idx]
                        world_landmarks = [
                            {"x": float(lm.x), "y": float(lm.y), "z": float(getattr(lm, "z", 0.0))}
                            for lm in w_lms
                        ]

                    hand_dict = self._format_hand(pts, side, side_score, w, h, ml_gesture, ml_gesture_score)
                    hand_dict["world_landmarks"] = world_landmarks
                    tracked_hands.append(hand_dict)

                # Clean stale smoothing
                for stale in [s for s in self._smoothed if s not in seen_sides]:
                    self._smoothed.pop(stale, None)

                return tracked_hands
            except Exception as e:
                return []

        # 2. Legacy MediaPipe solutions path
        if hasattr(self.recognizer, "process"):
            try:
                results = self.recognizer.process(frame_rgb)
                if not results.multi_hand_landmarks:
                    self._smoothed.clear()
                    return []

                tracked_hands = []
                seen_sides = []
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    side = "Right"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        side = results.multi_handedness[idx].classification[0].label
                    if side in seen_sides:
                        side = "Left" if seen_sides[0] == "Right" else "Right"
                    seen_sides.append(side)

                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                    pts = self._smooth(side, pts)
                    hand_dict = self._format_hand(pts, side, 0.85, w, h, "NONE", 0.0)
                    tracked_hands.append(hand_dict)

                return tracked_hands
            except Exception as e:
                print('HAND TRACKER ERROR:', e)
                return []

        return []

    def _format_hand(
        self,
        pts: np.ndarray,
        side: str,
        side_score: float,
        w: int,
        h: int,
        ml_gesture: str,
        ml_gesture_score: float,
    ) -> Dict[str, Any]:
        """Convert landmarks to standard AETHON hand perception schema."""
        xs = pts[:, 0] * w
        ys = pts[:, 1] * h

        span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
        pad = max(6.0, span * 0.10)
        x1 = max(0, int(xs.min() - pad))
        y1 = max(0, int(ys.min() - pad))
        x2 = min(w - 1, int(xs.max() + pad))
        y2 = min(h - 1, int(ys.max() + pad))

        def pt(i: int) -> Dict[str, float]:
            return {"x": float(pts[i][0]), "y": float(pts[i][1])}

        wrist = pt(WRIST)
        thumb_tip = pt(THUMB_TIP)
        index_tip = pt(INDEX_TIP)
        palm_center = pt(MIDDLE_MCP)

        hand_scale = max(
            1e-4,
            math.hypot(
                palm_center["x"] - wrist["x"],
                palm_center["y"] - wrist["y"],
            ),
        )

        pinch_dist = math.hypot(
            thumb_tip["x"] - index_tip["x"],
            thumb_tip["y"] - index_tip["y"],
        )
        pinch_ratio = pinch_dist / hand_scale
        curled = self._count_curled_fingers(pts)

        is_pinching = pinch_ratio < PINCH_RATIO_THRESHOLD
        is_grasping = curled >= GRASP_CURLED_FINGERS or is_pinching

        # Map and refine gesture
        gesture, gesture_conf = self._classify_hand_gesture(
            pts, ml_gesture, ml_gesture_score, curled, pinch_ratio
        )

        return {
            "side": side,
            "side_confidence": round(side_score, 2),
            "bbox": [x1, y1, x2, y2],
            "wrist": wrist,
            "thumb_tip": thumb_tip,
            "index_tip": index_tip,
            "palm_center": palm_center,
            "fingertips": [pt(i) for i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)],
            "hand_scale": round(hand_scale, 4),
            "pinch_distance": pinch_dist,
            "pinch_ratio": round(pinch_ratio, 3),
            "is_pinching": is_pinching,
            "curled_fingers": curled,
            "is_grasping": is_grasping,
            "gesture": gesture,
            "gesture_confidence": round(gesture_conf, 2),
            "landmarks": [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts
            ],
        }

    def _classify_hand_gesture(
        self,
        pts: np.ndarray,
        ml_gesture: str,
        ml_score: float,
        curled: int,
        pinch_ratio: float,
    ) -> Tuple[str, float]:
        """Combine MediaPipe ML gesture classification with fast geometric heuristics."""
        # 1. OK Sign: thumb tip close to index tip, other 3 fingers extended
        if pinch_ratio < 0.46 and curled <= 1:
            return "OK_SIGN", 0.94

        # 2. Pinch gesture: thumb & index close together, other fingers can be curled or relaxed
        if pinch_ratio < 0.38:
            return "PINCH", 0.93

        # 3. Direct ML gesture mapping if confident (>= 0.58 threshold for fast response)
        if ml_gesture != "NONE" and ml_score >= 0.58:
            mapping = {
                "Closed_Fist": "FIST",
                "Open_Palm": "OPEN_PALM",
                "Pointing_Up": "POINTING",
                "Thumb_Down": "THUMBS_DOWN",
                "Thumb_Up": "THUMBS_UP",
                "Victory": "VICTORY",
                "ILoveYou": "I_LOVE_YOU",
            }
            if ml_gesture in mapping:
                return mapping[ml_gesture], ml_score

        # 4. Fast geometric fallback rules (O(1))
        wrist = pts[WRIST][:2]
        d_idx = float(np.linalg.norm(pts[INDEX_TIP][:2] - wrist))
        d_mid = float(np.linalg.norm(pts[MIDDLE_TIP][:2] - wrist))
        d_rng = float(np.linalg.norm(pts[RING_TIP][:2] - wrist))
        d_pnk = float(np.linalg.norm(pts[PINKY_TIP][:2] - wrist))
        d_thb = float(np.linalg.norm(pts[THUMB_TIP][:2] - wrist))

        # Victory (V sign): index & middle extended, ring & pinky curled
        if d_idx > d_rng * 1.15 and d_mid > d_pnk * 1.15 and curled in (2, 3):
            return "VICTORY", 0.90

        # Pointing: index extended, middle/ring/pinky curled
        if d_idx > d_mid * 1.15 and curled >= 2:
            return "POINTING", 0.92

        # Thumbs up / down
        if curled >= 3 and d_thb > 0.08:
            if pts[THUMB_TIP][1] < pts[WRIST][1] - 0.05:
                return "THUMBS_UP", 0.92
            elif pts[THUMB_TIP][1] > pts[WRIST][1] + 0.05:
                return "THUMBS_DOWN", 0.92

        # Fist
        if curled >= 3:
            return "FIST", 0.90

        # Open palm (fingers spread out)
        if curled <= 1:
            return "OPEN_PALM", 0.88

        return "NONE", 0.0

    def _smooth(self, side: str, pts: np.ndarray) -> np.ndarray:
        prev = self._smoothed.get(side)
        if prev is None or prev.shape != pts.shape:
            self._smoothed[side] = pts
            return pts
        blended = prev * (1.0 - HAND_SMOOTHING_ALPHA) + pts * HAND_SMOOTHING_ALPHA
        self._smoothed[side] = blended
        return blended

    def _count_curled_fingers(self, pts: np.ndarray) -> int:
        wrist = pts[WRIST][:2]
        curled = 0
        for tip, pip, _mcp in FINGER_JOINTS:
            d_tip = float(np.linalg.norm(pts[tip][:2] - wrist))
            d_pip = float(np.linalg.norm(pts[pip][:2] - wrist))
            if d_tip < d_pip * 1.02:
                curled += 1
        return curled


hand_tracker = HandTracker()
