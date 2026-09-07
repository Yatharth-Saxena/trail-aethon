import math
from typing import List, Dict, Any, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from backend.config import (
    CONFIDENCE_THRESHOLD_HAND,
    HAND_TRACKING_CONFIDENCE,
    HAND_MODEL_COMPLEXITY,
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

# Thumb-to-index separation, as a fraction of the hand's own size, below
# which the hand counts as pinching. Normalising by hand size makes this
# independent of how far the operator is from the camera.
PINCH_RATIO_THRESHOLD = 0.55
# Number of curled fingers required for a power grasp.
GRASP_CURLED_FINGERS = 3
# Minimum handedness classification score before the Left/Right label is
# trusted; below this the previous frame's assignment is kept.
HANDEDNESS_MIN_SCORE = 0.60


class HandTracker:
    def __init__(self):
        self.mp_hands = None
        self.hands = None
        # side -> smoothed landmark array, used to damp per-frame jitter.
        self._smoothed: Dict[str, np.ndarray] = {}

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands
                # All thresholds come from backend/config.py so they can be
                # retuned for cabin lighting without editing tracker code.
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=MAX_TRACKED_HANDS,
                    # Complexity 1 gives noticeably better fingertip accuracy
                    # than the lite model; affordable now that no body pose
                    # model runs in the pipeline.
                    model_complexity=HAND_MODEL_COMPLEXITY,
                    min_detection_confidence=CONFIDENCE_THRESHOLD_HAND,
                    min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
                )
            except Exception as e:
                print(f"[HandTracker Warning] Could not init mediapipe hands: {e}")
                self.hands = None
        else:
            print("[HandTracker Info] MediaPipe solutions.hands not available in current environment; running in fallback mode.")

    def process(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self.hands is None:
            return []

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            self._smoothed.clear()
            return []

        tracked_hands: List[Dict[str, Any]] = []
        seen_sides: List[str] = []

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            side, side_score = self._resolve_side(results, idx, seen_sides)
            seen_sides.append(side)

            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            pts = self._smooth(side, pts)

            xs = pts[:, 0] * w
            ys = pts[:, 1] * h

            # Pad the box proportionally to the hand's size so the margin is
            # correct at any distance from the camera.
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

            # Hand scale: wrist to middle-finger knuckle. Stable under
            # rotation and independent of finger articulation.
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

            tracked_hands.append({
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
                "is_pinching": pinch_ratio < PINCH_RATIO_THRESHOLD,
                "curled_fingers": curled,
                "is_grasping": curled >= GRASP_CURLED_FINGERS or pinch_ratio < PINCH_RATIO_THRESHOLD,
                "landmarks": [
                    {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts
                ],
            })

        # Drop smoothing state for hands that left the frame.
        for stale in [s for s in self._smoothed if s not in seen_sides]:
            self._smoothed.pop(stale, None)

        return tracked_hands

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _resolve_side(self, results, idx: int, seen_sides: List[str]) -> Tuple[str, float]:
        """
        Read MediaPipe's handedness classification, falling back to the
        opposite of the already-assigned hand when the score is too low to
        trust (which otherwise causes Left/Right to flip mid-gesture).
        """
        label, score = "Right", 0.0
        if results.multi_handedness and idx < len(results.multi_handedness):
            classification = results.multi_handedness[idx].classification[0]
            label = classification.label
            score = float(classification.score)

        if score < HANDEDNESS_MIN_SCORE or label in seen_sides:
            if seen_sides:
                label = "Left" if seen_sides[0] == "Right" else "Right"
        return label, score

    def _smooth(self, side: str, pts: np.ndarray) -> np.ndarray:
        prev = self._smoothed.get(side)
        if prev is None or prev.shape != pts.shape:
            self._smoothed[side] = pts
            return pts
        blended = prev * (1.0 - HAND_SMOOTHING_ALPHA) + pts * HAND_SMOOTHING_ALPHA
        self._smoothed[side] = blended
        return blended

    def _count_curled_fingers(self, pts: np.ndarray) -> int:
        """
        A finger is curled when its tip sits closer to the wrist than its
        middle joint does — a rotation-invariant test that works whichever
        way the hand is facing.
        """
        wrist = pts[WRIST][:2]
        curled = 0
        for tip, pip, _mcp in FINGER_JOINTS:
            d_tip = float(np.linalg.norm(pts[tip][:2] - wrist))
            d_pip = float(np.linalg.norm(pts[pip][:2] - wrist))
            if d_tip < d_pip * 1.02:
                curled += 1
        return curled


hand_tracker = HandTracker()
