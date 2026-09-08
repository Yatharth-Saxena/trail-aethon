from typing import Dict, Any, List, Optional
from collections import deque
import time

class GesturePipeline:
    """
    Unifies real-time hand gestures and full-body gestures with temporal debouncing.
    Runs in < 0.1ms after pose and hand landmark extraction.
    """

    def __init__(self, history_len: int = 3):
        self.history: deque = deque(maxlen=history_len)
        self.last_primary_gesture = "NONE"
        self.last_change_time = time.time()

    def analyze(
        self,
        hands: List[Dict[str, Any]],
        pose: Optional[Dict[str, Any]],
        current_action: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        hand_gestures = []
        for h in (hands or []):
            g = h.get("gesture")
            conf = h.get("gesture_confidence", 0.8)
            side = h.get("side", "Right")
            if g and g not in ("NONE", "OPEN_HAND", "REACHING"):
                hand_gestures.append({
                    "side": side,
                    "gesture": g,
                    "confidence": conf,
                    "is_pinching": h.get("is_pinching", False),
                    "is_grasping": h.get("is_grasping", False)
                })

        body_gestures = []
        posture = "Seated"
        if pose:
            posture = pose.get("posture", "Seated")
            body_gestures = pose.get("gestures", [])

        # Prioritize prominent gestures
        candidates = []
        for bg in body_gestures:
            candidates.append((bg["confidence"] + 0.05, bg["gesture"], f"{bg['gesture'].replace('_', ' ').title()}"))

        for hg in hand_gestures:
            candidates.append((hg["confidence"], hg["gesture"], f"{hg['gesture'].replace('_', ' ').title()} ({hg['side']} Hand)"))

        primary_gesture = "NONE"
        gesture_summary = "None"
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            primary_gesture = candidates[0][1]
            gesture_summary = candidates[0][2]

        # Zero-latency instant snap for confident gestures (>= 0.84)
        if candidates and candidates[0][0] >= 0.84 and primary_gesture != "NONE":
            if self.last_primary_gesture != primary_gesture:
                self.last_primary_gesture = primary_gesture
                self.last_change_time = time.time()
            self.history.clear()
            self.history.append(primary_gesture)
        else:
            # 2-frame debouncing confirmation for lower-confidence gestures or release to NONE
            self.history.append(primary_gesture)
            if len(self.history) >= 2:
                votes = {}
                for g in self.history:
                    votes[g] = votes.get(g, 0) + 1
                most_voted, vote_count = max(votes.items(), key=lambda x: x[1])
                if vote_count >= 2 and most_voted != self.last_primary_gesture:
                    self.last_primary_gesture = most_voted
                    self.last_change_time = time.time()

        stable_duration = round(time.time() - self.last_change_time, 1)

        return {
            "hand_gestures": hand_gestures,
            "body_gestures": body_gestures,
            "primary_gesture": self.last_primary_gesture,
            "gesture_summary": gesture_summary if self.last_primary_gesture != "NONE" else "None",
            "posture": posture,
            "has_gesture": self.last_primary_gesture != "NONE",
            "stable_duration": stable_duration
        }

gesture_pipeline = GesturePipeline()
