import math
import time
from typing import List, Dict, Any, Optional, Tuple

# Approach/contact radii are expressed as a fraction of the frame width so the
# thresholds hold at any camera resolution.
CONTACT_REACH_FRAC = 0.045
APPROACH_REACH_FRAC = 0.085
# Per-frame closing/opening speed (pixels) that counts as intentional motion.
APPROACH_SPEED = 1.5
RELEASE_SPEED = 4.0


def compute_bbox_center(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_bbox_distance(bbox1: List[int], bbox2: List[int]) -> float:
    cx1, cy1 = compute_bbox_center(bbox1)
    cx2, cy2 = compute_bbox_center(bbox2)
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def check_bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)


def bbox_gap(bbox1: List[int], bbox2: List[int]) -> float:
    """Shortest edge-to-edge distance between two boxes; 0 when they overlap."""
    dx = max(bbox2[0] - bbox1[2], bbox1[0] - bbox2[2], 0)
    dy = max(bbox2[1] - bbox1[3], bbox1[1] - bbox2[3], 0)
    return math.hypot(dx, dy)


def point_in_bbox(px: float, py: float, bbox: List[int], margin: float = 0.0) -> bool:
    return (bbox[0] - margin <= px <= bbox[2] + margin) and (bbox[1] - margin <= py <= bbox[3] + margin)


class HandObjectInteraction:
    """
    Derives a contact state for every (hand, object) pair.

    Proximity is measured edge-to-edge between the hand and object boxes, and
    contact is confirmed with fingertip positions, so large objects no longer
    read as "far" simply because their centre is far from the hand's centre.
    """

    def __init__(self):
        self.prev_gaps: Dict[str, float] = {}
        self._last_seen: Dict[str, float] = {}

    def analyze(
        self,
        hands: List[Dict[str, Any]],
        objects: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        interactions: List[Dict[str, Any]] = []
        h, w = frame_shape[:2]
        now = time.time()

        contact_reach = max(22.0, w * CONTACT_REACH_FRAC)
        approach_reach = max(45.0, w * APPROACH_REACH_FRAC)

        for hand in hands:
            hand_side = hand.get("side", "Right")
            hand_bbox = hand.get("bbox")
            if not hand_bbox or len(hand_bbox) < 4:
                continue

            is_grasping = hand.get("is_grasping", hand.get("is_pinching", False))

            # Fingertips arrive normalised; convert to this frame's pixels.
            tips: List[Tuple[float, float]] = []
            for tip in hand.get("fingertips", []) or []:
                tips.append((tip["x"] * w, tip["y"] * h))
            index_tip = hand.get("index_tip")
            index_px = (index_tip["x"] * w, index_tip["y"] * h) if index_tip else None
            if index_px and not tips:
                tips.append(index_px)

            for obj in objects:
                obj_label = obj.get("label", "")
                if obj_label.lower() == "person":
                    continue  # The operator's own body box is not an interaction target.

                obj_bbox = obj.get("bbox")
                if not obj_bbox or len(obj_bbox) < 4:
                    continue

                gap = bbox_gap(hand_bbox, obj_bbox)
                overlaps = check_bbox_overlap(hand_bbox, obj_bbox)
                tip_inside = any(point_in_bbox(tx, ty, obj_bbox) for tx, ty in tips)
                index_inside = bool(index_px and point_in_bbox(index_px[0], index_px[1], obj_bbox))

                # Track closing speed per hand/object pair. Using the track id
                # keeps pairs distinct when two objects share a label.
                pair_key = f"{hand_side}|{obj.get('track_id', obj_label)}"
                prev_gap = self.prev_gaps.get(pair_key, gap)
                delta = gap - prev_gap
                self.prev_gaps[pair_key] = gap
                self._last_seen[pair_key] = now

                if "button" in obj_label.lower():
                    if index_inside or (overlaps and gap <= contact_reach * 0.5):
                        state = "PRESSING"
                    elif gap < approach_reach:
                        state = "APPROACHING"
                    else:
                        state = "IDLE"
                else:
                    if tip_inside or overlaps:
                        state = "GRASPED" if is_grasping else "CONTACT"
                    elif gap <= contact_reach:
                        state = "CONTACT" if is_grasping else "NEAR"
                    elif gap <= approach_reach:
                        state = "APPROACHING" if delta < -APPROACH_SPEED else "NEAR"
                    elif delta > RELEASE_SPEED and prev_gap <= approach_reach:
                        state = "RELEASING"
                    else:
                        state = "FAR"

                interactions.append({
                    "hand": hand_side,
                    "object": obj_label,
                    "state": state,
                    "distance": round(gap, 1),
                    "overlaps": overlaps,
                    "timestamp": now,
                })

        self._prune(now)
        return interactions

    def _prune(self, now: float, max_age: float = 3.0):
        """Forget pairs that have not been observed recently."""
        stale = [k for k, t in self._last_seen.items() if now - t > max_age]
        for k in stale:
            self._last_seen.pop(k, None)
            self.prev_gaps.pop(k, None)


hand_object_interaction = HandObjectInteraction()
