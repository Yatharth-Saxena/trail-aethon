import math
import time
from typing import List, Dict, Any, Optional

def compute_bbox_center(bbox: List[int]) -> tuple[float, float]:
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

class HandObjectInteraction:
    def __init__(self):
        self.prev_distances: Dict[str, float] = {}

    def analyze(
        self,
        hands: List[Dict[str, Any]],
        objects: List[Dict[str, Any]],
        frame_shape: tuple[int, int]
    ) -> List[Dict[str, Any]]:
        interactions = []
        h, w = frame_shape[:2]

        for hand in hands:
            hand_side = hand.get("side", "Right")
            hand_bbox = hand.get("bbox")
            if not hand_bbox:
                continue

            index_tip = hand.get("index_tip")
            it_x = index_tip["x"] * w if index_tip else hand_bbox[0]
            it_y = index_tip["y"] * h if index_tip else hand_bbox[1]

            for obj in objects:
                obj_label = obj.get("label", "")
                if obj_label.lower() == "person":
                    continue # Skip self-interaction with person bbox

                obj_bbox = obj.get("bbox")
                if not obj_bbox:
                    continue

                dist = compute_bbox_distance(hand_bbox, obj_bbox)
                overlaps = check_bbox_overlap(hand_bbox, obj_bbox)
                
                # Check if index fingertip is inside object bounding box
                ox1, oy1, ox2, oy2 = obj_bbox
                tip_inside = (ox1 <= it_x <= ox2) and (oy1 <= it_y <= oy2)

                state = "FAR"
                pair_key = f"{hand_side}_{obj_label}"
                prev_d = self.prev_distances.get(pair_key, dist)
                delta_d = dist - prev_d
                self.prev_distances[pair_key] = dist

                if "button" in obj_label.lower():
                    if tip_inside or (overlaps and dist < 50):
                        state = "PRESSING"
                    elif dist < 80:
                        state = "APPROACHING"
                else:
                    if overlaps or tip_inside:
                        if hand.get("is_pinching", False) or dist < 45:
                            state = "GRASPED"
                        else:
                            state = "CONTACT"
                    elif dist < 75:
                        if delta_d < -2.0:
                            state = "APPROACHING"
                        else:
                            state = "NEAR"
                    else:
                        if delta_d > 5.0 and prev_d < 90:
                            state = "RELEASING"
                        else:
                            state = "IDLE"

                interactions.append({
                    "hand": hand_side,
                    "object": obj_label,
                    "state": state,
                    "distance": round(dist, 1),
                    "overlaps": overlaps,
                    "timestamp": time.time()
                })

        return interactions

hand_object_interaction = HandObjectInteraction()
