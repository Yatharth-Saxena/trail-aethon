import cv2
import numpy as np
from typing import List, Dict, Any, Optional

# Color palettes in BGR format
COLOR_MAP_BGR = {
    "Red": (40, 40, 235),
    "Crimson": (30, 20, 180),
    "Blue": (225, 135, 35),
    "Cyan": (220, 210, 40),
    "Green": (50, 195, 55),
    "Yellow": (40, 215, 245),
    "Orange": (30, 135, 245),
    "Purple": (195, 65, 185),
    "Pink": (175, 75, 235),
    "Black": (70, 70, 70),
    "White": (245, 245, 245),
    "Silver / Gray": (185, 185, 185),
    "Brown / Wood": (65, 105, 155),
    "Person": (240, 240, 240),
    "Tray": (195, 65, 185),
    "Complete Button": (40, 215, 245)
}

def draw_perception_overlays(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    hands: Optional[List[Dict[str, Any]]] = None,
    action_info: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Draw detection boxes, hand landmarks and the activity banner.

    Only hands are skeletonised — there is no body-pose overlay.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Draw Object & Everyday Item Detections
    for det in detections:
        label = det.get("label", "Object")
        raw_lbl = det.get("raw_label", label)
        conf = det.get("confidence", 0.0)
        bbox = det.get("bbox") # [x1, y1, x2, y2]
        color_name = det.get("color", "")
        is_held = det.get("is_held", False)

        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Select display color
        box_color = COLOR_MAP_BGR.get(color_name)
        if not box_color:
            if raw_lbl in COLOR_MAP_BGR:
                box_color = COLOR_MAP_BGR[raw_lbl]
            else:
                box_color = (0, 220, 200) # Fallback Cyan

        # If object is black, use visible slate gray border so it contrasts on dark backgrounds
        if color_name == "Black":
            box_color = (120, 120, 120)

        # Person bounding box: thinner border
        if raw_lbl == "Person":
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (230, 230, 230), 1)
            movement_text = action_info.get("movement", "Active") if action_info else "Active"
            text = f"ASTRONAUT {conf:.2f} // {movement_text.upper()}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1), (30, 35, 40), -1)
            cv2.putText(annotated, text, (x1 + 4, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            continue

        # Everyday & payload objects: clean rectangular bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Held / independent-motion tags. OpenCV's Hershey fonts have no
        # glyph for "●", so the annotated stream uses ASCII markers.
        tag_prefix = "[HELD] " if is_held else ""
        if det.get("moving") or det.get("is_moving"):
            tag_prefix += "[MOVE] "
        text = f"{tag_prefix}{det.get('display_name') or label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

        # Badge background
        badge_y1 = max(0, y1 - th - 8)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, y1), box_color, -1)

        # Contrast text
        badge_lum = 0.299 * box_color[2] + 0.587 * box_color[1] + 0.114 * box_color[0]
        text_color = (15, 18, 22) if badge_lum > 140 else (250, 250, 250)
        cv2.putText(annotated, text, (x1 + 5, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

    # 2. Draw Hand Landmarks & Skeleton
    if hands:
        for hand in hands:
            landmarks = hand.get("landmarks", [])
            side = hand.get("side", "Right")
            h_color = (80, 220, 230) if side == "Right" else (180, 230, 80)

            pts = []
            for lm in landmarks:
                px = int(lm["x"] * w)
                py = int(lm["y"] * h)
                pts.append((px, py))

            if len(pts) >= 21:
                hand_connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17)
                ]
                for p1, p2 in hand_connections:
                    cv2.line(annotated, pts[p1], pts[p2], h_color, 2, cv2.LINE_AA)

            # Fingertips get a larger marker than the intermediate joints
            fingertip_ids = {4, 8, 12, 16, 20}
            for i, (px, py) in enumerate(pts):
                radius = 5 if i in fingertip_ids else 3
                cv2.circle(annotated, (px, py), radius, h_color, -1)
                if i in fingertip_ids:
                    cv2.circle(annotated, (px, py), 2, (255, 255, 255), -1)

            wrist = hand.get("wrist")
            if wrist:
                wx, wy = int(wrist["x"] * w), int(wrist["y"] * h)
                grip = "GRASP" if hand.get("is_grasping") else ("PINCH" if hand.get("is_pinching") else "OPEN")
                hand_text = f"{side[0]} Hand // {grip}"
                (tw, th), _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tx = max(4, wx - tw // 2)
                ty = max(th + 6, wy - 12)
                cv2.rectangle(annotated, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (15, 20, 25), -1)
                cv2.putText(annotated, hand_text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, h_color, 1, cv2.LINE_AA)

    # 3. Draw Comprehensive Action & Movement HUD Banner
    if action_info:
        act_label = action_info.get("label", "Monitoring")
        movement = action_info.get("movement", "Stationary")
        posture = action_info.get("posture", "Seated")
        conf = action_info.get("confidence", 0.0)

        banner_text = f"ACTIVITY: {act_label}"
        if movement and movement != "Stationary":
            banner_text += f" | {movement}"
        if posture:
            banner_text += f" [{posture}]"

        bh = 34
        overlay = annotated.copy()
        cv2.rectangle(overlay, (12, h - bh - 12), (w - 12, h - 12), (12, 16, 20), -1)
        cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)
        cv2.rectangle(annotated, (12, h - bh - 12), (w - 12, h - 12), (220, 230, 235), 1)

        # Pulsing status dot
        dot_color = (0, 240, 120) if "holding" in act_label.lower() or "pressing" in act_label.lower() or "placing" in act_label.lower() else (0, 220, 240)
        cv2.circle(annotated, (30, h - 29), 6, dot_color, -1)
        cv2.putText(annotated, banner_text, (46, h - 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)

    return annotated
