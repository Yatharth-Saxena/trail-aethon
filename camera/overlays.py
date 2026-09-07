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
    pose: Optional[Dict[str, Any]] = None,
    action_info: Optional[Dict[str, Any]] = None
) -> np.ndarray:
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
            text = f"Astronaut {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1), (30, 35, 40), -1)
            cv2.putText(annotated, text, (x1 + 4, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            continue

        # Everyday & payload objects: clean rectangular bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Held tag indicator
        tag_prefix = "● HELD: " if is_held else ""
        text = f"{tag_prefix}{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

        # Badge background
        badge_y1 = max(0, y1 - th - 8)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, y1), box_color, -1)

        # Contrast text
        badge_lum = 0.299 * box_color[2] + 0.587 * box_color[1] + 0.114 * box_color[0]
        text_color = (15, 18, 22) if badge_lum > 140 else (250, 250, 250)
        cv2.putText(annotated, text, (x1 + 5, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

    # 2. Draw MediaPipe Human Pose Skeleton & Joints
    if pose and pose.get("landmarks"):
        lms = pose["landmarks"]
        if len(lms) >= 33:
            pose_pts = []
            for lm in lms:
                px = int(lm["x"] * w)
                py = int(lm["y"] * h)
                vis = lm.get("visibility", 1.0)
                pose_pts.append((px, py, vis))

            connections = [
                # Torso
                (11, 12), (11, 23), (12, 24), (23, 24),
                # Left Arm
                (11, 13), (13, 15),
                # Right Arm
                (12, 14), (14, 16),
                # Left Leg
                (23, 25), (25, 27), (27, 29), (29, 31),
                # Right Leg
                (24, 26), (26, 28), (28, 30), (30, 32),
                # Shoulders to head
                (11, 0), (12, 0)
            ]

            bone_color = (0, 230, 200)    # Neon Cyan
            joint_color = (255, 255, 255) # Bright White Core

            # Draw bones
            for p1, p2 in connections:
                if pose_pts[p1][2] > 0.25 and pose_pts[p2][2] > 0.25:
                    pt1 = (pose_pts[p1][0], pose_pts[p1][1])
                    pt2 = (pose_pts[p2][0], pose_pts[p2][1])
                    cv2.line(annotated, pt1, pt2, bone_color, 2, cv2.LINE_AA)

            # Draw key joints
            key_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            for j in key_joints:
                if pose_pts[j][2] > 0.25:
                    jp = (pose_pts[j][0], pose_pts[j][1])
                    cv2.circle(annotated, jp, 5, bone_color, -1)
                    cv2.circle(annotated, jp, 2, joint_color, -1)

            # Astronaut Pose HUD Badge
            head_p = pose_pts[0]
            if head_p[2] > 0.25:
                hx, hy = head_p[0], head_p[1]
                movement_text = action_info.get("movement", "Active") if action_info else "Active"
                badge_text = f"ASTRONAUT // {movement_text.upper()}"
                (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                bx = max(8, hx - bw // 2)
                by = max(18, hy - 30)
                cv2.rectangle(annotated, (bx - 4, by - bh - 4), (bx + bw + 4, by + 4), (15, 20, 25), -1)
                cv2.rectangle(annotated, (bx - 4, by - bh - 4), (bx + bw + 4, by + 4), (0, 230, 200), 1)
                cv2.putText(annotated, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 200), 1, cv2.LINE_AA)

    # 3. Draw Hand Landmarks
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
                cv2.circle(annotated, (px, py), 3, h_color, -1)

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
                    cv2.line(annotated, pts[p1], pts[p2], h_color, 1, cv2.LINE_AA)

            wrist = hand.get("wrist")
            if wrist:
                wx, wy = int(wrist["x"] * w), int(wrist["y"] * h)
                hand_text = f"Hand ({side[0]})"
                cv2.putText(annotated, hand_text, (wx - 25, max(15, wy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, h_color, 1, cv2.LINE_AA)

    # 4. Draw Comprehensive Action & Movement HUD Banner
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
