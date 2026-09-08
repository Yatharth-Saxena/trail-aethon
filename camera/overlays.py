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
    action_info: Optional[Dict[str, Any]] = None,
    pose: Optional[Dict[str, Any]] = None,
    gestures: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Draw detection boxes, hand skeletons & gestures, body pose skeleton, and activity banner.
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

        # Astronaut / Human Operator: Tactical corner brackets & cyber cyan badge
        is_astro = det.get("category") == "ASTRONAUT" or raw_lbl in ("Astronaut", "Person") or label in ("Astronaut", "Person")
        if is_astro:
            astro_color = (200, 230, 0) # Mint / Cyan BGR
            # Draw corner reticle brackets
            corner_len = min(24, max(10, min(x2 - x1, y2 - y1) // 4))
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), astro_color, 2, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), astro_color, 2, cv2.LINE_AA)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), astro_color, 2, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), astro_color, 2, cv2.LINE_AA)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), astro_color, 2, cv2.LINE_AA)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), astro_color, 2, cv2.LINE_AA)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), astro_color, 2, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), astro_color, 2, cv2.LINE_AA)
            # Subtle boundary
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 100, 80), 1)

            track_id = det.get("track_id", 1)
            posture_text = det.get("posture") or (action_info.get("posture") if action_info else "SEATED")
            action_text = det.get("action") or det.get("activity") or (action_info.get("label") if action_info else "ACTIVE")
            text = f"✦ ASTRONAUT #{track_id} // [{str(action_text).upper()}] ({str(posture_text).upper()})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), (15, 25, 20), -1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), astro_color, 1)
            cv2.putText(annotated, text, (x1 + 5, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, astro_color, 1, cv2.LINE_AA)
            continue

        # Everyday & payload objects: clean rectangular bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Category chip and state tags
        cat_badge = det.get("category_badge") or "ITEM"
        tag_prefix = f"[{cat_badge}] "
        if is_held:
            held_by = det.get("held_by") or "Hand"
            tag_prefix += f"[HELD: {held_by}] "
        elif det.get("moving") or det.get("is_moving"):
            tag_prefix += "[MOVE] "

        pos = det.get("position")
        pos_str = f" ({pos['cx']},{pos['cy']})" if pos else ""
        text = f"{tag_prefix}{det.get('display_name') or label}{pos_str} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        # Badge background
        badge_y1 = max(0, y1 - th - 8)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, y1), box_color, -1)

        # Contrast text
        badge_lum = 0.299 * box_color[2] + 0.587 * box_color[1] + 0.114 * box_color[0]
        text_color = (15, 18, 22) if badge_lum > 140 else (250, 250, 250)
        cv2.putText(annotated, text, (x1 + 5, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

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
                g = hand.get("gesture")
                if g and g not in ("NONE", "OPEN_HAND", "REACHING"):
                    grip = g
                else:
                    grip = "GRASP" if hand.get("is_grasping") else ("PINCH" if hand.get("is_pinching") else "OPEN")
                hand_text = f"{side[0]} Hand // {grip}"
                (tw, th), _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tx = max(4, wx - tw // 2)
                ty = max(th + 6, wy - 12)
                cv2.rectangle(annotated, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (15, 20, 25), -1)
                cv2.putText(annotated, hand_text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, h_color, 1, cv2.LINE_AA)

    # 3. Draw Anatomical Body Skeleton Mesh (Multi-Pose)
    poses_to_draw = []
    if pose:
        if "all_poses" in pose and isinstance(pose["all_poses"], list):
            poses_to_draw = pose["all_poses"]
        else:
            poses_to_draw = [pose]

    for p_item in poses_to_draw:
        if not p_item or not p_item.get("landmarks"):
            continue
        lms = p_item["landmarks"]
        num_lms = len(lms)
        mesh_color = (200, 230, 0)      # Cyan / Mint BGR
        spine_color = (245, 245, 245)    # High-contrast white/silver
        accent_color = (248, 189, 56)   # Sky blue BGR

        def get_pt(idx, _lms=lms, _nl=num_lms):
            if idx < _nl and _lms[idx].get("visibility", 1.0) > 0.25:
                return (int(_lms[idx]["x"] * w), int(_lms[idx]["y"] * h))
            return None

        p11 = get_pt(11) # L Shoulder
        p12 = get_pt(12) # R Shoulder
        p23 = get_pt(23) # L Hip
        p24 = get_pt(24) # R Hip

        # 3a. Torso Biometric Polygonal Mesh Facets
        if p11 and p12 and p23 and p24:
            sternum = (int((p11[0] + p12[0]) * 0.5), int((p11[1] + p12[1]) * 0.5))
            mid_hip = (int((p23[0] + p24[0]) * 0.5), int((p23[1] + p24[1]) * 0.5))
            solar_plexus = (int(sternum[0] * 0.4 + mid_hip[0] * 0.6), int(sternum[1] * 0.4 + mid_hip[1] * 0.6))

            # Translucent torso mesh fill
            overlay = annotated.copy()
            torso_poly = np.array([p11, p12, p24, p23], dtype=np.int32)
            cv2.fillPoly(overlay, [torso_poly], (210, 230, 50))
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

            # Torso lattice mesh struts
            cv2.line(annotated, p11, p24, (180, 210, 30), 1, cv2.LINE_AA)
            cv2.line(annotated, p12, p23, (180, 210, 30), 1, cv2.LINE_AA)
            cv2.line(annotated, sternum, p23, (160, 200, 30), 1, cv2.LINE_AA)
            cv2.line(annotated, sternum, p24, (160, 200, 30), 1, cv2.LINE_AA)

            # Vertebral Spine Column
            cv2.line(annotated, sternum, mid_hip, spine_color, 2, cv2.LINE_AA)
            cv2.circle(annotated, solar_plexus, 3, (255, 255, 255), -1)

        # 3b. Complete Human Anatomical Skeleton Connections
        connections = [
            # Shoulders / Clavicle
            (11, 12),
            # Left Arm & Forearm
            (11, 13), (13, 15),
            # Right Arm & Forearm
            (12, 14), (14, 16),
            # Left Hand Anchors
            (15, 17), (15, 19), (15, 21), (17, 19),
            # Right Hand Anchors
            (16, 18), (16, 20), (16, 22), (18, 20),
            # Flanks & Pelvis
            (11, 23), (12, 24), (23, 24),
            # Left Leg & Foot
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
            # Right Leg & Foot
            (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
            # Head / Facial Perimeter
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (0, 11), (0, 12)
        ]

        for idx1, idx2 in connections:
            pt1 = get_pt(idx1)
            pt2 = get_pt(idx2)
            if pt1 and pt2:
                c = accent_color if (idx1 >= 15 or idx2 >= 15) else mesh_color
                cv2.line(annotated, pt1, pt2, c, 2, cv2.LINE_AA)

        # 3c. Articulation Nodes (Biometric Joint Rings)
        for i in range(num_lms):
            pt = get_pt(i)
            if pt:
                if i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
                    cv2.circle(annotated, pt, 5, mesh_color, 1, cv2.LINE_AA)
                    cv2.circle(annotated, pt, 2, (255, 255, 255), -1)
                else:
                    cv2.circle(annotated, pt, 2, (255, 255, 255), -1)

    # 4. Draw Active Gesture HUD Badge (Top-Right)
    primary_gesture = (gestures or {}).get("primary_gesture", "NONE")
    if primary_gesture and primary_gesture not in ("NONE", "STATIONARY"):
        g_text = f"GESTURE // {primary_gesture.replace('_', ' ')}"
        (gtw, gth), _ = cv2.getTextSize(g_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        gx = w - gtw - 24
        gy = 28
        cv2.rectangle(annotated, (gx - 6, gy - gth - 6), (gx + gtw + 12, gy + 6), (25, 20, 15), -1)
        cv2.rectangle(annotated, (gx - 6, gy - gth - 6), (gx + gtw + 12, gy + 6), (248, 189, 56), 1)
        cv2.circle(annotated, (gx + 2, gy - gth // 2), 4, (248, 189, 56), -1)
        cv2.putText(annotated, g_text, (gx + 12, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (248, 189, 56), 1, cv2.LINE_AA)

    # 5. Draw Comprehensive Action & Movement HUD Banner
    if action_info:
        act_label = action_info.get("label", "Monitoring")
        movement = action_info.get("movement", "Stationary")
        posture = (gestures or {}).get("posture") or action_info.get("posture", "Seated")
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
