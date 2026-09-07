import cv2
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque
from ultralytics import YOLO
from backend.config import BASE_DIR, CONFIDENCE_THRESHOLD_OBJECT

# ---------------------------------------------------------------------------
# Temporal Smoothing Configuration
# ---------------------------------------------------------------------------
# Number of consecutive frames an object must be detected before it is
# reported to the pipeline.  This eliminates single-frame phantom detections
# caused by transient HSV-colour matches on skin, clothing, or background.
TEMPORAL_CONFIRM_FRAMES = 5


def extract_dominant_color(frame_bgr: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
    """
    Extracts the dominant perceived color of the object within bbox using HSV color analysis.
    Returns {"name": "Red", "hex": "#e53e3e"}
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_box = y2 - y1
    w_box = x2 - x1
    if h_box <= 0 or w_box <= 0:
        return {"name": "Unknown", "hex": "#888888"}
    
    # Inset by 12% to avoid background boundary contamination
    pad_y = int(h_box * 0.12)
    pad_x = int(w_box * 0.12)
    roi = frame_bgr[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
    if roi.size == 0:
        roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"name": "Unknown", "hex": "#888888"}
        
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    
    med_s = float(np.median(s_chan))
    med_v = float(np.median(v_chan))
    med_h = float(np.median(h_chan))
    
    # Achromatic / Grayscale
    if med_v < 50 or (med_v < 70 and med_s < 50):
        return {"name": "Black", "hex": "#2d3748"}
    elif med_v > 185 and med_s < 38:
        return {"name": "White", "hex": "#f7fafc"}
    elif med_s < 45 and 50 <= med_v <= 185:
        return {"name": "Silver / Gray", "hex": "#a0aec0"}
    
    # Chromatic Hues (0 - 180)
    if med_h < 11 or med_h >= 168:
        return {"name": "Red", "hex": "#e53e3e"}
    elif 11 <= med_h < 25:
        if med_v < 130 and med_s < 160:
            return {"name": "Brown / Wood", "hex": "#9c6b4e"}
        return {"name": "Orange", "hex": "#ed8936"}
    elif 25 <= med_h < 38:
        return {"name": "Yellow", "hex": "#ecc94b"}
    elif 38 <= med_h < 85:
        return {"name": "Green", "hex": "#38a169"}
    elif 85 <= med_h < 105:
        return {"name": "Cyan", "hex": "#319795"}
    elif 105 <= med_h < 135:
        return {"name": "Blue", "hex": "#3182ce"}
    elif 135 <= med_h < 155:
        return {"name": "Purple", "hex": "#805ad5"}
    else:
        return {"name": "Pink", "hex": "#d53f8c"}


class DetectedObject:
    def __init__(
        self,
        label: str,
        confidence: float,
        bbox: List[int],
        color: str = "Unknown",
        color_hex: str = "#888888",
        is_held: bool = False,
        timestamp: Optional[float] = None
    ):
        self.label = label
        self.confidence = float(confidence)
        self.bbox = [int(v) for v in bbox] # [x1, y1, x2, y2]
        self.color = color
        self.color_hex = color_hex
        self.is_held = is_held
        self.timestamp = timestamp or time.time()

    @property
    def display_name(self) -> str:
        if self.label.lower() in ["person", "astronaut"]:
            return "Person"
        # Prepend color if meaningful and not redundant
        if self.color and self.color not in ["Unknown", "Silver / Gray"]:
            c_low = self.color.lower().split(" ")[0]
            if not self.label.lower().startswith(c_low):
                return f"{self.color} {self.label}"
        elif self.color == "Silver / Gray" and "laptop" in self.label.lower():
            return "Silver Laptop"
        elif self.color == "Black" and "phone" in self.label.lower():
            return "Black Phone"
        return self.label

    def to_dict(self) -> Dict[str, Any]:
        disp = self.display_name
        return {
            "label": self.label,
            "raw_label": self.label,
            "display_name": disp,
            "color": self.color,
            "color_hex": self.color_hex,
            "confidence": round(self.confidence, 2),
            "bbox": self.bbox,
            "is_held": self.is_held,
            "timestamp": self.timestamp
        }


class CustomObjectDetector:
    # Everyday items from COCO model with high, confident detection thresholds
    YOLO_EVERYDAY_CLASSES = {
        0: ("Person", 0.40),
        67: ("Cell Phone", 0.50),
        41: ("Cup", 0.50),
        39: ("Bottle", 0.50),
        73: ("Book", 0.52),
        76: ("Scissors", 0.50),
        63: ("Laptop", 0.52),
        64: ("Mouse", 0.48),
        66: ("Keyboard", 0.50),
    }

    def __init__(self):
        self.yolo_model: Optional[YOLO] = None
        self._init_models()

        # -------------------------------------------------------------------
        # Temporal smoothing buffer – tracks per-label detection counts
        # across frames.  A label must appear in N consecutive frames before
        # it is emitted as an actual detection.
        # -------------------------------------------------------------------
        self._label_streak: Dict[str, int] = {}          # label -> consecutive frame count
        self._label_last_frame: Dict[str, int] = {}      # label -> last frame_id seen
        self._frame_id: int = 0

    def _init_models(self):
        model_paths = [
            BASE_DIR / "yolov8n.pt",
            Path("yolov8n.pt"),
            BASE_DIR / "models" / "object_detector" / "weights.pt"
        ]
        for p in model_paths:
            if p.exists():
                try:
                    self.yolo_model = YOLO(str(p))
                    print(f"[ObjectDetector] Loaded YOLO model from {p}")
                    break
                except Exception as e:
                    print(f"[ObjectDetector Error] Failed loading {p}: {e}")
        
        if self.yolo_model is None:
            try:
                self.yolo_model = YOLO("yolov8n.pt")
                print("[ObjectDetector] Initialized YOLOv8n detector")
            except Exception as e:
                print(f"[ObjectDetector Warning] YOLOv8n initialization fallback: {e}")

    # -----------------------------------------------------------------------
    # Temporal confirmation helpers
    # -----------------------------------------------------------------------
    def _confirm_label(self, label: str) -> bool:
        """
        Register that *label* was seen in the current frame. Returns True
        only when the label has been detected in at least TEMPORAL_CONFIRM_FRAMES
        consecutive frames.
        """
        fid = self._frame_id
        last_fid = self._label_last_frame.get(label, -999)

        if last_fid == fid - 1:
            # Consecutive frame
            self._label_streak[label] = self._label_streak.get(label, 0) + 1
        else:
            # Gap detected – reset streak
            self._label_streak[label] = 1

        self._label_last_frame[label] = fid
        return self._label_streak[label] >= TEMPORAL_CONFIRM_FRAMES

    def _begin_frame(self):
        """Called once per frame at the start of detect()."""
        self._frame_id += 1

    def detect(
        self,
        frame: np.ndarray,
        person_bbox: Optional[List[int]] = None,
        pose_data: Optional[Dict[str, Any]] = None,
        hands: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        self._begin_frame()
        detections: List[Dict[str, Any]] = []
        h, w = frame.shape[:2]

        # 1. Person anchor (Astronaut) – always emitted if detected
        if person_bbox is not None and len(person_bbox) >= 4:
            detections.append(
                DetectedObject("Person", 0.96, person_bbox, color="Suit / Clothes").to_dict()
            )

        # Build torso & head exclusion mask to prevent clothing/face being detected as objects
        torso_mask = np.zeros((h, w), dtype=np.uint8)
        pcx = w / 2.0
        pw = w * 0.4
        if person_bbox and len(person_bbox) >= 4:
            px1, py1, px2, py2 = person_bbox
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)
            pcx = (px1 + px2) / 2.0

            # Head region exclusion
            if pose_data and pose_data.get("head"):
                hx = int(pose_data["head"]["x"] * w)
                hy = int(pose_data["head"]["y"] * h)
                cv2.circle(torso_mask, (hx, hy), max(25, int(pw * 0.32)), 255, -1)
            else:
                cv2.rectangle(torso_mask, (px1, py1), (px2, int(py1 + ph * 0.40)), 255, -1)

            # Torso region exclusion (from shoulders to waist) – wider margin
            shoulders_y = int(pose_data.get("shoulders_mid_y", 0.45) * h) if pose_data else int(py1 + ph * 0.28)
            margin_x = int(pw * 0.04)  # reduced inset → wider exclusion
            tx1 = max(0, px1 - margin_x)
            tx2 = min(w - 1, px2 + margin_x)
            ty1 = max(0, shoulders_y - 15)
            ty2 = min(h - 1, py2)
            if ty2 > ty1 and tx2 > tx1:
                torso_mask[ty1:ty2, tx1:tx2] = 255

            # Forearm exclusion when hands are not pinching/grasping
            if pose_data and hands is not None:
                for side_key, wrist_key in [("left_wrist", "left_wrist"), ("right_wrist", "right_wrist")]:
                    wr = pose_data.get(wrist_key)
                    if wr:
                        # Check if this side's hand is pinching (actively grasping)
                        side_name = "Left" if "left" in wrist_key else "Right"
                        is_grasping = any(
                            hnd.get("side") == side_name and hnd.get("is_pinching", False)
                            for hnd in (hands or [])
                        )
                        if not is_grasping:
                            wx = int(wr["x"] * w)
                            wy = int(wr["y"] * h)
                            # Draw a circle around wrist/forearm area to exclude
                            cv2.circle(torso_mask, (wx, wy), max(20, int(pw * 0.18)), 255, -1)

        # Helper to check if a point is near tracked hands
        def is_near_hand(cx: float, cy: float) -> bool:
            if not hands:
                return False
            for hnd in hands:
                hb = hnd.get("bbox")
                if hb and len(hb) >= 4:
                    hcx = (hb[0] + hb[2]) / 2.0
                    hcy = (hb[1] + hb[3]) / 2.0
                    # Inside hand box or within 55px radius
                    if (hb[0] - 12 <= cx <= hb[2] + 12) and (hb[1] - 12 <= cy <= hb[3] + 12):
                        return True
                    if np.hypot(cx - hcx, cy - hcy) < 55:
                        return True
            return False

        # Helper to check if candidate is inside person torso/head
        def is_in_torso(cx: int, cy: int) -> bool:
            if 0 <= cy < h and 0 <= cx < w:
                return torso_mask[cy, cx] > 0
            return False

        # 2. Everyday Object Detection via YOLOv8 (Strict thresholds)
        if self.yolo_model is not None:
            try:
                target_ids = list(self.YOLO_EVERYDAY_CLASSES.keys())
                results = self.yolo_model.predict(
                    frame,
                    classes=target_ids,
                    conf=0.42,
                    verbose=False,
                    imgsz=480
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls_id in self.YOLO_EVERYDAY_CLASSES:
                            label, min_c = self.YOLO_EVERYDAY_CLASSES[cls_id]
                            if conf >= min_c:
                                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                                xyxy = [max(0, xyxy[0]), max(0, xyxy[1]), min(w - 1, xyxy[2]), min(h - 1, xyxy[3])]
                                
                                if cls_id == 0:
                                    # Direct Person (Astronaut) detection via YOLO
                                    detections.append(
                                        DetectedObject(
                                            label="Person",
                                            confidence=conf,
                                            bbox=xyxy,
                                            color="Suit / Clothes",
                                            color_hex="#f0f0f0"
                                        ).to_dict()
                                    )
                                    continue

                                bcx = (xyxy[0] + xyxy[2]) // 2
                                bcy = (xyxy[1] + xyxy[3]) // 2
                                # Exclude if inside torso unless held in hand
                                if is_in_torso(bcx, bcy) and not is_near_hand(bcx, bcy):
                                    continue
                                color_info = extract_dominant_color(frame, xyxy)
                                detections.append(
                                    DetectedObject(
                                        label=label,
                                        confidence=conf,
                                        bbox=xyxy,
                                        color=color_info["name"],
                                        color_hex=color_info["hex"]
                                    ).to_dict()
                                )
            except Exception:
                pass

        # 3. Specialized BAS Payload Objects (Block A, Block B, Tray, Button)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -------------------------------------------------------------
        # Object A: Red Payload Block  (TIGHTENED)
        # -------------------------------------------------------------
        lower_red1 = np.array([0, 160, 110])
        upper_red1 = np.array([7, 255, 255])
        lower_red2 = np.array([173, 160, 110])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours_a, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cand_a = None
        best_score_a = 0.0

        for c in contours_a:
            area = cv2.contourArea(c)
            rx, ry, rw, rh = cv2.boundingRect(c)
            cx, cy = rx + rw // 2, ry + rh // 2

            in_hand = is_near_hand(cx, cy)
            min_area = 550 if in_hand else 800
            if area < min_area or area > 7000:
                continue

            # Torso rejection: if on torso and not held by hand, reject (it's clothing)
            if is_in_torso(cx, cy) and not in_hand:
                continue

            # Spatial reach: if not in hand, must be on lower table surface within operator corridor
            if not in_hand:
                if cy < h * 0.58:  # Above desk level -> background/wall/curtains/posters
                    continue
                if abs(cx - pcx) > pw * 0.65:  # Outside operator reach corridor -> reject
                    continue

            # Geometric check: solidity, extent, aspect ratio (TIGHTENED)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            extent = area / float(rw * rh)
            aspect = rw / float(max(1, rh))

            if solidity < 0.86 or extent < 0.65 or not (0.60 <= aspect <= 1.65):
                continue

            # BGR color purity verification (TIGHTENED)
            roi = frame[ry:ry+rh, rx:rx+rw]
            if roi.size == 0:
                continue
            mean_b = float(np.mean(roi[:, :, 0]))
            mean_g = float(np.mean(roi[:, :, 1]))
            mean_r = float(np.mean(roi[:, :, 2]))
            if not (mean_r > 1.50 * mean_g and mean_r > 1.50 * mean_b and mean_r > 100):
                continue

            score = solidity * 0.4 + extent * 0.4 + (1.0 if in_hand else 0.5) * 0.2
            if score > best_score_a:
                best_score_a = score
                conf = round(min(0.96, 0.85 + (area / 7000.0) * 0.11), 2)
                best_cand_a = DetectedObject(
                    label="Object A",
                    confidence=conf,
                    bbox=[rx, ry, rx + rw, ry + rh],
                    color="Red",
                    color_hex="#ef4444"
                ).to_dict()

        if best_cand_a:
            if self._confirm_label("Object A"):
                detections.append(best_cand_a)
            else:
                # Still accumulating confirmation frames – don't emit yet
                pass
        else:
            # Object A not seen this frame → reset streak
            self._label_streak["Object A"] = 0

        # -------------------------------------------------------------
        # Object B: Blue Assembly Block  (TIGHTENED — Wood fallback removed)
        # -------------------------------------------------------------
        lower_blue = np.array([100, 130, 90])
        upper_blue = np.array([126, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours_b, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_b = False
        for c in contours_b:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            cx, cy = bx + bw // 2, by + bh // 2
            in_hand = is_near_hand(cx, cy)
            min_area = 600 if in_hand else 800
            if area < min_area or area > 7000:
                continue
            if is_in_torso(cx, cy) and not in_hand:
                continue
            if not in_hand and (cy < h * 0.58 or abs(cx - pcx) > pw * 0.65):
                continue

            # Ensure not inside a detected Laptop screen or background monitor
            inside_laptop = False
            for det in detections:
                if det.get("raw_label") == "Laptop":
                    lb = det.get("bbox", [])
                    if lb and lb[0] <= cx <= lb[2] and lb[1] <= cy <= lb[3]:
                        inside_laptop = True
                        break
            if inside_laptop and not in_hand:
                continue

            hull = cv2.convexHull(c)
            solidity = area / max(1, cv2.contourArea(hull))
            extent = area / float(bw * bh)
            aspect = bw / float(max(1, bh))
            if solidity >= 0.88 and extent >= 0.65 and (0.60 <= aspect <= 1.65):
                # BGR color purity check for blue dominance
                roi_b = frame[by:by+bh, bx:bx+bw]
                if roi_b.size > 0:
                    mean_b_ch = float(np.mean(roi_b[:, :, 0]))
                    mean_g_ch = float(np.mean(roi_b[:, :, 1]))
                    mean_r_ch = float(np.mean(roi_b[:, :, 2]))
                    if not (mean_b_ch > mean_r_ch * 1.15 and mean_b_ch > mean_g_ch * 1.05 and mean_b_ch > 70):
                        continue

                if self._confirm_label("Object B"):
                    detections.append(
                        DetectedObject(
                            label="Object B",
                            confidence=0.92,
                            bbox=[bx, by, bx + bw, by + bh],
                            color="Blue",
                            color_hex="#3b82f6"
                        ).to_dict()
                    )
                found_b = True
                break

        if not found_b:
            # Fallback to Wooden Block (with strict human skin exclusion)
            lower_wood = np.array([14, 95, 80])
            upper_wood = np.array([23, 180, 190])
            wood_mask = cv2.inRange(hsv, lower_wood, upper_wood)
            wood_mask = cv2.morphologyEx(wood_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours_wb, _ = cv2.findContours(wood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours_wb:
                area = cv2.contourArea(c)
                bx, by, bw, bh = cv2.boundingRect(c)
                cx, cy = bx + bw // 2, by + bh // 2
                in_hand = is_near_hand(cx, cy)
                min_area = 700 if in_hand else 900
                if area < min_area or area > 5500:
                    continue
                # Strict: cannot be inside torso, head, or arms
                if is_in_torso(cx, cy):
                    continue
                if not in_hand and (cy < h * 0.58 or abs(cx - pcx) > pw * 0.65):
                    continue
                hull = cv2.convexHull(c)
                solidity = area / max(1, cv2.contourArea(hull))
                extent = area / float(bw * bh)
                aspect = bw / float(max(1, bh))
                if solidity >= 0.90 and extent >= 0.70 and (0.70 <= aspect <= 1.45):
                    if self._confirm_label("Object B"):
                        detections.append(
                            DetectedObject(
                                label="Object B",
                                confidence=0.90,
                                bbox=[bx, by, bx + bw, by + bh],
                                color="Brown / Wood",
                                color_hex="#9c6b4e"
                            ).to_dict()
                        )
                    found_b = True
                    break

        if not found_b:
            self._label_streak["Object B"] = 0

        # -------------------------------------------------------------
        # Tray: Distinct Assembly Container on Table Surface  (TIGHTENED)
        # -------------------------------------------------------------
        lower_tray_purple = np.array([128, 70, 55])
        upper_tray_purple = np.array([165, 255, 225])
        tray_mask = cv2.inRange(hsv, lower_tray_purple, upper_tray_purple)
        tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        contours_tray, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_tray = False
        for c in contours_tray:
            area = cv2.contourArea(c)
            if area < 5000 or area > 30000:
                continue
            tx, ty, tw, th = cv2.boundingRect(c)
            tcx, tcy = tx + tw // 2, ty + th // 2
            if ty < h * 0.55:  # Must be on lower workspace desk
                continue
            if abs(tcx - pcx) > pw * 0.80:  # Must be in front of astronaut
                continue
            if is_in_torso(tcx, tcy):
                continue
            aspect = tw / float(max(1, th))
            if not (1.5 <= aspect <= 3.2):
                continue
            hull = cv2.convexHull(c)
            solidity = area / max(1, cv2.contourArea(hull))
            extent = area / float(tw * th)
            if solidity >= 0.86 and extent >= 0.62:
                # BGR color purity: ensure purple tint (blue > red > green)
                roi_t = frame[ty:ty+th, tx:tx+tw]
                if roi_t.size > 0:
                    mean_b_t = float(np.mean(roi_t[:, :, 0]))
                    mean_g_t = float(np.mean(roi_t[:, :, 1]))
                    mean_r_t = float(np.mean(roi_t[:, :, 2]))
                    # Purple should have meaningful blue channel
                    if mean_b_t < 40 and mean_r_t < 40:
                        continue

                if self._confirm_label("Tray"):
                    detections.append(
                        DetectedObject(
                            label="Tray",
                            confidence=0.90,
                            bbox=[tx, ty, tx + tw, ty + th],
                            color="Purple",
                            color_hex="#a855f7"
                        ).to_dict()
                    )
                found_tray = True
                break

        if not found_tray:
            self._label_streak["Tray"] = 0

        # -------------------------------------------------------------
        # Complete Button: Yellow Console Button  (TIGHTENED)
        # -------------------------------------------------------------
        lower_btn = np.array([24, 150, 140])
        upper_btn = np.array([35, 255, 255])
        btn_mask = cv2.inRange(hsv, lower_btn, upper_btn)
        btn_mask = cv2.morphologyEx(btn_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours_btn, _ = cv2.findContours(btn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_btn = False
        for c in contours_btn:
            area = cv2.contourArea(c)
            if area < 500 or area > 4000:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            bcx, bcy = bx + bw // 2, by + bh // 2
            if is_in_torso(bcx, bcy):
                continue
            if bcy < h * 0.52:  # Must be on lower workstation console
                continue
            if abs(bcx - pcx) > pw * 0.80:
                continue
            aspect = bw / float(max(1, bh))
            if not (0.75 <= aspect <= 1.30):
                continue
            hull = cv2.convexHull(c)
            solidity = area / max(1, cv2.contourArea(hull))
            extent = area / float(bw * bh)
            if solidity >= 0.87 and extent >= 0.65:
                # BGR purity: yellow should have high R+G, low B
                roi_btn = frame[by:by+bh, bx:bx+bw]
                if roi_btn.size > 0:
                    mean_b_btn = float(np.mean(roi_btn[:, :, 0]))
                    mean_g_btn = float(np.mean(roi_btn[:, :, 1]))
                    mean_r_btn = float(np.mean(roi_btn[:, :, 2]))
                    if not (mean_r_btn > 100 and mean_g_btn > 100 and mean_b_btn < mean_r_btn * 0.7):
                        continue

                if self._confirm_label("Complete Button"):
                    detections.append(
                        DetectedObject(
                            label="Complete Button",
                            confidence=0.92,
                            bbox=[bx, by, bx + bw, by + bh],
                            color="Yellow",
                            color_hex="#eab308"
                        ).to_dict()
                    )
                found_btn = True
                break

        if not found_btn:
            self._label_streak["Complete Button"] = 0

        # 4. Remove duplicate/overlapping bounding boxes (IoU > 0.45)
        return self._nms(detections, iou_thresh=0.45)

    def _nms(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        if len(detections) <= 1:
            return detections
        # Sort by confidence descending
        dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
        keep = []
        for d in dets:
            box_a = d.get("bbox")
            if not box_a or len(box_a) < 4:
                continue
            overlap = False
            for kept in keep:
                # Don't suppress Person
                if d.get("raw_label") == "Person" or kept.get("raw_label") == "Person":
                    continue
                box_b = kept.get("bbox")
                iou = self._calc_iou(box_a, box_b)
                if iou > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                keep.append(d)
        return keep

    def _calc_iou(self, b1: List[int], b2: List[int]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area <= 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter_area / float(area1 + area2 - inter_area)

object_detector = CustomObjectDetector()
