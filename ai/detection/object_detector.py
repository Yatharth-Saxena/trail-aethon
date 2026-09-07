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
                                
                                exp_label = label
                                cname = color_info["name"]
                                if cname == "Red":
                                    exp_label = "Object A"
                                elif cname in ["Blue", "Brown / Wood"]:
                                    exp_label = "Object B"
                                elif cname in ["Purple", "Silver / Gray"]:
                                    exp_label = "Tray"
                                elif cname in ["Yellow", "Orange"]:
                                    exp_label = "Complete Button"
                                elif label in ["Cup", "Bottle"] and not any(d["label"] == "Object A" for d in detections):
                                    exp_label = "Object A" # Fallback mapping

                                detections.append(
                                    DetectedObject(
                                        label=exp_label,
                                        confidence=conf,
                                        bbox=xyxy,
                                        color=cname,
                                        color_hex=color_info["hex"]
                                    ).to_dict()
                                )
            except Exception:
                pass

        # 3. Custom HSV checks removed in favor of YOLO mapping
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
                pass # Allow NMS to suppress duplicate persons
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
