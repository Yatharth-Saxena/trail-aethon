import cv2
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from backend.config import BASE_DIR, CONFIDENCE_THRESHOLD_OBJECT

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
    # Everyday items from COCO model
    YOLO_EVERYDAY_CLASSES = {
        67: ("Cell Phone", 0.28),
        41: ("Cup", 0.30),
        39: ("Bottle", 0.30),
        73: ("Book", 0.30),
        76: ("Scissors", 0.30),
        63: ("Laptop", 0.32),
        64: ("Mouse", 0.32),
        66: ("Keyboard", 0.32),
    }

    def __init__(self):
        self.yolo_model: Optional[YOLO] = None
        self._init_models()

    def _init_models(self):
        # Look for local yolov8n.pt
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

    def detect(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        h, w = frame.shape[:2]

        # 1. Person anchor
        if person_bbox is not None:
            detections.append(
                DetectedObject("Person", 0.96, person_bbox, color="Suit / Clothes").to_dict()
            )

        # 2. Everyday Object Detection via YOLOv8
        if self.yolo_model is not None:
            try:
                target_ids = list(self.YOLO_EVERYDAY_CLASSES.keys())
                results = self.yolo_model.predict(
                    frame,
                    classes=target_ids,
                    conf=0.25,
                    verbose=False,
                    imgsz=320
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls_id in self.YOLO_EVERYDAY_CLASSES:
                            label, min_c = self.YOLO_EVERYDAY_CLASSES[cls_id]
                            if conf >= min_c:
                                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                                # Clip to frame
                                xyxy = [max(0, xyxy[0]), max(0, xyxy[1]), min(w - 1, xyxy[2]), min(h - 1, xyxy[3])]
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
            except Exception as e:
                pass

        # 3. Specialized BAS Payload Objects (Block A, Block B, Tray, Button)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Object A: Red Payload Block
        lower_red1 = np.array([0, 130, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([168, 130, 80])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for c in contours:
                area = cv2.contourArea(c)
                if 200 < area < 2800:
                    rx, ry, rw, rh = cv2.boundingRect(c)
                    if rw < 70 and rh < 70 and ry > 20:
                        aspect = rw / max(1, rh)
                        if 0.45 <= aspect <= 2.2:
                            conf = min(0.97, 0.82 + (area / 2000.0) * 0.15)
                            color_info = extract_dominant_color(frame, [rx, ry, rx + rw, ry + rh])
                            detections.append(
                                DetectedObject(
                                    label="Object A",
                                    confidence=conf,
                                    bbox=[rx, ry, rx + rw, ry + rh],
                                    color=color_info.get("name", "Red"),
                                    color_hex=color_info.get("hex", "#e53e3e")
                                ).to_dict()
                            )
                            break

        # Object B: Wooden Assembly Block
        lower_wood = np.array([14, 70, 65])
        upper_wood = np.array([26, 210, 220])
        wood_mask = cv2.inRange(hsv, lower_wood, upper_wood)
        wood_mask = cv2.morphologyEx(wood_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        wood_mask = cv2.morphologyEx(wood_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        
        contours_b, _ = cv2.findContours(wood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_b:
            for c in contours_b:
                area = cv2.contourArea(c)
                if 300 < area < 3200:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    if bw < 80 and bh < 80 and by > 30:
                        conf = min(0.95, 0.78 + (area / 2500.0) * 0.15)
                        color_info = extract_dominant_color(frame, [bx, by, bx + bw, by + bh])
                        detections.append(
                            DetectedObject(
                                label="Object B",
                                confidence=conf,
                                bbox=[bx, by, bx + bw, by + bh],
                                color=color_info.get("name", "Brown / Wood"),
                                color_hex=color_info.get("hex", "#9c6b4e")
                            ).to_dict()
                        )
                        break

        # Tray: Wide container surface
        lower_tray = np.array([0, 0, 40])
        upper_tray = np.array([180, 55, 180])
        tray_mask = cv2.inRange(hsv, lower_tray, upper_tray)
        tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        contours_tray, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_tray:
            for c in contours_tray:
                area = cv2.contourArea(c)
                if area > 4000:
                    tx, ty, tw, th = cv2.boundingRect(c)
                    if tw > th * 1.4:
                        color_info = extract_dominant_color(frame, [tx, ty, tx + tw, ty + th])
                        detections.append(
                            DetectedObject(
                                label="Tray",
                                confidence=0.88,
                                bbox=[tx, ty, tx + tw, ty + th],
                                color=color_info.get("name", "Silver / Gray"),
                                color_hex=color_info.get("hex", "#a0aec0")
                            ).to_dict()
                        )
                        break

        # Complete Button
        lower_btn = np.array([22, 100, 100])
        upper_btn = np.array([38, 255, 255])
        btn_mask = cv2.inRange(hsv, lower_btn, upper_btn)
        contours_btn, _ = cv2.findContours(btn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_btn:
            for c in contours_btn:
                area = cv2.contourArea(c)
                if 200 < area < 4500:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = bw / max(1, bh)
                    if 0.7 <= aspect <= 1.4:
                        color_info = extract_dominant_color(frame, [bx, by, bx + bw, by + bh])
                        detections.append(
                            DetectedObject(
                                label="Complete Button",
                                confidence=0.91,
                                bbox=[bx, by, bx + bw, by + bh],
                                color=color_info.get("name", "Yellow"),
                                color_hex=color_info.get("hex", "#ecc94b")
                            ).to_dict()
                        )
                        break

        # 4. Remove duplicate/overlapping bounding boxes (IoU > 0.5)
        return self._nms(detections)

    def _nms(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
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
