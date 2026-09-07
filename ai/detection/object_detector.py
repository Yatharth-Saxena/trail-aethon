import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import (
    BASE_DIR,
    MODELS_DIR,
    CONFIDENCE_THRESHOLD_OBJECT,
    OBJECT_CLASS_CONFIDENCE,
    YOLO_INFERENCE_IMGSZ,
    YOLO_NMS_IOU,
    YOLO_MAX_DETECTIONS,
    CUSTOM_DETECTOR_WEIGHTS,
    AUTODISCOVER_MODEL_WEIGHTS,
    FALLBACK_DETECTOR_WEIGHTS,
    DETECTION_VOTE_WINDOW,
    DETECTION_VOTE_MIN_HITS,
    DETECTION_MAX_MISSES,
    TRACK_IOU_MATCH,
    TRACK_VOTE_WINDOW,
    OBJECT_MOVING_SPEED,
    OBJECT_STILL_SPEED,
    OBJECT_MOTION_CONFIRM_FRAMES,
    OBJECT_MOTION_RELEASE_FRAMES,
    OBJECT_MOTION_HISTORY,
)

# ---------------------------------------------------------------------------
# Detection / tracking configuration
# ---------------------------------------------------------------------------
# Extra confidence demanded from an object whose centre sits inside a person
# box with no hand nearby. Guards against clothing/print patterns being
# reported as handheld items.
ON_BODY_CONF_PENALTY = 0.10
# Cross-class overlap suppression.
CROSS_CLASS_IOU = 0.75


# ---------------------------------------------------------------------------
# Colour estimation
# ---------------------------------------------------------------------------
_HUE_NAMES: List[Tuple[int, int, str, str]] = [
    (0, 10, "Red", "#e53e3e"),
    (10, 22, "Orange", "#ed8936"),
    (22, 34, "Yellow", "#ecc94b"),
    (34, 85, "Green", "#38a169"),
    (85, 100, "Cyan", "#319795"),
    (100, 130, "Blue", "#3182ce"),
    (130, 150, "Purple", "#805ad5"),
    (150, 168, "Pink", "#d53f8c"),
    (168, 181, "Red", "#e53e3e"),
]

_COLOR_HEX = {
    "Red": "#e53e3e",
    "Orange": "#ed8936",
    "Yellow": "#ecc94b",
    "Green": "#38a169",
    "Cyan": "#319795",
    "Blue": "#3182ce",
    "Purple": "#805ad5",
    "Pink": "#d53f8c",
    "Brown / Wood": "#9c6b4e",
    "Black": "#2d3748",
    "White": "#f7fafc",
    "Silver / Gray": "#a0aec0",
    "Unknown": "#888888",
}


def _name_for_hue(hue: float, sat: float, val: float) -> str:
    for lo, hi, name, _hex in _HUE_NAMES:
        if lo <= hue < hi:
            # Dark, muted oranges/yellows read as wood or brown, not orange.
            if name in ("Orange", "Yellow") and val < 140 and sat < 170:
                return "Brown / Wood"
            return name
    return "Unknown"


def extract_dominant_color(frame_bgr: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
    """
    Estimate the dominant surface colour of the object inside *bbox*.

    Only the central region of the box is sampled so that background pixels
    around the object's silhouette do not pull the result. The hue is taken
    from a saturation-weighted histogram peak rather than a plain median,
    which is far more stable on textured or shaded objects.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return {"name": "Unknown", "hex": _COLOR_HEX["Unknown"], "confidence": 0.0}

    # Sample the central 60% of the box.
    inset_x = int((x2 - x1) * 0.20)
    inset_y = int((y2 - y1) * 0.20)
    roi = frame_bgr[y1 + inset_y: y2 - inset_y, x1 + inset_x: x2 - inset_x]
    if roi.size == 0:
        roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"name": "Unknown", "hex": _COLOR_HEX["Unknown"], "confidence": 0.0}

    # Downsample for speed; colour statistics do not need full resolution.
    if roi.shape[0] > 48 or roi.shape[1] > 48:
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)
    v_chan = hsv[:, :, 2].astype(np.float32)

    total = h_chan.size
    chromatic = (s_chan > 65) & (v_chan > 45) & (v_chan < 252)
    chroma_ratio = float(np.count_nonzero(chromatic)) / float(total)

    if chroma_ratio < 0.20:
        # Achromatic surface: classify by brightness.
        med_v = float(np.median(v_chan))
        if med_v < 65:
            name = "Black"
        elif med_v > 190:
            name = "White"
        else:
            name = "Silver / Gray"
        return {"name": name, "hex": _COLOR_HEX[name], "confidence": round(1.0 - chroma_ratio, 2)}

    hues = h_chan[chromatic]
    sats = s_chan[chromatic]
    vals = v_chan[chromatic]

    # Saturation-weighted hue histogram (5-degree bins over the 0-180 range).
    hist, edges = np.histogram(hues, bins=36, range=(0, 180), weights=sats)
    peak = int(np.argmax(hist))
    peak_hue = float((edges[peak] + edges[peak + 1]) / 2.0)

    # Agreement = share of weight in the peak bin and its neighbours.
    window = hist[max(0, peak - 1): peak + 2].sum()
    agreement = float(window / hist.sum()) if hist.sum() > 0 else 0.0

    name = _name_for_hue(peak_hue, float(np.median(sats)), float(np.median(vals)))
    return {
        "name": name,
        "hex": _COLOR_HEX.get(name, _COLOR_HEX["Unknown"]),
        "confidence": round(min(1.0, agreement * chroma_ratio * 2.2), 2),
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _iou(b1: List[int], b2: List[int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / float(a1 + a2 - inter)


def _contains_ratio(inner: List[int], outer: List[int]) -> float:
    """Fraction of *inner*'s area that lies inside *outer*."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / float(area)


def _smooth_box(prev: List[int], new: List[int], alpha: float = 0.55) -> List[int]:
    """Exponential box smoothing to remove per-frame jitter."""
    return [int(prev[i] * (1.0 - alpha) + new[i] * alpha) for i in range(4)]


# ---------------------------------------------------------------------------
# Experiment prop aliasing
# ---------------------------------------------------------------------------
# Classes that can plausibly stand in for a handheld experiment payload.
_PROP_CLASSES = {
    "Cup", "Bottle", "Bowl", "Wine Glass", "Sports Ball", "Apple", "Orange",
    "Banana", "Remote", "Mouse", "Cell Phone", "Book", "Teddy Bear", "Vase",
}
# Classes that can plausibly stand in for the flat component tray.
_TRAY_CLASSES = {"Bowl", "Book", "Suitcase", "Handbag", "Laptop", "Keyboard"}
# Minimum colour agreement before a prop is aliased to an experiment label.
ALIAS_COLOR_CONFIDENCE = 0.35


def _experiment_alias(
    class_label: str,
    color_name: str,
    color_conf: float,
    bbox: List[int],
    frame_area: float,
) -> Optional[str]:
    """
    Map a recognised everyday object onto the experiment's prop vocabulary
    (Object A / Object B / Tray / Complete Button).

    Aliasing is deliberately conservative: it requires a plausible object
    class, a confidently measured colour, and a plausible size. Anything that
    does not clearly qualify keeps its real class name.
    """
    if color_conf < ALIAS_COLOR_CONFIDENCE:
        return None

    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    area_ratio = area / max(1.0, frame_area)
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]

    if class_label in _PROP_CLASSES and area_ratio < 0.22:
        if color_name == "Red":
            return "Object A"
        if color_name in ("Blue", "Brown / Wood"):
            return "Object B"
        if color_name in ("Yellow", "Orange") and area_ratio < 0.04:
            return "Complete Button"

    if class_label in _TRAY_CLASSES and area_ratio >= 0.05 and box_w >= box_h * 0.9:
        if color_name in ("Silver / Gray", "Purple", "White"):
            return "Tray"

    return None


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------
class _Track:
    """
    One tracked object across frames.

    Carries the detection-commit vote window, the colour/label vote history,
    and the centroid history used to estimate whether the object is moving.
    """

    __slots__ = (
        "track_id", "bbox", "confidence", "class_label", "misses",
        "last_frame", "hit_window", "color_votes", "alias_votes", "color_conf",
        "centroids", "speed", "moving", "motion_streak", "still_streak",
    )

    def __init__(self, track_id: int, bbox: List[int], confidence: float, class_label: str, frame_id: int):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.class_label = class_label
        self.misses = 0
        self.last_frame = frame_id
        # Rolling window of per-frame matched/missed outcomes (N-of-M voting).
        self.hit_window: deque = deque([1], maxlen=DETECTION_VOTE_WINDOW)
        self.color_votes: deque = deque(maxlen=TRACK_VOTE_WINDOW)
        self.alias_votes: deque = deque(maxlen=TRACK_VOTE_WINDOW)
        self.color_conf = 0.0
        # (timestamp, cx, cy) samples for velocity estimation.
        self.centroids: deque = deque(maxlen=OBJECT_MOTION_HISTORY)
        self.speed = 0.0
        self.moving = False
        self.motion_streak = 0
        self.still_streak = 0

    @property
    def confirmed(self) -> bool:
        """Committed once seen in N of the last M frames."""
        return sum(self.hit_window) >= DETECTION_VOTE_MIN_HITS

    def majority(self, votes: deque, default: Any = None) -> Any:
        if not votes:
            return default
        return Counter(votes).most_common(1)[0][0]

    def record_motion(self, frame_width: int, now: float):
        """
        Update the speed estimate and the debounced `moving` flag from the
        centroid history.

        Speed is centroid travel per second as a fraction of frame width, so
        it does not change meaning with camera resolution. Separate enter and
        exit thresholds plus streak counters mean brief detector jitter or
        camera shake cannot flip the flag.
        """
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.centroids.append((now, cx, cy))

        if len(self.centroids) < 3:
            self.speed = 0.0
            return

        t0, x0, y0 = self.centroids[0]
        t1, x1, y1 = self.centroids[-1]
        dt = t1 - t0
        if dt <= 1e-3:
            return

        travel = math.hypot(x1 - x0, y1 - y0) / max(1, frame_width)
        # Exponential blend keeps the reading steady between frames.
        self.speed = self.speed * 0.4 + (travel / dt) * 0.6

        if self.speed >= OBJECT_MOVING_SPEED:
            self.motion_streak += 1
            self.still_streak = 0
        elif self.speed <= OBJECT_STILL_SPEED:
            self.still_streak += 1
            self.motion_streak = 0
        else:
            # Between thresholds: hold the current state (hysteresis band).
            return

        if not self.moving and self.motion_streak >= OBJECT_MOTION_CONFIRM_FRAMES:
            self.moving = True
        elif self.moving and self.still_streak >= OBJECT_MOTION_RELEASE_FRAMES:
            self.moving = False


class CustomObjectDetector:
    """
    YOLOv8-based detector for the human operator and everyday objects, with
    IoU tracking so that reported detections are temporally stable rather
    than per-frame noise.
    """

    # Per-class labels and confidence floors live in backend/config.py so they
    # can be tuned without touching detector code.
    YOLO_EVERYDAY_CLASSES: Dict[int, Tuple[str, float]] = OBJECT_CLASS_CONFIDENCE

    PERSON_CLASS_ID = 0

    def __init__(self):
        self.yolo_model: Optional[YOLO] = None
        self.weights_path: Optional[str] = None
        self.using_custom_weights: bool = False
        self._init_models()

        self._tracks: Dict[str, List[_Track]] = {}   # class label -> tracks
        self._next_track_id: int = 1
        self._frame_id: int = 0

    def _candidate_weights(self) -> List[Path]:
        """
        Build the weight search order: fine-tuned weights first (so a model
        produced by scripts/train_detector.py is used automatically), then any
        other .pt in models/, then the stock base model.
        """
        candidates: List[Path] = []

        for entry in CUSTOM_DETECTOR_WEIGHTS:
            p = Path(entry)
            candidates.append(p if p.is_absolute() else MODELS_DIR / p)

        if AUTODISCOVER_MODEL_WEIGHTS and MODELS_DIR.exists():
            for found in sorted(MODELS_DIR.rglob("*.pt")):
                if found not in candidates:
                    candidates.append(found)

        candidates.append(BASE_DIR / FALLBACK_DETECTOR_WEIGHTS)
        candidates.append(Path(FALLBACK_DETECTOR_WEIGHTS))
        return candidates

    def _init_models(self):
        base_name = Path(FALLBACK_DETECTOR_WEIGHTS).name

        for p in self._candidate_weights():
            if not p.exists():
                continue
            try:
                self.yolo_model = YOLO(str(p))
                self.weights_path = str(p)
                self.using_custom_weights = p.name != base_name
                kind = "fine-tuned" if self.using_custom_weights else "stock base"
                print(f"[ObjectDetector] Loaded {kind} YOLO weights from {p}")
                break
            except Exception as e:
                print(f"[ObjectDetector Error] Failed loading {p}: {e}")

        if self.yolo_model is None:
            try:
                # Last resort: let Ultralytics download the base model.
                self.yolo_model = YOLO(FALLBACK_DETECTOR_WEIGHTS)
                self.weights_path = FALLBACK_DETECTOR_WEIGHTS
                print(f"[ObjectDetector] Initialized {FALLBACK_DETECTOR_WEIGHTS} detector")
            except Exception as e:
                print(f"[ObjectDetector Warning] Detector initialization fallback: {e}")
                return

        if not self.using_custom_weights:
            print(
                "[ObjectDetector] Using stock COCO weights. For better accuracy on "
                "payload components, collect frames with scripts/collect_dataset.py "
                "and fine-tune with scripts/train_detector.py — the resulting "
                "weights in models/ are picked up automatically on next start."
            )
        print(f"[ObjectDetector] Inference resolution: imgsz={YOLO_INFERENCE_IMGSZ}")

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    def _run_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.yolo_model is None:
            return []

        h, w = frame.shape[:2]
        floor_conf = min(
            min(c for _lbl, c in self.YOLO_EVERYDAY_CLASSES.values()),
            CONFIDENCE_THRESHOLD_OBJECT,
        )
        try:
            results = self.yolo_model.predict(
                frame,
                classes=list(self.YOLO_EVERYDAY_CLASSES.keys()),
                conf=floor_conf,
                iou=YOLO_NMS_IOU,
                max_det=YOLO_MAX_DETECTIONS,
                imgsz=YOLO_INFERENCE_IMGSZ,
                verbose=False,
            )
        except Exception as e:
            print(f"[ObjectDetector Error] Inference failed: {e}")
            return []

        raw: List[Dict[str, Any]] = []
        if not results or results[0].boxes is None:
            return raw

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            spec = self.YOLO_EVERYDAY_CLASSES.get(cls_id)
            if spec is None:
                continue
            label, min_conf = spec
            conf = float(box.conf[0])
            xy = box.xyxy[0].tolist()
            bbox = [
                max(0, int(xy[0])), max(0, int(xy[1])),
                min(w - 1, int(xy[2])), min(h - 1, int(xy[3])),
            ]
            if bbox[2] - bbox[0] < 8 or bbox[3] - bbox[1] < 8:
                continue
            raw.append({
                "cls_id": cls_id,
                "class_label": label,
                "confidence": conf,
                "min_conf": min_conf,
                "bbox": bbox,
            })
        return raw

    # -----------------------------------------------------------------------
    # Association
    # -----------------------------------------------------------------------
    def _associate(
        self,
        class_label: str,
        candidates: List[Dict[str, Any]],
        frame_width: int,
        now: float,
    ) -> List[_Track]:
        """Greedy IoU association of *candidates* to the tracks of one class."""
        tracks = self._tracks.setdefault(class_label, [])
        candidates = sorted(candidates, key=lambda d: d["confidence"], reverse=True)
        unmatched_tracks = list(tracks)
        matched: List[_Track] = []

        for cand in candidates:
            best_track, best_iou = None, 0.0
            for tr in unmatched_tracks:
                score = _iou(cand["bbox"], tr.bbox)
                if score > best_iou:
                    best_track, best_iou = tr, score

            if best_track is not None and best_iou >= TRACK_IOU_MATCH:
                unmatched_tracks.remove(best_track)
                best_track.bbox = _smooth_box(best_track.bbox, cand["bbox"])
                best_track.confidence = max(cand["confidence"], best_track.confidence * 0.9)
                best_track.hit_window.append(1)
                best_track.misses = 0
                best_track.last_frame = self._frame_id
                best_track.record_motion(frame_width, now)
                matched.append(best_track)
            else:
                tr = _Track(self._next_track_id, cand["bbox"], cand["confidence"], class_label, self._frame_id)
                self._next_track_id += 1
                tr.record_motion(frame_width, now)
                tracks.append(tr)
                matched.append(tr)

        # Age tracks that received no detection this frame. They still vote, so
        # a run of misses eventually un-commits the track.
        for tr in unmatched_tracks:
            tr.hit_window.append(0)
            tr.misses += 1
            tr.confidence *= 0.85

        self._tracks[class_label] = [
            tr for tr in tracks
            if tr.misses <= (DETECTION_MAX_MISSES if tr.confirmed else 1)
        ]
        return matched

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        hands: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect the operator and everyday objects in *frame*.

        *hands* are the tracked hands for the same frame (in this frame's
        coordinate space). They are used to decide whether an object sitting
        in front of the operator's body is genuinely handheld.
        """
        self._frame_id += 1
        now = time.time()
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        raw = self._run_yolo(frame)

        # Person boxes first: they gate the on-body confidence penalty.
        person_raw = [
            d for d in raw
            if d["cls_id"] == self.PERSON_CLASS_ID and d["confidence"] >= d["min_conf"]
        ]
        person_boxes = [d["bbox"] for d in person_raw]

        def near_hand(bbox: List[int]) -> bool:
            if not hands:
                return False
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            reach = max(40.0, 0.075 * w)
            for hnd in hands:
                hb = hnd.get("bbox")
                if not hb or len(hb) < 4:
                    continue
                if _iou(bbox, hb) > 0.02:
                    return True
                hcx = (hb[0] + hb[2]) / 2.0
                hcy = (hb[1] + hb[3]) / 2.0
                if math.hypot(cx - hcx, cy - hcy) < reach:
                    return True
            return False

        def on_body(bbox: List[int]) -> bool:
            return any(_contains_ratio(bbox, pb) > 0.85 for pb in person_boxes)

        # Group surviving detections by class, applying per-class thresholds.
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for det in raw:
            required = det["min_conf"]
            if det["cls_id"] != self.PERSON_CLASS_ID:
                if on_body(det["bbox"]) and not near_hand(det["bbox"]):
                    required += ON_BODY_CONF_PENALTY
            if det["confidence"] < required:
                continue
            by_class.setdefault(det["class_label"], []).append(det)

        # Age out classes that disappeared entirely this frame.
        for class_label in list(self._tracks.keys()):
            if class_label not in by_class:
                self._associate(class_label, [], w, now)

        detections: List[Dict[str, Any]] = []
        for class_label, candidates in by_class.items():
            for tr in self._associate(class_label, candidates, w, now):
                if not tr.confirmed:
                    continue

                if class_label == "Person":
                    detections.append({
                        "label": "Person",
                        "raw_label": "Person",
                        "class_label": "Person",
                        "display_name": "Person",
                        "color": "Suit / Clothes",
                        "color_hex": "#f0f0f0",
                        "confidence": round(min(0.99, tr.confidence), 2),
                        "bbox": list(tr.bbox),
                        "is_held": False,
                        "held": False,
                        "held_by": "",
                        "velocity": round(tr.speed, 4),
                        "is_moving": tr.moving,
                        "moving": tr.moving,
                        "track_id": tr.track_id,
                        "timestamp": now,
                    })
                    continue

                # Colour is voted over the track's recent history so the name
                # does not flip between frames.
                sample = extract_dominant_color(frame, tr.bbox)
                tr.color_votes.append(sample["name"])
                tr.color_conf = max(tr.color_conf * 0.9, sample["confidence"])
                color_name = tr.majority(tr.color_votes, "Unknown")
                color_agreement = tr.color_votes.count(color_name) / max(1, len(tr.color_votes))

                alias = _experiment_alias(
                    class_label,
                    color_name,
                    tr.color_conf * color_agreement,
                    tr.bbox,
                    frame_area,
                )
                tr.alias_votes.append(alias)
                alias = tr.majority(tr.alias_votes, None)

                label = alias or class_label
                if color_name in ("Unknown", None):
                    display_name = class_label
                else:
                    display_name = f"{color_name} {class_label}"
                if alias:
                    display_name = f"{alias} ({display_name})"

                detections.append({
                    "label": label,
                    "raw_label": label,
                    "class_label": class_label,
                    "display_name": display_name,
                    "color": color_name,
                    "color_hex": _COLOR_HEX.get(color_name, _COLOR_HEX["Unknown"]),
                    "confidence": round(min(0.99, tr.confidence), 2),
                    "bbox": list(tr.bbox),
                    "is_held": False,
                    "held": False,
                    "held_by": "",
                    # Centroid speed as a fraction of frame width per second,
                    # plus the debounced independent-movement flag.
                    "velocity": round(tr.speed, 4),
                    "is_moving": tr.moving,
                    "moving": tr.moving,
                    "track_id": tr.track_id,
                    "timestamp": now,
                })

        return self._suppress_overlaps(detections)

    def reset(self):
        """Clear all tracks (used when a new experiment run starts)."""
        self._tracks.clear()
        self._frame_id = 0

    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------
    def _suppress_overlaps(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate boxes across different classes. Per-class
        duplicates are already handled by YOLO's own NMS and by tracking, so
        this only fires when two classes claim almost exactly the same region
        (e.g. Cup and Bowl on one mug). Person boxes are never suppressed by
        an object box.
        """
        if len(detections) <= 1:
            return detections

        dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep: List[Dict[str, Any]] = []
        for det in dets:
            is_person = det["class_label"] == "Person"
            duplicate = False
            for kept in keep:
                if (kept["class_label"] == "Person") != is_person:
                    continue
                if _iou(det["bbox"], kept["bbox"]) > CROSS_CLASS_IOU:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(det)
        return keep


object_detector = CustomObjectDetector()
