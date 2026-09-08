import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter

import cv2
import numpy as np
import torch
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
    DETECTION_MAX_MISSES_CONFIRMED,
    DETECTION_MAX_MISSES_CANDIDATE,
    PERSON_MIN_ASPECT_RATIO,
    PERSON_MAX_ASPECT_RATIO,
    PERSON_MIN_AREA_FRAC,
    PERSON_MAX_AREA_FRAC,
    PERSON_CEILING_ZONE_FRAC,
    TRACK_IOU_MATCH,
    TRACK_VOTE_WINDOW,
    OBJECT_MOVING_SPEED,
    OBJECT_STILL_SPEED,
    OBJECT_MOTION_CONFIRM_FRAMES,
    OBJECT_MOTION_RELEASE_FRAMES,
    OBJECT_MOTION_HISTORY,
    OBJECT_DOMAIN_TAXONOMY,
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
# Multi-Factor Person/Astronaut Validation Helper
# ---------------------------------------------------------------------------
def _fuse_person_bbox_with_pose(bbox: List[int], pose: Dict[str, Any], frame_shape: Tuple[int, int]) -> List[int]:
    """
    Envelop both the person detection box and visible pose landmarks with slight
    anatomical padding so that the bounding box tightly, accurately and stably
    frames the human operator.
    """
    h, w = frame_shape
    lms = pose.get("landmarks", [])
    valid_pts = []
    for lm in lms:
        v = lm.get("visibility", 1.0)
        if v is not None and v >= 0.28:
            valid_pts.append((lm["x"] * w, lm["y"] * h))
    if not valid_pts:
        return bbox

    pxs = [pt[0] for pt in valid_pts]
    pys = [pt[1] for pt in valid_pts]
    pose_min_x = min(pxs)
    pose_max_x = max(pxs)
    pose_min_y = min(pys)
    pose_max_y = max(pys)

    bx1, by1, bx2, by2 = bbox
    pad_x = max(8, int((bx2 - bx1) * 0.04))
    pad_y = max(10, int((by2 - by1) * 0.04))

    fx1 = max(0, int(min(bx1, pose_min_x - pad_x)))
    fy1 = max(0, int(min(by1, pose_min_y - pad_y)))
    fx2 = min(w - 1, int(max(bx2, pose_max_x + pad_x)))
    fy2 = min(h - 1, int(max(by2, pose_max_y + pad_y)))

    return [fx1, fy1, fx2, fy2]


def _validate_person_candidate(
    bbox: List[int],
    confidence: float,
    frame_shape: Tuple[int, int],
    hands: Optional[List[Dict[str, Any]]] = None,
    pose: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Strict multi-factor human gate.
    Requires:
    1. Valid human aspect ratio and minimum area.
    2. Not in the ceiling / upper zone.
    3. Not a hand bounding box misclassified as a full person.
    4. Explicit human skeleton / pose corroboration: candidate box MUST contain
       at least 4 visible pose landmarks (including key upper torso anchors:
       nose, shoulders, or hips) or have strong IoU/containment with verified pose bbox.
    If no valid pose exists in frame, candidates are rejected.
    """
    h, w = frame_shape
    frame_area = float(h * w)
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    area = box_w * box_h
    aspect_ratio = box_h / float(box_w)
    area_frac = area / frame_area

    # 1. Aspect ratio check (height / width should be human-like)
    if aspect_ratio < PERSON_MIN_ASPECT_RATIO or aspect_ratio > PERSON_MAX_ASPECT_RATIO:
        return False

    # 2. Area fraction check (too small or nearly full frame)
    if area_frac < PERSON_MIN_AREA_FRAC or area_frac > PERSON_MAX_AREA_FRAC:
        return False

    # 3. Ceiling zone check: reject ceiling lights / upper background false positives
    if y2 < 0.38 * h:
        return False
    cy = (y1 + y2) / 2.0
    if cy < PERSON_CEILING_ZONE_FRAC * h:
        return False

    # 4. Hand-as-person check (hand detection wrongly classified as a full person)
    if hands:
        for hnd in hands:
            hb = hnd.get("bbox")
            if hb and len(hb) >= 4:
                if _contains_ratio(bbox, hb) > 0.60 or _iou(bbox, hb) > 0.50:
                    return False

    # 5. Pose corroboration: MUST have valid human pose/skeleton corroboration
    if not pose:
        return False

    visible = pose.get("visible_landmarks_count", 0)
    if visible < 6:
        return False

    pose_bbox = pose.get("bbox")
    if not pose_bbox or len(pose_bbox) < 4:
        return False

    # Check visible landmarks inside this candidate bbox
    lms = pose.get("landmarks", [])
    contained_count = 0
    key_torso_count = 0
    for idx, lm in enumerate(lms):
        v = lm.get("visibility", 1.0)
        if v is not None and v > 0.28:
            lx = lm["x"] * w
            ly = lm["y"] * h
            if (x1 - 12) <= lx <= (x2 + 12) and (y1 - 12) <= ly <= (y2 + 12):
                contained_count += 1
                if idx in (0, 11, 12, 13, 14, 23, 24): # Nose, shoulders, elbows, hips
                    key_torso_count += 1

    iou_with_pose = _iou(bbox, pose_bbox)
    contains_pose = _contains_ratio(bbox, pose_bbox)
    contained_by_pose = _contains_ratio(pose_bbox, bbox)

    has_landmark_evidence = (contained_count >= 4 and key_torso_count >= 1)
    has_bbox_overlap = (iou_with_pose >= 0.15 or contains_pose >= 0.25 or contained_by_pose >= 0.25)

    if not (has_landmark_evidence or has_bbox_overlap):
        return False

    return True


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------
class _Track:
    """
    One tracked object across frames.

    Carries the detection-commit vote window, the colour/label vote history,
    centroid history for velocity estimation, and velocity extrapolation for occlusions.
    """

    __slots__ = (
        "track_id", "bbox", "confidence", "class_label", "misses",
        "last_frame", "hit_window", "color_votes", "alias_votes", "color_conf",
        "centroids", "speed", "moving", "motion_streak", "still_streak",
        "vx", "vy",
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
        self.vx = 0.0
        self.vy = 0.0

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

        self.vx = (x1 - x0) / max(1, len(self.centroids) - 1)
        self.vy = (y1 - y0) / max(1, len(self.centroids) - 1)

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

    YOLO_EVERYDAY_CLASSES: Dict[int, Tuple[str, float]] = OBJECT_CLASS_CONFIDENCE
    PERSON_CLASS_ID = 0

    def __init__(self):
        self.yolo_model: Optional[YOLO] = None
        self.weights_path: Optional[str] = None
        self.using_custom_weights: bool = False
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()

        self._tracks: Dict[str, List[_Track]] = {}   # class label -> tracks
        self._astronaut_tracks: Dict[int, _Track] = {}  # astronaut track_id -> track
        self._next_track_id: int = 2
        self._frame_id: int = 0

    def _candidate_weights(self) -> List[Path]:
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
                self.yolo_model = YOLO(FALLBACK_DETECTOR_WEIGHTS)
                self.weights_path = FALLBACK_DETECTOR_WEIGHTS
                print(f"[ObjectDetector] Initialized {FALLBACK_DETECTOR_WEIGHTS} detector")
            except Exception as e:
                print(f"[ObjectDetector Warning] Detector initialization fallback: {e}")
                return

        print(f"[ObjectDetector] Inference resolution: imgsz={YOLO_INFERENCE_IMGSZ}")

    def _run_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.yolo_model is None:
            return []

        h, w = frame.shape[:2]
        floor_conf = min(
            min(c for _lbl, c in self.YOLO_EVERYDAY_CLASSES.values()),
            CONFIDENCE_THRESHOLD_OBJECT,
        )
        try:
            # Predict all classes unconstrained so any object in frame is detected
            results = self.yolo_model.predict(
                frame,
                classes=None,
                conf=floor_conf,
                iou=YOLO_NMS_IOU,
                max_det=YOLO_MAX_DETECTIONS,
                imgsz=YOLO_INFERENCE_IMGSZ,
                device=self.device,
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
            if spec is not None:
                label, min_conf = spec
            else:
                raw_name = self.yolo_model.names.get(cls_id, f"object_{cls_id}")
                label = raw_name.replace("_", " ").title()
                min_conf = CONFIDENCE_THRESHOLD_OBJECT

            conf = float(box.conf[0])
            if conf < min_conf:
                continue

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

    def _associate(
        self,
        class_label: str,
        candidates: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
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

        # Age tracks that received no detection this frame. Extrapolate bbox for occlusions.
        for tr in unmatched_tracks:
            tr.hit_window.append(0)
            tr.misses += 1
            tr.confidence *= 0.85
            if tr.confirmed and (abs(tr.vx) > 0.5 or abs(tr.vy) > 0.5):
                dx, dy = int(tr.vx), int(tr.vy)
                tr.bbox = [
                    max(0, tr.bbox[0] + dx),
                    max(0, tr.bbox[1] + dy),
                    min(frame_width - 1, tr.bbox[2] + dx),
                    min(frame_height - 1, tr.bbox[3] + dy),
                ]

        self._tracks[class_label] = [
            tr for tr in tracks
            if tr.misses <= (DETECTION_MAX_MISSES_CONFIRMED if tr.confirmed else DETECTION_MAX_MISSES_CANDIDATE)
        ]
        return matched

    def _track_astronauts(
        self,
        candidates: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> List[_Track]:
        """
        Maintains stable identities for each detected astronaut (Person #1, Person #2, etc.).
        Associates candidates to existing astronaut tracks via IoU and centroid proximity.
        """
        unmatched_tracks = list(self._astronaut_tracks.values())
        matched: List[_Track] = []
        candidates_sorted = sorted(candidates, key=lambda c: c.get("confidence", 0.0), reverse=True)

        for cand in candidates_sorted:
            cand_bbox = cand["bbox"]
            cand_conf = cand.get("confidence", 0.95)
            best_track, best_score = None, 0.0

            for tr in unmatched_tracks:
                score = _iou(cand_bbox, tr.bbox)
                c_dist = math.hypot(
                    (cand_bbox[0] + cand_bbox[2]) / 2.0 - (tr.bbox[0] + tr.bbox[2]) / 2.0,
                    (cand_bbox[1] + cand_bbox[3]) / 2.0 - (tr.bbox[1] + tr.bbox[3]) / 2.0,
                )
                if score > best_score:
                    best_track, best_score = tr, score
                elif best_score < 0.15 and c_dist < max(frame_width, frame_height) * 0.18:
                    best_track, best_score = tr, 0.20

            if best_track is not None and best_score >= 0.15:
                unmatched_tracks.remove(best_track)
                best_track.bbox = _smooth_box(best_track.bbox, cand_bbox)
                best_track.confidence = max(cand_conf, best_track.confidence * 0.92)
                best_track.hit_window.append(1)
                best_track.misses = 0
                best_track.last_frame = self._frame_id
                best_track.record_motion(frame_width, now)
                best_track.pose = cand.get("pose")
                best_track.posture = cand.get("posture", "Seated")
                matched.append(best_track)
            else:
                existing_ids = set(self._astronaut_tracks.keys())
                new_id = 1
                while new_id in existing_ids:
                    new_id += 1
                tr = _Track(new_id, cand_bbox, cand_conf, "Astronaut", self._frame_id)
                tr.hit_window = deque([1, 1], maxlen=DETECTION_VOTE_WINDOW)
                tr.record_motion(frame_width, now)
                tr.pose = cand.get("pose")
                tr.posture = cand.get("posture", "Seated")
                self._astronaut_tracks[new_id] = tr
                matched.append(tr)

        # Age tracks that received no detection this frame
        dead_ids = []
        for tr in unmatched_tracks:
            tr.hit_window.append(0)
            tr.misses += 1
            tr.confidence *= 0.88
            if tr.misses > 2:
                dead_ids.append(tr.track_id)

        for tid in dead_ids:
            self._astronaut_tracks.pop(tid, None)

        return matched

    def detect(
        self,
        frame: np.ndarray,
        hands: Optional[List[Dict[str, Any]]] = None,
        pose: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect all human operators and all everyday objects in *frame* with multi-factor validation.
        """
        self._frame_id += 1
        now = time.time()
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        raw = self._run_yolo(frame)

        # Extract all verified human poses
        poses: List[Dict[str, Any]] = []
        if pose:
            if "all_poses" in pose and isinstance(pose["all_poses"], list):
                poses = pose["all_poses"]
            else:
                poses = [pose]
        valid_poses = [p for p in poses if p and p.get("visible_landmarks_count", 0) >= 5]

        # 1. Multi-factor Person/Astronaut Validation
        person_raw = []
        for d in raw:
            if d["cls_id"] == self.PERSON_CLASS_ID and d["confidence"] >= max(0.45, d["min_conf"]):
                if _validate_person_candidate(d["bbox"], d["confidence"], (h, w), hands=hands, pose=pose):
                    person_raw.append(d)

        # Corroborate valid poses with YOLO person detections
        corroborated_persons: List[Dict[str, Any]] = []
        unmatched_yolo = list(person_raw)

        for p_item in valid_poses:
            pb = p_item.get("bbox")
            if not pb or len(pb) < 4:
                continue

            best_yolo, best_score = None, 0.0
            for yd in unmatched_yolo:
                iou = _iou(yd["bbox"], pb)
                contains = _contains_ratio(yd["bbox"], pb)
                contained = _contains_ratio(pb, yd["bbox"])
                score = max(iou, contains * 0.7, contained * 0.7)
                if score > best_score:
                    best_yolo, best_score = yd, score

            if best_yolo is not None and best_score >= 0.12:
                unmatched_yolo.remove(best_yolo)
                fused_box = _fuse_person_bbox_with_pose(best_yolo["bbox"], p_item, (h, w))
                corroborated_persons.append({
                    "cls_id": self.PERSON_CLASS_ID,
                    "class_label": "Astronaut",
                    "confidence": max(best_yolo["confidence"], 0.95),
                    "bbox": fused_box,
                    "min_conf": 0.45,
                    "pose": p_item,
                    "posture": p_item.get("posture", "Seated"),
                })
            else:
                # MediaPipe clearly tracks human pose
                lms = p_item.get("landmarks", [])
                has_upper = any(
                    (lms[i].get("visibility", 1.0) or 1.0) >= 0.28
                    for i in (0, 11, 12) if i < len(lms)
                )
                if has_upper:
                    fused_box = _fuse_person_bbox_with_pose(list(pb), p_item, (h, w))
                    corroborated_persons.append({
                        "cls_id": self.PERSON_CLASS_ID,
                        "class_label": "Astronaut",
                        "confidence": 0.95,
                        "bbox": fused_box,
                        "min_conf": 0.45,
                        "pose": p_item,
                        "posture": p_item.get("posture", "Seated"),
                    })

        # If no human pose is present, purge all astronaut tracks immediately
        if not valid_poses:
            self._astronaut_tracks.clear()

        # Track all verified astronauts
        astro_tracks = self._track_astronauts(corroborated_persons, w, h, now)
        person_boxes = [list(tr.bbox) for tr in astro_tracks]

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

        # Group surviving detections by class (non-person everyday objects)
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for det in raw:
            if det["cls_id"] == self.PERSON_CLASS_ID:
                continue
            required = det["min_conf"]
            if on_body(det["bbox"]) and not near_hand(det["bbox"]):
                required += ON_BODY_CONF_PENALTY
            if det["confidence"] >= required:
                by_class.setdefault(det["class_label"], []).append(det)

        # Age out classes that disappeared entirely this frame
        for class_label in list(self._tracks.keys()):
            if class_label not in by_class:
                self._associate(class_label, [], w, h, now)

        detections: List[Dict[str, Any]] = []

        # Emit all tracked astronauts
        for tr in astro_tracks:
            if not tr.confirmed:
                continue
            track_status = "TRACKED" if tr.misses == 0 else "OCCLUDED"
            cx, cy = int((tr.bbox[0] + tr.bbox[2]) / 2.0), int((tr.bbox[1] + tr.bbox[3]) / 2.0)
            posture_str = getattr(tr, "posture", "Seated")
            matched_pose = getattr(tr, "pose", None)
            detections.append({
                "label": "Astronaut",
                "raw_label": "Astronaut",
                "class_label": "Astronaut",
                "category": "ASTRONAUT",
                "category_badge": "CREW",
                "display_name": f"✦ ASTRONAUT #{tr.track_id}",
                "role": f"Mission Operator #{tr.track_id} / EVA Specialist",
                "color": "EVA Spacesuit",
                "color_hex": "#00e6c8",
                "confidence": round(min(0.99, tr.confidence), 2),
                "bbox": list(tr.bbox),
                "is_held": False,
                "held": False,
                "held_by": "",
                "velocity": round(tr.speed, 4),
                "is_moving": tr.moving,
                "moving": tr.moving,
                "track_id": tr.track_id,
                "track_status": track_status,
                "position": {"cx": cx, "cy": cy},
                "posture": posture_str,
                "pose": matched_pose,
                "timestamp": now,
            })

        for class_label, candidates in by_class.items():
            for tr in self._associate(class_label, candidates, w, h, now):
                if not tr.confirmed:
                    continue

                track_status = "TRACKED" if tr.misses == 0 else "OCCLUDED"
                cx, cy = int((tr.bbox[0] + tr.bbox[2]) / 2.0), int((tr.bbox[1] + tr.bbox[3]) / 2.0)

                # Colour is voted over the track's recent history
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

                # Domain Taxonomy resolution
                tax = OBJECT_DOMAIN_TAXONOMY.get(class_label.lower(), {})
                category = tax.get("category", "PAYLOAD")
                category_badge = tax.get("category_badge", "ITEM")
                domain_name = tax.get("domain_name")

                if alias:
                    label = alias
                    category = "PAYLOAD"
                    category_badge = "CONTROL" if "Button" in alias else ("TRAY" if "Tray" in alias else "PAYLOAD")
                    if color_name not in ("Unknown", None):
                        display_name = f"{alias} ({color_name} {class_label})"
                    else:
                        display_name = alias
                    color_hex = _COLOR_HEX.get(color_name, tax.get("color_hex", "#a855f7"))
                elif domain_name:
                    label = domain_name
                    if color_name not in ("Unknown", None):
                        display_name = f"{domain_name} ({color_name} {class_label})"
                    else:
                        display_name = f"{domain_name} ({class_label})"
                    color_hex = tax.get("color_hex") or _COLOR_HEX.get(color_name, "#38bdf8")
                else:
                    label = class_label
                    if color_name not in ("Unknown", None):
                        display_name = f"{color_name} {class_label}"
                    else:
                        display_name = class_label
                    color_hex = _COLOR_HEX.get(color_name, "#38bdf8")

                detections.append({
                    "label": label,
                    "raw_label": class_label,
                    "class_label": class_label,
                    "category": category,
                    "category_badge": category_badge,
                    "display_name": display_name,
                    "color": color_name,
                    "color_hex": color_hex,
                    "confidence": round(min(0.99, tr.confidence), 2),
                    "bbox": list(tr.bbox),
                    "is_held": False,
                    "held": False,
                    "held_by": "",
                    "velocity": round(tr.speed, 4),
                    "is_moving": tr.moving,
                    "moving": tr.moving,
                    "track_id": tr.track_id,
                    "track_status": track_status,
                    "position": {"cx": cx, "cy": cy},
                    "timestamp": now,
                })

        return self._suppress_overlaps(detections)

    def reset(self):
        """Clear all tracks (used when a new experiment run starts)."""
        self._tracks.clear()
        self._astronaut_tracks.clear()
        self._frame_id = 0

    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------
    def _suppress_overlaps(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate boxes across different classes. Per-class
        duplicates are already handled by YOLO's own NMS and by tracking, so
        this only fires when two classes claim almost exactly the same region
        (e.g. Cup and Bowl on one mug). Astronaut boxes are never suppressed by
        an object box.
        """
        if len(detections) <= 1:
            return detections

        dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep: List[Dict[str, Any]] = []
        for det in dets:
            is_astro = det.get("category") == "ASTRONAUT" or det.get("class_label") in ("Person", "Astronaut")
            duplicate = False
            for kept in keep:
                kept_is_astro = kept.get("category") == "ASTRONAUT" or kept.get("class_label") in ("Person", "Astronaut")
                if kept_is_astro != is_astro:
                    continue
                if _iou(det["bbox"], kept["bbox"]) > CROSS_CLASS_IOU:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(det)
        return keep


object_detector = CustomObjectDetector()
