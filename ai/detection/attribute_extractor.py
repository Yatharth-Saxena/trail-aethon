import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from ai.schemas import ObjectAttributes, ColorInfo, SizeEstimate, ShapeInfo, StabilityInfo
from ai.detection.object_detector import extract_dominant_color

# Adaptive crop sizing flag: automatically reduces crop size if latency spikes
_ADAPTIVE_DOWNSAMPLE = False
_LAST_DURATION_MS = 0.0


def estimate_texture(crop_bgr: np.ndarray) -> str:
    """
    Coarse texture heuristic based on local variance of Laplacian inside the bbox.
    
    NOTE: Texture classification is a variance heuristic, NOT ML-grade material detection.
    It distinguishes coarse surface character (smooth/matte vs textured/patterned vs reflective/specular).
    """
    if crop_bgr.size == 0:
        return "smooth"

    target_size = 48 if _ADAPTIVE_DOWNSAMPLE else 96
    h, w = crop_bgr.shape[:2]
    if h > target_size or w > target_size:
        crop_bgr = cv2.resize(crop_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(laplacian.var())

    # Specular highlight detection: presence of very bright saturated pixels
    specular_ratio = float(np.mean(gray > 240))

    if lap_var < 50.0:
        return "smooth"
    elif lap_var > 300.0 or (specular_ratio > 0.08 and lap_var > 120.0):
        return "reflective"
    else:
        return "textured"


def estimate_shape(frame_bgr: np.ndarray, bbox: List[int]) -> Dict[str, float]:
    """
    Computes aspect ratio and orientation (degrees) on the largest contour
    inside the bbox region via cv2.minAreaRect.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    w_px = max(1, x2 - x1)
    h_px = max(1, y2 - y1)
    aspect_ratio = round(float(w_px) / float(h_px), 3)

    if w_px < 4 or h_px < 4:
        return {"aspect_ratio": aspect_ratio, "orientation_deg": 0.0}

    # Extract crop with slight padding so contour boundaries are detectable
    pad_x = max(3, int(w_px * 0.08))
    pad_y = max(3, int(h_px * 0.08))
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(fw, x2 + pad_x), min(fh, y2 + pad_y)
    crop = frame_bgr[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        return {"aspect_ratio": aspect_ratio, "orientation_deg": 0.0}

    target_size = 48 if _ADAPTIVE_DOWNSAMPLE else 96
    if crop.shape[0] > target_size or crop.shape[1] > target_size:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        edges = cv2.Canny(blurred, 30, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5 and cv2.contourArea(largest) >= 12:
            rect = cv2.minAreaRect(largest)
            angle = float(rect[-1])
            return {"aspect_ratio": aspect_ratio, "orientation_deg": round(angle, 2)}

    return {"aspect_ratio": aspect_ratio, "orientation_deg": 0.0}


def derive_motion_state(
    is_held: bool,
    is_moving: bool,
    track_status: str,
    prev_motion_state: Optional[str] = None,
) -> str:
    """
    Derives motion state ("STATIC", "MOVING", "HELD", "PREDICTED")
    reusing existing action recognizer and interaction signals without secondary tracking.
    """
    if is_held:
        return "HELD"
    if is_moving:
        return "MOVING"
    if prev_motion_state in ("STATIC", "MOVING", "HELD") and track_status == "PREDICTED":
        return prev_motion_state
    if track_status == "PREDICTED":
        return "PREDICTED"
    return "STATIC"


def extract_attributes(
    frame_bgr: np.ndarray,
    bbox: List[int],
    track_age_frames: int = 1,
    confidence_ema: float = 0.9,
    confirmed: bool = True,
    is_held: bool = False,
    is_moving: bool = False,
    track_status: str = "TRACKED",
    interactions: Optional[List[Dict[str, Any]]] = None,
    object_label: str = "",
    prev_motion_state: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract standardized attributes for a single confirmed object detection.
    Always returns the complete, non-optional ObjectAttributes schema.
    """
    global _ADAPTIVE_DOWNSAMPLE, _LAST_DURATION_MS
    t0 = time.perf_counter()

    fh, fw = frame_bgr.shape[:2]
    frame_area = max(1.0, float(fw * fh))

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    w_px = max(0, x2 - x1)
    h_px = max(0, y2 - y1)
    area_frac = round(float(w_px * h_px) / frame_area, 5)

    # 1. Color extraction (reused from object_detector)
    color_raw = extract_dominant_color(frame_bgr, bbox)
    color_dict = {
        "name": str(color_raw.get("name", "Unknown")),
        "hex": str(color_raw.get("hex", "#888888")),
        "confidence": float(color_raw.get("confidence", 0.0)),
    }

    # 2. Size estimate
    size_dict = {
        "width_px": w_px,
        "height_px": h_px,
        "area_frac": area_frac,
    }

    # 3. Shape analysis
    shape_dict = estimate_shape(frame_bgr, [x1, y1, x2, y2])

    # 4. Texture analysis
    crop = frame_bgr[y1:y2, x1:x2]
    texture_val = estimate_texture(crop)

    # 5. Motion state determination
    # Wire interaction signals if provided
    held_state = is_held
    if interactions and object_label:
        for inter in interactions:
            if inter.get("object") == object_label and inter.get("state") in ("GRASPED", "CONTACT", "PRESSING"):
                held_state = True
                break

    motion_val = derive_motion_state(
        is_held=held_state,
        is_moving=is_moving,
        track_status=track_status,
        prev_motion_state=prev_motion_state,
    )

    # 6. Stability info
    stability_dict = {
        "track_age_frames": int(track_age_frames),
        "confidence_ema": round(float(confidence_ema), 3),
        "confirmed": bool(confirmed),
    }

    raw_output = {
        "color": color_dict,
        "size_estimate": size_dict,
        "shape": shape_dict,
        "texture": texture_val,
        "motion_state": motion_val,
        "stability": stability_dict,
    }

    # Validate against strict Pydantic model
    validated = ObjectAttributes(**raw_output)
    result = validated.model_dump()

    # Latency tracking & adaptive downsampling guard (<4ms per object)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _LAST_DURATION_MS = elapsed_ms
    if elapsed_ms > 4.0:
        _ADAPTIVE_DOWNSAMPLE = True
    elif elapsed_ms < 1.5:
        _ADAPTIVE_DOWNSAMPLE = False

    return result
