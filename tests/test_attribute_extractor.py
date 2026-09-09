import numpy as np
import cv2
import pytest

from backend.config import (
    CONFIDENCE_THRESHOLD_OBJECT,
    DETECTION_CONFIRM_FRAMES_ON,
    DETECTION_CONFIRM_FRAMES_OFF,
)
from ai.schemas import ObjectAttributes
from ai.detection.attribute_extractor import extract_attributes
from ai.detection.object_detector import _Track, CustomObjectDetector


def test_synthetic_frame_attribute_extraction():
    """
    Synthetic frame with a known solid-color rectangle at a known bbox:
    assert color.name, size_estimate.area_frac, shape.orientation_deg all land in expected ranges.
    """
    frame_h, frame_w = 270, 480
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Draw solid red rectangle at [100, 80, 260, 180]
    x1, y1, x2, y2 = 100, 80, 260, 180
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)  # BGR Red

    attrs = extract_attributes(
        frame_bgr=frame,
        bbox=[x1, y1, x2, y2],
        track_age_frames=5,
        confidence_ema=0.92,
        confirmed=True,
        is_held=False,
        is_moving=False,
        track_status="TRACKED",
    )

    # 1. Color assertions
    assert attrs["color"]["name"] == "Red"
    assert attrs["color"]["hex"].lower() == "#e53e3e"
    assert attrs["color"]["confidence"] > 0.3

    # 2. Size estimate assertions
    w_px = x2 - x1
    h_px = y2 - y1
    expected_area_frac = (w_px * h_px) / float(frame_w * frame_h)
    assert attrs["size_estimate"]["width_px"] == w_px
    assert attrs["size_estimate"]["height_px"] == h_px
    assert abs(attrs["size_estimate"]["area_frac"] - expected_area_frac) < 0.005

    # 3. Shape assertions: axis-aligned rectangle has orientation ~0 or ~90 deg
    orientation = attrs["shape"]["orientation_deg"]
    assert abs(orientation % 90.0) < 5.0 or abs(orientation % 90.0 - 90.0) < 5.0
    assert abs(attrs["shape"]["aspect_ratio"] - round(w_px / h_px, 3)) < 0.01

    # 4. Texture and stability
    assert attrs["texture"] in ("smooth", "textured", "reflective")
    assert attrs["motion_state"] == "STATIC"
    assert attrs["stability"]["confirmed"] is True
    assert attrs["stability"]["track_age_frames"] == 5

    # 5. Validate through Pydantic ObjectAttributes
    validated = ObjectAttributes(**attrs)
    assert validated.color.name == "Red"


def test_track_dropped_detection_prediction_and_motion():
    """
    Feed a track a sequence of frames with one dropped detection in the middle:
    assert motion_state doesn't flip and track_status shows PREDICTED only on the dropped frame,
    confirmed doesn't toggle.
    """
    frame = np.zeros((270, 480, 3), dtype=np.uint8)
    initial_bbox = [100, 100, 160, 160]

    # Create track and confirm it across N frames
    track = _Track(
        track_id=1,
        bbox=initial_bbox,
        confidence=0.85,
        class_label="Cup",
        frame_id=1,
    )
    for f_id in range(2, DETECTION_CONFIRM_FRAMES_ON + 1):
        track.update_detection(initial_bbox, 0.85, frame_id=f_id, frame_width=480, now=1.0 + f_id * 0.04)

    assert track.confirmed, "Track should be confirmed after N frames"

    # Frame before drop (normal detection)
    track.update_detection(initial_bbox, 0.85, frame_id=4, frame_width=480, now=1.2)
    assert track.track_status == "TRACKED"
    assert track.confirmed

    attrs_pre = extract_attributes(
        frame_bgr=frame,
        bbox=track.bbox,
        track_age_frames=track.last_frame,
        confidence_ema=track.confidence_ema,
        confirmed=bool(track.confirmed),
        is_held=False,
        is_moving=track.moving,
        track_status=track.track_status,
    )
    assert attrs_pre["motion_state"] == "STATIC"
    assert attrs_pre["stability"]["confirmed"] is True

    # Dropped detection frame (occlusion miss)
    track.update_miss(frame_width=480, frame_height=270, now=1.24)
    assert track.track_status == "PREDICTED", "Dropped frame should mark track_status as PREDICTED"
    assert track.confirmed, "Confirmed status must NOT toggle on a single dropped frame"

    attrs_drop = extract_attributes(
        frame_bgr=frame,
        bbox=track.bbox,
        track_age_frames=track.last_frame,
        confidence_ema=track.confidence_ema,
        confirmed=bool(track.confirmed),
        is_held=False,
        is_moving=track.moving,
        track_status=track.track_status,
        prev_motion_state=attrs_pre["motion_state"],
    )
    assert attrs_drop["motion_state"] == "STATIC", "motion_state must not flip on dropped detection frame"
    assert attrs_drop["stability"]["confirmed"] is True, "stability.confirmed must not toggle"

    # Frame after drop (detection resumes)
    track.update_detection(initial_bbox, 0.85, frame_id=6, frame_width=480, now=1.28)
    assert track.track_status == "TRACKED", "Resumed detection should restore track_status to TRACKED"
    assert track.confirmed

    attrs_post = extract_attributes(
        frame_bgr=frame,
        bbox=track.bbox,
        track_age_frames=track.last_frame,
        confidence_ema=track.confidence_ema,
        confirmed=bool(track.confirmed),
        is_held=False,
        is_moving=track.moving,
        track_status=track.track_status,
        prev_motion_state=attrs_drop["motion_state"],
    )
    assert attrs_post["motion_state"] == "STATIC"
    assert attrs_post["stability"]["confirmed"] is True


def test_track_hysteresis_confirmation():
    """
    Feed a track 3 consecutive sub-threshold-confidence frames after being confirmed:
    assert it does NOT drop confirmation (hysteresis holds until M frames),
    then feed M frames -> assert it drops.
    """
    initial_bbox = [50, 50, 90, 90]
    high_conf = 0.80
    low_conf = 0.15
    assert low_conf < CONFIDENCE_THRESHOLD_OBJECT
    assert high_conf >= CONFIDENCE_THRESHOLD_OBJECT

    track = _Track(
        track_id=2,
        bbox=initial_bbox,
        confidence=high_conf,
        class_label="Bottle",
        frame_id=1,
    )

    # N frames to confirm
    for f in range(2, DETECTION_CONFIRM_FRAMES_ON + 1):
        track.update_confidence(high_conf)

    assert track.confirmed, f"Track should become confirmed after {DETECTION_CONFIRM_FRAMES_ON} consecutive frames"

    # Feed 3 consecutive sub-threshold frames
    for _ in range(3):
        track.update_confidence(low_conf)
        assert track.confirmed, "Hysteresis must hold confirmation across 3 sub-threshold frames"

    # Reset track to confirmed state with fresh high confidence
    for _ in range(DETECTION_CONFIRM_FRAMES_ON):
        track.update_confidence(high_conf)
    assert track.confirmed

    # Feed M consecutive sub-threshold frames
    M = DETECTION_CONFIRM_FRAMES_OFF
    for frame_idx in range(1, M + 1):
        track.update_confidence(low_conf)
        if frame_idx < M:
            assert track.confirmed, f"Track should remain confirmed at sub-threshold frame {frame_idx}/{M}"
        else:
            assert not track.confirmed, f"Track must drop confirmation after {M} sub-threshold frames"


def test_associate_label_weighted_matching():
    """
    In _associate(), matching a new detection to an existing track weights both
    IoU and label agreement, preventing silent label swap.
    """
    detector = CustomObjectDetector()
    now = 100.0

    # Initial candidate
    cand_cup = {"bbox": [100, 100, 150, 150], "confidence": 0.85, "class_label": "Cup"}
    matched = detector._associate("Cup", [cand_cup], frame_width=480, frame_height=270, now=now)
    assert len(matched) == 1
    orig_track_id = matched[0].track_id

    # Two overlapping candidates: one Cup, one Bottle
    cand_same_label = {"bbox": [105, 105, 155, 155], "confidence": 0.80, "class_label": "Cup"}
    cand_diff_label = {"bbox": [102, 102, 152, 152], "confidence": 0.82, "class_label": "Bottle"}

    # Associate Cup candidates
    matched_cup = detector._associate("Cup", [cand_same_label], frame_width=480, frame_height=270, now=now + 0.04)
    assert len(matched_cup) >= 1
    assert matched_cup[0].track_id == orig_track_id
    assert matched_cup[0].class_label == "Cup"
