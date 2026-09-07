# CHANGED: Split into two independent inference threads so MediaPipe hand
# tracking runs at ~25 Hz while YOLO object detection runs at ~10 Hz.
# The hand skeleton overlay is now updated at 25 Hz instead of 15 Hz,
# eliminating the visual lag between the real hand position and the drawn
# skeleton. Object bboxes update at 10 Hz and are carried forward (same as
# before). camera_service.update_perception_state() is called from the hand
# thread so every MJPEG frame gets the freshest possible landmark data.

import time
import threading
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from ai.detection.object_detector import object_detector
from ai.tracking.hand_tracker import hand_tracker
from ai.interaction.hand_object_interaction import hand_object_interaction
from ai.actions.action_recognizer import action_recognizer
from camera.camera_service import camera_service
from experiment.manager import experiment_manager
from event_logging.event_logger import event_logger
from backend.config import (
    AI_HAND_TRACKING_FPS,
    AI_INFERENCE_FPS,
    AI_MIN_IDLE_SECONDS,
    OBJECT_DISPLACED_DEBOUNCE_SECONDS,
)

# ---------------------------------------------------------------------------
# Warmup guard – skip event emission for the first N frames after experiment
# start to let the detection pipeline stabilise and avoid false triggers.
# ---------------------------------------------------------------------------
WARMUP_FRAMES = 30

# Inference resolution. Detection and hand tracking run on frames downsampled
# to this size; results are scaled back to the source frame's coordinate space.
AI_FRAME_WIDTH = 640
AI_FRAME_HEIGHT = 360


class PerceptionPipeline:
    """
    Two-thread perception pipeline.

    • ``_hand_loop``      — MediaPipe Hands only, ~25 Hz.  Pushes fresh hand
                           data to camera_service so the drawn skeleton tracks
                           the real hand at video speed.

    • ``_detection_loop`` — YOLOv8 object detection, ~10 Hz.  Runs interaction
                           analysis + action recognition after each YOLO pass
                           and may emit experiment events.

    Between detection frames the last known object state is reused by the
    overlay system, so video stays smooth regardless of YOLO throughput.
    """

    def __init__(self):
        self.is_running = False
        self.hand_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None

        # Shared state — written by both threads, read by the API layer.
        self.lock = threading.Lock()
        self.latest_objects: List[Dict[str, Any]] = []
        self.latest_hands: List[Dict[str, Any]] = []
        self.latest_interactions: List[Dict[str, Any]] = []
        self.latest_action: Dict[str, Any] = action_recognizer.current_action_display
        self.hand_fps = 0.0
        self.detection_fps = 0.0

        # Latest scaled hand data shared from the hand thread to the detection
        # thread so interaction + action recognition always has current hands.
        self._hand_lock = threading.Lock()
        self._latest_scaled_hands: List[Dict[str, Any]] = []
        self._latest_small_hands: List[Dict[str, Any]] = []

        # Warmup tracking
        self._frames_since_start: int = 0
        self._experiment_was_running: bool = False
        self._last_displacement_log: float = 0.0

        self.start()

    # ------------------------------------------------------------------
    # Public properties for backward-compat with API consumers
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        """Reported as YOLO detection FPS (the slower, heavier stage)."""
        return self.detection_fps

    def start(self):
        if self.is_running:
            return
        self.is_running = True

        self.hand_thread = threading.Thread(target=self._hand_loop, daemon=True,
                                            name="aethon-hands")
        self.hand_thread.start()

        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True,
                                                 name="aethon-yolo")
        self.detection_thread.start()

    @staticmethod
    def _scale_bbox(bbox: List[int], scale_x: float, scale_y: float) -> List[int]:
        return [
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y),
        ]

    # ------------------------------------------------------------------
    # Thread 1 — MediaPipe Hands at AI_HAND_TRACKING_FPS (~25 Hz)
    # ------------------------------------------------------------------
    def _hand_loop(self):
        """
        Run MediaPipe Hands on every frame at up to AI_HAND_TRACKING_FPS.

        Results are published to camera_service immediately so the MJPEG
        overlay encoder always has sub-40ms-old landmark data, not 100ms-old.
        """
        interval = 1.0 / AI_HAND_TRACKING_FPS
        frame_count = 0
        timer = time.time()

        while self.is_running:
            t0 = time.time()
            frame = camera_service.get_latest_frame(annotated=False)

            if frame is not None:
                try:
                    h_orig, w_orig = frame.shape[:2]
                    ai_frame = cv2.resize(
                        frame,
                        (AI_FRAME_WIDTH, AI_FRAME_HEIGHT),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    scale_x = w_orig / AI_FRAME_WIDTH
                    scale_y = h_orig / AI_FRAME_HEIGHT

                    hands_small = hand_tracker.process(ai_frame)
                    hands = [
                        {**hnd, "bbox": self._scale_bbox(hnd.get("bbox", [0, 0, 0, 0]), scale_x, scale_y)}
                        for hnd in hands_small
                    ]

                    # Share with the detection thread (no heavy lock needed —
                    # the worst case is one stale frame read).
                    with self._hand_lock:
                        self._latest_scaled_hands = hands
                        self._latest_small_hands = hands_small

                    # Push updated hands into the live overlay immediately so
                    # the skeleton tracks the hand at hand-loop rate, not YOLO rate.
                    with self.lock:
                        self.latest_hands = hands
                        current_objects = self.latest_objects
                        current_action = self.latest_action

                    camera_service.update_perception_state(
                        current_objects,
                        hands,
                        current_action,
                    )

                except Exception as e:
                    print(f"[PerceptionPipeline/Hands Error] {e}")

            # FPS counter
            frame_count += 1
            now = time.time()
            if now - timer >= 1.0:
                self.hand_fps = round(frame_count / (now - timer), 1)
                frame_count = 0
                timer = now

            elapsed = time.time() - t0
            sleep_time = max(AI_MIN_IDLE_SECONDS, interval - elapsed)
            time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Thread 2 — YOLOv8 + interaction + action at AI_INFERENCE_FPS (~10 Hz)
    # ------------------------------------------------------------------
    def _detection_loop(self):
        """
        Run YOLOv8 object detection, hand-object interaction, and action
        recognition at AI_INFERENCE_FPS.  Uses the freshest hand data from
        the hand thread so interaction scoring is always current.
        """
        interval = 1.0 / AI_INFERENCE_FPS
        frame_count = 0
        timer = time.time()

        while self.is_running:
            t0 = time.time()
            frame = camera_service.get_latest_frame(annotated=False)

            if frame is not None:
                try:
                    h_orig, w_orig = frame.shape[:2]
                    ai_frame = cv2.resize(
                        frame,
                        (AI_FRAME_WIDTH, AI_FRAME_HEIGHT),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    scale_x = w_orig / AI_FRAME_WIDTH
                    scale_y = h_orig / AI_FRAME_HEIGHT

                    # Grab the freshest hand data from the hand thread.
                    with self._hand_lock:
                        hands_small = list(self._latest_small_hands)
                        hands = list(self._latest_scaled_hands)

                    # 1. Object detection — uses small hands for on-body gating.
                    objects_small = object_detector.detect(ai_frame, hands=hands_small)
                    objects = [
                        {**obj, "bbox": self._scale_bbox(obj.get("bbox", [0, 0, 0, 0]), scale_x, scale_y)}
                        for obj in objects_small
                    ]

                    # 2. Hand-object interaction
                    interactions = hand_object_interaction.analyze(hands, objects, (h_orig, w_orig))

                    # 3. Temporal action recognition
                    event = action_recognizer.update(
                        objects,
                        hands,
                        interactions,
                        frame_shape=(h_orig, w_orig),
                    )

                    with self.lock:
                        self.latest_objects = objects
                        self.latest_interactions = interactions
                        self.latest_action = action_recognizer.current_action_display
                        current_hands = self.latest_hands  # already set by hand thread

                    # Push full state (objects + fresh hands + action).
                    camera_service.update_perception_state(
                        objects,
                        current_hands,
                        action_recognizer.current_action_display,
                    )

                    self._log_object_displacement(action_recognizer.current_action_display)

                    # 4. Feed to experiment manager if a discrete event occurred.
                    exp_running = experiment_manager.experiment_manager.status.value == "RUNNING"
                    if exp_running and not self._experiment_was_running:
                        self._frames_since_start = 0
                    self._experiment_was_running = exp_running

                    if exp_running:
                        self._frames_since_start += 1

                    if event is not None:
                        if not exp_running or self._frames_since_start > WARMUP_FRAMES:
                            event_logger.log(
                                "ACTION_DETECTED",
                                f"{event.event} on {event.object}" + (f" -> {event.target}" if event.target else ""),
                                event.model_dump()
                            )
                            experiment_manager.experiment_manager.process_event(event)
                        else:
                            event_logger.log(
                                "ACTION_SUPPRESSED",
                                f"Warmup frame {self._frames_since_start}/{WARMUP_FRAMES}: "
                                f"{event.event} on {event.object} suppressed",
                                {"warmup_frame": self._frames_since_start}
                            )

                except Exception as e:
                    print(f"[PerceptionPipeline/YOLO Error] {e}")

            # FPS counter
            frame_count += 1
            now = time.time()
            if now - timer >= 1.0:
                self.detection_fps = round(frame_count / (now - timer), 1)
                frame_count = 0
                timer = now

            elapsed = time.time() - t0
            sleep_time = max(AI_MIN_IDLE_SECONDS, interval - elapsed)
            time.sleep(sleep_time)

    def _log_object_displacement(self, action: Dict[str, Any]):
        """
        Record unattended object movement in the audit trail.

        This is intentionally *not* routed through the experiment validator:
        the state machine only understands PICK_UP / PLACE / PRESS, so feeding
        it a displacement would trigger spurious out-of-order voice alerts.
        """
        if not action or action.get("action") != "OBJECT_DISPLACED":
            return

        now = time.time()
        if now - self._last_displacement_log < OBJECT_DISPLACED_DEBOUNCE_SECONDS:
            return
        self._last_displacement_log = now

        event_logger.log(
            "OBJECT_DISPLACED",
            action.get("narration", "Unattended object movement detected"),
            {
                "object": action.get("object"),
                "velocity": action.get("object_velocity"),
            },
            level="WARN",
        )

    def person_detected(self) -> bool:
        with self.lock:
            return any(obj.get("raw_label") == "Person" for obj in self.latest_objects)

    def get_perception_state(self) -> Dict[str, Any]:
        with self.lock:
            act = self.latest_action or {}
            has_person = any(obj.get("raw_label") == "Person" for obj in self.latest_objects)
            return {
                # Report the hand tracking FPS since it's the higher-rate stage
                # that directly determines overlay smoothness.
                "fps": self.hand_fps,
                "detection_fps": self.detection_fps,
                "objects": self.latest_objects,
                "hands_count": len(self.latest_hands),
                "person_detected": has_person,
                # Body pose tracking is intentionally not part of this
                # pipeline; the key is kept so existing consumers stay valid.
                "pose": None,
                "current_action": act,
                "movement": act.get("movement", "Active" if has_person else "Stationary"),
                "posture": act.get("posture", "Seated" if has_person else "Standby"),
                "narration": act.get("narration", "Monitoring"),
                "interactions": self.latest_interactions
            }

    def stop(self):
        self.is_running = False


perception_pipeline = PerceptionPipeline()
