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
    AI_INFERENCE_FPS,
    AI_MIN_IDLE_SECONDS,
    OBJECT_DISPLACED_DEBOUNCE_SECONDS,
)

# ---------------------------------------------------------------------------
# Warmup guard – skip event emission for the first N frames after experiment
# start to let the detection pipeline stabilise and avoid false triggers.
# ---------------------------------------------------------------------------
WARMUP_FRAMES = 30

# Inference resolution. Detection and hand tracking run here, then results are
# scaled back to the source frame's coordinate space.
AI_FRAME_WIDTH = 640
AI_FRAME_HEIGHT = 360


class PerceptionPipeline:
    """
    Perception stages, in order: hand tracking, object/person detection,
    hand-object interaction, then temporal action recognition.

    There is no body-pose stage — the pipeline tracks hands only.
    """

    def __init__(self):
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.latest_objects: List[Dict[str, Any]] = []
        self.latest_hands: List[Dict[str, Any]] = []
        self.latest_interactions: List[Dict[str, Any]] = []
        self.latest_action: Dict[str, Any] = action_recognizer.current_action_display
        self.fps = 0.0

        # Warmup tracking – counts frames since experiment was last started
        self._frames_since_start: int = 0
        self._experiment_was_running: bool = False
        self._last_displacement_log: float = 0.0

        self.start()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self.worker_thread.start()

    @staticmethod
    def _scale_bbox(bbox: List[int], scale_x: float, scale_y: float) -> List[int]:
        return [
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y),
        ]

    def _pipeline_loop(self):
        # Inference is paced independently of (and slower than) the video
        # stream loop; camera_service reuses the last detections for the
        # frames in between so the feed stays smooth.
        frame_interval = 1.0 / AI_INFERENCE_FPS
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

                    # 1. Hand Tracking (runs first so the detector knows where
                    #    the hands are and can judge whether an object on the
                    #    operator's body is actually being held)
                    hands_small = hand_tracker.process(ai_frame)
                    hands = [
                        {**hnd, "bbox": self._scale_bbox(hnd.get("bbox", [0, 0, 0, 0]), scale_x, scale_y)}
                        for hnd in hands_small
                    ]

                    # 2. Person & Object Detection
                    objects_small = object_detector.detect(ai_frame, hands=hands_small)
                    objects = [
                        {**obj, "bbox": self._scale_bbox(obj.get("bbox", [0, 0, 0, 0]), scale_x, scale_y)}
                        for obj in objects_small
                    ]

                    # 3. Hand-Object Interaction
                    interactions = hand_object_interaction.analyze(hands, objects, (h_orig, w_orig))

                    # 4. Temporal Action Recognition & Movement Analysis
                    event = action_recognizer.update(
                        objects,
                        hands,
                        interactions,
                        frame_shape=(h_orig, w_orig),
                    )

                    with self.lock:
                        self.latest_objects = objects
                        self.latest_hands = hands
                        self.latest_interactions = interactions
                        self.latest_action = action_recognizer.current_action_display

                    # Update camera overlays
                    camera_service.update_perception_state(
                        objects,
                        hands,
                        action_recognizer.current_action_display
                    )

                    self._log_object_displacement(action_recognizer.current_action_display)

                    # 5. Feed to Experiment Manager if a discrete event occurred
                    #    Warmup guard: suppress events for first WARMUP_FRAMES
                    #    after experiment starts to let detection settle.
                    exp_running = experiment_manager.experiment_manager.status.value == "RUNNING"
                    if exp_running and not self._experiment_was_running:
                        # Experiment just started → reset warmup counter
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
                            # Process through validator and state machine
                            experiment_manager.experiment_manager.process_event(event)
                        else:
                            # Suppressed during warmup
                            event_logger.log(
                                "ACTION_SUPPRESSED",
                                f"Warmup frame {self._frames_since_start}/{WARMUP_FRAMES}: "
                                f"{event.event} on {event.object} suppressed",
                                {"warmup_frame": self._frames_since_start}
                            )

                except Exception as e:
                    print(f"[PerceptionPipeline Error] {e}")

            # Measure inference FPS
            frame_count += 1
            now = time.time()
            if now - timer >= 1.0:
                self.fps = round(frame_count / (now - timer), 1)
                frame_count = 0
                timer = now

            # Always yield at least AI_MIN_IDLE_SECONDS so this thread cannot
            # starve the capture and stream threads when inference is slower
            # than the target interval.
            elapsed = time.time() - t0
            sleep_time = max(AI_MIN_IDLE_SECONDS, frame_interval - elapsed)
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
                "fps": self.fps,
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
