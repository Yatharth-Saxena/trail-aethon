# CHANGED: Split into two independent inference threads so MediaPipe hand
# tracking runs at ~25 Hz while YOLO object detection runs at ~10 Hz.
# The hand skeleton overlay is now updated at 25 Hz instead of 15 Hz,
# eliminating the visual lag between the real hand position and the drawn
# skeleton. Object bboxes update at 10 Hz and are carried forward (same as
# before). camera_service.update_perception_state() is called from the hand
# thread so every MJPEG frame gets the freshest possible landmark data.

import time
import math
import threading
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from ai.detection.object_detector import object_detector
from ai.tracking.hand_tracker import hand_tracker
from ai.tracking.pose_tracker import pose_tracker
from ai.gestures.gesture_pipeline import gesture_pipeline
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
AI_FRAME_WIDTH = 480
AI_FRAME_HEIGHT = 270


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
        self.latest_pose: Optional[Dict[str, Any]] = None
        self.latest_all_poses: List[Dict[str, Any]] = []
        self.latest_gestures: Dict[str, Any] = {}
        self.latest_interactions: List[Dict[str, Any]] = []
        self.latest_action: Dict[str, Any] = action_recognizer.current_action_display
        self.hand_fps = 0.0
        self.detection_fps = 0.0

        # Latest scaled hand data shared from the hand thread to the detection
        # thread so interaction + action recognition always has current hands.
        self._hand_lock = threading.Lock()
        self._latest_scaled_hands: List[Dict[str, Any]] = []
        self._latest_small_hands: List[Dict[str, Any]] = []

        # Frame dimension tracking for resolution-independent coordinate normalization
        self._frame_width: int = 1280
        self._frame_height: int = 720

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
            # Wake up immediately when a new frame is delivered, or timeout at interval to maintain cadence
            if hasattr(camera_service, "_ai_frame_event"):
                camera_service._ai_frame_event.wait(timeout=interval)
                camera_service._ai_frame_event.clear()

            t0 = time.time()
            frame = camera_service.get_latest_frame(annotated=False)

            if frame is not None:
                try:
                    h_orig, w_orig = frame.shape[:2]
                    self._frame_width = w_orig
                    self._frame_height = h_orig
                    if w_orig == AI_FRAME_WIDTH and h_orig == AI_FRAME_HEIGHT:
                        ai_frame = frame
                        scale_x, scale_y = 1.0, 1.0
                    else:
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

                    # 2. Ultra-fast Body Pose Estimation + Body Gestures (~13.5ms)
                    pose_small = pose_tracker.process(ai_frame)
                    scaled_pose = None
                    scaled_all_poses = []
                    if pose_small:
                        scaled_pose = {
                            **pose_small,
                            "bbox": self._scale_bbox(pose_small.get("bbox", [0, 0, 0, 0]), scale_x, scale_y)
                        }
                        raw_all = pose_small.get("all_poses", [pose_small])
                        for p_item in raw_all:
                            scaled_all_poses.append({
                                **p_item,
                                "bbox": self._scale_bbox(p_item.get("bbox", [0, 0, 0, 0]), scale_x, scale_y)
                            })
                        scaled_pose["all_poses"] = scaled_all_poses

                    # 3. Real-time Gesture Synthesis & Debouncing (<0.1ms)
                    with self.lock:
                        current_action = self.latest_action
                    gestures = gesture_pipeline.analyze(hands, scaled_pose, current_action)

                    # Share hands and pose with the detection thread
                    with self._hand_lock:
                        self._latest_scaled_hands = hands
                        self._latest_small_hands = hands_small
                        self._latest_small_pose = pose_small

                    # Push updated hands, pose and gestures into the live overlay immediately
                    with self.lock:
                        self.latest_hands = hands
                        self.latest_pose = scaled_pose
                        self.latest_all_poses = scaled_all_poses
                        self.latest_gestures = gestures
                        current_objects = self.latest_objects

                    camera_service.update_perception_state(
                        current_objects,
                        hands,
                        current_action,
                        pose=scaled_pose,
                        gestures=gestures,
                    )

                except Exception as e:
                    print(f"[PerceptionPipeline/Hands&Pose Error] {e}")
            else:
                with self.lock:
                    self.latest_hands = []
                    self.latest_pose = None
                    self.latest_all_poses = []
                    self.latest_gestures = {}
                with self._hand_lock:
                    self._latest_scaled_hands = []
                    self._latest_small_hands = []
                    self._latest_small_pose = None
                camera_service.update_perception_state(
                    self.latest_objects,
                    [],
                    self.latest_action,
                    pose=None,
                    gestures={},
                )

            # FPS counter
            frame_count += 1
            now = time.time()
            if now - timer >= 1.0:
                self.hand_fps = round(frame_count / (now - timer), 1)
                frame_count = 0
                timer = now

            elapsed = time.time() - t0
            sleep_time = max(0.001, interval - elapsed)
            time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Thread 2 — YOLOv8 + interaction + action at AI_INFERENCE_FPS (~10 Hz)
    # ------------------------------------------------------------------
    def _detection_loop(self):
        """
        Run YOLOv8 object detection, hand-object interaction, and action
        recognition at AI_INFERENCE_FPS. Uses the freshest hand & pose data from
        the hand thread so interaction scoring and person validation are current.
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
                    self._frame_width = w_orig
                    self._frame_height = h_orig
                    if w_orig == AI_FRAME_WIDTH and h_orig == AI_FRAME_HEIGHT:
                        ai_frame = frame
                        scale_x, scale_y = 1.0, 1.0
                    else:
                        ai_frame = cv2.resize(
                            frame,
                            (AI_FRAME_WIDTH, AI_FRAME_HEIGHT),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        scale_x = w_orig / AI_FRAME_WIDTH
                        scale_y = h_orig / AI_FRAME_HEIGHT

                    # Grab the freshest hand and pose data from the hand thread.
                    with self._hand_lock:
                        hands_small = list(self._latest_small_hands)
                        hands = list(self._latest_scaled_hands)
                        pose_small = getattr(self, "_latest_small_pose", None)

                    # 1. Object detection — uses small hands & pose for multi-factor gating.
                    objects_small = object_detector.detect(ai_frame, hands=hands_small, pose=pose_small)
                    objects = [
                        {**obj, "bbox": self._scale_bbox(obj.get("bbox", [0, 0, 0, 0]), scale_x, scale_y)}
                        for obj in objects_small
                    ]

                    # 2. Hand-object interaction
                    interactions = hand_object_interaction.analyze(hands, objects, (h_orig, w_orig))

                    # Map GRASPED / CONTACT interaction states to object held metadata
                    for obj in objects:
                        for inter in interactions:
                            target_obj = inter.get("object")
                            if target_obj in (obj.get("label"), obj.get("raw_label"), obj.get("class_label")) and inter.get("state") in ("GRASPED", "CONTACT", "PRESSING"):
                                obj["is_held"] = True
                                obj["held"] = True
                                obj["held_by"] = f"{inter.get('hand')} Hand"
                                break

                    # Attribute action and activity to each astronaut
                    with self.lock:
                        active_scaled_pose = self.latest_pose

                    for obj in objects:
                        if obj.get("category") == "ASTRONAUT":
                            astro_id = obj.get("track_id", 1)
                            astro_posture = obj.get("posture", "Seated")
                            matched_p = obj.get("pose") or active_scaled_pose

                            # Check held items
                            held_items = [o.get("display_name") or o.get("label") for o in objects if o.get("held") or o.get("is_held")]
                            if held_items:
                                obj["action"] = f"Holding {held_items[0]}"
                                obj["activity"] = f"Manipulating {held_items[0]}"
                            else:
                                p_gest = None
                                if matched_p:
                                    p_gest = matched_p.get("primary_gesture")
                                    if not p_gest or p_gest in ("NONE", "STATIONARY"):
                                        g_list = matched_p.get("gestures", [])
                                        if g_list:
                                            p_gest = g_list[0].get("gesture")
                                if p_gest and p_gest not in ("NONE", "STATIONARY"):
                                    obj["action"] = p_gest.replace("_", " ").title()
                                    obj["activity"] = p_gest.replace("_", " ").title()
                                elif hands and any((h.get("bbox") or [0, 0, 0, 0])[1] > h_orig * 0.45 for h in hands):
                                    obj["action"] = "Typing / Operating Controls"
                                    obj["activity"] = "Operating Workstation"
                                else:
                                    obj["action"] = f"{astro_posture} at Station"
                                    obj["activity"] = "Monitoring Experiment Station"

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
                        current_hands = self.latest_hands
                        current_pose = self.latest_pose
                        current_gestures = self.latest_gestures

                    # Push full state
                    camera_service.update_perception_state(
                        objects,
                        current_hands,
                        action_recognizer.current_action_display,
                        pose=current_pose,
                        gestures=current_gestures,
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
            else:
                with self.lock:
                    self.latest_objects = []
                    self.latest_interactions = []
                camera_service.update_perception_state(
                    [],
                    self.latest_hands,
                    action_recognizer.current_action_display,
                    pose=self.latest_pose,
                    gestures=self.latest_gestures,
                )

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
            return any(
                obj.get("category") == "ASTRONAUT" or
                obj.get("raw_label") in ("Person", "Astronaut") or
                obj.get("class_label") in ("Person", "Astronaut")
                for obj in self.latest_objects
            ) or (self.latest_pose is not None)

    def get_perception_state(self) -> Dict[str, Any]:
        with self.lock:
            act = self.latest_action or {}
            gestures = getattr(self, "latest_gestures", {}) or {}
            primary_gesture = gestures.get("primary_gesture", "NONE")
            
            pose = self.latest_pose
            all_poses = getattr(self, "latest_all_poses", []) or ([pose] if pose else [])
            valid_poses = [p for p in all_poses if p and p.get("visible_landmarks_count", 0) >= 5]
            has_person = len(valid_poses) > 0

            fw = max(1, self._frame_width or 1280)
            fh = max(1, self._frame_height or 720)

            valid_hands = list(self.latest_hands)
            posture = gestures.get("posture") or act.get("posture", "Seated" if has_person else "Standby")

            bound_objects = []
            seen_astro_ids = set()

            # Process detected objects: retain all verified astronauts and everyday items
            for obj in self.latest_objects:
                is_astro = (
                    obj.get("category") == "ASTRONAUT" or
                    obj.get("raw_label") in ("Astronaut", "Person") or
                    obj.get("class_label") in ("Astronaut", "Person") or
                    obj.get("label") in ("Astronaut", "Person")
                )
                if is_astro:
                    if not has_person:
                        continue
                    astro_obj = dict(obj)
                    astro_id = astro_obj.get("track_id", 1)
                    if astro_id in seen_astro_ids:
                        continue
                    seen_astro_ids.add(astro_id)

                    astro_obj["category"] = "ASTRONAUT"
                    astro_obj["category_badge"] = "CREW"
                    astro_obj["display_name"] = f"✦ ASTRONAUT #{astro_id}"
                    astro_obj["role"] = f"Mission Operator #{astro_id} / EVA Specialist"
                    astro_obj["color_hex"] = "#00e6c8"
                    astro_obj["hands"] = valid_hands
                    astro_obj["skeleton_tracked"] = True
                    astro_obj["posture"] = astro_obj.get("posture", posture)
                    if not astro_obj.get("action"):
                        astro_obj["action"] = act.get("label", f"{posture} at Station")
                    bound_objects.append(astro_obj)
                else:
                    bound_objects.append(obj)
            
            # If YOLO missed any verified pose, synthesize astronaut entries for untracked poses
            if has_person and len(seen_astro_ids) == 0:
                for idx, p_item in enumerate(valid_poses):
                    astro_id = idx + 1
                    pb = p_item.get("bbox", [0, 0, 0, 0])
                    p_posture = p_item.get("posture", posture)
                    bound_objects.append({
                        "label": "Astronaut",
                        "raw_label": "Astronaut",
                        "class_label": "Astronaut",
                        "category": "ASTRONAUT",
                        "category_badge": "CREW",
                        "display_name": f"✦ ASTRONAUT #{astro_id}",
                        "role": f"Mission Operator #{astro_id} / EVA Specialist",
                        "color": "EVA Spacesuit",
                        "color_hex": "#00e6c8",
                        "confidence": 0.96,
                        "bbox": pb,
                        "track_id": astro_id,
                        "track_status": "TRACKED",
                        "hands": valid_hands,
                        "skeleton_tracked": True,
                        "posture": p_posture,
                        "action": act.get("label", f"{p_posture} at Station"),
                    })

            # Attach resolution-independent normalized_bbox [0.0 .. 1.0] to all objects
            for obj in bound_objects:
                b = obj.get("bbox")
                if b and len(b) >= 4:
                    obj["normalized_bbox"] = [
                        round(max(0.0, min(1.0, b[0] / fw)), 5),
                        round(max(0.0, min(1.0, b[1] / fh)), 5),
                        round(max(0.0, min(1.0, b[2] / fw)), 5),
                        round(max(0.0, min(1.0, b[3] / fh)), 5),
                    ]

            # Attach normalized_bbox to all poses
            for p_item in valid_poses:
                pb = p_item.get("bbox")
                if pb and len(pb) >= 4:
                    p_item["normalized_bbox"] = [
                        round(max(0.0, min(1.0, pb[0] / fw)), 5),
                        round(max(0.0, min(1.0, pb[1] / fh)), 5),
                        round(max(0.0, min(1.0, pb[2] / fw)), 5),
                        round(max(0.0, min(1.0, pb[3] / fh)), 5),
                    ]

            if pose and pose.get("bbox") and len(pose["bbox"]) >= 4:
                pb = pose["bbox"]
                pose["normalized_bbox"] = [
                    round(max(0.0, min(1.0, pb[0] / fw)), 5),
                    round(max(0.0, min(1.0, pb[1] / fh)), 5),
                    round(max(0.0, min(1.0, pb[2] / fw)), 5),
                    round(max(0.0, min(1.0, pb[3] / fh)), 5),
                ]

            state = {
                "frame_width": fw,
                "frame_height": fh,
                "fps": self.hand_fps,
                "detection_fps": self.detection_fps,
                "objects": bound_objects,
                "hands": valid_hands,
                "hands_count": len(valid_hands),
                "person_detected": has_person,
                "pose": pose if has_person else None,
                "poses": valid_poses if has_person else [],
                "gestures": gestures,
                "hand_gestures": gestures.get("hand_gestures", []),
                "body_gestures": gestures.get("body_gestures", []),
                "primary_gesture": primary_gesture,
                "gesture_summary": gestures.get("gesture_summary", "None"),
                "current_action": act,
                "movement": act.get("movement", "Active" if has_person else "Stationary"),
                "posture": posture,
                "narration": act.get("narration", "Monitoring"),
                "interactions": self.latest_interactions
            }
            return state

    def stop(self):
        self.is_running = False


perception_pipeline = PerceptionPipeline()
