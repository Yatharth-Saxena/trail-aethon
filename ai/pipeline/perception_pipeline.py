import time
import threading
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from ai.detection.object_detector import object_detector
from ai.pose.pose_estimator import pose_estimator
from ai.tracking.hand_tracker import hand_tracker
from ai.interaction.hand_object_interaction import hand_object_interaction
from ai.actions.action_recognizer import action_recognizer
from camera.camera_service import camera_service
from experiment.manager import experiment_manager
from event_logging.event_logger import event_logger
from backend.config import AI_TARGET_FPS

class PerceptionPipeline:
    def __init__(self):
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.latest_objects: List[Dict[str, Any]] = []
        self.latest_hands: List[Dict[str, Any]] = []
        self.latest_pose: Optional[Dict[str, Any]] = None
        self.latest_interactions: List[Dict[str, Any]] = []
        self.latest_action: Dict[str, Any] = action_recognizer.current_action_display
        self.fps = 0.0

        self.start()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self.worker_thread.start()

    def _pipeline_loop(self):
        frame_interval = 1.0 / AI_TARGET_FPS
        frame_count = 0
        timer = time.time()

        while self.is_running:
            t0 = time.time()
            frame = camera_service.get_latest_frame(annotated=False)

            if frame is not None:
                try:
                    h_orig, w_orig = frame.shape[:2]
                    # Resize to 640x360 for high-speed real-time MediaPipe and object detection
                    ai_w, ai_h = 640, 360
                    ai_frame = cv2.resize(frame, (ai_w, ai_h), interpolation=cv2.INTER_LINEAR)
                    scale_x = w_orig / ai_w
                    scale_y = h_orig / ai_h

                    # 1. Pose Estimation
                    pose_data = pose_estimator.process(ai_frame)
                    person_bbox = None
                    if pose_data and pose_data.get("bbox"):
                        bx1, by1, bx2, by2 = pose_data["bbox"]
                        person_bbox = [
                            int(bx1 * scale_x),
                            int(by1 * scale_y),
                            int(bx2 * scale_x),
                            int(by2 * scale_y)
                        ]
                        pose_data["bbox"] = person_bbox

                    # 2. Object Detection on ai_frame
                    pb_small = None
                    if person_bbox:
                        pb_small = [int(v / scale_x) for v in person_bbox]
                    objects_small = object_detector.detect(ai_frame, person_bbox=pb_small)
                    objects = []
                    for obj in objects_small:
                        ob = obj.get("bbox", [0, 0, 0, 0])
                        objects.append({
                            **obj,
                            "bbox": [
                                int(ob[0] * scale_x),
                                int(ob[1] * scale_y),
                                int(ob[2] * scale_x),
                                int(ob[3] * scale_y)
                            ]
                        })

                    # 3. Hand Tracking on ai_frame
                    hands_small = hand_tracker.process(ai_frame)
                    hands = []
                    for hnd in hands_small:
                        hb = hnd.get("bbox", [0, 0, 0, 0])
                        hands.append({
                            **hnd,
                            "bbox": [
                                int(hb[0] * scale_x),
                                int(hb[1] * scale_y),
                                int(hb[2] * scale_x),
                                int(hb[3] * scale_y)
                            ]
                        })

                    # 4. Hand-Object Interaction
                    interactions = hand_object_interaction.analyze(hands, objects, (h_orig, w_orig))

                    # 5. Temporal Action Recognition & Movement Analysis
                    event = action_recognizer.update(objects, hands, interactions, pose=pose_data)

                    with self.lock:
                        self.latest_objects = objects
                        self.latest_hands = hands
                        self.latest_pose = pose_data
                        self.latest_interactions = interactions
                        self.latest_action = action_recognizer.current_action_display

                    # Update camera overlays
                    camera_service.update_perception_state(
                        objects,
                        hands,
                        pose_data,
                        action_recognizer.current_action_display
                    )

                    # 6. Feed to Experiment Manager if a discrete event occurred
                    if event is not None:
                        event_logger.log(
                            "ACTION_DETECTED",
                            f"{event.event} on {event.object}" + (f" -> {event.target}" if event.target else ""),
                            event.model_dump()
                        )
                        # Process through validator and state machine
                        experiment_manager.experiment_manager.process_event(event)

                except Exception as e:
                    print(f"[PerceptionPipeline Error] {e}")

            # Measure inference FPS
            frame_count += 1
            now = time.time()
            if now - timer >= 1.0:
                self.fps = round(frame_count / (now - timer), 1)
                frame_count = 0
                timer = now

            elapsed = time.time() - t0
            sleep_time = max(0.002, frame_interval - elapsed)
            time.sleep(sleep_time)

    def get_perception_state(self) -> Dict[str, Any]:
        with self.lock:
            act = self.latest_action or {}
            return {
                "fps": self.fps,
                "objects": self.latest_objects,
                "hands_count": len(self.latest_hands),
                "person_detected": self.latest_pose is not None,
                "current_action": act,
                "movement": act.get("movement", "Stationary"),
                "posture": act.get("posture", "Seated"),
                "narration": act.get("narration", "Monitoring"),
                "interactions": self.latest_interactions
            }

    def stop(self):
        self.is_running = False

perception_pipeline = PerceptionPipeline()
