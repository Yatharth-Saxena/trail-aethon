import time
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from experiment.state_machine.experiment_state import ExperimentEvent
from ai.interaction.hand_object_interaction import bbox_gap, point_in_bbox
from backend.config import (
    ACTION_DEBOUNCE_SECONDS,
    TEMPORAL_BUFFER_SIZE,
    OBJECT_DISPLACED_DEBOUNCE_SECONDS,
)

# Interaction states that mean a tracked hand is driving the object. Anything
# moving *without* one of these is movement AETHON did not cause.
HAND_DRIVEN_STATES = {"GRASPED", "CONTACT", "PRESSING"}
# Fraction of recent frames an object must be flagged moving before it is
# reported as displaced. Layered on top of the detector's own motion
# debounce so a single jittery track cannot raise a false alarm.
DISPLACED_HISTORY_RATIO = 0.5
DISPLACED_HISTORY_FRAMES = 6

# ---------------------------------------------------------------------------
# Stabilisation constants
# ---------------------------------------------------------------------------
# Consecutive frames the same object must be held before an event is emitted.
HOLD_CONFIRM_FRAMES = 6
# Minimum detection confidence on the held object before an event is emitted.
HOLD_CONFIDENCE_GATE = 0.60
# Hand/object gap (fraction of frame width) that counts as contact.
CONTACT_REACH_FRAC = 0.045
# Normalised distance between two object centres that counts as "placed on".
PLACE_ON_TARGET_FRAC = 0.075
PLACE_ON_TRAY_FRAC = 0.095
# Hand centre above this normalised height counts as a raised hand.
RAISED_HAND_Y = 0.35
# Normalised per-frame hand travel above which the hand is considered moving.
HAND_MOTION_SPEED = 0.012
# Frames of hand history inspected for waving.
WAVE_WINDOW = 8
# Normalised distance from the head proxy for "phone at ear" / "drinking".
HEAD_PROXIMITY = 0.14


class ActionRecognizer:
    """
    Temporal action recogniser driven entirely by hand tracking and object
    detection. There is no body-pose stage: gestures are inferred from hand
    landmark motion, and head position is approximated from the operator's
    detected person box when an action needs it (phone at ear, drinking).
    """

    def __init__(self):
        self.history: deque = deque(maxlen=TEMPORAL_BUFFER_SIZE)
        self.last_triggered_time = 0.0
        self.last_triggered_action = ""
        self.pickup_counter = 0
        self.place_counter = 0
        # Track label consistency across frames
        self._held_label_streak: str = ""
        self._held_label_count: int = 0
        self._last_displaced_time = 0.0
        self.current_action_display: Dict[str, Any] = self._idle_display(
            "Ready / Monitoring",
            "Monitoring experiment station. No active manipulation.",
        )

    @staticmethod
    def _idle_display(label: str, narration: str) -> Dict[str, Any]:
        return {
            "action": "IDLE",
            "label": label,
            "movement": "Stationary",
            "posture": "Seated",
            "actor": "Astronaut",
            "hand": "None",
            "object": "None",
            "color": "None",
            "confidence": 0.90,
            "narration": narration,
            "object_moving": False,
            "object_velocity": 0.0,
        }

    def reset(self):
        self.history.clear()
        self.last_triggered_time = 0.0
        self.last_triggered_action = ""
        self.pickup_counter = 0
        self.place_counter = 0
        self._held_label_streak = ""
        self._held_label_count = 0
        self._last_displaced_time = 0.0
        self.current_action_display = self._idle_display(
            "Ready / Monitoring",
            "Monitoring experiment station. Ready.",
        )

    def _is_stable_hold(self, label: str) -> bool:
        """Return True only when the same label has been held for HOLD_CONFIRM_FRAMES consecutive calls."""
        if label == self._held_label_streak:
            self._held_label_count += 1
        else:
            self._held_label_streak = label
            self._held_label_count = 1
        return self._held_label_count >= HOLD_CONFIRM_FRAMES

    # -----------------------------------------------------------------------
    # Hand-derived gesture analysis
    # -----------------------------------------------------------------------
    def _hand_centers(self, hands: List[Dict[str, Any]], w: int, h: int) -> Dict[str, Tuple[float, float]]:
        """Normalised centre of each tracked hand, keyed by side."""
        centers: Dict[str, Tuple[float, float]] = {}
        for hnd in hands:
            box = hnd.get("bbox")
            if not box or len(box) < 4:
                continue
            cx = ((box[0] + box[2]) / 2.0) / max(1, w)
            cy = ((box[1] + box[3]) / 2.0) / max(1, h)
            centers[hnd.get("side", "Right")] = (cx, cy)
        return centers

    def _analyze_gestures(self, centers: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Classify hand movement from the recent centre history: raised hands,
        lateral waving, and general reaching motion.
        """
        raised = [side for side, (_cx, cy) in centers.items() if cy < RAISED_HAND_Y]
        raised_side = ""
        if len(raised) >= 2:
            raised_side = "Both"
        elif raised:
            raised_side = raised[0]

        # Per-hand travel speed over the last few frames.
        speed = 0.0
        waving_side = ""
        recent = [s for s in list(self.history)[-WAVE_WINDOW:] if s.get("hand_centers")]

        for side, (cx, cy) in centers.items():
            xs = [s["hand_centers"][side][0] for s in recent if side in s["hand_centers"]]
            ys = [s["hand_centers"][side][1] for s in recent if side in s["hand_centers"]]
            if len(xs) < 3:
                continue

            steps = [
                math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
                for i in range(1, len(xs))
            ]
            speed = max(speed, sum(steps) / len(steps))

            # Waving: repeated left/right direction changes with real travel.
            if side in raised or raised_side == "Both":
                diffs = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
                sign_changes = sum(
                    1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0
                )
                travel = sum(abs(d) for d in diffs)
                if len(diffs) >= 4 and sign_changes >= 2 and travel > 0.10:
                    waving_side = side

        if waving_side:
            movement = f"Waving {waving_side} Hand"
        elif raised_side:
            movement = f"Raising {'Both Hands' if raised_side == 'Both' else raised_side + ' Hand'}"
        elif speed > HAND_MOTION_SPEED:
            movement = "Reaching / Hand Motion"
        elif centers:
            movement = "Hands Steady"
        else:
            movement = "Stationary"

        return {
            "movement": movement,
            "raised_side": raised_side,
            "waving_side": waving_side,
            "speed": speed,
        }

    @staticmethod
    def _head_proxy(objects: List[Dict[str, Any]], w: int, h: int) -> Optional[Dict[str, float]]:
        """
        Approximate normalised head position from the largest detected person
        box. Used only for coarse "is the object up at the face" checks.
        """
        best, best_area = None, 0
        for obj in objects:
            if (obj.get("raw_label") or "").lower() != "person":
                continue
            box = obj.get("bbox")
            if not box or len(box) < 4:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > best_area:
                best, best_area = box, area
        if best is None:
            return None
        box_h = max(1, best[3] - best[1])
        return {
            "x": ((best[0] + best[2]) / 2.0) / max(1, w),
            "y": (best[1] + box_h * 0.14) / max(1, h),
        }

    # -----------------------------------------------------------------------
    # Independent object movement
    # -----------------------------------------------------------------------
    def _find_displaced_object(
        self,
        objects: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        held_object: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Find an object that is moving on its own — sliding, tipping, drifting
        or nudged by something other than a tracked hand.

        An object is only reported once the detector's debounced motion flag
        has held for a majority of recent frames, and only when no tracked
        hand is in contact with it, which is what separates this from a
        normal hand-driven PICK_UP or PLACE.
        """
        hand_driven = {
            i.get("object")
            for i in interactions
            if i.get("state") in HAND_DRIVEN_STATES
        }
        held_label = None
        if held_object is not None:
            held_label = held_object.get("raw_label") or held_object.get("label")

        recent = [s for s in list(self.history)[-DISPLACED_HISTORY_FRAMES:] if s.get("motion_flags")]
        best, best_speed = None, 0.0

        for obj in objects:
            label = obj.get("raw_label") or obj.get("label") or ""
            if label.lower() == "person":
                continue
            if not obj.get("is_moving"):
                continue
            if label in hand_driven or label == held_label:
                continue

            track_id = obj.get("track_id", label)
            if recent:
                votes = [1 for s in recent if s["motion_flags"].get(track_id)]
                if len(votes) / len(recent) < DISPLACED_HISTORY_RATIO:
                    continue

            speed = float(obj.get("velocity", 0.0))
            if speed > best_speed:
                best, best_speed = obj, speed

        return best

    # -----------------------------------------------------------------------
    # Held-object resolution
    # -----------------------------------------------------------------------
    def _find_held_object(
        self,
        objects: List[Dict[str, Any]],
        hands: List[Dict[str, Any]],
        w: int,
        h: int,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Pick the object the operator is actually holding.

        Contact is measured edge-to-edge between the hand and object boxes and
        confirmed by fingertips inside the object box, so it stays correct for
        both small and large items. A hand that is neither grasping nor
        touching the object with a fingertip is not treated as holding it.
        """
        contact_reach = max(22.0, w * CONTACT_REACH_FRAC)
        best_obj, best_side, best_score = None, "Right", -1.0

        for hnd in hands:
            hand_box = hnd.get("bbox")
            if not hand_box or len(hand_box) < 4:
                continue
            side = hnd.get("side", "Right")
            grasping = hnd.get("is_grasping", hnd.get("is_pinching", False))
            tips = [
                (t["x"] * w, t["y"] * h)
                for t in (hnd.get("fingertips") or [])
            ]

            for obj in objects:
                raw_lbl = (obj.get("raw_label") or obj.get("label") or "").lower()
                if raw_lbl in ("person", "tray"):
                    continue
                obj_box = obj.get("bbox")
                if not obj_box or len(obj_box) < 4:
                    continue

                gap = bbox_gap(hand_box, obj_box)
                tip_inside = any(point_in_bbox(tx, ty, obj_box) for tx, ty in tips)
                if gap > contact_reach and not tip_inside:
                    continue
                if not (grasping or tip_inside):
                    continue

                # Prefer a grasped, fingertip-confirmed, tightly-overlapping pair.
                score = (2.0 if grasping else 0.0) + (1.5 if tip_inside else 0.0)
                score += max(0.0, 1.0 - gap / max(1.0, contact_reach))
                if score > best_score:
                    best_obj, best_side, best_score = obj, side, score

        return best_obj, best_side

    # -----------------------------------------------------------------------
    # Main update
    # -----------------------------------------------------------------------
    def update(
        self,
        objects: List[Dict[str, Any]],
        hands: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> Optional[ExperimentEvent]:
        now = time.time()
        h, w = frame_shape[0], frame_shape[1]

        centers = self._hand_centers(hands, w, h)
        snapshot = {
            "time": now,
            "objects": {o.get("raw_label", o.get("label")): o for o in objects},
            "objects_list": objects,
            "hands": hands,
            "hand_centers": centers,
            "interactions": interactions,
            # Per-track motion flags, so independent movement can be confirmed
            # over the temporal buffer rather than a single frame.
            "motion_flags": {
                o.get("track_id", o.get("raw_label", o.get("label"))): bool(o.get("is_moving"))
                for o in objects
            },
        }
        self.history.append(snapshot)

        # 1. Gesture & movement analysis from hand tracking
        gestures = self._analyze_gestures(centers)
        movement_label = gestures["movement"]
        # Without a body-pose stage there is no reliable seated/standing
        # signal, so posture reports the station default.
        posture_label = "Seated"
        is_waving = bool(gestures["waving_side"])
        hand_raised = bool(gestures["raised_side"])
        raised_side = gestures["raised_side"] or "Right"

        head_data = self._head_proxy(objects, w, h)

        # 2. Resolve which object (if any) is genuinely held
        held_object, holding_hand_side = self._find_held_object(objects, hands, w, h)
        if held_object is not None:
            held_object["is_held"] = True
            # `held` / `held_by` are the keys the voice service and dashboard
            # read; `is_held` is kept for the overlay renderer.
            held_object["held"] = True
            held_object["held_by"] = f"{holding_hand_side} Hand"
        else:
            self.pickup_counter = 0
            self.place_counter = 0
            self._held_label_streak = ""
            self._held_label_count = 0

        # 3. Detect movement that no tracked hand caused
        displaced_object = self._find_displaced_object(objects, interactions, held_object)

        # 4. Handle Button Pressing
        pressing_interactions = [
            i for i in interactions
            if i.get("state") == "PRESSING" and "button" in i.get("object", "").lower()
        ]
        if pressing_interactions:
            btn_int = pressing_interactions[0]
            self.current_action_display = {
                "action": "PRESS",
                "label": "Pressing Complete Button",
                "movement": "Pressing Downwards",
                "posture": posture_label,
                "actor": "Astronaut",
                "hand": f"{btn_int['hand']} hand",
                "object": "Complete Button",
                "color": "Yellow",
                "confidence": 0.95,
                "narration": f"Astronaut is pressing the Complete Button with their {btn_int['hand']} hand."
            }
            press_count = sum(
                1 for frame in list(self.history)[-7:]
                if any(i["state"] == "PRESSING" and "button" in i["object"].lower() for i in frame["interactions"])
            )
            if press_count >= 4 and (now - self.last_triggered_time > ACTION_DEBOUNCE_SECONDS):
                self.last_triggered_time = now
                self.last_triggered_action = "PRESS_BUTTON"
                return ExperimentEvent(
                    event="PRESS",
                    object="Complete Button",
                    actor="person",
                    hand=btn_int["hand"],
                    confidence=0.95
                )

        # 5. Handle Held Everyday Objects & Payload Objects
        if held_object:
            raw_lbl = held_object.get("raw_label", held_object.get("label", ""))
            disp_name = held_object.get("label", raw_lbl)
            color_name = held_object.get("color", "")
            obj_confidence = held_object.get("confidence", 0.0)
            obj_box = held_object.get("bbox", [0, 0, 0, 0])
            obj_center = (
                ((obj_box[0] + obj_box[2]) / 2.0) / max(1, w),
                ((obj_box[1] + obj_box[3]) / 2.0) / max(1, h),
            )

            # Check label stability for payload objects before emitting events
            label_stable = self._is_stable_hold(raw_lbl)
            confidence_ok = obj_confidence >= HOLD_CONFIDENCE_GATE

            near_head = False
            if head_data:
                near_head = math.hypot(
                    obj_center[0] - head_data["x"],
                    obj_center[1] - head_data["y"],
                ) < HEAD_PROXIMITY

            # Cell Phone
            if "phone" in raw_lbl.lower():
                action_verb = "PHONE_CALL" if near_head else "HOLDING_PHONE"
                action_text = f"On Phone Call with {disp_name}" if near_head else f"Holding / Operating {disp_name}"
                self.current_action_display = {
                    "action": action_verb,
                    "label": f"{action_text} ({holding_hand_side} Hand)",
                    "movement": movement_label,
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{holding_hand_side} hand",
                    "object": disp_name,
                    "color": color_name,
                    "confidence": 0.93,
                    "narration": f"Astronaut is holding a {disp_name} in their {holding_hand_side.lower()} hand."
                }

            # Bottle or Cup (Drink or Hold)
            elif any(k in raw_lbl.lower() for k in ["bottle", "cup", "wine glass"]):
                action_verb = "DRINKING" if near_head else "HOLDING_CONTAINER"
                action_text = f"Drinking from {disp_name}" if near_head else f"Holding {disp_name}"
                self.current_action_display = {
                    "action": action_verb,
                    "label": f"{action_text} ({holding_hand_side} Hand)",
                    "movement": movement_label,
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{holding_hand_side} hand",
                    "object": disp_name,
                    "color": color_name,
                    "confidence": 0.92,
                    "narration": f"Astronaut is {action_text.lower()}."
                }

            # Book / Notebook
            elif "book" in raw_lbl.lower():
                self.current_action_display = {
                    "action": "READING_DOCUMENT",
                    "label": f"Examining / Holding {disp_name}",
                    "movement": movement_label,
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{holding_hand_side} hand",
                    "object": disp_name,
                    "color": color_name,
                    "confidence": 0.91,
                    "narration": f"Astronaut is examining {disp_name}."
                }

            # Scissors / Tool
            elif "scissors" in raw_lbl.lower():
                self.current_action_display = {
                    "action": "USING_TOOL",
                    "label": f"Operating {disp_name}",
                    "movement": movement_label,
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{holding_hand_side} hand",
                    "object": disp_name,
                    "color": color_name,
                    "confidence": 0.90,
                    "narration": f"Astronaut is operating {disp_name} with their {holding_hand_side.lower()} hand."
                }

            # Payload Object A (Experiment Procedure Tracking)
            elif "object a" in raw_lbl.lower():
                target_b = snapshot["objects"].get("Object B")
                target_tray = snapshot["objects"].get("Tray")

                def center_gap(target: Optional[Dict[str, Any]]) -> Optional[float]:
                    if not target:
                        return None
                    tb = target.get("bbox")
                    if not tb or len(tb) < 4:
                        return None
                    tcx = ((tb[0] + tb[2]) / 2.0) / max(1, w)
                    tcy = ((tb[1] + tb[3]) / 2.0) / max(1, h)
                    return math.hypot(obj_center[0] - tcx, obj_center[1] - tcy)

                gap_b = center_gap(target_b)
                gap_tray = center_gap(target_tray)
                near_b = gap_b is not None and gap_b < PLACE_ON_TARGET_FRAC
                near_tray = gap_tray is not None and gap_tray < PLACE_ON_TRAY_FRAC

                if near_b:
                    self.place_counter += 1
                    self.current_action_display = {
                        "action": "PLACE",
                        "label": f"Placing {disp_name} on Object B",
                        "movement": "Placing Downwards",
                        "posture": posture_label,
                        "actor": "Astronaut",
                        "hand": f"{holding_hand_side} hand",
                        "object": f"{disp_name} → Object B",
                        "color": color_name,
                        "confidence": 0.93,
                        "narration": f"Placing {disp_name} onto Object B."
                    }
                    if (self.place_counter >= HOLD_CONFIRM_FRAMES
                            and label_stable and confidence_ok
                            and (now - self.last_triggered_time > ACTION_DEBOUNCE_SECONDS)):
                        self.last_triggered_time = now
                        self.last_triggered_action = "PLACE_A_ON_B"
                        self.place_counter = 0
                        return ExperimentEvent(
                            event="PLACE",
                            object="Object A",
                            target="Object B",
                            actor="person",
                            hand=holding_hand_side,
                            confidence=0.94
                        )
                elif near_tray:
                    self.place_counter += 1
                    self.current_action_display = {
                        "action": "PLACE",
                        "label": f"Returning {disp_name} to Tray",
                        "movement": "Returning to Surface",
                        "posture": posture_label,
                        "actor": "Astronaut",
                        "hand": f"{holding_hand_side} hand",
                        "object": f"{disp_name} → Tray",
                        "color": color_name,
                        "confidence": 0.92,
                        "narration": f"Returning {disp_name} to Tray."
                    }
                    if (self.place_counter >= HOLD_CONFIRM_FRAMES
                            and label_stable and confidence_ok
                            and (now - self.last_triggered_time > ACTION_DEBOUNCE_SECONDS)):
                        self.last_triggered_time = now
                        self.last_triggered_action = "PLACE_A_TRAY"
                        self.place_counter = 0
                        return ExperimentEvent(
                            event="PLACE",
                            object="Object A",
                            target="Tray",
                            actor="person",
                            hand=holding_hand_side,
                            confidence=0.92
                        )
                else:
                    self.pickup_counter += 1
                    self.place_counter = 0
                    self.current_action_display = {
                        "action": "PICK_UP",
                        "label": f"Holding {disp_name} ({holding_hand_side} Hand)",
                        "movement": movement_label,
                        "posture": posture_label,
                        "actor": "Astronaut",
                        "hand": f"{holding_hand_side} hand",
                        "object": disp_name,
                        "color": color_name,
                        "confidence": 0.94,
                        "narration": f"Astronaut has picked up {disp_name} with their {holding_hand_side.lower()} hand."
                    }
                    if (self.pickup_counter >= HOLD_CONFIRM_FRAMES
                            and label_stable and confidence_ok
                            and (now - self.last_triggered_time > ACTION_DEBOUNCE_SECONDS)):
                        self.last_triggered_time = now
                        self.last_triggered_action = "PICK_UP_A"
                        self.pickup_counter = 0
                        return ExperimentEvent(
                            event="PICK_UP",
                            object="Object A",
                            actor="person",
                            hand=holding_hand_side,
                            confidence=0.94
                        )

            # Generic item
            else:
                self.current_action_display = {
                    "action": "HOLDING_OBJECT",
                    "label": f"Holding {disp_name} in {holding_hand_side} Hand",
                    "movement": movement_label,
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{holding_hand_side} hand",
                    "object": disp_name,
                    "color": color_name,
                    "confidence": 0.91,
                    "narration": f"Astronaut is holding a {disp_name} in their {holding_hand_side.lower()} hand."
                }

        # 6. No Object Held -> Independent object motion, else gesture activity
        else:
            hands_low = any(
                (hnd.get("bbox") or [0, 0, 0, 0])[1] > h * 0.45 for hnd in hands
            )
            if displaced_object is not None:
                disp_label = displaced_object.get("label", "Object")
                speed = float(displaced_object.get("velocity", 0.0))
                self.current_action_display = {
                    "action": "OBJECT_DISPLACED",
                    "label": f"{disp_label} Moving Unattended",
                    "movement": "Object Displacement",
                    "posture": posture_label,
                    "actor": "Environment",
                    "hand": "None",
                    "object": disp_label,
                    "color": displaced_object.get("color", "None"),
                    "confidence": 0.88,
                    "narration": (
                        f"{disp_label} is moving without hand contact. "
                        "Possible drift, slide or external disturbance."
                    ),
                }
                self._last_displaced_time = now
            elif is_waving:
                self.current_action_display = {
                    "action": "WAVING",
                    "label": f"Waving {raised_side} Hand to Camera",
                    "movement": "Waving Hand",
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{raised_side} hand",
                    "object": "None",
                    "color": "None",
                    "confidence": 0.95,
                    "narration": f"Astronaut is waving their {raised_side.lower()} hand."
                }
            elif hand_raised:
                self.current_action_display = {
                    "action": "GESTURE",
                    "label": f"Raising {raised_side} Hand",
                    "movement": movement_label,
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{raised_side} hand",
                    "object": "None",
                    "color": "None",
                    "confidence": 0.92,
                    "narration": f"Astronaut has raised their {raised_side.lower()} arm."
                }
            elif hands and hands_low:
                # Hands resting/typing in front of the workstation controls
                self.current_action_display = {
                    "action": "TYPING",
                    "label": "Typing / Operating Workstation Controls",
                    "movement": "Hand Dexterity / Typing",
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": "Both hands" if len(hands) > 1 else f"{hands[0].get('side', 'Right')} hand",
                    "object": "Workstation Controls",
                    "color": "None",
                    "confidence": 0.90,
                    "narration": "Astronaut is operating workstation controls."
                }
            else:
                self.current_action_display = {
                    "action": "IDLE",
                    "label": "Monitoring / Seated at Workstation",
                    "movement": movement_label if hands else "Stationary",
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": "None",
                    "object": "None",
                    "color": "None",
                    "confidence": 0.92,
                    "narration": "Astronaut is seated attentively at the workstation."
                }

        # Attach the object-motion signal to whichever action was selected, so
        # consumers can tell a static object from one that is in motion
        # regardless of which branch above produced the display.
        motion_source = displaced_object or held_object
        self.current_action_display["object_moving"] = bool(
            motion_source.get("is_moving") if motion_source else False
        )
        self.current_action_display["object_velocity"] = round(
            float(motion_source.get("velocity", 0.0)) if motion_source else 0.0, 4
        )

        return None

    def displaced_since(self, seconds: float = OBJECT_DISPLACED_DEBOUNCE_SECONDS) -> bool:
        """True when an unattended-movement display was set within *seconds*."""
        return (time.time() - self._last_displaced_time) <= seconds


action_recognizer = ActionRecognizer()
