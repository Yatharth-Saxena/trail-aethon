import time
import math
from typing import List, Dict, Any, Optional
from collections import deque
from experiment.state_machine.experiment_state import ExperimentEvent
from backend.config import ACTION_DEBOUNCE_SECONDS, TEMPORAL_BUFFER_SIZE

# ---------------------------------------------------------------------------
# Stabilisation constants
# ---------------------------------------------------------------------------
# Consecutive frames the same object must be held before an event is emitted.
HOLD_CONFIRM_FRAMES = 6
# Minimum detection confidence on the held object before an event is emitted.
HOLD_CONFIDENCE_GATE = 0.88


class ActionRecognizer:
    def __init__(self):
        self.history: deque = deque(maxlen=TEMPORAL_BUFFER_SIZE)
        self.last_triggered_time = 0.0
        self.last_triggered_action = ""
        self.pickup_counter = 0
        self.place_counter = 0
        # Track label consistency across frames
        self._held_label_streak: str = ""
        self._held_label_count: int = 0
        self.current_action_display: Dict[str, Any] = {
            "action": "IDLE",
            "label": "Ready / Monitoring",
            "movement": "Stationary",
            "posture": "Seated",
            "actor": "Astronaut",
            "hand": "None",
            "object": "None",
            "color": "None",
            "confidence": 0.90,
            "narration": "Monitoring experiment station. No active manipulation."
        }

    def reset(self):
        self.history.clear()
        self.last_triggered_time = 0.0
        self.last_triggered_action = ""
        self.pickup_counter = 0
        self.place_counter = 0
        self._held_label_streak = ""
        self._held_label_count = 0
        self.current_action_display = {
            "action": "IDLE",
            "label": "Ready / Monitoring",
            "movement": "Stationary",
            "posture": "Seated",
            "actor": "Astronaut",
            "hand": "None",
            "object": "None",
            "color": "None",
            "confidence": 0.90,
            "narration": "Monitoring experiment station. Ready."
        }

    def _is_stable_hold(self, label: str) -> bool:
        """Return True only when the same label has been held for HOLD_CONFIRM_FRAMES consecutive calls."""
        if label == self._held_label_streak:
            self._held_label_count += 1
        else:
            self._held_label_streak = label
            self._held_label_count = 1
        return self._held_label_count >= HOLD_CONFIRM_FRAMES

    def update(
        self,
        objects: List[Dict[str, Any]],
        hands: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        pose: Optional[Dict[str, Any]] = None
    ) -> Optional[ExperimentEvent]:
        now = time.time()
        snapshot = {
            "time": now,
            "objects": {o.get("raw_label", o.get("label")): o for o in objects},
            "objects_list": objects,
            "hands": hands,
            "interactions": interactions,
            "pose": pose
        }
        self.history.append(snapshot)

        # 1. Analyze Movements & Posture from MediaPipe Pose
        movement_label = "Stationary"
        posture_label = "Seated"
        is_waving = False
        arm_elevated = False
        elevated_arm_side = ""

        if pose:
            shoulders_y = pose.get("shoulders_mid_y", 0.5)
            l_wrist = pose.get("left_wrist")
            r_wrist = pose.get("right_wrist")
            head = pose.get("head")

            if shoulders_y < 0.32:
                posture_label = "Standing"
            else:
                posture_label = "Seated"

            # Arm elevation
            r_raised = r_wrist and (r_wrist["y"] < shoulders_y - 0.04)
            l_raised = l_wrist and (l_wrist["y"] < shoulders_y - 0.04)

            if r_raised and l_raised:
                movement_label = "Raising Both Arms"
                arm_elevated = True
                elevated_arm_side = "Both"
            elif r_raised:
                movement_label = "Raising Right Arm"
                arm_elevated = True
                elevated_arm_side = "Right"
            elif l_raised:
                movement_label = "Raising Left Arm"
                arm_elevated = True
                elevated_arm_side = "Left"

            # Waving detection (lateral oscillation of wrist over recent frames)
            if len(self.history) >= 8 and arm_elevated:
                wrist_key = "right_wrist" if elevated_arm_side == "Right" else "left_wrist"
                xs = [
                    h["pose"].get(wrist_key, {}).get("x", 0)
                    for h in list(self.history)[-8:]
                    if h.get("pose")
                ]
                if len(xs) >= 6:
                    diffs = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
                    sign_changes = sum(1 for i in range(1, len(diffs)) if (diffs[i] * diffs[i - 1]) < 0)
                    total_travel = sum(abs(d) for d in diffs)
                    if sign_changes >= 2 and total_travel > 0.08:
                        is_waving = True
                        movement_label = f"Waving {elevated_arm_side} Hand"

        # 2. Match Hands with Objects (Verified Holding & Grasp Detection)

        if not hands and pose:
            # Emulate hand positions from wrists if Hand Tracking is disabled
            for side, w_key in [("Left", "left_wrist"), ("Right", "right_wrist")]:
                wrist = pose.get(w_key)
                if wrist:
                    wx, wy = wrist["x"] * 1280, wrist["y"] * 720
                    hands.append({
                        "side": side,
                        "bbox": [wx - 25, wy - 25, wx + 25, wy + 25],
                        "is_pinching": False
                    })

        held_object = None

        holding_hand_side = "Right"

        for hnd in hands:
            h_box = hnd.get("bbox", [0, 0, 0, 0])
            hcx = (h_box[0] + h_box[2]) / 2.0
            hcy = (h_box[1] + h_box[3]) / 2.0
            side = hnd.get("side", "Right")
            is_pinching = hnd.get("is_pinching", False)

            for obj in objects:
                raw_lbl = obj.get("raw_label", obj.get("label", ""))
                if raw_lbl.lower() in ["person", "tray"]:
                    continue
                o_box = obj.get("bbox", [0, 0, 0, 0])
                ocx = (o_box[0] + o_box[2]) / 2.0
                ocy = (o_box[1] + o_box[3]) / 2.0

                dist = math.hypot(hcx - ocx, hcy - ocy)
                # Bounding box overlap
                overlap = (
                    h_box[0] < o_box[2] and h_box[2] > o_box[0] and
                    h_box[1] < o_box[3] and h_box[3] > o_box[1]
                )
                # Genuine hand hold requires either direct bounding overlap or close fingertip contact (<42px)
                if (overlap and (is_pinching or dist < 55)) or (dist < 40):
                    held_object = obj
                    holding_hand_side = side
                    obj["is_held"] = True
                    break
            if held_object:
                break

        if not held_object:
            self.pickup_counter = 0
            self.place_counter = 0
            self._held_label_streak = ""
            self._held_label_count = 0

        # 3. Handle Button Pressing
        pressing_interactions = [i for i in interactions if i.get("state") == "PRESSING" and "button" in i.get("object", "").lower()]
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

        # 4. Handle Held Everyday Objects & Payload Objects
        if held_object:
            raw_lbl = held_object.get("raw_label", held_object.get("label", ""))
            disp_name = held_object.get("label", raw_lbl)
            color_name = held_object.get("color", "")
            obj_confidence = held_object.get("confidence", 0.0)
            head_data = pose.get("head") if pose else None

            # Check label stability for payload objects before emitting events
            is_payload = raw_lbl.lower() in ["object a", "object b"]
            label_stable = self._is_stable_hold(raw_lbl)
            confidence_ok = obj_confidence >= HOLD_CONFIDENCE_GATE

            # Cell Phone
            if "phone" in raw_lbl.lower():
                near_ear = False
                if head_data:
                    hx, hy = head_data["x"], head_data["y"]
                    o_box = held_object.get("bbox", [0, 0, 0, 0])
                    near_ear = math.hypot(hx - (o_box[0]+o_box[2])/2400.0, hy - (o_box[1]+o_box[3])/1440.0) < 0.25

                action_verb = "PHONE_CALL" if near_ear else "HOLDING_PHONE"
                action_text = f"On Phone Call with {disp_name}" if near_ear else f"Holding / Operating {disp_name}"
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
            elif any(k in raw_lbl.lower() for k in ["bottle", "cup"]):
                near_mouth = False
                if head_data:
                    near_mouth = (arm_elevated and head_data["y"] > 0.15)
                action_verb = "DRINKING" if near_mouth else "HOLDING_CONTAINER"
                action_text = f"Drinking from {disp_name}" if near_mouth else f"Holding {disp_name}"
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
            elif "object a" in raw_lbl.lower() or "red" in disp_name.lower():
                last_obj = held_object

                target_b = snapshot["objects"].get("Object B")
                target_tray = snapshot["objects"].get("Tray")

                near_b = False
                if target_b:
                    tb_box = target_b["bbox"]
                    ob_box = last_obj["bbox"]
                    dist = math.hypot((ob_box[0]+ob_box[2])/2 - (tb_box[0]+tb_box[2])/2,
                                      (ob_box[1]+ob_box[3])/2 - (tb_box[1]+tb_box[3])/2)
                    near_b = dist < 75

                near_tray = False
                if target_tray:
                    tt_box = target_tray["bbox"]
                    ob_box = last_obj["bbox"]
                    dist = math.hypot((ob_box[0]+ob_box[2])/2 - (tt_box[0]+tt_box[2])/2,
                                      (ob_box[1]+ob_box[3])/2 - (tt_box[1]+tt_box[3])/2)
                    near_tray = dist < 95

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

        # 5. No Object Held -> Movement-Driven Activity
        else:
            if is_waving:
                self.current_action_display = {
                    "action": "WAVING",
                    "label": f"Waving {elevated_arm_side} Hand to Camera",
                    "movement": "Waving Hand",
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{elevated_arm_side} hand",
                    "object": "None",
                    "color": "None",
                    "confidence": 0.95,
                    "narration": f"Astronaut is waving their {elevated_arm_side.lower()} hand."
                }
            elif arm_elevated:
                self.current_action_display = {
                    "action": "GESTURE",
                    "label": f"Raising {elevated_arm_side} Hand",
                    "movement": movement_label,
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": f"{elevated_arm_side} hand",
                    "object": "None",
                    "color": "None",
                    "confidence": 0.92,
                    "narration": f"Astronaut has raised their {elevated_arm_side.lower()} arm."
                }
            elif len(hands) >= 1 and any(h.get("bbox", [0,0,0,0])[1] > 320 for h in hands):
                # Hands resting/typing in front of keyboard
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
                    "label": f"Monitoring / Seated at Workstation",
                    "movement": "Stationary",
                    "posture": posture_label,
                    "actor": "Astronaut",
                    "hand": "None",
                    "object": "None",
                    "color": "None",
                    "confidence": 0.92,
                    "narration": "Astronaut is seated attentively at the workstation."
                }

        return None

action_recognizer = ActionRecognizer()
