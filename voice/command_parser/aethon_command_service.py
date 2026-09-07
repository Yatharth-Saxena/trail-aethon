from typing import Dict, Any, List, Optional
from datetime import datetime
from voice.command_parser.command_parser import command_parser, Intent
from experiment.manager.experiment_manager import experiment_manager
from voice.text_to_speech.tts import tts_service
from event_logging.event_logger import event_logger

class AethonCommandService:
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = [
            {
                "id": 1,
                "role": "assistant",
                "speaker": "Aethon",
                "time": datetime.now().strftime("%I:%M %p"),
                "text": "AETHON online. Ready for Payload Assembly experiment. Say 'Start the experiment' or press Start."
            }
        ]

    def handle_command(self, raw_text: str, current_perception: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        time_str = datetime.now().strftime("%I:%M %p")
        # Add user message
        user_msg = {
            "id": len(self.conversation_history) + 1,
            "role": "user",
            "speaker": "You",
            "time": time_str,
            "text": raw_text
        }
        self.conversation_history.append(user_msg)
        event_logger.log("VOICE_COMMAND", raw_text, {"raw_text": raw_text})

        intent = command_parser.parse(raw_text)
        response_text = ""

        # Extract perception details
        action_dict = {}
        objects_list = []
        movement_str = "Stationary"
        posture_str = "Seated"
        narration_str = "Monitoring"

        if isinstance(current_perception, dict):
            if "current_action" in current_perception and isinstance(current_perception["current_action"], dict):
                action_dict = current_perception["current_action"]
                objects_list = current_perception.get("objects", [])
                movement_str = current_perception.get("movement", action_dict.get("movement", "Stationary"))
                posture_str = current_perception.get("posture", action_dict.get("posture", "Seated"))
                narration_str = current_perception.get("narration", action_dict.get("narration", "Monitoring"))
            else:
                action_dict = current_perception
                movement_str = action_dict.get("movement", "Stationary")
                posture_str = action_dict.get("posture", "Seated")
                narration_str = action_dict.get("narration", "Monitoring")

        if intent == Intent.START_EXPERIMENT:
            res = experiment_manager.start()
            response_text = "Experiment started.\nStep 1: Pick up Object A (Red Block)."
            tts_service.speak(response_text)
        elif intent == Intent.WHAT_AM_I_DOING:
            label = action_dict.get("label", "Idle / Monitoring")
            act_obj = action_dict.get("object", "")
            act_color = action_dict.get("color", "")
            
            if narration_str and narration_str not in ("Monitoring", "Standing", "Seated"):
                response_text = f"Action detected: {narration_str} Movement: {movement_str} ({posture_str})."
            elif act_obj:
                response_text = f"You are currently {label.lower()}. Interacting with {act_color + ' ' if act_color else ''}{act_obj} while {posture_str.lower()}."
            else:
                response_text = f"You are currently {posture_str.lower()} and {movement_str.lower()}. No active handheld item detected."
            tts_service.speak(response_text)
        elif intent == Intent.IDENTIFY_OBJECT:
            act_obj = action_dict.get("object")
            act_col = action_dict.get("color")
            if act_obj in (None, "None", "", "none"):
                act_obj = None
            if act_col in (None, "None", "", "none"):
                act_col = None
            held_objs = [o for o in objects_list if o.get("held")]
            
            if held_objs:
                held = held_objs[0]
                hand = held.get("held_by", "hand")
                response_text = f"You are holding a {held.get('display_name', held.get('label'))} in your {hand}."
            elif act_obj:
                response_text = f"Active object: {act_col + ' ' if act_col else ''}{act_obj}."
            elif objects_list:
                item_names = [
                    o.get("display_name") or f"{o.get('color', '')} {o.get('label', '')}".strip()
                    for o in objects_list
                    if (o.get("label") or "").lower() not in ("person", "astronaut")
                ]
                if item_names:
                    response_text = f"I detect: {', '.join(item_names[:4])} in the work area."
                else:
                    response_text = "Astronaut operator detected. No separate objects currently in the work area."
            else:
                response_text = "No distinct objects are currently identified in your hands or immediate view. Try holding up a daily item like a phone, bottle, cup, or block."
            tts_service.speak(response_text)
        elif intent == Intent.IDENTIFY_COLOR:
            act_col = action_dict.get("color")
            act_obj = action_dict.get("object")
            if act_obj in (None, "None", "", "none"):
                act_obj = None
            if act_col in (None, "None", "", "none"):
                act_col = None
            held_objs = [o for o in objects_list if o.get("held")]
            
            if held_objs:
                h = held_objs[0]
                response_text = f"The {h.get('label')} in your {h.get('held_by', 'hand')} is {h.get('color')}."
            elif act_col and act_obj:
                response_text = f"The {act_obj} is {act_col}."
            elif objects_list:
                colors_desc = []
                for o in objects_list:
                    lbl = o.get("raw_label") or o.get("label") or "item"
                    if lbl.lower() in ("person", "astronaut"):
                        continue
                    col = o.get("color")
                    if col and col not in ("Neutral", "Unknown"):
                        colors_desc.append(f"{lbl} is {col}")
                if colors_desc:
                    response_text = f"Identified colors: {', '.join(colors_desc[:4])}."
                else:
                    response_text = "Detected items have standard neutral coloration."
            else:
                response_text = "No objects detected. Hold an everyday item (such as your phone, bottle, or block) in front of the camera to inspect its color."
            tts_service.speak(response_text)
        elif intent == Intent.WHAT_MOVEMENTS:
            response_text = f"Movement tracked: {movement_str}. Posture: {posture_str}. Activity state: {action_dict.get('label', 'Idle')}."
            tts_service.speak(response_text)
        elif intent == Intent.NEXT_STEP:
            guidance = experiment_manager.get_guidance()
            response_text = guidance
            tts_service.speak(guidance)
        elif intent == Intent.CHECK_CURRENT_ACTION:
            check_msg = experiment_manager.check_current_action(action_dict)
            response_text = check_msg
            tts_service.speak(check_msg)
        elif intent == Intent.REPEAT_STEP:
            step_instruction = experiment_manager.get_current_step_instruction()
            response_text = f"Step {experiment_manager.current_step}: {step_instruction}."
            tts_service.speak(response_text)
        elif intent == Intent.PAUSE_EXPERIMENT:
            experiment_manager.pause()
            response_text = "Experiment paused."
        elif intent == Intent.RESUME_EXPERIMENT:
            experiment_manager.resume()
            response_text = f"Resumed. Step {experiment_manager.current_step}: {experiment_manager.get_current_step_instruction()}."
        elif intent == Intent.RESET_EXPERIMENT:
            experiment_manager.reset()
            response_text = "Experiment has been reset to Step 1."
        elif intent == Intent.STOP_EXPERIMENT:
            experiment_manager.stop()
            response_text = "Experiment stopped."
        elif intent == Intent.HELP:
            response_text = "You can ask: 'What am I doing?', 'What is this object?', 'What color is this?', 'What are my movements?', 'What is the next step?', 'Am I doing it right?', or control the experiment."
            tts_service.speak(response_text)
        else:
            # Fallback natural guidance
            response_text = f"Received: \"{raw_text}\". {experiment_manager.get_guidance()}"
            tts_service.speak(response_text)

        assistant_msg = {
            "id": len(self.conversation_history) + 1,
            "role": "assistant",
            "speaker": "Aethon",
            "time": datetime.now().strftime("%I:%M %p"),
            "text": response_text
        }
        self.conversation_history.append(assistant_msg)
        event_logger.log("AETHON_RESPONSE", response_text, {"intent": intent.value})

        return {
            "intent": intent.value,
            "response": response_text,
            "experiment_state": experiment_manager.get_state(),
            "conversation": self.conversation_history
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history

aethon_command_service = AethonCommandService()
