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
                "text": "AETHON online. Ready for Payload Assembly experiment. Say 'Hey AETHON' or press the mic button to interact."
            }
        ]
        self.last_active_time: float = 0.0
        self.last_wake_time: float = 0.0

    def is_active(self) -> bool:
        import time
        return (time.time() - self.last_active_time) < 4.5

    def is_in_active_window(self) -> bool:
        import time
        # 25-second active conversation window after being called
        return (time.time() - self.last_wake_time) < 25.0

    @staticmethod
    def _normalize_aethon_spelling(text: str) -> str:
        if not text:
            return text
        import re
        from voice.command_parser.command_parser import WAKE_TWO_WORD_VARIANTS, WAKE_SINGLE_WORD_VARIANTS
        cleaned = text.strip()
        # Repeated wake calls like "ethane than", "ethan ethan", "he tane tane tan"
        cleaned = re.sub(rf"^\s*(?:ethane|ethan|ae?thon|ae?than|athan|tane)\s+(?:than|then|ethan|ethane|ae?thon|ae?than|athan|tane|tan)\b[,.\s]*", "Hey AETHON, ", cleaned, flags=re.IGNORECASE)
        # Hey / Hi / Hello / He / A / Suno + variant (including thanks, then, aethan, ton, etc.)
        cleaned = re.sub(rf"^\s*(?:hey|hi|hello|he|a|ey|ay|suno)\s+(?:{WAKE_TWO_WORD_VARIANTS})\b[,\s.]*", "Hey AETHON, ", cleaned, flags=re.IGNORECASE)
        # Okay / Ok + variant
        cleaned = re.sub(rf"^\s*okay?\s+(?:{WAKE_TWO_WORD_VARIANTS})\b[,\s.]*", "OK AETHON, ", cleaned, flags=re.IGNORECASE)
        # hetan -> Hey AETHON
        cleaned = re.sub(r"^\s*hetan\b[,\s.]*", "Hey AETHON, ", cleaned, flags=re.IGNORECASE)
        # Single word fast call
        cleaned = re.sub(rf"^\s*(?:{WAKE_SINGLE_WORD_VARIANTS})\b[,\s.]*", "AETHON, ", cleaned, flags=re.IGNORECASE)
        # Clean trailing commas if nothing follows
        cleaned = re.sub(r"^Hey AETHON,\s*$", "Hey AETHON", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^OK AETHON,\s*$", "OK AETHON", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^AETHON,\s*$", "AETHON", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def handle_command(self, raw_text: str, current_perception: Optional[Dict[str, Any]] = None, source: str = "web") -> Dict[str, Any]:
        if not raw_text or not raw_text.strip():
            return {
                "intent": "IGNORE",
                "response": "",
                "experiment_state": experiment_manager.get_state(),
                "conversation": self.conversation_history
            }

        import time
        from voice.command_parser.command_parser import has_wake_word
        has_wake = has_wake_word(raw_text)
        in_window = self.is_in_active_window()

        # Parse intent early to check if user spoke an experiment command
        intent = command_parser.parse(raw_text)

        # For physical microphone:
        # 1. If wake word present -> wake up & respond!
        # 2. If in active conversation window AND recognized an intent -> respond to follow-up!
        # 3. Otherwise -> ignore background room chatter
        if source == "hardware_mic":
            if not has_wake and not in_window:
                return {
                    "intent": "IGNORE",
                    "response": "",
                    "experiment_state": experiment_manager.get_state(),
                    "conversation": self.conversation_history
                }
            if not has_wake and in_window and intent == Intent.UNKNOWN:
                # Check for explicit question / command words before giving guidance
                has_q = any(q in raw_text.lower() for q in [
                    "what", "how", "am i", "start", "stop", "reset", "pause", "step", "color", "colour",
                    "action", "doing", "holding", "repeat", "help", "status", "who", "next", "which",
                    "procedure", "instruction", "guide", "tell", "explain", "check", "verify"
                ])
                if not has_q:
                    return {
                        "intent": "IGNORE",
                        "response": "",
                        "experiment_state": experiment_manager.get_state(),
                        "conversation": self.conversation_history
                    }

        now_t = time.time()
        self.last_active_time = now_t
        if has_wake or in_window:
            self.last_wake_time = now_t

        time_str = datetime.now().strftime("%I:%M %p")
        # Normalize spelling of AETHON in user bubble so chat always displays "AETHON"
        clean_user_text = self._normalize_aethon_spelling(raw_text)

        # Deduplicate simultaneous duplicate commands across web and hardware mic
        if hasattr(self, "_last_handled_text") and hasattr(self, "_last_handled_time") and hasattr(self, "_last_handled_source"):
            if (
                clean_user_text.lower() == self._last_handled_text.lower()
                and (now_t - self._last_handled_time) < 1.5
                and (source != self._last_handled_source or raw_text.lower() == getattr(self, "_last_handled_raw", "").lower())
            ):
                return {
                    "intent": "DUPLICATE",
                    "response": getattr(self, "_last_handled_response", ""),
                    "experiment_state": experiment_manager.get_state(),
                    "conversation": self.conversation_history
                }
        self._last_handled_text = clean_user_text
        self._last_handled_raw = raw_text
        self._last_handled_source = source
        self._last_handled_time = now_t

        # Add user message
        user_msg = {
            "id": len(self.conversation_history) + 1,
            "role": "user",
            "speaker": "You",
            "time": time_str,
            "text": clean_user_text,
            "source": source
        }
        self.conversation_history.append(user_msg)
        event_logger.log("VOICE_COMMAND", clean_user_text, {"raw_text": raw_text, "source": source})

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

        if intent == Intent.GREETING:
            lower_text = raw_text.lower()
            if any(w in lower_text for w in ["who are you", "what is your name", "what are you", "introduce yourself", "about yourself"]):
                response_text = (
                    "I am AETHON, your AI Mission Control Assistant. "
                    "I monitor payload assembly experiments, track object interactions, "
                    "and ensure step-by-step procedural safety."
                )
            elif any(w in lower_text for w in ["how are you", "how are things", "how's it going", "how it going", "what's up", "whats up", "kaise ho", "kya haal", "kya chal"]):
                response_text = (
                    "All telemetry systems are nominal and operating smoothly. "
                    "Ready to monitor your experiment, Commander."
                )
            elif any(w in lower_text for w in ["namaste", "namaskar"]):
                response_text = "Namaste, Commander. AETHON is online and ready for your experiment."
            elif any(w in lower_text for w in ["thank", "thanks", "appreciate", "good job", "great job"]):
                response_text = "You're welcome, Commander. Standing by for your next instruction."
            elif any(w in lower_text for w in ["are you there", "can you hear", "are you listening", "are you ready"]):
                response_text = "Yes, Commander, I can hear you clearly and all perception systems are active."
            elif any(w in lower_text for w in ["bye", "goodbye", "see you", "good night", "standby"]):
                response_text = "Acknowledged. Standing by in monitoring mode. Have a safe mission, Commander."
                self.last_wake_time = 0.0  # Immediately return to standby
            else:
                response_text = "Hello, Commander. AETHON is online and ready. How can I assist you with your experiment today?"
        elif intent == Intent.STATUS:
            exp_state = experiment_manager.get_state()
            exp_status = exp_state.get("status", "IDLE")
            person_det = current_perception.get("person_detected", False) if current_perception else False
            obj_count = len([o for o in objects_list if (o.get("raw_label") or "").lower() != "person"])
            fps = current_perception.get("fps", 0) if current_perception else 0
            response_text = (
                f"AETHON status report: All systems nominal. "
                f"Experiment status: {exp_status}. "
                f"{'Person detected' if person_det else 'No person in view'}. "
                f"{obj_count} object{'s' if obj_count != 1 else ''} tracked. "
                f"Perception running at {fps} FPS."
            )
        elif intent == Intent.START_EXPERIMENT:
            res = experiment_manager.start()
            response_text = "Experiment started.\nStep 1: Pick up Object A (Red Block)."
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
        elif intent == Intent.WHAT_MOVEMENTS:
            response_text = f"Movement tracked: {movement_str}. Posture: {posture_str}. Activity state: {action_dict.get('label', 'Idle')}."
        elif intent == Intent.NEXT_STEP:
            guidance = experiment_manager.get_guidance()
            response_text = guidance
        elif intent == Intent.CHECK_CURRENT_ACTION:
            check_msg = experiment_manager.check_current_action(action_dict)
            response_text = check_msg
        elif intent == Intent.REPEAT_STEP:
            step_instruction = experiment_manager.get_current_step_instruction()
            response_text = f"Step {experiment_manager.current_step}: {step_instruction}."
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
            response_text = (
                "You can say: 'Hey AETHON, what am I doing?', 'What is this object?', "
                "'What color is this?', 'What are my movements?', 'What is the next step?', "
                "'Am I doing it right?', 'Status report', or control the experiment with "
                "'Start', 'Pause', 'Resume', 'Reset'."
            )
        elif intent == Intent.MUTE:
            response_text = "Voice feedback muted."
            tts_service.muted = True
        else:
            # Unrelated question or statement addressed to Aethon
            response_text = (
                "I am focused on your experiment mission. Please ask questions related to the experiment, "
                "assembly procedures, or object tracking. You can say 'Help' for available commands."
            )

        if intent != Intent.MUTE:
            tts_service.muted = False

        self._last_handled_response = response_text

        # Output speech via hardware TTS only for standalone non-web commands
        # (Web client voices responses using high-quality natural neural browser speech)
        if response_text and not tts_service.muted and source != "web":
            tts_service.speak(response_text)

        assistant_msg = {
            "id": len(self.conversation_history) + 1,
            "role": "assistant",
            "speaker": "Aethon",
            "time": datetime.now().strftime("%I:%M %p"),
            "text": response_text,
            "source": source
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
