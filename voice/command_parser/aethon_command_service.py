from typing import Dict, Any, List, Optional
from datetime import datetime
from voice.command_parser.command_parser import command_parser, Intent
from experiment.manager.experiment_manager import experiment_manager
from voice.text_to_speech.tts import tts_service
from event_logging.event_logger import event_logger
from voice.llm.ollama_client import query_ollama, build_perception_context, is_ollama_available

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

        # Trigger experiment state actions & dashboard UI signals if keywords are present
        lower_cmd = raw_text.lower()
        action_signal = None

        if "start" in lower_cmd and "experiment" in lower_cmd:
            experiment_manager.start()
            action_signal = "START_EXPERIMENT"
        elif "pause" in lower_cmd and "experiment" in lower_cmd:
            experiment_manager.pause()
            action_signal = "PAUSE_EXPERIMENT"
        elif "resume" in lower_cmd and "experiment" in lower_cmd:
            experiment_manager.resume()
            action_signal = "RESUME_EXPERIMENT"
        elif "reset" in lower_cmd and "experiment" in lower_cmd:
            experiment_manager.reset()
            action_signal = "RESET_EXPERIMENT"
        elif "stop" in lower_cmd and "experiment" in lower_cmd:
            experiment_manager.stop()
            action_signal = "STOP_EXPERIMENT"
        elif "log" in lower_cmd or "logs" in lower_cmd:
            action_signal = "SHOW_LOGS"
        elif "monitor" in lower_cmd or "camera feed" in lower_cmd:
            action_signal = "SHOW_MONITOR"
        elif "step" in lower_cmd or "experiment view" in lower_cmd:
            action_signal = "SHOW_EXPERIMENT"
        elif "snapshot" in lower_cmd or "take photo" in lower_cmd or "picture" in lower_cmd:
            action_signal = "TAKE_SNAPSHOT"
        elif "start recording" in lower_cmd or "record video" in lower_cmd:
            action_signal = "RECORD_START"
        elif "stop recording" in lower_cmd:
            action_signal = "RECORD_STOP"

        # Check if user ONLY said the wake word ("AETHON" / "Ethan") without a command
        from voice.command_parser.command_parser import strip_wake_word
        cmd_only = strip_wake_word(clean_user_text).strip()

        if not cmd_only:
            response_text = "AETHON online. Standing by, Commander. How can I assist you with your mission?"
        else:
            # Query the trained Ollama Qwen2.5:3b model for ALL voice responses
            try:
                perception_ctx = build_perception_context(current_perception)
                llm_response = query_ollama(
                    user_message=clean_user_text,
                    context=perception_ctx,
                    conversation_history=self.conversation_history[-8:],
                    timeout=35,
                )
                response_text = llm_response
            except Exception as e:
                print(f"[AETHON LLM] Fallback triggered: {e}")
                response_text = (
                    "AETHON standing by, Commander. All perception and mission control systems remain active."
                )

        self._last_handled_response = response_text

        # Speak response via hardware TTS for standalone hardware mic commands
        if response_text and not tts_service.muted and source == "hardware_mic":
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
        event_logger.log("AETHON_RESPONSE", response_text, {"source": source, "action": action_signal})

        return {
            "intent": "LLM_RESPONSE",
            "action": action_signal,
            "response": response_text,
            "experiment_state": experiment_manager.get_state(),
            "conversation": self.conversation_history
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history

aethon_command_service = AethonCommandService()
