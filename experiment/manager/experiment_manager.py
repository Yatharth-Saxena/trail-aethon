import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from experiment.state_machine.experiment_state import (
    ExperimentStatus,
    ValidationStatus,
    ExperimentEvent,
    StepState,
    ValidationResult
)
from experiment.validator.sequence_validator import SequenceValidator
from event_logging.event_logger import event_logger
from data.session_manager import session_manager
from voice.text_to_speech.tts import tts_service
from backend.config import CONFIGS_DIR

class ExperimentManager:
    def __init__(self, config_name: str = "payload_assembly_demo.json"):
        self.config_path = CONFIGS_DIR / config_name
        self.config_data = self._load_config()
        self.status = ExperimentStatus.IDLE
        self.current_step = 1
        self.validator = SequenceValidator(self.config_data.get("steps", []))
        self.steps_state: List[Dict[str, Any]] = self._init_steps_state()
        self.last_validation_result: Optional[ValidationResult] = None
        self.state_listeners: List[Callable[[Dict[str, Any]], None]] = []

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ExperimentManager Error] Failed to load config: {e}")
            return {"id": "default", "name": "Payload Assembly (Demo)", "steps": []}

    def _init_steps_state(self) -> List[Dict[str, Any]]:
        states = []
        for step in self.config_data.get("steps", []):
            states.append({
                "number": step["step_number"],
                "label": step["label"],
                "instruction": step["instruction"],
                "status": "pending", # "pending", "current", "complete"
                "meta": "Pending",
                "complete": False,
                "current": False,
                "completed_at": None,
                "expected_action": step.get("expected_action"),
                "expected_object": step.get("expected_object"),
                "expected_target": step.get("expected_target"),
                "guidance": step.get("guidance"),
                "success_announcement": step.get("success_announcement")
            })
        return states

    def add_listener(self, callback: Callable[[Dict[str, Any]], None]):
        self.state_listeners.append(callback)

    def _notify(self):
        state = self.get_state()
        for listener in self.state_listeners:
            try:
                listener(state)
            except Exception as e:
                print(f"[ExperimentManager Error] in listener: {e}")

    def get_progress_percentage(self) -> int:
        total = len(self.steps_state)
        if total == 0:
            return 0
        if self.status == ExperimentStatus.COMPLETED:
            return 100
        completed_count = sum(1 for s in self.steps_state if s["complete"])
        return int((completed_count / total) * 100)

    def get_state(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.config_data.get("id"),
            "experiment_name": self.config_data.get("name"),
            "status": self.status.value,
            "running": self.status == ExperimentStatus.RUNNING,
            "current_step": self.current_step,
            "total_steps": len(self.steps_state),
            "progress_percentage": self.get_progress_percentage(),
            "steps": self.steps_state,
            "next_step_label": self.get_current_step_instruction(),
            "last_validation": self.last_validation_result.model_dump() if self.last_validation_result else None
        }

    def get_current_step_instruction(self) -> str:
        if self.status == ExperimentStatus.COMPLETED:
            return "Experiment completed successfully."
        if 1 <= self.current_step <= len(self.steps_state):
            return self.steps_state[self.current_step - 1]["label"]
        return "No active step."

    def start(self) -> Dict[str, Any]:
        self.status = ExperimentStatus.RUNNING
        session_id = session_manager.create_new_session(self.config_data.get("id"))
        event_logger.set_session(session_id)
        
        # Reset steps
        self.steps_state = self._init_steps_state()
        self.current_step = 1
        if self.steps_state:
            self.steps_state[0]["status"] = "current"
            self.steps_state[0]["current"] = True
            self.steps_state[0]["meta"] = "In Progress..."

        msg = "Experiment started. Please pick up Object A."
        tts_service.speak(msg)
        event_logger.log("EXPERIMENT_STARTED", msg, {"session_id": session_id})
        self._notify()
        return {"status": "success", "message": msg, "state": self.get_state()}

    def pause(self) -> Dict[str, Any]:
        if self.status == ExperimentStatus.RUNNING:
            self.status = ExperimentStatus.PAUSED
            msg = "Experiment paused."
            tts_service.speak(msg)
            event_logger.log("EXPERIMENT_PAUSED", msg)
            self._notify()
            return {"status": "success", "message": msg, "state": self.get_state()}
        return {"status": "noop", "state": self.get_state()}

    def resume(self) -> Dict[str, Any]:
        if self.status == ExperimentStatus.PAUSED:
            self.status = ExperimentStatus.RUNNING
            msg = f"Experiment resumed. Step {self.current_step}: {self.get_current_step_instruction()}."
            tts_service.speak(msg)
            event_logger.log("EXPERIMENT_RESUMED", msg)
            self._notify()
            return {"status": "success", "message": msg, "state": self.get_state()}
        return {"status": "noop", "state": self.get_state()}

    def reset(self) -> Dict[str, Any]:
        self.status = ExperimentStatus.IDLE
        self.current_step = 1
        self.steps_state = self._init_steps_state()
        self.last_validation_result = None
        msg = "Experiment reset."
        tts_service.speak(msg)
        event_logger.log("EXPERIMENT_RESET", msg)
        self._notify()
        return {"status": "success", "message": msg, "state": self.get_state()}

    def stop(self) -> Dict[str, Any]:
        self.status = ExperimentStatus.STOPPED
        msg = "Experiment ended."
        tts_service.speak(msg)
        event_logger.log("EXPERIMENT_STOPPED", msg)
        self._notify()
        return {"status": "success", "message": msg, "state": self.get_state()}

    def process_event(self, event: ExperimentEvent) -> ValidationResult:
        if self.status != ExperimentStatus.RUNNING:
            return ValidationResult(
                status=ValidationStatus.NO_ACTION,
                message=f"Experiment is currently {self.status.value}. Actions are not evaluated.",
                advance_step=False,
                current_step=self.current_step,
                expected={},
                detected=event.model_dump()
            )

        result = self.validator.validate(self.current_step, event)
        self.last_validation_result = result

        if result.advance_step:
            step_idx = self.current_step - 1
            now_str = datetime.now().strftime("%H:%M:%S")
            self.steps_state[step_idx]["complete"] = True
            self.steps_state[step_idx]["current"] = False
            self.steps_state[step_idx]["status"] = "complete"
            self.steps_state[step_idx]["meta"] = f"Completed  ·  {now_str}"
            self.steps_state[step_idx]["completed_at"] = now_str

            event_logger.log(
                "STEP_COMPLETED",
                f"Step {self.current_step} completed: {self.steps_state[step_idx]['label']}",
                {"step": self.current_step, "event": event.model_dump()}
            )

            self.current_step += 1
            if self.current_step > len(self.steps_state):
                self.status = ExperimentStatus.COMPLETED
                compl_msg = "Experiment completed successfully. All steps verified."
                tts_service.speak(compl_msg)
                event_logger.log("EXPERIMENT_COMPLETED", compl_msg)
            else:
                next_idx = self.current_step - 1
                self.steps_state[next_idx]["current"] = True
                self.steps_state[next_idx]["status"] = "current"
                self.steps_state[next_idx]["meta"] = "In Progress..."
                if result.voice_alert:
                    tts_service.speak(result.voice_alert)
        else:
            # Step was skipped, out of order, or wrong object
            if result.status in [ValidationStatus.STEP_SKIPPED, ValidationStatus.OUT_OF_ORDER, ValidationStatus.WRONG_OBJECT]:
                if result.voice_alert:
                    tts_service.speak(result.voice_alert)
                event_logger.log(
                    "VIOLATION",
                    result.message,
                    {"violation_type": result.status.value, "detected": event.model_dump()},
                    level="WARN"
                )

        self._notify()
        return result

    def get_guidance(self) -> str:
        if self.status == ExperimentStatus.COMPLETED:
            return "The experiment is completed. You can review the logs or reset the experiment."
        if self.status != ExperimentStatus.RUNNING:
            return "The experiment is currently paused or stopped. Press Start to proceed."
        step_idx = self.current_step - 1
        if 0 <= step_idx < len(self.steps_state):
            step = self.steps_state[step_idx]
            return f"The next step is: {step['label']}. {step['guidance']}"
        return "No further steps."

    def check_current_action(self, current_action_summary: Optional[Dict[str, Any]] = None) -> str:
        if self.status != ExperimentStatus.RUNNING:
            return f"The experiment is currently {self.status.value}. Start the experiment to begin validation."
        step_idx = self.current_step - 1
        step = self.steps_state[step_idx]
        if current_action_summary and current_action_summary.get("action"):
            act = current_action_summary.get("action")
            obj = current_action_summary.get("object", "")
            return f"You're on step {self.current_step}: {step['label']}. Current action detected is {act} on {obj}."
        return f"You're on step {self.current_step}. {step['guidance']}"

experiment_manager = ExperimentManager()
