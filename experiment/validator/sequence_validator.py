from typing import List, Dict, Any, Optional
from experiment.state_machine.experiment_state import (
    ExperimentEvent,
    ValidationResult,
    ValidationStatus,
    StepState
)

class SequenceValidator:
    def __init__(self, steps_config: List[Dict[str, Any]]):
        self.steps = steps_config

    def _normalize(self, val: Optional[str]) -> str:
        if not val:
            return ""
        return val.strip().lower().replace("_", " ")

    def _matches_step(self, step: Dict[str, Any], event: ExperimentEvent) -> bool:
        # Check action
        action_match = self._normalize(step.get("expected_action")) == self._normalize(event.event)
        
        # Check primary object
        step_obj = self._normalize(step.get("expected_object"))
        event_obj = self._normalize(event.object)
        obj_match = (step_obj in event_obj) or (event_obj in step_obj)
        
        # Check target if step expects one
        expected_target = step.get("expected_target")
        if expected_target:
            step_target = self._normalize(expected_target)
            event_target = self._normalize(event.target)
            target_match = (step_target in event_target) or (event_target in step_target)
        else:
            target_match = True

        return action_match and obj_match and target_match

    def validate(self, current_step_num: int, event: ExperimentEvent) -> ValidationResult:
        if current_step_num < 1 or current_step_num > len(self.steps):
            return ValidationResult(
                status=ValidationStatus.NO_ACTION,
                message=f"Experiment is not at an active step (step {current_step_num})",
                advance_step=False,
                current_step=current_step_num,
                expected={},
                detected=event.model_dump()
            )

        current_step = self.steps[current_step_num - 1]
        expected_dict = {
            "action": current_step.get("expected_action"),
            "object": current_step.get("expected_object"),
            "target": current_step.get("expected_target")
        }

        # 1. Check if it matches current step exactly
        if self._matches_step(current_step, event):
            success_msg = current_step.get("success_announcement") or f"Step {current_step_num} completed."
            return ValidationResult(
                status=ValidationStatus.CORRECT,
                message=f"Step {current_step_num} successfully verified: {current_step.get('label')}",
                voice_alert=success_msg,
                advance_step=True,
                current_step=current_step_num,
                expected=expected_dict,
                detected=event.model_dump()
            )

        # 2. Check if this matches a subsequent step (Skipped or Out of Order)
        for future_step in self.steps[current_step_num:]:
            if self._matches_step(future_step, event):
                future_num = future_step.get("step_number")
                if future_num == current_step_num + 1:
                    return ValidationResult(
                        status=ValidationStatus.STEP_SKIPPED,
                        message=f"Step {current_step_num} skipped. Detected step {future_num} action.",
                        voice_alert=f"Step {current_step_num} was skipped. {current_step.get('guidance')}",
                        advance_step=False,
                        current_step=current_step_num,
                        expected=expected_dict,
                        detected=event.model_dump()
                    )
                else:
                    return ValidationResult(
                        status=ValidationStatus.OUT_OF_ORDER,
                        message=f"Out of sequence action. Detected step {future_num} action while on step {current_step_num}.",
                        voice_alert=f"That action is out of sequence. The current step is to {current_step.get('label').lower()}.",
                        advance_step=False,
                        current_step=current_step_num,
                        expected=expected_dict,
                        detected=event.model_dump()
                    )

        # 3. Check if the user interacted with the wrong object for the current step's action
        step_action = self._normalize(current_step.get("expected_action"))
        event_action = self._normalize(event.event)
        if step_action == event_action:
            step_obj = current_step.get("expected_object")
            return ValidationResult(
                status=ValidationStatus.WRONG_OBJECT,
                message=f"Incorrect object detected ({event.object}). Expected {step_obj}.",
                voice_alert=f"Incorrect object. Please {step_action} {step_obj}.",
                advance_step=False,
                current_step=current_step_num,
                expected=expected_dict,
                detected=event.model_dump()
            )

        # 4. If action occurred on an irrelevant object or unknown action
        return ValidationResult(
            status=ValidationStatus.UNKNOWN_ACTION,
            message=f"Action '{event.event}' on '{event.object}' does not correspond to the current task.",
            voice_alert=f"Unrecognized step. Current step: {current_step.get('label')}.",
            advance_step=False,
            current_step=current_step_num,
            expected=expected_dict,
            detected=event.model_dump()
        )
