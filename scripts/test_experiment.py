"""
AETHON - Experiment Engine Diagnostics
Tests starting, advancing steps, simulating violations, and completing experiment.
"""
from experiment.manager.experiment_manager import experiment_manager
from experiment.state_machine.experiment_state import ExperimentEvent
import time

def test():
    print("[Experiment Test] Initializing Payload Assembly...")
    experiment_manager.reset()
    assert experiment_manager.status.value == "IDLE"

    print("[Experiment Test] Starting experiment...")
    res = experiment_manager.start()
    assert experiment_manager.status.value == "RUNNING"
    assert experiment_manager.current_step == 1
    print(f"[PASS] Experiment started: Step {experiment_manager.current_step}")

    # Step 1: PICK_UP Object A
    e1 = ExperimentEvent(event="PICK_UP", object="Object A", confidence=0.94)
    r1 = experiment_manager.process_event(e1)
    print(f"[PASS] Event 1 processed: {r1.status.value} -> Advance: {r1.advance_step}, Current Step: {experiment_manager.current_step}")
    assert experiment_manager.current_step == 2

    # Violation test: Skipped step (send Step 4 action while on Step 2)
    e_skip = ExperimentEvent(event="PLACE", object="Object A", target="Tray", confidence=0.90)
    r_skip = experiment_manager.process_event(e_skip)
    print(f"[PASS] Violation test (Skip): {r_skip.status.value} -> Advance: {r_skip.advance_step}, Current Step: {experiment_manager.current_step}")
    assert r_skip.status.value == "OUT_OF_ORDER" or r_skip.status.value == "STEP_SKIPPED"
    assert experiment_manager.current_step == 2 # Did NOT advance!

    # Step 2: PLACE Object A on Object B
    e2 = ExperimentEvent(event="PLACE", object="Object A", target="Object B", confidence=0.92)
    r2 = experiment_manager.process_event(e2)
    print(f"[PASS] Event 2 processed: {r2.status.value}, Current Step: {experiment_manager.current_step}")
    assert experiment_manager.current_step == 3

    # Step 3: PICK_UP Object A
    e3 = ExperimentEvent(event="PICK_UP", object="Object A", confidence=0.91)
    r3 = experiment_manager.process_event(e3)
    assert experiment_manager.current_step == 4

    # Step 4: PLACE Object A on Tray
    e4 = ExperimentEvent(event="PLACE", object="Object A", target="Tray", confidence=0.89)
    r4 = experiment_manager.process_event(e4)
    assert experiment_manager.current_step == 5

    # Step 5: PRESS Complete Button
    e5 = ExperimentEvent(event="PRESS", object="Complete Button", confidence=0.96)
    r5 = experiment_manager.process_event(e5)
    print(f"[PASS] Event 5 processed: {r5.status.value}, Status: {experiment_manager.status.value}")
    assert experiment_manager.status.value == "COMPLETED"
    print("[ALL PASS] Experiment test completed successfully!")

if __name__ == "__main__":
    test()
