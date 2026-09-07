import json
import pytest
from pathlib import Path
from experiment.validator.sequence_validator import SequenceValidator
from experiment.state_machine.experiment_state import ExperimentEvent, ValidationStatus

CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiment" / "configs" / "payload_assembly_demo.json"

@pytest.fixture
def validator():
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    return SequenceValidator(data["steps"])

def test_correct_sequence(validator):
    # Step 1: PICK_UP Object A
    e1 = ExperimentEvent(event="PICK_UP", object="Object A", confidence=0.95)
    r1 = validator.validate(1, e1)
    assert r1.status == ValidationStatus.CORRECT
    assert r1.advance_step is True

    # Step 2: PLACE Object A on Object B
    e2 = ExperimentEvent(event="PLACE", object="Object A", target="Object B", confidence=0.92)
    r2 = validator.validate(2, e2)
    assert r2.status == ValidationStatus.CORRECT
    assert r2.advance_step is True

    # Step 3: PICK_UP Object A again
    e3 = ExperimentEvent(event="PICK_UP", object="Object A", confidence=0.91)
    r3 = validator.validate(3, e3)
    assert r3.status == ValidationStatus.CORRECT
    assert r3.advance_step is True

    # Step 4: PLACE Object A in Tray
    e4 = ExperimentEvent(event="PLACE", object="Object A", target="Tray", confidence=0.88)
    r4 = validator.validate(4, e4)
    assert r4.status == ValidationStatus.CORRECT
    assert r4.advance_step is True

    # Step 5: PRESS Complete Button
    e5 = ExperimentEvent(event="PRESS", object="Complete Button", confidence=0.97)
    r5 = validator.validate(5, e5)
    assert r5.status == ValidationStatus.CORRECT
    assert r5.advance_step is True

def test_skipped_step(validator):
    # Currently at step 3, but user performs step 4: PLACE Object A on Tray
    e = ExperimentEvent(event="PLACE", object="Object A", target="Tray", confidence=0.90)
    res = validator.validate(3, e)
    assert res.status == ValidationStatus.STEP_SKIPPED
    assert res.advance_step is False
    assert "skipped" in res.message.lower()

def test_out_of_order(validator):
    # Currently at step 1 or 2, but user presses Complete Button (step 5)
    e = ExperimentEvent(event="PRESS", object="Complete Button", confidence=0.95)
    res = validator.validate(2, e)
    assert res.status == ValidationStatus.OUT_OF_ORDER
    assert res.advance_step is False
    assert "out of sequence" in res.message.lower()

def test_wrong_object(validator):
    # Currently at step 1 (expected PICK_UP Object A), but user picks up Object B
    e = ExperimentEvent(event="PICK_UP", object="Object B", confidence=0.91)
    res = validator.validate(1, e)
    assert res.status == ValidationStatus.WRONG_OBJECT
    assert res.advance_step is False
    assert "incorrect object" in res.message.lower()
