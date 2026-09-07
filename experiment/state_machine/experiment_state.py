from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import time

class ExperimentStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"

class ValidationStatus(str, Enum):
    CORRECT = "CORRECT"
    STEP_SKIPPED = "STEP_SKIPPED"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    WRONG_OBJECT = "WRONG_OBJECT"
    NO_ACTION = "NO_ACTION"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"

class ExperimentEvent(BaseModel):
    event: str  # e.g., "PICK_UP", "PLACE", "PRESS", "MOVE", "HOLD", "RELEASE"
    object: str # e.g., "Object A", "Object B", "Tray", "Complete Button"
    target: Optional[str] = None # e.g., "Object B", "Tray"
    actor: str = "person"
    hand: Optional[str] = "right" # "left", "right", "both"
    confidence: float = 0.90
    timestamp: float = Field(default_factory=time.time)
    source: str = "AI" # "AI" or "SIMULATION"

class StepState(BaseModel):
    number: int
    label: str
    instruction: str
    status: str = "pending" # "pending", "in_progress", "completed"
    completed_at: Optional[str] = None
    expected_action: str
    expected_object: str
    expected_target: Optional[str] = None

class ValidationResult(BaseModel):
    status: ValidationStatus
    message: str
    voice_alert: Optional[str] = None
    advance_step: bool = False
    current_step: int
    expected: Dict[str, Any]
    detected: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
