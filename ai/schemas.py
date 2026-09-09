from typing import Literal
from pydantic import BaseModel, Field


class ColorInfo(BaseModel):
    name: str = "Unknown"
    hex: str = "#888888"
    confidence: float = 0.0


class SizeEstimate(BaseModel):
    width_px: int = 0
    height_px: int = 0
    area_frac: float = 0.0


class ShapeInfo(BaseModel):
    aspect_ratio: float = 1.0
    orientation_deg: float = 0.0


class StabilityInfo(BaseModel):
    track_age_frames: int = 0
    confidence_ema: float = 0.0
    confirmed: bool = False


class ObjectAttributes(BaseModel):
    """
    Standardized attributes schema for tracked objects in AETHON perception.
    Guaranteed to be fully populated on every object without optional or missing keys.
    """
    color: ColorInfo = Field(default_factory=ColorInfo)
    size_estimate: SizeEstimate = Field(default_factory=SizeEstimate)
    shape: ShapeInfo = Field(default_factory=ShapeInfo)
    # Texture is a coarse variance-of-Laplacian heuristic, not ML material classification.
    texture: Literal["smooth", "textured", "reflective"] = "smooth"
    motion_state: Literal["STATIC", "MOVING", "HELD", "PREDICTED"] = "STATIC"
    stability: StabilityInfo = Field(default_factory=StabilityInfo)
