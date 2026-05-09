from dataclasses import dataclass
from typing import Optional


@dataclass
class Hit:
    x: float
    y: float
    distance_from_center: Optional[float] = None
    angle: Optional[float] = None
    ring: Optional[int] = None
    valid: bool = True
    confidence: Optional[float] = None
