from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Target:
    center_x: float
    center_y: float
    radius: float
    rings: Optional[List[float]] = None
    type: str = "circular"
    confidence: Optional[float] = None
