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
    # Punkt celowania, jako przesunięcie (dx, dy) od (center_x, center_y).
    # None = celowanie w wykryty środek (domyślne zachowanie). Dla sylwetek
    # (np. IDPA) wykryty środek konturu nie musi być punktem celowania -
    # użytkownik może go ustawić ręcznie (patrz services/manual_correction.py).
    aim_offset_x: Optional[float] = None
    aim_offset_y: Optional[float] = None
