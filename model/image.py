from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class Image:
    path: str
    original_data: Optional[Any] = None
    processed_data: Optional[Any] = None
    width: Optional[int] = None
    height: Optional[int] = None
    scale: Optional[float] = None
    metadata: Optional[dict] = None