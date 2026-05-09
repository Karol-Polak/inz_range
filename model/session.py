from dataclasses import dataclass
from typing import List, Optional
from .image import Image
from .target import Target
from .hit import Hit


@dataclass
class Session:
    id: Optional[int]
    image: Image
    target: Target
    hits: List[Hit]
    statistics: dict
    metadata: dict
