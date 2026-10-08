from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class TrainingSession(SQLModel, table=True):
    __tablename__ = "training_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    image_path: str

    hit_count: int

    cep_50: float
    mean_radius: float
    extreme_spread: float
    std_radius: float

    mean_x: float
    mean_y: float

    outliers_count: int

    weapon_type: Optional[str] = None
    distance_m: Optional[float] = None

    notes: Optional[str] = None