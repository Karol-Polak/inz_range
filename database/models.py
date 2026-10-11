from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class TrainingSession(SQLModel, table=True):
    __tablename__ = "training_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    image_path: str

    target_center_x: float
    target_center_y: float
    target_radius: float
    target_type: str

    hit_count: int

    accuracy_radius: float
    precision_radius: float
    precision_std: float
    cep_50: float
    extreme_spread: float
    outliers_count: int

    mean_x: float
    mean_y: float

    # Przesunięcie punktu celowania od target_center_x/y (patrz model/target.py).
    # None = celowanie w wykryty środek (stare sesje, zapisane przed tą kolumną,
    # mają tu NULL i zachowują się tak jak dawniej).
    aim_offset_x: Optional[float] = None
    aim_offset_y: Optional[float] = None

    scale_mm_per_px: Optional[float] = None
    weapon_type: Optional[str] = None
    distance_m: Optional[float] = None

    notes: Optional[str] = None


class HitRecord(SQLModel, table=True):
    __tablename__ = "hit_records"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="training_sessions.id", index=True)

    x: float
    y: float
    distance_from_center: Optional[float] = None
    valid: bool = True
    confidence: Optional[float] = None
    source: str = "auto"  # "auto" (detection) or "manual" (user-added/corrected)
