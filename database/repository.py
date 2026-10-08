from __future__ import annotations

from typing import List, Optional

from sqlmodel import Session as DbSession, select

from database.db import engine as default_engine
from database.models import HitRecord, TrainingSession
from model.hit import Hit
from model.session import Session
from model.target import Target
from services.image_loader import load_image
from services.statistics import calculate_statistics


def save_session(
    session: Session,
    weapon_type: Optional[str] = None,
    distance_m: Optional[float] = None,
    scale_mm_per_px: Optional[float] = None,
    notes: Optional[str] = None,
    engine=None,
) -> TrainingSession:
    """
    Zapisuje sesję treningową (statystyki + każde trafienie) w bazie danych.
    Zakłada, że session.statistics jest aktualne (np. po ręcznej korekcie trafień).
    """

    engine = engine or default_engine
    stats = session.statistics
    target = session.target

    training_session = TrainingSession(
        image_path=session.image.path,
        target_center_x=target.center_x,
        target_center_y=target.center_y,
        target_radius=target.radius,
        target_type=target.type,
        hit_count=stats.get("count", len(session.hits)),
        accuracy_radius=stats.get("accuracy_radius", 0.0),
        precision_radius=stats.get("precision_radius", 0.0),
        precision_std=stats.get("precision_std", 0.0),
        cep_50=stats.get("cep_50", 0.0),
        extreme_spread=stats.get("extreme_spread", 0.0),
        outliers_count=stats.get("outliers_count", 0),
        mean_x=stats.get("mean_x", 0.0),
        mean_y=stats.get("mean_y", 0.0),
        scale_mm_per_px=scale_mm_per_px,
        weapon_type=weapon_type,
        distance_m=distance_m,
        notes=notes,
    )

    with DbSession(engine) as db_session:
        db_session.add(training_session)
        db_session.commit()
        db_session.refresh(training_session)

        for hit in session.hits:
            db_session.add(
                HitRecord(
                    session_id=training_session.id,
                    x=hit.x,
                    y=hit.y,
                    distance_from_center=hit.distance_from_center,
                    valid=hit.valid,
                    confidence=hit.confidence,
                    source=hit.source,
                )
            )

        db_session.commit()
        db_session.refresh(training_session)

        return training_session


def load_session(session_id: int, engine=None) -> Optional[Session]:
    """
    Wczytuje zapisaną sesję i odtwarza ją jako obiekt Session.
    Statystyki są przeliczane na nowo na podstawie zapisanych trafień
    (a nie po prostu kopiowane z zapisanych agregatów).
    """

    engine = engine or default_engine

    with DbSession(engine) as db_session:
        training_session = db_session.get(TrainingSession, session_id)

        if training_session is None:
            return None

        hit_records = list(
            db_session.exec(
                select(HitRecord).where(HitRecord.session_id == session_id)
            )
        )

    hits = [
        Hit(
            x=record.x,
            y=record.y,
            distance_from_center=record.distance_from_center,
            valid=record.valid,
            confidence=record.confidence,
            source=record.source,
        )
        for record in hit_records
    ]

    target = Target(
        center_x=training_session.target_center_x,
        center_y=training_session.target_center_y,
        radius=training_session.target_radius,
        type=training_session.target_type,
    )

    image = load_image(training_session.image_path)
    statistics = calculate_statistics(hits)

    return Session(
        id=training_session.id,
        image=image,
        target=target,
        hits=hits,
        statistics=statistics,
        metadata={
            "weapon_type": training_session.weapon_type,
            "distance_m": training_session.distance_m,
            "scale_mm_per_px": training_session.scale_mm_per_px,
            "notes": training_session.notes,
        },
    )


def list_sessions(engine=None) -> List[TrainingSession]:
    engine = engine or default_engine

    with DbSession(engine) as db_session:
        statement = select(TrainingSession).order_by(TrainingSession.created_at.desc())
        return list(db_session.exec(statement))


def delete_session(session_id: int, engine=None) -> bool:
    engine = engine or default_engine

    with DbSession(engine) as db_session:
        training_session = db_session.get(TrainingSession, session_id)

        if training_session is None:
            return False

        hit_records = db_session.exec(
            select(HitRecord).where(HitRecord.session_id == session_id)
        )
        for record in hit_records:
            db_session.delete(record)

        db_session.delete(training_session)
        db_session.commit()

        return True
