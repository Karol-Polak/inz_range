from __future__ import annotations

from typing import List, Optional

from sqlmodel import Session, select

from database.db import engine
from database.models import TrainingSession


def save_training_session(
    training_session: TrainingSession
) -> TrainingSession:
    with Session(engine) as session:
        session.add(training_session)
        session.commit()
        session.refresh(training_session)

        return training_session


def get_training_sessions() -> List[TrainingSession]:
    with Session(engine) as session:
        statement = (
            select(TrainingSession)
            .order_by(TrainingSession.created_at.desc())
        )

        return list(session.exec(statement))


def get_training_session_by_id(
    session_id: int
) -> Optional[TrainingSession]:
    with Session(engine) as session:
        return session.get(TrainingSession, session_id)


def delete_training_session(
    session_id: int
) -> bool:
    with Session(engine) as session:
        training_session = session.get(
            TrainingSession,
            session_id
        )

        if training_session is None:
            return False

        session.delete(training_session)
        session.commit()

        return True