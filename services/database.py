from typing import List
from model.session import Session


def save_session(session: Session) -> None:
    raise NotImplementedError


def load_session(session_id: int) -> Session:
    raise NotImplementedError


def list_sessions() -> List[Session]:
    raise NotImplementedError
