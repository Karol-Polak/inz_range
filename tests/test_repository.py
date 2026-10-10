import cv2
import numpy as np
import pytest
from sqlmodel import SQLModel, create_engine

from database.models import TrainingSession, HitRecord
from database.repository import (
    save_session,
    load_session,
    list_sessions,
    delete_session,
)
from model.hit import Hit
from model.image import Image
from model.target import Target
from model.session import Session
from services.statistics import calculate_statistics


@pytest.fixture
def engine():
    test_engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(test_engine)
    return test_engine


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "target.jpg"
    cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))
    return str(path)


def _build_session(image_path) -> Session:
    target = Target(center_x=25.0, center_y=25.0, radius=20.0, type="circular", confidence=1.0)
    hits = [
        Hit(x=1.0, y=0.0, distance_from_center=1.0, valid=True, confidence=0.9, source="auto"),
        Hit(x=0.0, y=-1.0, distance_from_center=1.0, valid=True, confidence=0.8, source="manual"),
    ]
    stats = calculate_statistics(hits)
    image = Image(path=image_path, original_data=np.zeros((50, 50, 3), dtype=np.uint8))

    return Session(id=None, image=image, target=target, hits=hits, statistics=stats, metadata={})


def test_save_session_persists_training_session_and_hit_records(engine, sample_image_path):
    session = _build_session(sample_image_path)

    saved = save_session(session, weapon_type="Glock 17", distance_m=10.0, engine=engine)

    assert saved.id is not None
    assert saved.hit_count == 2
    assert saved.weapon_type == "Glock 17"
    assert saved.distance_m == 10.0

    with_hits = list_sessions(engine=engine)
    assert len(with_hits) == 1


def test_save_session_persists_each_hit_with_its_source(engine, sample_image_path):
    session = _build_session(sample_image_path)

    saved = save_session(session, engine=engine)

    from sqlmodel import Session as DbSession, select

    with DbSession(engine) as db_session:
        records = list(
            db_session.exec(
                select(HitRecord).where(HitRecord.session_id == saved.id)
            )
        )

    sources = sorted(record.source for record in records)
    assert sources == ["auto", "manual"]


def test_load_session_reconstructs_session_and_recomputes_statistics(engine, sample_image_path):
    original = _build_session(sample_image_path)
    saved = save_session(original, engine=engine)

    loaded = load_session(saved.id, engine=engine)

    assert loaded is not None
    assert len(loaded.hits) == 2
    assert loaded.target.center_x == 25.0
    assert loaded.target.radius == 20.0
    # statistics must be recomputed from the stored hits, not just copied
    recomputed = calculate_statistics(loaded.hits)
    assert loaded.statistics["cep_50"] == recomputed["cep_50"]


def test_load_session_returns_none_for_unknown_id(engine):
    assert load_session(999, engine=engine) is None


def test_list_sessions_orders_newest_first(engine, sample_image_path):
    session = _build_session(sample_image_path)
    first = save_session(session, engine=engine)
    second = save_session(_build_session(sample_image_path), engine=engine)

    sessions = list_sessions(engine=engine)

    assert [s.id for s in sessions] == [second.id, first.id]


def test_delete_session_removes_session_and_its_hits(engine, sample_image_path):
    session = _build_session(sample_image_path)
    saved = save_session(session, engine=engine)

    deleted = delete_session(saved.id, engine=engine)

    assert deleted is True
    assert load_session(saved.id, engine=engine) is None

    from sqlmodel import Session as DbSession, select

    with DbSession(engine) as db_session:
        remaining_hits = list(
            db_session.exec(
                select(HitRecord).where(HitRecord.session_id == saved.id)
            )
        )
    assert remaining_hits == []


def test_delete_session_returns_false_for_unknown_id(engine):
    assert delete_session(999, engine=engine) is False
