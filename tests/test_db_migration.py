"""
Testuje migrację starej bazy (bez kolumn aim_offset_x/aim_offset_y) oraz to,
że punkt celowania przeżywa zapis + wczytanie sesji bez zmiany statystyk.
"""

import cv2
import numpy as np
import pytest
from sqlmodel import create_engine

from database.db import _migrate_legacy_cwd_db_if_present, _resolve_db_path, create_db
from database.repository import load_session, save_session
from model.hit import Hit
from model.image import Image
from model.session import Session
from model.target import Target
from services.manual_correction import toggle_hit_at
from services.statistics import calculate_statistics, recompute_statistics

OLD_SCHEMA_SQL = """
CREATE TABLE training_sessions (
    id INTEGER PRIMARY KEY,
    created_at DATETIME,
    image_path VARCHAR NOT NULL,
    target_center_x FLOAT NOT NULL,
    target_center_y FLOAT NOT NULL,
    target_radius FLOAT NOT NULL,
    target_type VARCHAR NOT NULL,
    hit_count INTEGER NOT NULL,
    accuracy_radius FLOAT NOT NULL,
    precision_radius FLOAT NOT NULL,
    precision_std FLOAT NOT NULL,
    cep_50 FLOAT NOT NULL,
    extreme_spread FLOAT NOT NULL,
    outliers_count INTEGER NOT NULL,
    mean_x FLOAT NOT NULL,
    mean_y FLOAT NOT NULL,
    scale_mm_per_px FLOAT,
    weapon_type VARCHAR,
    distance_m FLOAT,
    notes VARCHAR
);
CREATE TABLE hit_records (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    distance_from_center FLOAT,
    valid BOOLEAN NOT NULL,
    confidence FLOAT,
    source VARCHAR NOT NULL
);
"""


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "target.jpg"
    cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))
    return str(path)


def test_create_db_adds_aim_offset_columns_to_an_old_schema_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", echo=False)
    with engine.connect() as connection:
        for statement in OLD_SCHEMA_SQL.strip().split(";"):
            if statement.strip():
                connection.exec_driver_sql(statement)
        connection.commit()

    create_db(engine)

    with engine.connect() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(training_sessions)")}

    assert "aim_offset_x" in columns
    assert "aim_offset_y" in columns


def test_create_db_migration_is_idempotent(tmp_path):
    # Running it twice must not raise (covers the "duplicate column" path).
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", echo=False)
    with engine.connect() as connection:
        for statement in OLD_SCHEMA_SQL.strip().split(";"):
            if statement.strip():
                connection.exec_driver_sql(statement)
        connection.commit()

    create_db(engine)
    create_db(engine)  # must not raise


def test_old_schema_row_still_loads_with_identical_statistics(tmp_path, sample_image_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", echo=False)
    with engine.connect() as connection:
        for statement in OLD_SCHEMA_SQL.strip().split(";"):
            if statement.strip():
                connection.exec_driver_sql(statement)
        connection.commit()

        connection.exec_driver_sql(
            "INSERT INTO training_sessions (id, created_at, image_path, target_center_x, "
            "target_center_y, target_radius, target_type, hit_count, accuracy_radius, "
            "precision_radius, precision_std, cep_50, extreme_spread, outliers_count, "
            "mean_x, mean_y) VALUES (1, '2026-01-01 00:00:00', :image_path, 25.0, 25.0, "
            "20.0, 'circular', 2, 0.0, 10.0, 0.0, 10.0, 0.0, 0, 10.0, 0.0)",
            {"image_path": sample_image_path},
        )
        connection.exec_driver_sql(
            "INSERT INTO hit_records (session_id, x, y, distance_from_center, valid, "
            "confidence, source) VALUES (1, 10.0, 0.0, 10.0, 1, 0.9, 'auto')"
        )
        connection.exec_driver_sql(
            "INSERT INTO hit_records (session_id, x, y, distance_from_center, valid, "
            "confidence, source) VALUES (1, 10.0, 0.0, 10.0, 1, 0.9, 'auto')"
        )
        connection.commit()

    create_db(engine)  # migrate: adds aim_offset_x/y as NULL for the old row

    loaded = load_session(1, engine=engine)

    assert loaded is not None
    # No aim point was ever set for this old row -> must fall back to the
    # target center, reproducing the exact statistics it had before.
    expected = calculate_statistics(loaded.hits)  # default aim_point = center
    assert loaded.statistics["accuracy_radius"] == expected["accuracy_radius"]
    assert loaded.statistics["precision_radius"] == expected["precision_radius"]


def test_accuracy_survives_save_and_load_with_a_custom_aim_point(tmp_path, sample_image_path):
    from sqlmodel import SQLModel

    engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}", echo=False)
    SQLModel.metadata.create_all(engine)

    # Aim point is offset from the detected center - e.g. a silhouette
    # target where the shooter aims at the A-zone, not the contour centroid.
    target = Target(center_x=25.0, center_y=25.0, radius=20.0, type="foreground",
                     aim_offset_x=5.0, aim_offset_y=-5.0)
    hits = [Hit(x=5.0, y=-5.0, distance_from_center=7.07, source="auto")]
    aim_point = (target.aim_offset_x, target.aim_offset_y)
    statistics = calculate_statistics(hits, aim_point=aim_point)

    assert statistics["accuracy_radius"] == 0.0  # MPI lands exactly on the aim point

    session = Session(
        id=None,
        image=Image(path=sample_image_path, original_data=np.zeros((50, 50, 3), dtype=np.uint8)),
        target=target,
        hits=hits,
        statistics=statistics,
        metadata={},
    )

    saved = save_session(session, engine=engine)
    loaded = load_session(saved.id, engine=engine)

    assert loaded.target.aim_offset_x == 5.0
    assert loaded.target.aim_offset_y == -5.0
    # Recomputed on load, but using the SAME (persisted) aim point - must
    # match exactly, not silently fall back to center-relative accuracy.
    assert loaded.statistics["accuracy_radius"] == 0.0


def test_accuracy_after_hit_correction_matches_between_save_and_load(tmp_path, sample_image_path):
    # Regression test for the bug where the hit-correction handler called
    # calculate_statistics(hits) without the aim point: save_session() then
    # persisted a center-relative accuracy_radius while load_session()
    # recomputed an aim-point-relative one, so the two disagreed.
    from sqlmodel import SQLModel

    engine = create_engine(f"sqlite:///{tmp_path / 'corrected.db'}", echo=False)
    SQLModel.metadata.create_all(engine)

    target = Target(center_x=25.0, center_y=25.0, radius=20.0, type="foreground",
                     aim_offset_x=5.0, aim_offset_y=-5.0)
    session = Session(
        id=None,
        image=Image(path=sample_image_path, original_data=np.zeros((50, 50, 3), dtype=np.uint8)),
        target=target,
        hits=[],
        statistics={},
        metadata={},
    )

    # Simulate a manual hit correction the way view/main_window.py's
    # correction-mode handler does it: toggle_hit_at() + recompute_statistics().
    session.hits = toggle_hit_at(session.hits, target, image_point=(30.0, 20.0))
    session.statistics = recompute_statistics(session)

    saved = save_session(session, engine=engine)
    loaded = load_session(saved.id, engine=engine)

    assert saved.accuracy_radius == loaded.statistics["accuracy_radius"]


# ---------------------------------------------------------------------------
# Item 4: DB anchored to the project root, not the CWD - legacy-file
# migration and the INZ_RANGE_DB override.
# ---------------------------------------------------------------------------

def test_migrate_legacy_cwd_db_copies_when_target_missing(tmp_path, monkeypatch):
    legacy_dir = tmp_path / "legacy_cwd"
    legacy_dir.mkdir()
    monkeypatch.chdir(legacy_dir)

    legacy_db = legacy_dir / "training_sessions.db"
    legacy_db.write_bytes(b"old-sqlite-bytes")

    target_path = tmp_path / "anchored" / "training_sessions.db"
    target_path.parent.mkdir()

    _migrate_legacy_cwd_db_if_present(target_path)

    assert target_path.exists()
    assert target_path.read_bytes() == b"old-sqlite-bytes"
    assert legacy_db.exists()  # original must survive untouched
    assert legacy_db.read_bytes() == b"old-sqlite-bytes"


def test_migrate_legacy_cwd_db_does_not_overwrite_existing_target(tmp_path, monkeypatch):
    legacy_dir = tmp_path / "legacy_cwd"
    legacy_dir.mkdir()
    monkeypatch.chdir(legacy_dir)

    legacy_db = legacy_dir / "training_sessions.db"
    legacy_db.write_bytes(b"old-bytes")

    target_path = tmp_path / "anchored" / "training_sessions.db"
    target_path.parent.mkdir()
    target_path.write_bytes(b"current-bytes-must-survive")

    _migrate_legacy_cwd_db_if_present(target_path)

    assert target_path.read_bytes() == b"current-bytes-must-survive"


def test_migrate_legacy_cwd_db_does_nothing_when_no_legacy_file(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty_cwd"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)

    target_path = tmp_path / "anchored" / "training_sessions.db"
    target_path.parent.mkdir()

    _migrate_legacy_cwd_db_if_present(target_path)

    assert not target_path.exists()


def test_migrate_legacy_cwd_db_is_a_noop_when_cwd_equals_target_dir(tmp_path, monkeypatch):
    # The common case: CWD is the project root itself, so the "legacy" path
    # and the anchored path are literally the same file - nothing to copy.
    monkeypatch.chdir(tmp_path)
    same_file = tmp_path / "training_sessions.db"
    same_file.write_bytes(b"only-copy")

    _migrate_legacy_cwd_db_if_present(same_file)

    assert same_file.read_bytes() == b"only-copy"


def test_resolve_db_path_uses_env_override_when_set(tmp_path, monkeypatch):
    custom_path = tmp_path / "custom_location" / "my.db"
    monkeypatch.setenv("INZ_RANGE_DB", str(custom_path))

    resolved = _resolve_db_path()

    assert resolved == custom_path.resolve()


def test_resolve_db_path_falls_back_to_project_root_without_override(monkeypatch):
    monkeypatch.delenv("INZ_RANGE_DB", raising=False)

    resolved = _resolve_db_path()

    assert resolved.name == "training_sessions.db"
    assert (resolved.parent / "main.py").exists()  # anchored at the project root
