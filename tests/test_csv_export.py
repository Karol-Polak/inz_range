import csv
from datetime import datetime

from database.models import TrainingSession
from model.hit import Hit
from model.image import Image
from model.session import Session
from model.target import Target
from services.csv_export import export_session_to_csv, export_sessions_list_to_csv
from services.statistics import calculate_statistics


def _read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.reader(csv_file, delimiter=";"))


def _build_session():
    hits = [
        Hit(x=1.0, y=0.0, distance_from_center=1.0, valid=True, confidence=0.9, source="auto"),
        Hit(x=0.0, y=-2.5, distance_from_center=2.5, valid=True, confidence=0.8, source="manual"),
    ]
    statistics = calculate_statistics(hits)
    target = Target(center_x=100.0, center_y=100.0, radius=80.0, type="circular")
    image = Image(path="x.jpg")
    return Session(id=1, image=image, target=target, hits=hits, statistics=statistics, metadata={})


def test_export_session_uses_semicolon_delimiter_and_utf8_sig(tmp_path):
    path = tmp_path / "session.csv"
    export_session_to_csv(_build_session(), str(path))

    raw_bytes = path.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM (utf-8-sig)

    text = raw_bytes.decode("utf-8-sig")
    assert ";" in text.splitlines()[0] or ";" in text  # delimiter is ';', not ','


def test_export_session_writes_summary_and_hit_rows(tmp_path):
    path = tmp_path / "session.csv"
    export_session_to_csv(_build_session(), str(path))

    rows = _read_rows(path)
    flat = [cell for row in rows for cell in row]

    assert any("Liczba trafien" in cell for cell in flat)
    assert any("CEP 50%" in cell for cell in flat)
    # two hit data rows should be present (plus a header row)
    hit_header_index = next(i for i, row in enumerate(rows) if row and row[0] == "Nr")
    hit_rows = [row for row in rows[hit_header_index + 1:] if row and row[0].strip()]
    assert len(hit_rows) == 2


def test_export_session_formats_numbers_with_decimal_comma(tmp_path):
    # Agreed format: ';' delimiter + ',' decimal separator, matching the
    # semicolon-delimited CSV convention Polish Excel expects for numbers
    # to be recognised as numeric rather than imported as text.
    path = tmp_path / "session.csv"
    export_session_to_csv(_build_session(), str(path))

    rows = _read_rows(path)
    hit_header_index = next(i for i, row in enumerate(rows) if row and row[0] == "Nr")
    first_hit_row = rows[hit_header_index + 1]

    # x_px for the first hit (1.0) must render as "1,000" style, not "1.000"
    assert "," in first_hit_row[1]
    assert "." not in first_hit_row[1]


def test_export_session_includes_mm_columns_when_scale_given(tmp_path):
    path = tmp_path / "session.csv"
    export_session_to_csv(_build_session(), str(path), scale_mm_per_px=2.0)

    rows = _read_rows(path)
    hit_header_index = next(i for i, row in enumerate(rows) if row and row[0] == "Nr")
    header = rows[hit_header_index]
    first_hit_row = rows[hit_header_index + 1]

    x_mm_index = header.index("X_mm")
    # hit 1 has x=1.0px -> 2.0mm with scale 2.0
    assert first_hit_row[x_mm_index].replace(",", ".") == "2.000"


def test_export_session_leaves_mm_columns_blank_without_scale(tmp_path):
    path = tmp_path / "session.csv"
    export_session_to_csv(_build_session(), str(path))  # no scale

    rows = _read_rows(path)
    hit_header_index = next(i for i, row in enumerate(rows) if row and row[0] == "Nr")
    header = rows[hit_header_index]
    first_hit_row = rows[hit_header_index + 1]

    x_mm_index = header.index("X_mm")
    assert first_hit_row[x_mm_index] == ""


def test_export_sessions_list_writes_one_row_per_session(tmp_path):
    sessions = [
        TrainingSession(
            id=1, created_at=datetime(2026, 1, 1), image_path="a.jpg",
            target_center_x=0, target_center_y=0, target_radius=100, target_type="circular",
            hit_count=5, accuracy_radius=2.0, precision_radius=8.0, precision_std=1.0,
            cep_50=7.5, extreme_spread=15.0, outliers_count=0, mean_x=1.0, mean_y=1.0,
            weapon_type="Glock 17", distance_m=10.0,
        ),
        TrainingSession(
            id=2, created_at=datetime(2026, 1, 2), image_path="b.jpg",
            target_center_x=0, target_center_y=0, target_radius=100, target_type="circular",
            hit_count=8, accuracy_radius=1.0, precision_radius=6.0, precision_std=0.5,
            cep_50=5.5, extreme_spread=12.0, outliers_count=1, mean_x=0.5, mean_y=0.5,
        ),
    ]

    path = tmp_path / "sessions.csv"
    export_sessions_list_to_csv(sessions, str(path))

    rows = _read_rows(path)
    data_rows = [row for row in rows[1:] if row and row[0].strip()]
    assert len(data_rows) == 2
