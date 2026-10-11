import csv
from typing import List, Optional

from database.models import TrainingSession
from model.session import Session

_DELIMITER = ";"
_ENCODING = "utf-8-sig"  # BOM so Excel on Windows/Polish locale picks UTF-8 automatically


def _format_number(value: Optional[float], decimals: int = 3) -> str:
    """
    Formats a number with a decimal COMMA (e.g. "1,000"), not a point.

    This pairs with the ';' delimiter: semicolon-delimited CSVs are the
    convention Polish/European Excel expects specifically because it then
    reads ',' as the decimal separator - writing '.' here would make Excel
    import these numbers as text instead of numbers.
    """

    if value is None:
        return ""
    return f"{value:.{decimals}f}".replace(".", ",")


def export_session_to_csv(session: Session, path: str, scale_mm_per_px: Optional[float] = None) -> None:
    """
    Zapisuje jedną sesję do CSV: podsumowanie statystyk + jeden wiersz na
    trafienie. Kolumny *_mm są wypełnione tylko, gdy podano scale_mm_per_px.
    """

    stats = session.statistics

    def px_and_mm(value: float):
        mm = value * scale_mm_per_px if scale_mm_per_px is not None else None
        return _format_number(value), _format_number(mm)

    with open(path, "w", encoding=_ENCODING, newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=_DELIMITER)

        writer.writerow(["Podsumowanie sesji"])
        writer.writerow(["Pole", "Wartosc_px", "Wartosc_mm"])
        writer.writerow(["Liczba trafien", stats.get("count", 0), ""])

        for label, key in (
            ("Celnosc (MPI od punktu celowania)", "accuracy_radius"),
            ("Precyzja (odleglosc od MPI)", "precision_radius"),
            ("Odchylenie std. precyzji", "precision_std"),
            ("CEP 50%", "cep_50"),
            ("Rozrzut maks.", "extreme_spread"),
            ("Sr. odchylenie X", "mean_x"),
            ("Sr. odchylenie Y", "mean_y"),
        ):
            px, mm = px_and_mm(stats.get(key, 0.0))
            writer.writerow([label, px, mm])

        writer.writerow(["Liczba outlierow", stats.get("outliers_count", 0), ""])

        writer.writerow([])

        writer.writerow(["Trafienia"])
        writer.writerow(["Nr", "X_px", "Y_px", "X_mm", "Y_mm", "Odleglosc_px", "Odleglosc_mm", "Valid", "Confidence", "Source"])

        for index, hit in enumerate(session.hits, start=1):
            x_px, x_mm = px_and_mm(hit.x)
            y_px, y_mm = px_and_mm(hit.y)
            distance_px, distance_mm = px_and_mm(hit.distance_from_center or 0.0)

            writer.writerow([
                index,
                x_px, y_px, x_mm, y_mm,
                distance_px, distance_mm,
                "1" if hit.valid else "0",
                _format_number(hit.confidence) if hit.confidence is not None else "",
                hit.source,
            ])


def export_sessions_list_to_csv(sessions: List[TrainingSession], path: str) -> None:
    """Zapisuje listę zapisanych sesji (jedna sesja = jeden wiersz) do CSV."""

    with open(path, "w", encoding=_ENCODING, newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=_DELIMITER)

        writer.writerow([
            "Id", "Data", "Liczba_trafien", "Celnosc_px", "Precyzja_px", "CEP50_px",
            "Rozrzut_maks_px", "Outliery", "Bron", "Odleglosc_m", "Skala_mm_na_px", "Notatki",
        ])

        for session in sessions:
            writer.writerow([
                session.id,
                session.created_at.strftime("%Y-%m-%d %H:%M"),
                session.hit_count,
                _format_number(session.accuracy_radius),
                _format_number(session.precision_radius),
                _format_number(session.cep_50),
                _format_number(session.extreme_spread),
                session.outliers_count,
                session.weapon_type or "",
                _format_number(session.distance_m) if session.distance_m is not None else "",
                _format_number(session.scale_mm_per_px, decimals=4) if session.scale_mm_per_px is not None else "",
                session.notes or "",
            ])
