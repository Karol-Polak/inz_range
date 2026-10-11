import os
import shutil
from pathlib import Path

from sqlalchemy.exc import OperationalError
from sqlmodel import SQLModel, create_engine

# Baza byla wczesniej otwierana wzgledem bierzacego katalogu roboczego
# (sqlite:///training_sessions.db), co tworzylo nowa, pusta baze przy kazdym
# uruchomieniu aplikacji z innego katalogu. Teraz jest zakotwiczona w
# katalogu glownym projektu (ten plik lezy w <root>/database/db.py), chyba
# ze zmienna srodowiskowa INZ_RANGE_DB wskazuje inna lokalizacje.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "training_sessions.db"


def _resolve_db_path() -> Path:
    override = os.environ.get("INZ_RANGE_DB")
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_DB_PATH


DB_PATH = _resolve_db_path()
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    echo=False
)

# Kolumny dodane do TrainingSession po tym, jak baza mogła już istnieć na
# dysku użytkownika. create_all() tworzy brakujące TABELE, ale nie dodaje
# brakujących KOLUMN do tabel, które już istnieją - stąd ręczna migracja.
_NEW_TRAINING_SESSION_COLUMNS = {
    "aim_offset_x": "REAL",
    "aim_offset_y": "REAL",
}


def create_db(engine_=None) -> None:
    """
    Tworzy tabele (jeśli ich jeszcze nie ma) i domigrowuje do istniejącej
    tabeli training_sessions kolumny, które zostały dodane do modelu
    później (patrz _NEW_TRAINING_SESSION_COLUMNS).

    Gdy wołane bez argumentu (czyli na prawdziwej, domyślnej bazie - tak jak
    robi to main.py), dodatkowo sprawdza, czy w bieżącym katalogu roboczym
    nie leży stary plik training_sessions.db z czasów, gdy baza była
    otwierana względnie, i jeśli tak, kopiuje go do nowej lokalizacji (patrz
    _migrate_legacy_cwd_db_if_present). Testy, które zawsze podają własny
    silnik, nigdy nie trafiają w tę ścieżkę.
    """

    if engine_ is None:
        _migrate_legacy_cwd_db_if_present(DB_PATH)

    target_engine = engine_ or engine
    SQLModel.metadata.create_all(target_engine)
    _add_missing_columns(target_engine, "training_sessions", _NEW_TRAINING_SESSION_COLUMNS)


def _migrate_legacy_cwd_db_if_present(target_path: Path) -> None:
    """
    Przed tą poprawką baza była tworzona względem katalogu, z którego
    uruchomiono aplikację. Jeśli w bieżącym katalogu roboczym leży taki
    stary plik, a w nowej (zakotwiczonej) lokalizacji nie ma jeszcze żadnej
    bazy, kopiuje go tam, żeby zapisane sesje nie zostały zgubione.

    Nigdy nie usuwa ani nie modyfikuje oryginalnego pliku. Jeśli docelowa
    baza już istnieje (np. migracja już się odbyła wcześniej), nic nie robi
    - nie ostrzega przy każdym starcie.
    """

    legacy_path = Path.cwd() / "training_sessions.db"

    try:
        same_file = legacy_path.exists() and legacy_path.resolve() == target_path.resolve()
    except OSError:
        same_file = False

    if same_file or not legacy_path.exists() or target_path.exists():
        return

    shutil.copy2(legacy_path, target_path)
    print(
        f"[inz_range] Znaleziono istniejacy plik bazy danych '{legacy_path}' "
        f"(z katalogu, z ktorego uruchomiono aplikacje). Skopiowano go do "
        f"'{target_path}', aby zapisane sesje byly dostepne niezaleznie od "
        f"katalogu startowego. Oryginalny plik NIE zostal usuniety."
    )


def _add_missing_columns(engine_, table: str, columns: dict) -> None:
    with engine_.connect() as connection:
        existing_columns = {
            row[1] for row in connection.exec_driver_sql(f"PRAGMA table_info({table})")
        }

        for column_name, sql_type in columns.items():
            if column_name in existing_columns:
                continue

            try:
                connection.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {column_name} {sql_type}"
                )
            except OperationalError as error:
                # Another call already added it (e.g. re-running create_db) -
                # anything else is a real problem and must not be swallowed.
                if "duplicate column" not in str(error).lower():
                    raise

        connection.commit()
