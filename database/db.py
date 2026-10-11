from sqlalchemy.exc import OperationalError
from sqlmodel import SQLModel, create_engine

DATABASE_URL = "sqlite:///training_sessions.db"

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
    """

    target_engine = engine_ or engine
    SQLModel.metadata.create_all(target_engine)
    _add_missing_columns(target_engine, "training_sessions", _NEW_TRAINING_SESSION_COLUMNS)


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