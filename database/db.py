from sqlmodel import SQLModel, create_engine

DATABASE_URL = "sqlite:///training_sessions.db"

engine = create_engine(
    DATABASE_URL,
    echo=False
)


def create_db() -> None:
    SQLModel.metadata.create_all(engine)