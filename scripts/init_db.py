"""Create the pgvector extension and RAG tables. Run after the database is up."""

from sqlalchemy import text

from app.core.database import engine
from app.models import Base


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(engine)
    print("Extension enabled and tables created:", list(Base.metadata.tables.keys()))


if __name__ == "__main__":
    main()
