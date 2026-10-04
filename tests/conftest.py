import pytest

from app.core.database import engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
