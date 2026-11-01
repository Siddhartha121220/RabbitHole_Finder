from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from rabbithole.config import get_settings
from rabbithole.db import loader, session
from rabbithole.ingest.clickstream import parse

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_clickstream(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cs") / "clickstream.parquet"
    parse(FIXTURES / "clickstream_tiny.tsv", out)
    return out


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def engine(
    monkeypatch_module: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Engine]:
    """A fresh, migrated database named by the test module's DB_NAME on the compose Postgres,
    with DATABASE_URL pointed at it. Skips the module if Postgres is not reachable."""
    name: str = request.module.DB_NAME
    mp = monkeypatch_module
    url = make_url(get_settings().database_url)
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)"))
            conn.execute(text(f"CREATE DATABASE {name}"))
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {e}")
    mp.setenv("DATABASE_URL", url.set(database=name).render_as_string(hide_password=False))
    get_settings.cache_clear()
    session.get_engine.cache_clear()
    loader.migrate()
    yield session.get_engine()
    session.get_engine.cache_clear()
    get_settings.cache_clear()
