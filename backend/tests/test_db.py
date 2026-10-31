"""Integration test against a real Postgres + pgvector (the compose `db` service).

Uses a separate `rabbithole_test` database and skips if Postgres is not reachable.
"""

from collections.abc import Iterator

import numpy as np
import polars as pl
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from rabbithole.config import get_settings
from rabbithole.db import loader, session
from rabbithole.embed.search import nearest_articles

DIM = get_settings().embedding_dim


@pytest.fixture(scope="module")
def engine(monkeypatch_module: pytest.MonkeyPatch) -> Iterator[Engine]:
    url = make_url(get_settings().database_url)
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text("DROP DATABASE IF EXISTS rabbithole_test WITH (FORCE)"))
            conn.execute(text("CREATE DATABASE rabbithole_test"))
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {e}")
    test_url = url.set(database="rabbithole_test").render_as_string(hide_password=False)
    monkeypatch_module.setenv("DATABASE_URL", test_url)
    get_settings.cache_clear()
    session.get_engine.cache_clear()
    loader.migrate()
    yield session.get_engine()
    session.get_engine.cache_clear()
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def unit(i: int) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v.tolist()


def frames(titles: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n = len(titles)
    comms = pl.DataFrame(
        {
            "id": [0],
            "label": ["Test"],
            "size": [n],
            "eligible": [True],
            "keywords": [["a"]],
            "central_article_id": [0],
            "top_article_ids": [list(range(n))],
            "centroid": [unit(0)],
            "umap_x": [0.5],
            "umap_y": [0.5],
        }
    )
    arts = pl.DataFrame(
        {
            "id": list(range(n)),
            "title": titles,
            "url": [loader.wiki_url(t) for t in titles],
            "views": [100] * n,
            "community_id": [0] * n,
            "lead_text": ["x"] * n,
            "embedding": [unit(i) for i in range(n)],
            "umap_x": [0.1] * n,
            "umap_y": [0.2] * n,
        }
    )
    chunks = pl.DataFrame(
        {
            "id": [0],
            "article_id": [0],
            "ord": [0],
            "text": ["x"],
            "embedding": [unit(0)],
        }
    )
    return comms, arts, chunks


def counts(engine: Engine) -> tuple[int, ...]:
    with engine.connect() as conn:
        return tuple(
            int(conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one())
            for t in ("communities", "articles", "chunks", "pipeline_runs")
        )


def test_load_is_idempotent_and_prunes_stale_rows(engine: Engine) -> None:
    info = {"dataset_mode": "dev", "months": ["2026-07"]}
    with engine.begin() as conn:
        loader.load(conn, *frames(["Chess", "Cricket", "Sushi"]), info)
    with engine.begin() as conn:
        loader.load(conn, *frames(["Chess", "Cricket", "Sushi"]), info)
    assert counts(engine) == (1, 3, 1, 2)
    # Reload with fewer articles and a title moved to another id (like DEV -> FULL).
    with engine.begin() as conn:
        loader.load(conn, *frames(["Cricket", "Chess"]), info)
    assert counts(engine)[:3] == (1, 2, 1)


def test_nearest_articles_orders_by_cosine(engine: Engine) -> None:
    with engine.begin() as conn:
        loader.load(
            conn, *frames(["Chess", "Cricket", "Sushi"]), {"dataset_mode": "dev", "months": []}
        )
    q = np.array(unit(1), dtype=np.float32) * 0.9 + np.array(unit(2), dtype=np.float32) * 0.1
    q /= np.linalg.norm(q)
    with engine.connect() as conn:
        hits = nearest_articles(conn, q, k=3, min_similarity=0.05)
    assert [h.title for h in hits] == ["Cricket", "Sushi"]  # Chess is orthogonal: filtered out
    assert hits[0].url == "https://en.wikipedia.org/wiki/Cricket"
