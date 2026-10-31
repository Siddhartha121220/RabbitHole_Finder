"""Parquet -> Postgres. Idempotent: upserts every row, then removes rows from a previous load
that no longer exist (e.g. after switching DEV <-> FULL). Never touches feedback or trails."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
from sqlalchemy import Connection, Table, delete, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rabbithole.db.models import Article, Chunk, Community, PipelineRun

log = logging.getLogger(__name__)
BATCH = 1000

ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"


def migrate() -> None:
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(ALEMBIC_INI)), "head")


def upsert(conn: Connection, table: Table, rows: list[dict[str, Any]]) -> None:
    """INSERT ... ON CONFLICT (id) DO UPDATE in batches.

    Only columns present in `rows` are updated, so columns filled by a later phase
    (e.g. communities.rh_score) survive a reload of the base data.
    ponytail: multi-row INSERTs; switch to COPY into a temp table if FULL loads are slow.
    """
    if not rows:
        return
    cols = [c for c in rows[0] if c != "id"]
    for i in range(0, len(rows), BATCH):
        stmt = pg_insert(table).values(rows[i : i + BATCH])
        conn.execute(
            stmt.on_conflict_do_update(
                index_elements=["id"], set_={c: stmt.excluded[c] for c in cols}
            )
        )


def prune(conn: Connection, table: Table, keep_ids: list[int]) -> None:
    """Delete rows left over from a previous load (e.g. after switching DEV <-> FULL)."""
    conn.execute(delete(table).where(table.c.id.not_in(keep_ids)))


def wiki_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title}"


def load(
    conn: Connection,
    communities: pl.DataFrame,
    articles: pl.DataFrame,
    chunks: pl.DataFrame,
    run_info: dict[str, Any],
) -> None:
    """Load all three content tables in FK order, inside the caller's transaction."""
    started = datetime.now(UTC)
    tables: list[tuple[Table, pl.DataFrame]] = [
        (cast(Table, Community.__table__), communities),
        (cast(Table, Article.__table__), articles),
        (cast(Table, Chunk.__table__), chunks),
    ]
    # An id that now belongs to a different title (e.g. DEV -> FULL) would clash with the
    # unique title constraint during upsert, so drop those articles first (chunks cascade).
    atab = cast(Table, Article.__table__)
    new_pairs = set(zip(articles["id"].to_list(), articles["title"].to_list(), strict=True))
    old_pairs = conn.execute(select(atab.c.id, atab.c.title)).all()
    changed = [i for i, t in old_pairs if (i, t) not in new_pairs]
    if changed:
        conn.execute(delete(atab).where(atab.c.id.in_(changed)))

    # Upsert parents before children, prune children before parents (foreign keys).
    for table, df in tables:
        upsert(conn, table, df.to_dicts())
    for table, df in reversed(tables):
        prune(conn, table, df["id"].to_list())

    stats = {
        "communities": communities.height,
        "articles": articles.height,
        "chunks": chunks.height,
    }
    conn.execute(
        insert(cast(Table, PipelineRun.__table__)).values(
            dataset_mode=run_info["dataset_mode"],
            clickstream_months=run_info["months"],
            started_at=started,
            finished_at=datetime.now(UTC),
            stats=stats,
        )
    )
    log.info("loaded %s", stats)
