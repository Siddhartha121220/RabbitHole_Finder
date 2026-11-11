"""Parquet -> Postgres. Idempotent: upserts every row, then removes rows from a previous load
that no longer exist (e.g. after switching DEV <-> FULL). Never touches feedback or trails."""

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
from pgvector.psycopg import register_vector
from sqlalchemy import Connection, Table, delete, insert, select, text
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
    Used for communities (arrays, JSONB); the big tables go through `copy_upsert`.
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


# Binary COPY type names for the column types used by articles and chunks.
COPY_TYPES = {"INTEGER": "int4", "TEXT": "text", "REAL": "float4", "FLOAT": "float8"}


def _copy_type(col: Any) -> str:
    name = type(col.type).__name__.upper()
    return "vector" if name == "VECTOR" else COPY_TYPES[str(col.type).split("(")[0].upper()]


def copy_upsert(conn: Connection, table: Table, df: pl.DataFrame) -> None:
    """Bulk upsert for the big tables: binary COPY into a temp table, then one
    INSERT ... SELECT ... ON CONFLICT. ~50x faster than multi-row INSERTs for FULL."""
    if df.is_empty():
        return
    cols = df.columns
    tmp = f"tmp_{table.name}"
    conn.execute(text(f"CREATE TEMP TABLE IF NOT EXISTS {tmp} (LIKE {table.name}) ON COMMIT DROP"))
    conn.execute(text(f"TRUNCATE {tmp}"))
    raw: Any = conn.connection.driver_connection
    register_vector(raw)
    vec_cols = {c for c in cols if _copy_type(table.c[c]) == "vector"}
    with (
        raw.cursor() as cur,
        cur.copy(f"COPY {tmp} ({', '.join(cols)}) FROM STDIN WITH (FORMAT BINARY)") as cp,
    ):
        cp.set_types([_copy_type(table.c[c]) for c in cols])
        vecs = {c: df[c].to_numpy() for c in vec_cols}
        plain = df.drop(vec_cols)
        for i, row in enumerate(plain.iter_rows(named=True)):
            cp.write_row([vecs[c][i] if c in vec_cols else row[c] for c in cols])
    col_list = ", ".join(cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id")
    conn.execute(
        text(
            f"INSERT INTO {table.name} ({col_list}) SELECT {col_list} FROM {tmp} "
            f"ON CONFLICT (id) DO UPDATE SET {updates}"
        )
    )


def prune(conn: Connection, table: Table, keep_ids: list[int], small: int = 10_000) -> None:
    """Delete rows left over from a previous load (e.g. after switching DEV <-> FULL).
    Large keep-lists go through a temp table instead of a giant NOT IN."""
    if len(keep_ids) <= small:
        conn.execute(delete(table).where(table.c.id.not_in(keep_ids)))
        return
    conn.execute(
        text("CREATE TEMP TABLE IF NOT EXISTS keep_ids (id int PRIMARY KEY) ON COMMIT DROP")
    )
    conn.execute(text("TRUNCATE keep_ids"))
    raw: Any = conn.connection.driver_connection
    with raw.cursor() as cur, cur.copy("COPY keep_ids (id) FROM STDIN") as cp:
        for i in keep_ids:
            cp.write_row((i,))
    conn.execute(
        text(
            f"DELETE FROM {table.name} t WHERE NOT EXISTS "
            "(SELECT 1 FROM keep_ids k WHERE k.id = t.id)"
        )
    )


def wiki_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title}"


def load(
    conn: Connection,
    communities: pl.DataFrame,
    articles: pl.DataFrame,
    chunks: pl.DataFrame | Iterable[pl.DataFrame],
    run_info: dict[str, Any],
) -> None:
    """Load all three content tables in FK order, inside the caller's transaction.
    `chunks` may be an iterable of batches, so FULL (millions of rows) never sits in memory."""
    started = datetime.now(UTC)
    ctab, atab, chtab = (
        cast(Table, Community.__table__),
        cast(Table, Article.__table__),
        cast(Table, Chunk.__table__),
    )
    # An id that now belongs to a different title (e.g. DEV -> FULL) would clash with the
    # unique title constraint during upsert, so drop those articles first (chunks cascade).
    new_pairs = set(zip(articles["id"].to_list(), articles["title"].to_list(), strict=True))
    old_pairs = conn.execute(select(atab.c.id, atab.c.title)).all()
    changed = [i for i, t in old_pairs if (i, t) not in new_pairs]
    if changed:
        conn.execute(delete(atab).where(atab.c.id.in_(changed)))

    # Upsert parents before children, prune children before parents (foreign keys).
    upsert(conn, ctab, communities.to_dicts())
    copy_upsert(conn, atab, articles)
    chunk_ids: list[int] = []
    for batch in [chunks] if isinstance(chunks, pl.DataFrame) else chunks:
        copy_upsert(conn, chtab, batch)
        chunk_ids += batch["id"].to_list()
        log.info("chunks loaded: %d", len(chunk_ids))
    prune(conn, chtab, chunk_ids)
    prune(conn, atab, articles["id"].to_list())
    prune(conn, ctab, communities["id"].to_list())

    stats = {
        "communities": communities.height,
        "articles": articles.height,
        "chunks": len(chunk_ids),
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
