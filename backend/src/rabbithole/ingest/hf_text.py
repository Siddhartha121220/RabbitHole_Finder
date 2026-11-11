"""FULL-mode article text: stream the Hugging Face `wikimedia/wikipedia` dump."""

import logging
from collections.abc import Iterator
from typing import Any

import polars as pl

from rabbithole.ingest.text import LEAD_CHARS, split_sections
from rabbithole.ingest.titles import normalize

BODY_CHARS = 20_000
log = logging.getLogger(__name__)


def latest_hf_config() -> str:
    from datasets import get_dataset_config_names

    return max(c for c in get_dataset_config_names("wikimedia/wikipedia") if c.endswith(".en"))


def stream_hf_articles(wanted: set[str], config: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield {title, lead_text, body_text, page_length} for wanted titles from the HF dump."""
    from datasets import load_dataset

    config = config or latest_hf_config()
    log.info("streaming wikimedia/wikipedia %s", config)
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    for i, row in enumerate(ds.select_columns(["title", "text"])):
        if i % 500_000 == 0:
            log.info("HF rows scanned: %d", i)
        title = normalize(str(row["title"]))
        if title not in wanted:
            continue
        text = str(row["text"])
        sections = split_sections(text)
        lead = sections[0][1] if sections and sections[0][0] == "Lead" else ""
        yield {
            "title": title,
            "lead_text": lead[:LEAD_CHARS],
            "body_text": text[:BODY_CHARS],
            "page_length": len(text.encode()),
        }


def article_text_full(ids: pl.DataFrame) -> pl.DataFrame:
    """article_text table for FULL (redirects are not resolved here; tracked via match rate).
    Rows are collected into DataFrames in batches, so 600k+ bodies don't sit in Python dicts."""
    schema = {
        "title": pl.String,
        "lead_text": pl.String,
        "body_text": pl.String,
        "page_length": pl.Int64,
    }
    frames: list[pl.DataFrame] = []
    batch: list[dict[str, Any]] = []
    for row in stream_hf_articles(set(ids["title"].to_list())):
        batch.append(row)
        if len(batch) == 20_000:
            frames.append(pl.DataFrame(batch, schema=schema))
            batch = []
    frames.append(pl.DataFrame(batch, schema=schema))
    # A title can appear twice in a dump; keep the first.
    found = pl.concat(frames).unique("title", keep="first", maintain_order=True)
    return ids.join(found, on="title", how="left").with_columns(pl.col("page_length").fill_null(0))
