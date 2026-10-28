# pyright: basic
# (`datasets` ships without complete type hints, so strict mode is relaxed for this file only.)
"""FULL-mode article text: stream the Hugging Face `wikimedia/wikipedia` dump."""

from collections.abc import Iterator
from typing import Any

import polars as pl

from rabbithole.ingest.text import LEAD_CHARS, split_sections
from rabbithole.ingest.titles import normalize

BODY_CHARS = 20_000


def latest_hf_config() -> str:
    from datasets import get_dataset_config_names

    return max(c for c in get_dataset_config_names("wikimedia/wikipedia") if c.endswith(".en"))


def stream_hf_articles(wanted: set[str], config: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield {title, lead_text, body_text, page_length} for wanted titles from the HF dump."""
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia", config or latest_hf_config(), split="train", streaming=True
    )
    for row in ds:
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
    """article_text table for FULL (redirects are not resolved here; tracked via match rate)."""
    rows = list(stream_hf_articles(set(ids["title"].to_list())))
    found = pl.DataFrame(
        rows,
        schema={
            "title": pl.String,
            "lead_text": pl.String,
            "body_text": pl.String,
            "page_length": pl.Int64,
        },
    )
    return ids.join(found, on="title", how="left").with_columns(pl.col("page_length").fill_null(0))
