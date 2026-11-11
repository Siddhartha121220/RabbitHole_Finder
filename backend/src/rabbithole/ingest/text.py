"""Article text: lead sections via the MediaWiki API (DEV) or Hugging Face dumps (FULL)."""

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from itertools import islice
from pathlib import Path
from typing import Any

import polars as pl

from rabbithole.config import Settings
from rabbithole.ingest.titles import normalize

log = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
BATCH = 20  # exintro extracts are limited to 20 pages per request
LEAD_CHARS = 2_000


def _api_batch(titles: list[str], settings: Settings, cache_dir: Path) -> dict[str, Any]:
    """One cached API call: >= 1 s between live requests, exponential backoff on 429/503/maxlag."""
    key = hashlib.sha1("\n".join(titles).encode()).hexdigest()
    cached = cache_dir / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts|info",
        "exintro": "1",
        "explaintext": "1",
        "redirects": "1",
        "maxlag": "5",
        "titles": "|".join(titles),
    }
    req = urllib.request.Request(
        f"{API_URL}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": settings.wiki_user_agent},
    )
    wait = 1.0  # politeness delay before every live request
    for _ in range(8):
        time.sleep(wait)
        wait = min(wait * 2, 60)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data: dict[str, Any] = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
            wait = max(wait, float(e.headers.get("Retry-After") or 0))
            log.warning("MediaWiki API HTTP %d, retrying in %.0f s", e.code, wait)
            continue
        if data.get("error", {}).get("code") == "maxlag":
            log.warning("MediaWiki API maxlag, retrying in %.0f s", wait)
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"MediaWiki API kept failing for batch starting {titles[0]!r}")


def parse_api_response(data: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """Map each *requested* title (underscored) to (lead text, page length in bytes)."""
    query = data.get("query", {})
    # Follow the API's own normalization and redirect chain back to what we asked for.
    back: dict[str, str] = {}
    for step in query.get("normalized", []) + query.get("redirects", []):
        src: str = step["from"]
        back[step["to"]] = back.get(src, src)
    out: dict[str, tuple[str, int]] = {}
    for page in query.get("pages", []):
        if page.get("missing") or "extract" not in page:
            continue
        title: str = page["title"]
        asked = back.get(title, title)
        out[normalize(asked)] = (page["extract"][:LEAD_CHARS], int(page.get("length", 0)))
    return out


def fetch_leads_api(titles: list[str], settings: Settings) -> dict[str, tuple[str, int]]:
    cache_dir = settings.data_dir / "raw" / "wiki_api"
    out: dict[str, tuple[str, int]] = {}
    for i in range(0, len(titles), BATCH):
        batch = titles[i : i + BATCH]
        out.update(parse_api_response(_api_batch(batch, settings, cache_dir)))
        if i // BATCH % 25 == 0:
            log.info("text: %d / %d titles", i + len(batch), len(titles))
    return out


def article_text_dev(ids: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    """article_text table for DEV: lead from the API; body is the lead (no full text fetched)."""
    found = fetch_leads_api(ids["title"].to_list(), settings)
    return ids.with_columns(
        pl.col("title")
        .replace_strict({t: v[0] for t, v in found.items()}, default=None, return_dtype=pl.String)
        .alias("lead_text"),
        pl.col("title")
        .replace_strict({t: v[1] for t, v in found.items()}, default=0, return_dtype=pl.Int64)
        .alias("page_length"),
    ).with_columns(pl.col("lead_text").alias("body_text"))


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split plain-text article dumps into (section title, text). The lead is "Lead".

    ponytail: heading = a short line with no final punctuation between blank lines; good
    enough for HF `wikimedia/wikipedia` text, swap for wikitext parsing if it misfires.
    """
    sections: list[tuple[str, list[str]]] = [("Lead", [])]
    for para in filter(None, (p.strip() for p in text.split("\n\n"))):
        first, _, rest = para.partition("\n")
        is_heading = len(first) < 60 and not first.endswith((".", ":", "!", "?", '"'))
        if is_heading and sections[0][1]:  # the lead always comes before any heading
            sections.append((first, [rest] if rest else []))
        else:
            sections[-1][1].append(para)
    return [(title, "\n\n".join(paras)) for title, paras in sections if paras]


def chunk_words(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split into windows of `size` words overlapping by `overlap`.

    ponytail: words stand in for tokens (~1.3 tokens/word for English); fine for retrieval.
    """
    words = text.split()
    if not words:
        return []
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def chunks(text_table: pl.DataFrame, max_per_article: int | None = None) -> pl.DataFrame:
    """chunks table: chunk_id, article_id, ord, section, text. Text starts with its heading.
    At most `max_per_article` chunks per article (None = all)."""
    rows: list[dict[str, Any]] = []
    for article_id, title, body in text_table.select(
        "article_id", "title", "body_text"
    ).iter_rows():
        if not body:
            continue
        pieces = (
            (section, piece)
            for section, sec_text in split_sections(body)
            for piece in chunk_words(sec_text)
        )
        # Keep the first chunks (lead and early sections); a trail only needs the best few.
        for ord_, (section, piece) in enumerate(islice(pieces, max_per_article)):
            heading = title.replace("_", " ") + ("" if section == "Lead" else f" — {section}")
            rows.append(
                {
                    "article_id": article_id,
                    "ord": ord_,
                    "section": section,
                    "text": f"{heading}\n{piece}",
                }
            )
    return pl.DataFrame(
        rows,
        schema={"article_id": pl.UInt32, "ord": pl.Int32, "section": pl.String, "text": pl.String},
    ).with_row_index("chunk_id")
