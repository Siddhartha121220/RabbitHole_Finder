"""Discover, download and parse Wikimedia clickstream dumps."""

import logging
import re
import urllib.request
from pathlib import Path

import duckdb
import polars as pl

from rabbithole.config import Settings

log = logging.getLogger(__name__)

BASE_URL = "https://dumps.wikimedia.org/other/clickstream/"


def _get(url: str, settings: Settings, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(
        url, method=method, headers={"User-Agent": settings.wiki_user_agent}
    )


def list_months(settings: Settings) -> list[str]:
    """All months (YYYY-MM) published on the dumps site, oldest first."""
    with urllib.request.urlopen(_get(BASE_URL, settings), timeout=30) as r:
        html = r.read().decode()
    return sorted(set(re.findall(r'href="(\d{4}-\d{2})/"', html)))


def resolve_months(settings: Settings, n: int = 3) -> list[str]:
    """Configured months, or the `n` most recent published ones."""
    return settings.months or list_months(settings)[-n:]


def month_url(month: str) -> str:
    return f"{BASE_URL}{month}/clickstream-enwiki-{month}.tsv.gz"


def download(month: str, settings: Settings) -> Path:
    """Download one month's dump into data/raw (skipped if already complete)."""
    url = month_url(month)
    with urllib.request.urlopen(_get(url, settings, "HEAD"), timeout=30) as r:
        size = int(r.headers["Content-Length"])
    if size > settings.max_download_bytes:
        raise RuntimeError(f"{url} is {size / 1e9:.1f} GB; over the limit, ask before downloading")

    dest = settings.data_dir / "raw" / f"clickstream-enwiki-{month}.tsv.gz"
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("downloading %s (%.0f MB)", url, size / 1e6)
    with urllib.request.urlopen(_get(url, settings), timeout=60) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def parse(tsv_gz: Path, out: Path) -> int:
    """Parse a clickstream TSV(.gz) into Parquet with columns prev, curr, type, n."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # quote='' because titles can contain double quotes.
    duckdb.sql(f"""
        COPY (
            SELECT prev, curr, type, n
            FROM read_csv('{tsv_gz}', delim='\t', header=false, quote='', escape='',
                columns={{'prev': 'VARCHAR', 'curr': 'VARCHAR', 'type': 'VARCHAR', 'n': 'BIGINT'}})
            WHERE prev IS NOT NULL AND curr IS NOT NULL AND n > 0
        ) TO '{out}' (FORMAT parquet)
    """)
    count = pl.scan_parquet(out).select(pl.len()).collect().item()
    log.info("parsed %s -> %s (%d rows)", tsv_gz.name, out.name, count)
    return count
