"""Human labels for validating the score: "would this be a rabbit hole?" rated 1-7."""

import csv
import random
import re
from pathlib import Path

import polars as pl

FIELDS = ["community_id", "label", "rating"]
SCALE = 7  # 1 = not at all ... 7 = absolutely; 7 points give more spread than 5 for Spearman


def first_sentence(text: str | None, max_chars: int = 200) -> str:
    """Opening sentence of a lead, shortened for one-line display.

    Splits at ". " + capital letter, but never inside parentheses, so the "(transl. ...)"
    and "(lit. ...)" asides in Wikipedia leads stay whole.
    ponytail: "J. P. Morgan" outside parentheses can still cut early; fine for a hint line.
    """
    if not text:
        return ""
    text = text.strip()
    sentence = text
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", text):
        head = text[: m.start()]
        if head.count("(") <= head.count(")"):
            sentence = head
            break
    return sentence if len(sentence) <= max_chars else sentence[: max_chars - 1].rstrip() + "…"


def read_labels(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(
            schema={"community_id": pl.Int64, "label": pl.String, "rating": pl.Int64}
        )
    return pl.read_csv(path, schema_overrides={"community_id": pl.Int64, "rating": pl.Int64})


def to_label(communities: pl.DataFrame, path: Path, n: int, seed: int = 11) -> list[int]:
    """Community ids still to label: a fixed random order of eligible communities, minus
    those already in `path`, capped so the total labelled is at most `n`."""
    ids = sorted(communities.filter("eligible")["community_id"].to_list())
    random.Random(seed).shuffle(ids)
    done = set(read_labels(path)["community_id"].to_list())
    remaining = max(n - len(done), 0)
    return [i for i in ids if i not in done][:remaining]


def append_label(path: Path, community_id: int, label: str, rating: int) -> None:
    if not 1 <= rating <= SCALE:
        raise ValueError(f"rating must be 1-{SCALE}")
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(FIELDS)
        writer.writerow([community_id, label, rating])
