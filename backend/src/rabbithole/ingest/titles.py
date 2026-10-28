"""Title normalization and the exclusion rules for the article subset (DATA_AND_ML §1.3)."""

import polars as pl

SPECIAL_SOURCES = ("other-search", "other-internal", "other-external", "other-empty", "other-other")

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# Main page, disambiguation pages, lists, and date/year pages ("1987", "44_BC", "1990s",
# "March_3", "2004_in_film").
EXCLUDE_PATTERN = (
    r"^(Main_Page"
    r"|.*_\(disambiguation\)"
    r"|List_of_.*|Lists_of_.*"
    r"|\d{1,4}(_BC|_AD|s)?"
    r"|(" + _MONTHS + r")_\d{1,2}"
    r"|(" + _MONTHS + r")_\d{4}"
    r"|\d{1,4}s?_in_.*)$"
)


def normalize(title: str) -> str:
    """Canonical title form: underscores, no surrounding whitespace, first letter upper."""
    t = title.strip().replace(" ", "_")
    return t[:1].upper() + t[1:]


def normalize_expr(col: str) -> pl.Expr:
    c = pl.col(col).str.strip_chars().str.replace_all(" ", "_", literal=True)
    return c.str.slice(0, 1).str.to_uppercase() + c.str.slice(1)


def is_excluded_expr(col: str) -> pl.Expr:
    return pl.col(col).str.contains(EXCLUDE_PATTERN)
