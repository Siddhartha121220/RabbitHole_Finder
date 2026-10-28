from pathlib import Path

import polars as pl
import pytest

from rabbithole.ingest import stats, subset, text, titles


def test_parse_keeps_all_rows(tiny_clickstream: Path) -> None:
    df = pl.read_parquet(tiny_clickstream)
    assert df.columns == ["prev", "curr", "type", "n"]
    assert df.height == 30
    assert df["n"].min() > 0  # pyright: ignore[reportOperatorIssue]


@pytest.mark.parametrize(
    ("title", "excluded"),
    [
        ("Main_Page", True),
        ("Chess_(disambiguation)", True),
        ("List_of_cricket_terms", True),
        ("1983", True),
        ("44_BC", True),
        ("1990s", True),
        ("March_3", True),
        ("2004_in_film", True),
        ("Cricket", False),
        ("1983_Cricket_World_Cup", False),
        ("Apollo_11", False),
    ],
)
def test_exclusion_rules(title: str, excluded: bool) -> None:
    out = pl.DataFrame({"t": [title]}).select(titles.is_excluded_expr("t")).item()
    assert out is excluded


def test_normalize() -> None:
    assert titles.normalize(" rust (programming language) ") == "Rust_(programming_language)"


def test_dev_subset_bfs_is_deterministic_and_bounded(tiny_clickstream: Path) -> None:
    got = subset.dev_subset(tiny_clickstream, ("Cricket", "Not_A_Page"), target=5, per_node=2)
    # Seed first, then Cricket's 2 most-clicked non-excluded links, then the next level by n.
    assert got[:3] == ["Cricket", "Sachin_Tendulkar", "Test_cricket"]
    assert len(got) == 5
    assert got == subset.dev_subset(tiny_clickstream, ("Cricket", "Not_A_Page"), 5, 2)
    assert "List_of_cricket_terms" not in got and "1983" not in got


def test_full_subset_threshold(tiny_clickstream: Path) -> None:
    got = subset.full_subset(tiny_clickstream, min_views=500)
    assert got == ["Chess", "Cricket", "India", "Magnus_Carlsen", "Sachin_Tendulkar"]


def test_article_stats(tiny_clickstream: Path) -> None:
    ids = subset.assign_ids(["Cricket", "Sachin_Tendulkar"])
    df = stats.article_stats(tiny_clickstream, ids)
    cricket = df.filter(pl.col("title") == "Cricket").row(0, named=True)
    assert cricket["views_in"] == 500 + 100 + 50 + 80 + 50
    assert cricket["onward_clicks"] == 300 + 200 + 20 + 150 + 40
    assert cricket["search_entry_share"] == pytest.approx(500 / 780)
    assert cricket["onward_ratio"] == pytest.approx(710 / 780)


def test_edges_probabilities_sum_to_one(tiny_clickstream: Path) -> None:
    ids = subset.assign_ids(["Cricket", "Sachin_Tendulkar", "Test_cricket", "Mumbai"])
    e = stats.edges(tiny_clickstream, ids)
    sums = e.group_by("src_id").agg(pl.col("p").sum())["p"].to_list()
    assert all(s == pytest.approx(1.0) for s in sums)
    # Cricket -> Sachin 300, Cricket -> Test_cricket 200 => p = 0.6 / 0.4
    assert sorted(e.filter(pl.col("src_id") == 0)["p"].to_list()) == pytest.approx([0.4, 0.6])


def test_parse_api_response_follows_redirects() -> None:
    data = {
        "query": {
            "normalized": [
                {"from": "Rust_(programming_language)", "to": "Rust (programming language)"}
            ],
            "redirects": [{"from": "Soccer", "to": "Association football"}],
            "pages": [
                {"title": "Rust (programming language)", "extract": "Rust is...", "length": 100},
                {"title": "Association football", "extract": "Football is...", "length": 200},
                {"title": "Nope", "missing": True},
            ],
        }
    }
    got = text.parse_api_response(data)
    assert got == {
        "Rust_(programming_language)": ("Rust is...", 100),
        "Soccer": ("Football is...", 200),
    }


def test_split_sections() -> None:
    doc = "Chess is a game.\n\nIt is old.\n\nHistory\nInvented in India.\n\nRules\n\nPieces move."
    assert text.split_sections(doc) == [
        ("Lead", "Chess is a game.\n\nIt is old."),
        ("History", "Invented in India."),
        ("Rules", "Pieces move."),
    ]


def test_chunk_words_overlap() -> None:
    words = [f"w{i}" for i in range(600)]
    got = text.chunk_words(" ".join(words), size=300, overlap=50)
    assert [len(c.split()) for c in got] == [300, 300, 100]
    assert got[1].split()[0] == "w250"
    assert text.chunk_words("") == []


def test_chunks_table() -> None:
    df = pl.DataFrame(
        {"article_id": [0, 1], "title": ["Chess", "Empty"], "body_text": ["Chess is fun.", None]}
    )
    out = text.chunks(df)
    assert out.height == 1
    assert out.row(0, named=True)["text"] == "Chess\nChess is fun."
