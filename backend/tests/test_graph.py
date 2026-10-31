import math

import polars as pl
import pytest
from scipy.sparse import csr_matrix

from rabbithole.graph import communities as comm
from rabbithole.graph import pruned, report


def two_cliques() -> pl.DataFrame:
    """Two directed 4-cliques {0-3} and {4-7} joined by one weak 3->4 edge."""
    rows: list[tuple[int, int, int]] = []
    for block in (range(0, 4), range(4, 8)):
        rows += [(a, b, 100) for a in block for b in block if a != b]
    rows.append((3, 4, 1))
    df = pl.DataFrame(rows, schema=["src_id", "dst_id", "n"], orient="row")
    return df.with_columns((pl.col("n") / pl.col("n").sum().over("src_id")).alias("p"))


def test_undirected_weights_sum_both_directions() -> None:
    e = pl.DataFrame({"src_id": [0, 1, 1], "dst_id": [1, 0, 2], "n": [5, 3, 2]})
    g = comm.undirected_graph(e, 3)
    assert g.ecount() == 2
    assert g.es[g.get_eid(0, 1)]["weight"] == 8


def test_leiden_is_deterministic_and_finds_cliques() -> None:
    g = comm.undirected_graph(two_cliques(), 8)
    a = comm.leiden(g, 1.0, seed=1)
    assert a == comm.leiden(g, 1.0, seed=1)
    assert len(set(a[:4])) == 1 and len(set(a[4:])) == 1 and a[0] != a[4]


def test_coherence_identical_vs_orthogonal() -> None:
    same = csr_matrix([[1.0, 0.0]] * 8)
    assert comm.coherence([0] * 8, same) == pytest.approx(1.0)
    ortho = csr_matrix([[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)])
    assert comm.coherence([0] * 8, ortho) == pytest.approx(0.0)
    assert comm.coherence([0] * 7 + [1], same) == 0.0  # no community reaches 8 members


def test_choose_resolution_prefers_in_band_balance() -> None:
    m = pl.DataFrame(
        {
            "resolution": [1.0, 2.0, 5.0],
            "median_size": [400.0, 100.0, 40.0],
            "modularity": [0.9, 0.7, 0.6],
            "coherence": [0.05, 0.07, 0.09],
        }
    )
    # 1.0 is out of band; 0.6*0.09 = 0.054 beats 0.7*0.07 = 0.049
    assert comm.choose_resolution(m) == 5.0


def test_central_article_uses_member_probability_votes() -> None:
    # Article 0 receives one huge edge from 1; article 2 is linked to by 1, 3 and 4.
    e = pl.DataFrame(
        {"src_id": [1, 1, 3, 4], "dst_id": [0, 2, 2, 2], "n": [10_000, 50, 40, 30]}
    ).with_columns((pl.col("n") / pl.col("n").sum().over("src_id")).alias("p"))
    m = pl.DataFrame({"article_id": [0, 1, 2, 3, 4], "community_id": [0] * 5})
    ranked = comm.central_articles(e, m)
    assert ranked.filter(pl.col("rank") == 0)["article_id"].item() == 2


def test_ctfidf_picks_distinguishing_words() -> None:
    docs = [
        "cricket bat wicket cricket bowler match match",
        "chess pawn bishop chess knight match match",
    ]
    kw = comm.ctfidf_keywords([10, 20], docs, top_n=2)
    # "match" appears equally in both, so the class-specific word wins.
    assert kw[10][0] == "cricket" and kw[20][0] == "chess"


def test_community_table_labels_and_eligibility() -> None:
    e = two_cliques()
    titles = [f"Topic_{i}" for i in range(8)]
    texts = pl.DataFrame(
        {
            "article_id": list(range(8)),
            "title": titles,
            "lead_text": ["alpha alpha beta"] * 4 + ["gamma gamma delta"] * 4,
        }
    )
    m = pl.DataFrame({"article_id": list(range(8)), "community_id": [0] * 4 + [1] * 4})
    t = comm.community_table(m, comm.central_articles(e, m), texts)
    assert t["size"].to_list() == [4, 4]
    assert t["eligible"].to_list() == [False, False]
    assert t.row(0, named=True)["label"].startswith("Topic ")
    assert "alpha" in t.row(0, named=True)["keywords"]


def test_prune_keeps_top_k_and_penalises_hubs() -> None:
    e = pl.DataFrame(
        {"src_id": [0, 0, 0, 1, 2, 3], "dst_id": [1, 2, 3, 3, 3, 0], "n": [5, 3, 2, 1, 1, 1]}
    ).with_columns((pl.col("n") / pl.col("n").sum().over("src_id")).alias("p"))
    out = pruned.prune(e, k=2, hub_penalty=0.5)
    assert out.filter(pl.col("src_id") == 0)["dst_id"].to_list() == [1, 2]
    row = out.filter((pl.col("src_id") == 1) & (pl.col("dst_id") == 3)).row(0, named=True)
    # p = 1, node 3 has degree 4 (3 in + 1 out): cost = 0 + 0.5 * log(5)
    assert row["cost"] == pytest.approx(0.5 * math.log(5))


def test_sample_report_lists_top_articles() -> None:
    table = pl.DataFrame(
        {
            "community_id": [0],
            "size": [9],
            "eligible": [True],
            "label": ["Chess (chess, pawn)"],
            "keywords": [["chess", "pawn"]],
            "top_article_ids": [[1, 0]],
        }
    )
    md = report.sample_report(
        table, pl.DataFrame({"article_id": [0, 1], "title": ["Pawn", "Chess"]})
    )
    assert "## Chess (chess, pawn)" in md
    assert "1. Chess\n2. Pawn" in md
