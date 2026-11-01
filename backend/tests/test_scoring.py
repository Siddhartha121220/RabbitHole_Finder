from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rabbithole.scoring import labels
from rabbithole.scoring.features import FEATURES, community_features, expected_depth
from rabbithole.scoring.score import breakdown, score


def test_expected_depth_three_node_chain_by_hand() -> None:
    # 0 -> 1 -> 2 with continue-probability 0.5 per step; 2 always leaves.
    # Pages viewed starting at 2: y2 = 1; at 1: y1 = 1 + 0.5*y2 = 1.5; at 0: y0 = 1 + 0.5*y1 = 1.75
    src, dst, q = np.array([0, 1]), np.array([1, 2]), np.array([0.5, 0.5])
    assert expected_depth(src, dst, q, np.array([1.0, 0.0, 0.0])) == pytest.approx(1.75)
    # Entry split evenly between 0 and 1: 0.5 * 1.75 + 0.5 * 1.5
    assert expected_depth(src, dst, q, np.array([1.0, 1.0, 0.0])) == pytest.approx(1.625)


def test_expected_depth_cycle_and_edge_cases() -> None:
    # Two pages linking to each other with p = 0.5: y = 1 + 0.5 y  ->  y = 2
    two = expected_depth(
        np.array([0, 1]), np.array([1, 0]), np.array([0.5, 0.5]), np.array([1.0, 1.0])
    )
    assert two == pytest.approx(2.0)
    lone = expected_depth(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([]), np.array([5.0])
    )
    assert lone == pytest.approx(1.0)  # one page, no internal links: you read it and leave
    assert expected_depth(np.array([]), np.array([]), np.array([]), np.array([])) == 0.0


def tiny_community() -> tuple[pl.DataFrame, ...]:
    """Community 0 = {0, 1}, community 1 = {2}."""
    articles = pl.DataFrame(
        {
            "article_id": [0, 1, 2],
            "views_in": [100, 50, 80],
            "search_in": [60, 10, 40],
            "internal_in": [20, 10, 20],
            "link_in": [20, 30, 20],
            "onward_clicks": [50, 20, 10],
            "onward_ratio": [0.5, 0.4, 0.125],
        }
    )
    edges = pl.DataFrame({"src_id": [0, 1, 0], "dst_id": [1, 0, 2], "n": [30, 10, 20]})
    membership = pl.DataFrame({"article_id": [0, 1, 2], "community_id": [0, 0, 1]})
    page_length = pl.DataFrame({"article_id": [0, 1, 2], "page_length": [1000, 3000, 500]})
    return articles, edges, membership, page_length


def test_community_features_by_hand() -> None:
    f = community_features(*tiny_community())
    c0 = f.filter(pl.col("community_id") == 0).row(0, named=True)
    assert c0["size"] == 2
    assert c0["stickiness"] == pytest.approx(40 / 70)  # internal clicks / all onward clicks
    assert c0["onward_ratio"] == pytest.approx((0.5 * 100 + 0.4 * 50) / 150)
    assert c0["content_richness"] == pytest.approx(np.log1p(2000))
    # wander in = other-internal 30 + link 50 - internal 40 = 40; search = 70
    assert c0["discovery_share"] == pytest.approx(40 / 110)
    # entries: article 0 = 100 - 10, article 1 = 50 - 30; q01 = 30/101, q10 = 10/51
    q01, q10 = 30 / 101, 10 / 51
    y = np.linalg.solve(np.eye(2) - np.array([[0, q01], [q10, 0]]), np.ones(2))
    assert c0["expected_depth"] == pytest.approx((90 * y[0] + 20 * y[1]) / 110)


def test_score_percentiles_and_weights() -> None:
    f = pl.DataFrame({"community_id": [0, 1, 2]} | {feat: [1.0, 2.0, 3.0] for feat in FEATURES})
    s = score(f, {feat: 1.0 for feat in FEATURES})
    assert s["rh_score"].to_list() == pytest.approx([100 / 3, 200 / 3, 100.0])
    # Weight only stickiness, and reverse it: ranking flips.
    flipped = f.with_columns(pl.Series("stickiness", [3.0, 2.0, 1.0]))
    s2 = score(flipped, {"stickiness": 1.0})
    assert s2["rh_score"].to_list() == pytest.approx([100.0, 200 / 3, 100 / 3])
    b = breakdown(s.row(2, named=True))
    assert set(b) == {*FEATURES, "rh_raw"} and b["stickiness"]["value"] == 3.0


def test_labels_resume_and_validate(tmp_path: Path) -> None:
    comms = pl.DataFrame({"community_id": list(range(6)), "eligible": [True] * 5 + [False]})
    path = tmp_path / "labels.csv"
    first = labels.to_label(comms, path, n=3)
    assert len(first) == 3 and 5 not in first
    labels.append_label(path, first[0], "Chess", 4)
    assert labels.to_label(comms, path, n=3) == first[1:]  # resumes in the same order
    assert labels.read_labels(path)["rating"].to_list() == [4]
    labels.append_label(path, first[1], "Sushi", 7)  # 7-point scale
    with pytest.raises(ValueError):
        labels.append_label(path, first[2], "x", 8)


def test_first_sentence() -> None:
    lead = "Chess is a board game for two players. It is played on an 8x8 board."
    assert labels.first_sentence(lead) == "Chess is a board game for two players."
    assert labels.first_sentence(lead, max_chars=10) == "Chess is…"
    assert labels.first_sentence(None) == ""
    film = "Adipurush (transl. First man) is a 2023 film. It was panned."
    assert labels.first_sentence(film) == "Adipurush (transl. First man) is a 2023 film."
