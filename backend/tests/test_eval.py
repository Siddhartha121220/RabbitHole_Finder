import numpy as np
import polars as pl
import pytest

from rabbithole.eval import labels as lab
from rabbithole.eval import metrics as m
from rabbithole.recommend import ranker
from rabbithole.recommend.candidates import bridge_path, candidates
from rabbithole.recommend.catalog import Catalog, from_frames

# --- metrics on toy inputs with hand-computed answers ---


def test_ndcg_perfect_and_hand_computed() -> None:
    rel = {1: 3, 2: 2, 3: 0}
    assert m.ndcg_at_k([1, 2, 3], rel, 3) == pytest.approx(1.0)
    # ranked [3, 1]: dcg = 0 + 7/log2(3); ideal = 7 + 3/log2(3)
    expected = (7 / np.log2(3)) / (7 + 3 / np.log2(3))
    assert m.ndcg_at_k([3, 1], rel, 2) == pytest.approx(expected)
    assert m.ndcg_at_k([1], {1: 0}, 5) == 0.0


def test_recall_and_mrr() -> None:
    rel = {1: 3, 2: 2, 3: 1, 4: 0}  # relevant (>= 2): {1, 2}
    assert m.recall_at_k([4, 2, 3], rel, 10) == 0.5
    assert m.mrr([4, 3, 2], rel) == pytest.approx(1 / 3)
    assert m.mrr([4, 3], rel) == 0.0


def test_novelty_diversity_serendipity_coverage() -> None:
    assert m.novelty([1, 2], {1: 0.5, 2: 0.25}) == pytest.approx(1.5)  # (1 + 2) / 2
    vec = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0]), 3: np.array([1.0, 0.0])}
    assert m.diversity([1, 2], vec) == pytest.approx(1.0)
    assert m.diversity([1, 3], vec) == pytest.approx(0.0)
    assert m.serendipity([1, 2], {1: 3, 2: 3}, obvious=[1]) == 0.5
    assert m.coverage([[1, 2], [2, 3]], 6) == 0.5


def test_bootstrap_ci_brackets_mean_and_paired_detects_shift() -> None:
    x = np.random.default_rng(0).normal(1.0, 0.1, 200)
    mean, lo, hi = m.bootstrap_ci(x, 500)
    assert lo < mean < hi
    d, lo, hi, p = m.paired_bootstrap(x + 0.5, x, 500)
    assert d == pytest.approx(0.5) and lo > 0 and p < 0.01


# --- ranker pieces ---


def test_goldilocks_peaks_at_target() -> None:
    d = np.linspace(0, 1, 101)
    assert d[np.argmax(ranker.goldilocks(d, surprise=0.0))] == pytest.approx(0.25)
    assert d[np.argmax(ranker.goldilocks(d, surprise=1.0))] == pytest.approx(0.75)


def test_mmr_skips_near_duplicate() -> None:
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    scores = np.array([1.0, 0.95, 0.8])
    assert ranker.top_k(scores, 2) == [0, 1]
    assert ranker.mmr(scores, vecs, 2, lam=0.7) == [0, 2]


# --- tiny catalog: 3 communities, a chain 0 -> 1 -> 2 -> 3 -> 4 -> 5 ---


@pytest.fixture
def catalog() -> Catalog:
    n = 6
    emb = np.eye(n, 4, dtype=np.float32) + 0.1
    return from_frames(
        pl.DataFrame({"article_id": range(n), "title": [f"A{i}" for i in range(n)]}),
        pl.DataFrame({"id": range(n), "embedding": list(emb)}),
        pl.DataFrame({"article_id": range(n), "community_id": [0, 0, 1, 1, 2, 2]}),
        pl.DataFrame(
            {"community_id": [0, 1, 2], "rh_score": [50.0, 80.0, 20.0], "popularity": [1.0] * 3}
        ),
        pl.DataFrame({"src_id": [0, 1, 2, 3, 4], "dst_id": [1, 2, 3, 4, 5], "cost": [0.5] * 5}),
    )


def test_candidates_exclude_seed_community_and_find_bridge(catalog: Catalog) -> None:
    c = candidates(catalog, seeds=[0], max_hops=6)
    comms = catalog.community_ids[c.idx].tolist()
    assert 0 not in comms
    j1, j2 = comms.index(1), comms.index(2)
    assert c.graph_dist[j1] == 2 and c.graph_dist[j2] == 4  # 0->1->2, 0->...->4
    assert c.bridge_strength[j1] == pytest.approx(np.exp(-1.0))
    assert bridge_path(c, j1) == [0, 1, 2]
    assert bridge_path(c, j2, max_hops=3) == []


def test_unreachable_beyond_hop_cap(catalog: Catalog) -> None:
    c = candidates(catalog, seeds=[0], max_hops=3)
    j2 = catalog.community_ids[c.idx].tolist().index(2)
    assert c.graph_dist[j2] == 4 and c.bridge_strength[j2] == 0.0


# --- labels from future walks ---


def test_walk_mass_excludes_one_hop_and_grades() -> None:
    e = pl.DataFrame({"src_id": [0, 1, 2], "dst_id": [1, 2, 3], "n": [10, 10, 10]})
    t = lab.transition_matrix(e, 4)
    mass = lab.walk_mass(t, [0])
    assert mass.tolist() == [0.0, 0.0, 1.0, 1.0]  # node 1 is a direct link: excluded
    assert lab.grade(np.array([0.0, 1.0, 2.0, 3.0])).tolist() == [0, 1, 2, 3]


def test_relevance_skips_seed_community(catalog: Catalog) -> None:
    e = pl.DataFrame({"src_id": [0, 1, 2, 3], "dst_id": [1, 2, 3, 4], "n": [5, 5, 5, 5]})
    rel = lab.relevance(catalog, lab.transition_matrix(e, 6), [0])
    assert set(rel) == {1, 2}
    assert rel[1] > 0 and rel[2] == 0  # node 4 is 4 steps away: beyond the walk
