"""API tests: FastAPI TestClient against a real Postgres test database loaded from fixtures.

Six articles in three communities, a click chain A0 -> A1 -> ... -> A5, one-hot embeddings,
and a fake encoder that maps the query "a<i>" onto article i's vector.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import Engine

from rabbithole.api.app import create_app
from rabbithole.api.observability import RateLimiter
from rabbithole.api.services import Services
from rabbithole.config import get_settings
from rabbithole.db import loader
from rabbithole.llm.provider import (
    DraftStep,
    RetrievedChunk,
    TemplateProvider,
    TrailDraft,
    TrailRequest,
    Usage,
)
from rabbithole.recommend.catalog import from_frames

DB_NAME = "rabbithole_api_test"  # read by the `engine` fixture in conftest
DIM = get_settings().embedding_dim
N = 6
TITLES = [f"A{i}" for i in range(N)]
COMMUNITY = [0, 0, 1, 1, 2, 2]


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class FakeEncoder:
    """'... a3' -> unit(3); anything else -> a direction no article has."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        def one(s: str) -> np.ndarray:
            word = s.split()[-1].lower()
            ok = word.startswith("a") and word[1:].isdigit() and int(word[1:]) < N
            return unit(int(word[1:]) if ok else DIM - 1)

        return np.stack([one(s) for s in sentences])


def load_fixture_data(engine: Engine) -> None:
    comms = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "label": ["Zero", "One", "Two"],
            "size": [2, 2, 2],
            "eligible": [True] * 3,
            "keywords": [["zero"], ["one"], ["two"]],
            "central_article_id": [0, 2, 4],
            "top_article_ids": [[0, 1], [2, 3], [4, 5]],
            "centroid": [unit(0).tolist(), unit(2).tolist(), unit(4).tolist()],
            "umap_x": [0.1, 0.5, 0.9],
            "umap_y": [0.1, 0.5, 0.9],
            "rh_score": [50.0, 80.0, 20.0],
            "rh_features": [{"stickiness": {"value": 0.5, "z": 0.1}, "rh_raw": 0.1}] * 3,
        }
    )
    arts = pl.DataFrame(
        {
            "id": list(range(N)),
            "title": TITLES,
            "url": [loader.wiki_url(t) for t in TITLES],
            "views": [600, 500, 400, 300, 200, 100],
            "community_id": COMMUNITY,
            "lead_text": [f"{t} is a test article. More text." for t in TITLES],
            "embedding": [unit(i).tolist() for i in range(N)],
            "umap_x": [0.1] * N,
            "umap_y": [0.2] * N,
        }
    )
    chunks = pl.DataFrame(
        {"id": [0], "article_id": [0], "ord": [0], "text": ["x"], "embedding": [unit(0).tolist()]}
    )
    # Community 3 ("Deep", not eligible so it stays off the map): 7 articles with a passage
    # each, enough for an LLM trail of 5-7 steps.
    deep = list(range(N, N + 7))
    comms = pl.concat(
        [
            comms,
            pl.DataFrame(
                {
                    "id": [3],
                    "label": ["Deep"],
                    "size": [7],
                    "eligible": [False],
                    "keywords": [["deep"]],
                    "central_article_id": [deep[0]],
                    "top_article_ids": [deep],
                    "centroid": [unit(10).tolist()],
                    "umap_x": [0.3],
                    "umap_y": [0.3],
                    "rh_score": [None],
                    "rh_features": [None],
                }
            ),
        ],
        how="diagonal_relaxed",
    )
    arts = pl.concat(
        [
            arts.with_columns(search_share=pl.lit(0.3)),
            pl.DataFrame(
                {
                    "id": deep,
                    "title": [f"D{i}" for i in deep],
                    "url": [loader.wiki_url(f"D{i}") for i in deep],
                    "views": [1000 - 10 * i for i in deep],
                    "search_share": [0.5] * 7,
                    "community_id": [3] * 7,
                    "lead_text": [f"D{i} is deep." for i in deep],
                    "embedding": [unit(i).tolist() for i in deep],
                    "umap_x": [0.3] * 7,
                    "umap_y": [0.3] * 7,
                }
            ),
        ],
        how="diagonal_relaxed",
    )
    chunks = pl.DataFrame(
        {
            "id": [0, *deep],
            "article_id": [0, *deep],
            "ord": [0] * 8,
            "text": ["x", *(f"Passage about D{i}." for i in deep)],
            "embedding": [unit(0).tolist(), *(unit(i).tolist() for i in deep)],
        }
    )
    with engine.begin() as conn:
        loader.load(conn, comms, arts, chunks, {"dataset_mode": "dev", "months": []})


def services(engine: Engine) -> Services:
    catalog = from_frames(
        pl.DataFrame({"article_id": range(N), "title": TITLES}),
        pl.DataFrame({"id": range(N), "embedding": [unit(i) for i in range(N)]}),
        pl.DataFrame({"article_id": range(N), "community_id": COMMUNITY}),
        pl.DataFrame(
            {"community_id": [0, 1, 2], "rh_score": [50.0, 80.0, 20.0], "popularity": [1.0] * 3}
        ),
        pl.DataFrame({"src_id": [0, 1, 2, 3, 4], "dst_id": [1, 2, 3, 4, 5], "cost": [0.5] * 5}),
    )
    return Services(get_settings(), engine, catalog, FakeEncoder(), TemplateProvider())


@pytest.fixture(scope="module")
def client(engine: Engine) -> Iterator[TestClient]:
    load_fixture_data(engine)
    settings = get_settings().model_copy(update={"trail_rate_per_minute": 2})
    app = create_app(settings, loader=lambda: services(engine))
    with TestClient(app) as c:
        yield c


def count(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "database": True,
        "model_loaded": True,
        "catalog_loaded": True,
        "communities": 3,
    }


def test_seed_search(client: TestClient) -> None:
    hits = client.post("/seeds/search", json={"query": "a3", "k": 2}).json()
    assert hits[0]["title"] == "A3" and hits[0]["url"] == "https://en.wikipedia.org/wiki/A3"


def test_recommend_returns_bridge_path_from_our_table_and_logs(
    client: TestClient, engine: Engine
) -> None:
    before = count(engine, "recommendation_log")
    r = client.post("/recommend", json={"interests": ["a0"], "surprise": 0.0, "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert [s["title"] for s in body["seeds"]] == ["A0"]
    holes = {h["id"]: h for h in body["rabbit_holes"]}
    assert set(holes) == {1, 2}  # never the seed's own community
    assert [a["title"] for a in holes[1]["bridge_path"]] == ["A0", "A1", "A2"]
    assert "reach A2 in 2 clicks via A1" in holes[1]["why"]
    assert holes[2]["bridge_path"][-1]["title"] == "A4"
    urls = {a["url"] for h in holes.values() for a in h["bridge_path"] + h["top_articles"]}
    assert urls <= {loader.wiki_url(t) for t in TITLES}
    assert holes[1]["rh_breakdown"]["stickiness"] == {"value": 0.5, "z": 0.1}
    assert holes[1]["hook"] == "A2 is a test article." and holes[1]["x"] == 0.5
    assert body["seeds"][0]["x"] == 0.1
    assert count(engine, "recommendation_log") == before + 1
    with engine.connect() as conn:
        stored = conn.execute(text("SELECT interests_hash FROM recommendation_log")).scalar_one()
    assert "a0" not in stored and len(stored) == 64  # hashed, never the raw text


def test_recommend_validation_and_no_match(client: TestClient) -> None:
    assert client.post("/recommend", json={"interests": []}).status_code == 422
    assert client.post("/recommend", json={"interests": ["a0"] * 6}).status_code == 422
    assert client.post("/recommend", json={"interests": ["a0"], "surprise": 2}).status_code == 422
    r = client.post("/recommend", json={"interests": ["nothing like it"]})
    assert r.status_code == 422 and "matched" in r.json()["detail"]


def test_rabbit_hole_detail_and_404(client: TestClient) -> None:
    d = client.get("/rabbit-holes/1").json()
    assert d["central_article"]["title"] == "A2"
    assert d["about"] == "A2 is a test article."
    assert client.get("/rabbit-holes/99").status_code == 404


def test_trail_is_cached_and_rate_limited(client: TestClient, engine: Engine) -> None:
    client.app.state.trail_limiter = RateLimiter(per_minute=2)  # type: ignore[attr-defined]
    first = client.post("/rabbit-holes/2/trail", json={"seed_titles": ["A5"]})
    assert first.status_code == 200
    assert [s["title"] for s in first.json()["steps"]] == ["A4"]  # seed A5 left out
    second = client.post("/rabbit-holes/2/trail", json={"seed_titles": ["A5"]})
    assert second.json() == first.json()
    assert count(engine, "trails") == 1
    assert first.json()["model"] == "template"
    assert client.post("/rabbit-holes/2/trail", json={}).status_code == 429  # limit is 2/min


def test_map_and_feedback(client: TestClient, engine: Engine) -> None:
    points = client.get("/map").json()
    assert {p["id"] for p in points} == {0, 1, 2}
    ok = {"session_id": "s1", "community_id": 1, "surprise": 0.5, "rating": "obsessed"}
    assert client.post("/feedback", json=ok).status_code == 201
    assert client.post("/feedback", json=ok | {"rating": "meh"}).status_code == 422
    assert count(engine, "feedback") == 1


def test_request_id_and_cors(client: TestClient) -> None:
    r = client.get("/health", headers={"X-Request-ID": "abc", "Origin": "http://localhost:3000"})
    assert r.headers["x-request-id"] == "abc"
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    evil = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


def test_openapi_lists_every_endpoint(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {
        "/health",
        "/seeds/search",
        "/recommend",
        "/rabbit-holes/{community_id}",
        "/rabbit-holes/{community_id}/trail",
        "/map",
        "/feedback",
    }


def test_rate_limiter_refills() -> None:
    now = [0.0]
    limiter = RateLimiter(per_minute=2, clock=lambda: now[0])
    assert limiter.allow("ip") and limiter.allow("ip") and not limiter.allow("ip")
    assert limiter.allow("other")
    now[0] = 30.0  # half a minute refills one token
    assert limiter.allow("ip") and not limiter.allow("ip")


class FakeLLM:
    """Returns invented ids, duplicates and mismatched chunks alongside valid steps."""

    name = "fake-llm"

    def __init__(self, n_valid: int) -> None:
        self.n_valid = n_valid
        self.seen: list[RetrievedChunk] = []

    def generate_trail(
        self, request: TrailRequest, context: list[RetrievedChunk]
    ) -> tuple[TrailDraft, Usage]:
        self.seen = context
        valid = [
            DraftStep(article_id=c.article_id, chunk_id=c.chunk_id, why=f"Read {c.title}.")
            for c in context[: self.n_valid]
        ]
        junk = [
            DraftStep(article_id=999_999, chunk_id=None, why="Invented article."),
            DraftStep(article_id=context[0].article_id, chunk_id=None, why="Duplicate."),
            DraftStep(article_id=context[1].article_id, chunk_id=context[2].chunk_id, why="x"),
            DraftStep(article_id=0, chunk_id=0, why="Article from another community."),
        ]
        draft = TrailDraft(intro="Into the deep.", steps=[*junk[:1], *valid, *junk[1:]])
        return draft, Usage(self.name, input_tokens=100, output_tokens=50)


def trail_with(client: TestClient, llm: FakeLLM, **body: Any) -> dict[str, Any]:
    app: Any = client.app
    app.state.trail_limiter = RateLimiter(per_minute=100)
    old = app.state.services.llm
    app.state.services.llm = llm
    try:
        r = client.post("/rabbit-holes/3/trail", json=body)
        assert r.status_code == 200
        return r.json()
    finally:
        app.state.services.llm = old


def test_llm_trail_keeps_only_retrieved_ids_and_our_urls(client: TestClient) -> None:
    llm = FakeLLM(n_valid=6)
    out = trail_with(client, llm, interests=["d7"], seed_titles=["D6"])
    assert out["model"] == "fake-llm" and out["intro"] == "Into the deep."
    ids = [s["id"] for s in out["steps"]]
    assert len(ids) == 6 and len(set(ids)) == 6
    assert 6 not in ids and {c.article_id for c in llm.seen} <= set(range(7, 13))  # seed left out
    for st in out["steps"]:  # link validity: every URL is the article table's URL
        assert st["url"] == loader.wiki_url(st["title"].replace(" ", "_"))
    assert all(st["why"].startswith("Read D") for st in out["steps"])


def test_too_few_valid_llm_steps_fall_back_to_template(client: TestClient) -> None:
    out = trail_with(client, FakeLLM(n_valid=3), interests=["other"])
    assert out["model"] == "template"
    assert [s["title"] for s in out["steps"]][:2] == ["D6", "D7"]  # most-read first
    assert all(s["url"].startswith("https://en.wikipedia.org/wiki/D") for s in out["steps"])


def test_template_trail_lists_members_only(client: TestClient) -> None:
    out = trail_with(client, FakeLLM(n_valid=0), bridge_article_ids=[0, 1])
    assert out["model"] == "template"
    assert {s["title"] for s in out["steps"]} <= {f"D{i}" for i in range(6, 13)}
