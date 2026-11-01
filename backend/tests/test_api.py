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
    return Services(get_settings(), engine, catalog, FakeEncoder())


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
    first = client.post("/rabbit-holes/2/trail", json={"seed_titles": ["A5"]})
    assert first.status_code == 200
    assert [s["title"] for s in first.json()["steps"]] == ["A4"]  # seed A5 left out
    second = client.post("/rabbit-holes/2/trail", json={"seed_titles": ["A5"]})
    assert second.json() == first.json()
    assert count(engine, "trails") == 1
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
