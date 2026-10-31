from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from rabbithole.config import Settings
from rabbithole.embed import embeddings as emb
from rabbithole.embed import umap_map


class FakeEncoder:
    """Deterministic 4-d 'embeddings' from text length and vowel count; records calls."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append(list(sentences))
        v = np.array(
            [[len(s), sum(c in "aeiou" for c in s), 1.0, 0.5][: self.dim] for s in sentences],
            dtype=np.float32,
        )
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def settings(tmp_path: Path, **kw: Any) -> Settings:
    base: dict[str, Any] = {"data_dir": tmp_path, "embedding_dim": 4, "embed_shard_rows": 2}
    return Settings(_env_file=None, **(base | kw))  # pyright: ignore[reportCallIssue]


def texts(n: int) -> pl.DataFrame:
    return pl.DataFrame({"id": list(range(n)), "text": [f"text number {i}" for i in range(n)]})


def test_embed_table_shards_and_resumes(tmp_path: Path) -> None:
    s = settings(tmp_path)
    model = FakeEncoder()
    out = emb.embed_table(texts(5), "articles", s, model)
    assert out.height == 5 and len(model.calls) == 3  # shards of 2, 2, 1
    assert np.allclose(np.linalg.norm(out["embedding"].to_numpy(), axis=1), 1.0)

    again = FakeEncoder()
    assert emb.embed_table(texts(5), "articles", s, again).equals(out)
    assert again.calls == []  # everything came from the shards on disk

    changed = texts(5).with_columns(
        pl.when(pl.col("id") == 4).then(pl.lit("new text")).otherwise("text").alias("text")
    )
    third = FakeEncoder()
    emb.embed_table(changed, "articles", s, third)
    assert third.calls == [["new text"]]  # only the shard whose contents changed


def test_embed_table_rejects_wrong_dimension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        emb.embed_table(texts(2), "articles", settings(tmp_path), FakeEncoder(dim=3))


def test_embed_table_prefers_imported_vectors(tmp_path: Path) -> None:
    s = settings(tmp_path)
    imported = emb._with_vectors(pl.Series("id", [0]), np.ones((1, 4), np.float32) / 2)  # pyright: ignore[reportPrivateUsage]
    (tmp_path / "import").mkdir()
    imported.write_parquet(tmp_path / "import" / "articles_embeddings.parquet")
    model = FakeEncoder()
    assert emb.embed_table(texts(1), "articles", s, model).equals(imported)
    assert model.calls == []


def test_query_gets_instruction_prefix(tmp_path: Path) -> None:
    model = FakeEncoder()
    emb.embed_query("  cricket ", settings(tmp_path), model)
    assert model.calls == [[Settings.model_fields["embedding_query_prefix"].default + "cricket"]]


def test_article_inputs_use_title_and_lead() -> None:
    df = pl.DataFrame(
        {"article_id": [1, 0], "title": ["Test_cricket", "Cricket"], "lead_text": [None, "A game."]}
    )
    assert emb.article_inputs(df)["text"].to_list() == ["Cricket. A game.", "Test cricket"]


def test_community_geometry_centroids_and_positions() -> None:
    vecs = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    membership = pl.DataFrame({"article_id": [0, 1, 2], "community_id": [7, 7, 9]})
    geo = umap_map.community_geometry(membership, vecs, xy)
    first = geo.row(0, named=True)
    assert first["community_id"] == 7
    assert np.allclose(first["centroid"], [2**-0.5, 2**-0.5])
    assert (first["umap_x"], first["umap_y"]) == (0.5, 0.5)
