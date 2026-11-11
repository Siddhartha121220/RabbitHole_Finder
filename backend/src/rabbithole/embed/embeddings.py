"""Sentence embeddings for article leads, chunks and user queries (CPU by default).

Long runs are written in shards so a crash resumes where it stopped. For a GPU run elsewhere
(Colab/Kaggle), `export_texts` writes the inputs and any `data/import/<name>_embeddings.parquet`
found later is used instead of embedding locally (see README).
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import polars as pl

from rabbithole.config import Settings

log = logging.getLogger(__name__)


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


@lru_cache
def load_model(name: str, device: str = "cpu") -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=device)
    if device.startswith("cuda"):
        model.half()  # fp16 on GPU: ~2x faster, cosine rankings unchanged in practice
    return cast(Encoder, model)


class MultiDevice:
    """One encoder process per GPU (sentence-transformers' multi-process pool), started
    once and reused for every shard. Close it when done."""

    def __init__(self, name: str, devices: list[str]) -> None:
        self.model: Any = load_model(name, devices[0])
        self.pool: Any = self.model.start_multi_process_pool(devices)

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        return self.model.encode(sentences, pool=self.pool, **kwargs)

    def close(self) -> None:
        self.model.stop_multi_process_pool(self.pool)


def batch_encoder(settings: Settings) -> Encoder:
    """Encoder for bulk embedding, from EMBED_DEVICES (CPU when empty)."""
    devices = settings.embed_devices
    if len(devices) > 1:
        return MultiDevice(settings.embedding_model, devices)
    return load_model(settings.embedding_model, devices[0] if devices else "cpu")


def encode(model: Encoder, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Unit-length float32 vectors, so cosine similarity is a dot product."""
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str, settings: Settings, model: Encoder | None = None) -> np.ndarray:
    model = model or load_model(settings.embedding_model)
    return encode(model, [settings.embedding_query_prefix + text.strip()])[0]


def _with_vectors(ids: pl.Series, vecs: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"id": ids, "embedding": vecs}).with_columns(
        pl.col("embedding").cast(pl.Array(pl.Float32, vecs.shape[1]))
    )


def embed_table(
    texts: pl.DataFrame, name: str, settings: Settings, model: Encoder | None = None
) -> pl.DataFrame:
    """Embed `texts` (columns id, text) -> (id, embedding). Resumable via shards on disk."""
    imported = settings.data_dir / "import" / f"{name}_embeddings.parquet"
    if imported.exists():
        log.info("using imported vectors %s", imported)
        return pl.read_parquet(imported)

    model_slug = settings.embedding_model.replace("/", "__")
    shard_dir = (
        settings.data_dir / "interim" / "embeddings" / settings.dataset_mode / model_slug / name
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    step = settings.embed_shard_rows
    parts: list[pl.DataFrame] = []
    encoder: Encoder | None = model
    for start in range(0, texts.height, step):
        part = texts.slice(start, step)
        # Shards are named by a hash of their contents, so a changed input never reuses
        # stale vectors and an unchanged one is never embedded twice.
        key = hashlib.sha1(part.select("id", "text").write_csv().encode()).hexdigest()[:16]
        shard = shard_dir / f"{key}.parquet"
        if not shard.exists():
            encoder = encoder or batch_encoder(settings)  # loaded only if a shard is missing
            vecs = encode(encoder, part["text"].to_list(), settings.embed_batch_size)
            if vecs.shape[1] != settings.embedding_dim:
                raise ValueError(
                    f"model gives {vecs.shape[1]}-d vectors, EMBEDDING_DIM is "
                    f"{settings.embedding_dim}"
                )
            _with_vectors(part["id"], vecs).write_parquet(shard)
            log.info("%s: embedded %d / %d", name, min(start + step, texts.height), texts.height)
        parts.append(pl.read_parquet(shard))
    if isinstance(encoder, MultiDevice):
        encoder.close()
    if not parts:
        return _with_vectors(
            pl.Series("id", [], pl.UInt32), np.zeros((0, settings.embedding_dim), np.float32)
        )
    return pl.concat(parts)


def export_texts(texts: pl.DataFrame, name: str, settings: Settings) -> Path:
    """Write (id, text) for embedding on a GPU machine; see README "GPU path"."""
    out = settings.data_dir / "export" / f"{name}_texts.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    texts.select("id", "text").write_parquet(out)
    return out


def article_inputs(text_table: pl.DataFrame) -> pl.DataFrame:
    """Embedding input per article: title + lead (title alone if the lead is missing)."""
    title = pl.col("title").str.replace_all("_", " ")
    return text_table.sort("article_id").select(
        pl.col("article_id").alias("id"),
        pl.when(pl.col("lead_text").fill_null("") != "")
        .then(title + pl.lit(". ") + pl.col("lead_text"))
        .otherwise(title)
        .alias("text"),
    )


def chunk_inputs(chunks: pl.DataFrame) -> pl.DataFrame:
    return chunks.sort("chunk_id").select(pl.col("chunk_id").alias("id"), "text")
