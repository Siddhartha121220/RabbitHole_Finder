"""Postgres schema (ARCHITECTURE §4). Vector width comes from EMBEDDING_DIM."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    REAL,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rabbithole.config import get_settings

DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    pass


class Community(Base):
    __tablename__ = "communities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    label: Mapped[str] = mapped_column(Text)
    hook: Mapped[str | None] = mapped_column(Text)
    size: Mapped[int] = mapped_column(Integer)
    eligible: Mapped[bool]
    keywords: Mapped[list[str]] = mapped_column(ARRAY(Text))
    central_article_id: Mapped[int] = mapped_column(Integer)
    top_article_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    rh_score: Mapped[float | None] = mapped_column(Float)
    rh_features: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    centroid: Mapped[Any] = mapped_column(Vector(DIM))
    umap_x: Mapped[float] = mapped_column(Float)
    umap_y: Mapped[float] = mapped_column(Float)


class Article(Base):
    __tablename__ = "articles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(Text, unique=True)
    url: Mapped[str] = mapped_column(Text)
    views: Mapped[int] = mapped_column(Integer)
    community_id: Mapped[int] = mapped_column(ForeignKey("communities.id"), index=True)
    lead_text: Mapped[str | None] = mapped_column(Text)
    search_share: Mapped[float | None] = mapped_column(REAL)  # visits from search / all visits
    embedding: Mapped[Any] = mapped_column(Vector(DIM))
    umap_x: Mapped[float] = mapped_column(Float)
    umap_y: Mapped[float] = mapped_column(Float)


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )
    ord: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(DIM))


class Trail(Base):
    """Cache of generated trails, keyed by (community_id, seed_key, model)."""

    __tablename__ = "trails"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    community_id: Mapped[int] = mapped_column(Integer, index=True)
    seed_key: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IN ('up', 'down', 'obsessed')", name="rating_values"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text)
    community_id: Mapped[int] = mapped_column(Integer)  # no FK: feedback outlives data reloads
    seed_titles: Mapped[list[str]] = mapped_column(ARRAY(Text))
    surprise: Mapped[float] = mapped_column(REAL)
    rating: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_mode: Mapped[str] = mapped_column(Text)
    clickstream_months: Mapped[list[str]] = mapped_column(ARRAY(Text))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stats: Mapped[dict[str, Any]] = mapped_column(JSONB)


class RecommendationLog(Base):
    """One row per /recommend call. Interests are stored only as a SHA-256 of the
    normalised strings: enough to count repeats, nothing a person typed is kept."""

    __tablename__ = "recommendation_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    interests_hash: Mapped[str] = mapped_column(Text)
    n_interests: Mapped[int] = mapped_column(Integer)
    surprise: Mapped[float] = mapped_column(REAL)
    community_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    latency_ms: Mapped[float] = mapped_column(REAL)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
