"""initial schema: communities, articles, chunks, trails, feedback, pipeline_runs

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from rabbithole.config import get_settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

DIM = get_settings().embedding_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "communities",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("hook", sa.Text),
        sa.Column("size", sa.Integer, nullable=False),
        sa.Column("eligible", sa.Boolean, nullable=False),
        sa.Column("keywords", ARRAY(sa.Text), nullable=False),
        sa.Column("central_article_id", sa.Integer, nullable=False),
        sa.Column("top_article_ids", ARRAY(sa.Integer), nullable=False),
        sa.Column("rh_score", sa.Float),
        sa.Column("rh_features", JSONB),
        sa.Column("centroid", Vector(DIM), nullable=False),
        sa.Column("umap_x", sa.Float, nullable=False),
        sa.Column("umap_y", sa.Float, nullable=False),
    )
    op.create_table(
        "articles",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("title", sa.Text, nullable=False, unique=True),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("views", sa.Integer, nullable=False),
        sa.Column(
            "community_id", sa.Integer, sa.ForeignKey("communities.id"), nullable=False, index=True
        ),
        sa.Column("lead_text", sa.Text),
        sa.Column("embedding", Vector(DIM), nullable=False),
        sa.Column("umap_x", sa.Float, nullable=False),
        sa.Column("umap_y", sa.Float, nullable=False),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column(
            "article_id",
            sa.Integer,
            sa.ForeignKey("articles.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(DIM), nullable=False),
    )
    for table, col in [
        ("articles", "embedding"),
        ("chunks", "embedding"),
        ("communities", "centroid"),
    ]:
        op.execute(
            f"CREATE INDEX ix_{table}_{col}_hnsw ON {table} USING hnsw ({col} vector_cosine_ops)"
        )
    op.create_table(
        "trails",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("community_id", sa.Integer, nullable=False, index=True),
        sa.Column("seed_key", sa.Text, nullable=False),
        sa.Column("model", sa.Text, nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("session_id", sa.Text, nullable=False),
        sa.Column("community_id", sa.Integer, nullable=False),
        sa.Column("seed_titles", ARRAY(sa.Text), nullable=False),
        sa.Column("surprise", sa.REAL, nullable=False),
        sa.Column("rating", sa.Text, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("rating IN ('up', 'down', 'obsessed')", name="rating_values"),
    )
    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_mode", sa.Text, nullable=False),
        sa.Column("clickstream_months", ARRAY(sa.Text), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("stats", JSONB, nullable=False),
    )


def downgrade() -> None:
    for table in ["pipeline_runs", "feedback", "trails", "chunks", "articles", "communities"]:
        op.drop_table(table)
