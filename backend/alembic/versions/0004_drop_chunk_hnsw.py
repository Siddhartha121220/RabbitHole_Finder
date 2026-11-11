"""drop the HNSW index on chunks.embedding

Trail retrieval always filters to ~30 candidate articles first (article_id btree), then
ranks their few hundred chunks exactly. The approximate index was never the right plan for
that query (filtering after an ANN scan can miss rows) and costs hours to build on FULL.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")


def downgrade() -> None:
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
