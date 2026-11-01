"""recommendation_log: one row per /recommend call, no personal data

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "recommendation_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("interests_hash", sa.Text, nullable=False),
        sa.Column("n_interests", sa.Integer, nullable=False),
        sa.Column("surprise", sa.REAL, nullable=False),
        sa.Column("community_ids", ARRAY(sa.Integer), nullable=False),
        sa.Column("latency_ms", sa.REAL, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("recommendation_log")
