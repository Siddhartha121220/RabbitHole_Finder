"""articles.search_share: share of an article's visits that come from search (trail ordering)

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("search_share", sa.REAL))


def downgrade() -> None:
    op.drop_column("articles", "search_share")
