from alembic import context

from rabbithole.db.models import Base
from rabbithole.db.session import get_engine

target_metadata = Base.metadata


def run_migrations_online() -> None:
    with get_engine().connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
