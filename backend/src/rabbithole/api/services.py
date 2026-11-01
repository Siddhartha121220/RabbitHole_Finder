"""What the API loads once at startup and shares across requests."""

from dataclasses import dataclass

from sqlalchemy import Engine, text

from rabbithole.config import Settings
from rabbithole.embed.embeddings import Encoder, load_model
from rabbithole.llm.provider import LLMProvider, make_provider
from rabbithole.recommend import catalog as cat


@dataclass
class Services:
    settings: Settings
    engine: Engine
    catalog: cat.Catalog
    encoder: Encoder
    llm: LLMProvider

    def database_ok(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


def load_services(settings: Settings, engine: Engine) -> Services:
    """Catalog (pruned graph, centroids, scores) from the pipeline's Parquet, and the query
    encoder, warmed up so the first user request is not the slow one."""
    processed = settings.data_dir / "processed" / settings.dataset_mode
    encoder = load_model(settings.embedding_model)
    encoder.encode(["warm up"])
    return Services(settings, engine, cat.load(processed), encoder, make_provider(settings))
