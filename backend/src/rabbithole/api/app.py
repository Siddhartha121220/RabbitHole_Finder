"""FastAPI app. Run with `make api`; OpenAPI docs at /docs."""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from rabbithole.api.observability import RateLimiter, configure_logging, request_context
from rabbithole.api.routes import router
from rabbithole.api.services import Services, load_services
from rabbithole.config import Settings, get_settings
from rabbithole.db.session import get_engine


def create_app(
    settings: Settings | None = None, loader: Callable[[], Services] | None = None
) -> FastAPI:
    """`loader` lets tests inject a small catalog and a fake encoder."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        configure_logging()
        app.state.services = (loader or (lambda: load_services(settings, get_engine())))()
        app.state.trail_limiter = RateLimiter(settings.trail_rate_per_minute)
        yield

    app = FastAPI(
        title="Rabbit Hole Finder API",
        version="0.1.0",
        description="Recommends Wikipedia rabbit holes from a few things you already love.",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.middleware("http")(request_context)
    app.include_router(router)
    return app


app = create_app()
