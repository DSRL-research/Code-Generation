"""The FastAPI app. Run it with: uvicorn src.main:app --port 8095"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from src.config import Settings, get_settings
from src.config.services import build_chain_service
from src.routers import chains, health, models, operators
from src.routers.errors import register_error_handlers
from src.store import ModelStore


def create_app(settings: Settings | None = None, transport: httpx.BaseTransport | None = None) -> FastAPI:
    """`settings` and `transport` replace the defaults in tests (transport: httpx.MockTransport for no network)."""
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # the source client logs each fetch itself

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One HTTP client for all runs. Redirects are not followed: they could lead outside the allowed links.
        with httpx.Client(follow_redirects=False, transport=transport) as http:
            app.state.model_store = ModelStore(settings.model_dir)
            app.state.chain_service = build_chain_service(settings, http, app.state.model_store)
            yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(chains.router)
    app.include_router(operators.router)
    app.include_router(models.router)
    return app


app = create_app()
