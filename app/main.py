"""
FastAPI application factory.

Responsibilities:
  - Create the FastAPI instance with metadata
  - Register routers
  - Mount middleware (CORS)
  - Manage startup / shutdown lifecycle (scheduler, HTTP client)
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.events import router as events_router


def _configure_logging(debug: bool = False) -> None:
    """Set up structured logging for production or verbose logging for dev."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Silence noisy third-party loggers in production.
    if not debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("apscheduler.executors").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# ── Shared resources managed by the lifespan ────────────────────
# These are set during startup and torn down during shutdown.
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle hook."""
    global http_client
    settings = get_settings()
    scheduler = None

    # ── Startup ──────────────────────────────────────────────
    http_client = httpx.AsyncClient(timeout=30.0)
    logger.info("HTTP client initialized")

    if settings.scheduler_enabled:
        from app.db.supabase import get_supabase_client
        from app.scheduler.jobs import create_scheduler

        db_client = get_supabase_client(settings)
        scheduler = create_scheduler(http_client, db_client, settings)
        scheduler.start()
        logger.info("Scheduler started")
    else:
        logger.info("Scheduler disabled via SCHEDULER_ENABLED=false")

    logger.info("%s v%s is ready", settings.app_name, settings.app_version)

    yield

    # ── Shutdown ─────────────────────────────────────────────
    if scheduler is not None:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")

    await http_client.aclose()
    logger.info("HTTP client closed")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()
    _configure_logging(settings.debug)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production backend that fetches historical events from Wikipedia, "
            "generates AI summaries via Gemini, stores them in Supabase, "
            "and serves them through REST APIs."
        ),
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────
    application.include_router(events_router)

    # ── Health check ─────────────────────────────────────────
    @application.get("/health", tags=["infra"])
    async def health_check():
        return {"status": "ok", "version": settings.app_version}

    return application


# ── Module-level app instance (used by `uvicorn app.main:app`) ──
app = create_app()
