"""
APScheduler daily job definitions.

Runs the Fetch → AI → Store pipeline once per day at a configurable
UTC hour.  The scheduler uses AsyncIOScheduler so it shares the
FastAPI event loop — no extra threads or processes.

Design notes:
  - max_instances=1 prevents overlapping runs if a job takes > 24h
  - misfire_grace_time=3600 allows the job to run if the app was down
    during the scheduled time (up to 1 hour late)
  - coalesce=True means if multiple misfires stack up, only one run
    happens

Scaling note:
  For multi-instance deployments (e.g. multiple Kubernetes pods),
  APScheduler will run the job on EVERY instance.  At that point,
  migrate to a distributed scheduler (e.g. Cloud Scheduler →
  POST /api/events/trigger) or use APScheduler's JobStore with a
  shared DB backend.
"""

from __future__ import annotations

import logging
from datetime import date

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import Settings

logger = logging.getLogger(__name__)

JOB_ID = "daily_history_pipeline"


async def daily_pipeline_job(
    http_client,
    db_client,
    settings: Settings,
) -> None:
    """
    Job function executed by APScheduler once per day.

    Runs the pipeline for today's date.  All errors are caught
    inside `run_pipeline`, so this function will not raise.
    """
    # Import here to avoid circular imports at module level.
    from app.services.pipeline import run_pipeline

    today = date.today()
    logger.info(
        "[Scheduler] Daily job triggered for %02d/%02d",
        today.month, today.day,
    )

    result = await run_pipeline(
        http_client=http_client,
        db_client=db_client,
        month=today.month,
        day=today.day,
        settings=settings,
    )

    if result.success:
        logger.info(
            "[Scheduler] Daily job complete — "
            "fetched=%d skipped=%d enriched=%d stored=%d",
            result.fetched, result.skipped, result.enriched, result.stored,
        )
    else:
        logger.error(
            "[Scheduler] Daily job finished with errors: %s",
            result.errors,
        )


def create_scheduler(
    http_client,
    db_client,
    settings: Settings,
) -> AsyncIOScheduler:
    """
    Create and configure the APScheduler instance.

    The scheduler is NOT started here — the caller (app lifespan)
    is responsible for calling scheduler.start() and scheduler.shutdown().
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    trigger = CronTrigger(
        hour=settings.scheduler_hour_utc,
        minute=settings.scheduler_minute_utc,
        timezone="UTC",
    )

    scheduler.add_job(
        daily_pipeline_job,
        trigger=trigger,
        id=JOB_ID,
        name="Daily History Pipeline",
        kwargs={
            "http_client": http_client,
            "db_client": db_client,
            "settings": settings,
        },
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
        replace_existing=True,
    )

    logger.info(
        "[Scheduler] Configured daily job at %02d:%02d UTC",
        settings.scheduler_hour_utc,
        settings.scheduler_minute_utc,
    )

    return scheduler
