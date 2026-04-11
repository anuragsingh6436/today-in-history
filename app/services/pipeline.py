"""
Pipeline orchestrator: Fetch → AI → Store.

This is the single entry point for processing a day's historical events.
It is designed to be called by:
  - The APScheduler daily job
  - A manual trigger via the REST API

Each stage is resilient — partial failures do not abort the pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from supabase import Client as SupabaseClient

from app.config import Settings, get_settings
from app.db.supabase import get_events_by_date, upsert_events
from app.models.event import HistoricalEvent
from app.models.pipeline import PipelineResult
from app.services.gemini import summarize_event
from app.services.prompts import SummaryStyle
from app.services.wikipedia import fetch_events

logger = logging.getLogger(__name__)


async def _fetch_stage(
    http_client: httpx.AsyncClient,
    month: int,
    day: int,
    result: PipelineResult,
) -> list[HistoricalEvent]:
    """Stage 1: Fetch events from Wikipedia."""
    logger.info("[Pipeline] Stage 1/4 — Fetching events for %02d/%02d", month, day)

    try:
        events = await fetch_events(http_client, month, day)
        result.fetched = len(events)
        logger.info("[Pipeline] Fetched %d events", len(events))
        return events

    except Exception as exc:
        msg = f"Wikipedia fetch failed: {exc}"
        logger.error("[Pipeline] %s", msg)
        result.errors.append(msg)
        result.success = False
        return []


async def _dedup_stage(
    db_client: SupabaseClient,
    events: list[HistoricalEvent],
    month: int,
    day: int,
    result: PipelineResult,
) -> list[HistoricalEvent]:
    """Stage 2: Filter out events that already exist in the DB."""
    logger.info("[Pipeline] Stage 2/4 — Checking for duplicates")

    try:
        existing = await get_events_by_date(db_client, month, day)
        existing_keys = {
            (e.year, e.title) for e in existing
        }

        new_events = [
            e for e in events
            if (e.year, e.title) not in existing_keys
        ]

        result.skipped = len(events) - len(new_events)
        logger.info(
            "[Pipeline] %d new, %d already in DB",
            len(new_events), result.skipped,
        )
        return new_events

    except Exception as exc:
        msg = f"Dedup check failed, proceeding with all events: {exc}"
        logger.warning("[Pipeline] %s", msg)
        result.errors.append(msg)
        # If dedup fails, process everything — upsert handles conflicts.
        return events


async def _enrich_stage(
    events: list[HistoricalEvent],
    settings: Settings,
    result: PipelineResult,
    style: Optional[SummaryStyle] = None,
) -> list[HistoricalEvent]:
    """Stage 3: Generate AI summaries for each event."""
    logger.info("[Pipeline] Stage 3/4 — Enriching %d events with AI (style=%s)", len(events), style or "default")

    enriched_count = 0
    failed_count = 0

    for i, event in enumerate(events, 1):
        logger.debug(
            "[Pipeline] Enriching %d/%d: %s (%d)",
            i, len(events), event.title, event.year,
        )

        summary = await summarize_event(event, settings, style=style)
        event.ai_summary = summary

        # Detect if the fallback was used (fallback starts with "On ").
        if summary.startswith(f"On {event.month}/{event.day}/{event.year}"):
            failed_count += 1
        else:
            enriched_count += 1

    result.enriched = enriched_count
    result.failed = failed_count
    logger.info(
        "[Pipeline] Enriched %d events, %d fell back to plain summary",
        enriched_count, failed_count,
    )

    return events


async def _store_stage(
    db_client: SupabaseClient,
    events: list[HistoricalEvent],
    result: PipelineResult,
) -> None:
    """Stage 4: Batch upsert events to Supabase."""
    logger.info("[Pipeline] Stage 4/4 — Storing %d events", len(events))

    if not events:
        logger.info("[Pipeline] Nothing to store")
        return

    try:
        stored = await upsert_events(db_client, events)
        result.stored = stored
        logger.info("[Pipeline] Stored %d events", stored)

        if stored < len(events):
            msg = f"Partial store: {stored}/{len(events)} events saved"
            logger.warning("[Pipeline] %s", msg)
            result.errors.append(msg)

    except Exception as exc:
        msg = f"Store failed: {exc}"
        logger.error("[Pipeline] %s", msg)
        result.errors.append(msg)
        result.success = False


async def run_pipeline(
    http_client: httpx.AsyncClient,
    db_client: SupabaseClient,
    month: int,
    day: int,
    settings: Optional[Settings] = None,
    style: Optional[SummaryStyle] = None,
) -> PipelineResult:
    """
    Run the full Fetch → AI → Store pipeline for a given date.

    Args:
        http_client: Shared httpx async client.
        db_client:   Supabase client.
        month:       Month (1-12).
        day:         Day (1-31).
        settings:    App settings (defaults to get_settings()).

    Returns:
        PipelineResult with stats from each stage.
    """
    settings = settings or get_settings()
    result = PipelineResult(month=month, day=day)

    logger.info(
        "=" * 60 + "\n[Pipeline] Starting pipeline for %02d/%02d\n" + "=" * 60,
        month, day,
    )

    # Stage 1: Fetch
    events = await _fetch_stage(http_client, month, day, result)
    if not events:
        logger.info("[Pipeline] No events fetched — pipeline complete")
        return result

    # Stage 2: Dedup
    new_events = await _dedup_stage(db_client, events, month, day, result)
    if not new_events:
        logger.info("[Pipeline] All events already in DB — pipeline complete")
        return result

    # Stage 3: Enrich
    enriched_events = await _enrich_stage(new_events, settings, result, style=style)

    # Stage 4: Store
    await _store_stage(db_client, enriched_events, result)

    logger.info(
        "[Pipeline] Complete for %02d/%02d — "
        "fetched=%d skipped=%d enriched=%d failed=%d stored=%d",
        month, day,
        result.fetched, result.skipped, result.enriched,
        result.failed, result.stored,
    )

    return result
