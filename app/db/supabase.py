"""
Supabase database abstraction layer.

All database access goes through this module. The rest of the app
never touches the Supabase client directly — swapping providers
means changing only this file.

The Supabase Python SDK is synchronous, so all calls are wrapped
with `asyncio.to_thread` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from supabase import Client, create_client

from app.config import Settings, get_settings
from app.models.event import HistoricalEvent

logger = logging.getLogger(__name__)

TABLE = "historical_events"


# ── Client factory ───────────────────────────────────────────────

def get_supabase_client(settings: Optional[Settings] = None) -> Client:
    """Create and return a Supabase client."""
    settings = settings or get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)


# ── Helpers ──────────────────────────────────────────────────────

def _event_to_row(event: HistoricalEvent) -> dict:
    """Convert a HistoricalEvent into a dict matching the DB schema."""
    return {
        "year": event.year,
        "month": event.month,
        "day": event.day,
        "title": event.title,
        "description": event.description,
        "wikipedia_url": event.wikipedia_url,
        "thumbnail_url": event.thumbnail_url,
        "ai_summary": event.ai_summary,
        "category": event.category,
        "region": event.region,
    }


def _row_to_event(row: dict) -> HistoricalEvent:
    """Convert a DB row dict back into a HistoricalEvent."""
    return HistoricalEvent(
        year=row["year"],
        month=row["month"],
        day=row["day"],
        title=row["title"],
        description=row["description"],
        wikipedia_url=row.get("wikipedia_url", ""),
        thumbnail_url=row.get("thumbnail_url", ""),
        ai_summary=row.get("ai_summary", ""),
        category=row.get("category", ""),
        region=row.get("region", ""),
    )


# ── CRUD operations ─────────────────────────────────────────────

async def upsert_event(
    client: Client,
    event: HistoricalEvent,
) -> Optional[dict]:
    """
    Insert or update a single event.

    Uses the UNIQUE(year, month, day, title) constraint to detect
    duplicates. On conflict, the description, wikipedia_url, and
    ai_summary are updated (the event may have been re-enriched).

    Returns the upserted row dict, or None on failure.
    """
    row = _event_to_row(event)
    try:
        result = await asyncio.to_thread(
            lambda: (
                client.table(TABLE)
                .upsert(row, on_conflict="year,month,day,title")
                .execute()
            )
        )
        if result.data:
            logger.debug("Upserted event: %s (%d)", event.title, event.year)
            return result.data[0]

        logger.warning("Upsert returned no data for: %s (%d)", event.title, event.year)
        return None

    except Exception as exc:
        logger.error("Failed to upsert event '%s' (%d): %s", event.title, event.year, exc)
        return None


async def upsert_events(
    client: Client,
    events: List[HistoricalEvent],
) -> int:
    """
    Batch upsert a list of events.

    Returns the count of successfully upserted rows.
    """
    if not events:
        return 0

    rows = [_event_to_row(e) for e in events]
    try:
        result = await asyncio.to_thread(
            lambda: (
                client.table(TABLE)
                .upsert(rows, on_conflict="year,month,day,title")
                .execute()
            )
        )
        count = len(result.data) if result.data else 0
        logger.info("Batch upserted %d/%d events", count, len(events))
        return count

    except Exception as exc:
        logger.error("Batch upsert failed (%d events): %s", len(events), exc)
        return 0


async def get_events_by_date(
    client: Client,
    month: int,
    day: int,
    year: Optional[int] = None,
) -> List[HistoricalEvent]:
    """
    Fetch events for a given date.

    Args:
        client: Supabase client.
        month:  Month (1-12).
        day:    Day (1-31).
        year:   Optional year filter. If None, returns all years.

    Returns:
        List of HistoricalEvent sorted by year ascending.
    """
    try:
        query = (
            client.table(TABLE)
            .select("*")
            .eq("month", month)
            .eq("day", day)
        )
        if year is not None:
            query = query.eq("year", year)

        query = query.order("year", desc=False)

        result = await asyncio.to_thread(lambda: query.execute())

        events = [_row_to_event(row) for row in (result.data or [])]
        logger.info(
            "Fetched %d events for %02d/%02d%s",
            len(events), month, day,
            f"/{year}" if year else "",
        )
        return events

    except Exception as exc:
        logger.error("Failed to fetch events for %02d/%02d: %s", month, day, exc)
        return []


async def event_exists(
    client: Client,
    year: int,
    month: int,
    day: int,
    title: str,
) -> bool:
    """Check if a specific event already exists in the DB."""
    try:
        result = await asyncio.to_thread(
            lambda: (
                client.table(TABLE)
                .select("id", count="exact")
                .eq("year", year)
                .eq("month", month)
                .eq("day", day)
                .eq("title", title)
                .execute()
            )
        )
        return (result.count or 0) > 0

    except Exception as exc:
        logger.error("Failed to check existence for '%s' (%d): %s", title, year, exc)
        return False


async def delete_events_by_date(
    client: Client,
    month: int,
    day: int,
) -> int:
    """
    Delete all events for a given date.

    Useful for re-fetching and re-enriching a day's events.
    Returns the count of deleted rows.
    """
    try:
        result = await asyncio.to_thread(
            lambda: (
                client.table(TABLE)
                .delete()
                .eq("month", month)
                .eq("day", day)
                .execute()
            )
        )
        count = len(result.data) if result.data else 0
        logger.info("Deleted %d events for %02d/%02d", count, month, day)
        return count

    except Exception as exc:
        logger.error("Failed to delete events for %02d/%02d: %s", month, day, exc)
        return 0
