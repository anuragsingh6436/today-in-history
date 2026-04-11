"""
Wikipedia 'On This Day' service.

Fetches historical events for a given date from the Wikipedia REST API,
normalizes the response, and returns a list of HistoricalEvent models.

API docs: https://en.wikipedia.org/api/rest_v1/#/Feed/onThisDay
"""

from __future__ import annotations

import logging
import re
from typing import List

import httpx

from app.models.event import HistoricalEvent, WikipediaEvent, WikipediaPage

logger = logging.getLogger(__name__)

_BASE_URL = "https://en.wikipedia.org/api/rest_v1/feed/onthisday/events"
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove any residual HTML tags from Wikipedia text fields."""
    return _HTML_TAG_RE.sub("", text).strip()


def _parse_event(raw: dict, month: int, day: int) -> WikipediaEvent | None:
    """
    Parse a single raw event dict into a WikipediaEvent.

    Returns None if the event is missing required fields so callers
    can skip it without crashing the whole batch.
    """
    year = raw.get("year")
    text = raw.get("text")
    if year is None or not text:
        logger.debug("Skipping event with missing year/text: %s", raw)
        return None

    pages: list[WikipediaPage] = []
    for p in raw.get("pages", []):
        title = (
            p.get("titles", {}).get("normalized")
            or p.get("title", "").replace("_", " ")
        )
        url = p.get("content_urls", {}).get("desktop", {}).get("page", "")
        thumbnail = p.get("thumbnail", {}).get("source", "")
        if not thumbnail:
            thumbnail = p.get("originalimage", {}).get("source", "")
        pages.append(WikipediaPage(title=_strip_html(title), url=url, thumbnail_url=thumbnail))

    return WikipediaEvent(
        year=int(year),
        text=_strip_html(text),
        pages=pages,
    )


def _to_historical_event(event: WikipediaEvent, month: int, day: int) -> HistoricalEvent:
    """Convert a parsed WikipediaEvent into our domain model."""
    # Use the first page's title as the event title; fall back to a
    # truncated version of the description text.
    if event.pages:
        title = event.pages[0].title
        url = event.pages[0].url
        thumbnail = event.pages[0].thumbnail_url
    else:
        title = event.text[:80] + ("..." if len(event.text) > 80 else "")
        url = ""
        thumbnail = ""

    return HistoricalEvent(
        year=event.year,
        title=title,
        description=event.text,
        wikipedia_url=url,
        thumbnail_url=thumbnail,
        month=month,
        day=day,
    )


async def fetch_events(
    client: httpx.AsyncClient,
    month: int,
    day: int,
) -> List[HistoricalEvent]:
    """
    Fetch historical events for the given month/day.

    Args:
        client: Shared httpx async client (managed by app lifespan).
        month: Month (1-12).
        day: Day of month (1-31).

    Returns:
        List of HistoricalEvent sorted by year ascending.

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx from Wikipedia.
        httpx.RequestError: On network-level failures.
    """
    url = f"{_BASE_URL}/{month:02d}/{day:02d}"
    logger.info("Fetching events from %s", url)

    response = await client.get(
        url,
        headers={"User-Agent": "TodayInHistoryBot/1.0 (educational project)"},
    )
    response.raise_for_status()

    data = response.json()
    raw_events = data.get("events", [])
    logger.info("Received %d raw events for %02d/%02d", len(raw_events), month, day)

    events: list[HistoricalEvent] = []
    for raw in raw_events:
        parsed = _parse_event(raw, month, day)
        if parsed is not None:
            events.append(_to_historical_event(parsed, month, day))

    # Sort chronologically.
    events.sort(key=lambda e: e.year)

    logger.info("Parsed %d valid events for %02d/%02d", len(events), month, day)
    return events
