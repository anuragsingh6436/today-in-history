"""
REST API endpoints for historical events.

All routes are thin — validation and response shaping only.
Business logic lives in services and the pipeline.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from app.db.supabase import get_events_by_date, get_supabase_client
from app.models.response import EventListResponse, TriggerResponse
from app.services.pipeline import run_pipeline
from app.services.prompts import SummaryStyle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])


# ── Validation helpers ───────────────────────────────────────────

def _validate_date(month: int, day: int) -> None:
    """Raise 422 if month/day are out of range."""
    if not (1 <= month <= 12):
        raise HTTPException(status_code=422, detail=f"month must be 1-12, got {month}")
    if not (1 <= day <= 31):
        raise HTTPException(status_code=422, detail=f"day must be 1-31, got {day}")


# ── GET /api/events/today ────────────────────────────────────────

@router.get(
    "/today",
    response_model=EventListResponse,
    summary="Get today's historical events",
)
async def get_today_events(
    skip: int = Query(default=0, ge=0, description="Number of events to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max events to return"),
    year: Optional[int] = Query(default=None, description="Filter by year"),
):
    """Fetch historical events that happened on today's date."""
    today = date.today()
    return await _get_events(today.month, today.day, skip, limit, year)


# ── GET /api/events/{month}/{day} ────────────────────────────────

@router.get(
    "/{month}/{day}",
    response_model=EventListResponse,
    summary="Get historical events for a specific date",
)
async def get_events_by_month_day(
    month: int = Path(ge=1, le=12, description="Month (1-12)"),
    day: int = Path(ge=1, le=31, description="Day (1-31)"),
    skip: int = Query(default=0, ge=0, description="Number of events to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max events to return"),
    year: Optional[int] = Query(default=None, description="Filter by year"),
):
    """Fetch historical events for the given month and day."""
    _validate_date(month, day)
    return await _get_events(month, day, skip, limit, year)


# ── POST /api/events/trigger/{month}/{day} ───────────────────────

@router.post(
    "/trigger/{month}/{day}",
    response_model=TriggerResponse,
    summary="Manually trigger the pipeline for a date",
)
async def trigger_pipeline(
    month: int = Path(ge=1, le=12, description="Month (1-12)"),
    day: int = Path(ge=1, le=31, description="Day (1-31)"),
    style: Optional[SummaryStyle] = Query(
        default=None,
        description="Summary style: short, detailed, or reel (defaults to env config)",
    ),
):
    """
    Run the full Fetch → AI → Store pipeline for the given date.

    This is the same pipeline the scheduler runs daily, exposed here
    for manual backfills and debugging.
    """
    _validate_date(month, day)

    # Import here to avoid circular import at module level.
    from app.main import http_client

    if http_client is None:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")

    db_client = get_supabase_client()

    result = await run_pipeline(http_client, db_client, month, day, style=style)

    status = "completed" if result.success else "completed with errors"
    return TriggerResponse(
        message=f"Pipeline {status} for {month:02d}/{day:02d}",
        result=result,
    )


# ── Shared query logic ──────────────────────────────────────────

async def _get_events(
    month: int,
    day: int,
    skip: int,
    limit: int,
    year: Optional[int],
) -> EventListResponse:
    """Query DB and return paginated response."""
    db_client = get_supabase_client()

    all_events = await get_events_by_date(db_client, month, day, year=year)

    total = len(all_events)
    paginated = all_events[skip : skip + limit]

    return EventListResponse(
        month=month,
        day=day,
        year=year,
        total=total,
        skip=skip,
        limit=limit,
        events=paginated,
    )
