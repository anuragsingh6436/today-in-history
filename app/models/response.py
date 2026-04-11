"""
API response models.

Every endpoint returns a consistent envelope so clients always know
what shape to expect.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.event import HistoricalEvent
from app.models.pipeline import PipelineResult


class EventResponse(BaseModel):
    """Single-event detail response."""
    event: HistoricalEvent


class EventListResponse(BaseModel):
    """Paginated list of events for a date."""
    month: int
    day: int
    year: Optional[int] = None
    total: int
    skip: int
    limit: int
    events: List[HistoricalEvent]


class TriggerResponse(BaseModel):
    """Response from a manual pipeline trigger."""
    message: str
    result: PipelineResult


class ErrorResponse(BaseModel):
    """Standard error body."""
    detail: str
