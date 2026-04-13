"""
Pydantic models for historical events.

Two layers:
  - WikipediaEvent: raw shape parsed from the Wikipedia API response.
  - HistoricalEvent: cleaned domain model used across the app.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WikipediaPage(BaseModel):
    """A single Wikipedia article linked to an event."""
    title: str = ""
    url: str = ""
    thumbnail_url: str = ""


class WikipediaEvent(BaseModel):
    """Raw event as returned by the Wikipedia 'On This Day' API."""
    year: int
    text: str
    pages: list[WikipediaPage] = Field(default_factory=list)


class HistoricalEvent(BaseModel):
    """
    Cleaned, app-level representation of a historical event.

    This is what services produce, routes return, and the DB stores.
    """
    year: int
    title: str
    description: str
    wikipedia_url: str = ""
    thumbnail_url: str = ""
    # Populated later by the Gemini service.
    ai_summary: str = ""
    category: str = ""
    region: str = ""
    month: int = 0
    day: int = 0
