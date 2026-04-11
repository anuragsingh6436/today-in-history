"""
Pipeline result model.

Captures stats from a full Fetch → AI → Store run so callers
(scheduler, API endpoint) can inspect what happened without
parsing logs.
"""

from __future__ import annotations

from pydantic import BaseModel


class PipelineResult(BaseModel):
    """Summary of a single pipeline run for one date."""
    month: int
    day: int
    fetched: int = 0       # Events received from Wikipedia.
    skipped: int = 0       # Events already in DB (duplicate avoidance).
    enriched: int = 0      # Events successfully summarized by Gemini.
    failed: int = 0        # Events where Gemini returned a fallback.
    stored: int = 0        # Events successfully upserted to Supabase.
    errors: list[str] = [] # Human-readable error messages for debugging.
    success: bool = True   # False if a critical stage failed entirely.
