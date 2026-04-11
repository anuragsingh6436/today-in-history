"""
Gemini AI summarization service.

Generates engaging summaries of historical events using Google's Gemini
API.  Includes exponential-backoff retry logic, rate-limit awareness,
and a graceful fallback when the API is unreachable.

Supports multiple summary styles (short, detailed, reel) via the
prompt registry in prompts.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from google import genai
from google.genai import types as genai_types

from app.config import Settings, get_settings
from app.models.event import HistoricalEvent
from app.services.prompts import (
    BATCH_DIGEST_PROMPT,
    PromptConfig,
    SummaryStyle,
    get_prompt_config,
)

logger = logging.getLogger(__name__)


# ── Retry helpers ────────────────────────────────────────────────

_RETRYABLE_KEYWORDS = ("429", "500", "502", "503", "overloaded", "rate limit", "quota")


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception looks like a transient / rate-limit error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _RETRYABLE_KEYWORDS)


# ── Core generation with retry ──────────────────────────────────

async def _generate_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    prompt_config: PromptConfig,
    max_retries: int,
    base_delay: float,
) -> Optional[str]:
    """
    Call Gemini's generate_content with exponential backoff.

    Uses temperature and max_output_tokens from the PromptConfig
    so each style controls its own generation behaviour.

    Returns the generated text, or None if all retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=prompt_config.temperature,
                    max_output_tokens=prompt_config.max_output_tokens,
                ),
            )

            text = response.text
            if text:
                return text.strip()

            logger.warning("Gemini returned empty text on attempt %d", attempt)

        except Exception as exc:
            if _is_retryable(exc) and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Retryable error on attempt %d/%d: %s — retrying in %.1fs",
                    attempt, max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
            elif attempt < max_retries and _is_retryable(exc):
                continue
            else:
                logger.error(
                    "Non-retryable error or retries exhausted on attempt %d/%d: %s",
                    attempt, max_retries, exc,
                )
                return None

    logger.error("All %d attempts exhausted", max_retries)
    return None


# ── Fallback ─────────────────────────────────────────────────────

def _fallback_summary(event: HistoricalEvent) -> str:
    """Minimal fallback when Gemini is unavailable."""
    return (
        f"On {event.month}/{event.day}/{event.year}, {event.description} "
        f"This event is associated with {event.title}."
    )


# ── Public API ───────────────────────────────────────────────────

async def summarize_event(
    event: HistoricalEvent,
    settings: Optional[Settings] = None,
    style: Optional[SummaryStyle] = None,
) -> str:
    """
    Generate an AI summary for a single historical event.

    Args:
        event:    The event to summarize.
        settings: App settings (defaults to get_settings()).
        style:    Summary style (defaults to settings.gemini_summary_style).

    Returns a fallback string if the Gemini API is unreachable.
    """
    settings = settings or get_settings()
    style = style or SummaryStyle(settings.gemini_summary_style)

    prompt_config = get_prompt_config(style)

    prompt = prompt_config.template.format(
        month=event.month,
        day=event.day,
        year=event.year,
        title=event.title,
        description=event.description,
    )

    client = genai.Client(api_key=settings.gemini_api_key)

    result = await _generate_with_retry(
        client=client,
        model=settings.gemini_model,
        prompt=prompt,
        prompt_config=prompt_config,
        max_retries=settings.gemini_max_retries,
        base_delay=settings.gemini_base_delay,
    )

    if result is None:
        logger.warning("Falling back to plain summary for '%s'", event.title)
        return _fallback_summary(event)

    return result


async def summarize_events(
    events: List[HistoricalEvent],
    settings: Optional[Settings] = None,
    style: Optional[SummaryStyle] = None,
) -> List[HistoricalEvent]:
    """
    Enrich a list of HistoricalEvent with AI-generated summaries.

    Each event's `ai_summary` field is populated in-place.
    Events that fail summarization receive a fallback summary.
    """
    settings = settings or get_settings()

    for event in events:
        event.ai_summary = await summarize_event(event, settings, style=style)

    return events


async def generate_daily_digest(
    events: List[HistoricalEvent],
    month: int,
    day: int,
    settings: Optional[Settings] = None,
) -> str:
    """
    Generate a curated daily digest from a list of events.

    Returns a fallback message if the API is unreachable.
    """
    settings = settings or get_settings()

    events_text = "\n".join(
        f"- {e.year}: {e.title} — {e.description}" for e in events
    )

    prompt = BATCH_DIGEST_PROMPT.format(
        month=month,
        day=day,
        events_text=events_text,
    )

    client = genai.Client(api_key=settings.gemini_api_key)

    # Digest always uses the detailed config for generation params.
    digest_config = get_prompt_config(SummaryStyle.DETAILED)

    result = await _generate_with_retry(
        client=client,
        model=settings.gemini_model,
        prompt=prompt,
        prompt_config=digest_config,
        max_retries=settings.gemini_max_retries,
        base_delay=settings.gemini_base_delay,
    )

    if result is None:
        return f"Today in History — {month}/{day}: {len(events)} events occurred on this date."

    return result
