"""
Unit tests for the pipeline orchestrator.

All external calls (Wikipedia, Gemini, Supabase) are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.event import HistoricalEvent
from app.models.pipeline import PipelineResult
from app.services.pipeline import (
    _fetch_stage,
    _dedup_stage,
    _enrich_stage,
    _store_stage,
    run_pipeline,
)


# ── Fixtures ─────────────────────────────────────────────────────

def _make_event(year: int = 1969, title: str = "Moon Landing", **kw) -> HistoricalEvent:
    defaults = dict(
        year=year,
        title=title,
        description="Apollo 11 lands on the Moon.",
        wikipedia_url="https://en.wikipedia.org/wiki/Apollo_11",
        month=7,
        day=20,
    )
    defaults.update(kw)
    return HistoricalEvent(**defaults)


def _mock_settings():
    mock = MagicMock()
    mock.gemini_api_key = "fake"
    mock.gemini_model = "gemini-2.0-flash"
    mock.gemini_summary_style = "detailed"
    mock.gemini_max_retries = 1
    mock.gemini_base_delay = 0.01
    return mock


# ── Stage 1: Fetch ───────────────────────────────────────────────

class TestFetchStage:
    @pytest.mark.asyncio
    @patch("app.services.pipeline.fetch_events", new_callable=AsyncMock)
    async def test_success(self, mock_fetch):
        events = [_make_event(), _make_event(year=1776, title="Independence")]
        mock_fetch.return_value = events

        result = PipelineResult(month=7, day=20)
        out = await _fetch_stage(MagicMock(), 7, 20, result)

        assert len(out) == 2
        assert result.fetched == 2
        assert result.success is True

    @pytest.mark.asyncio
    @patch("app.services.pipeline.fetch_events", new_callable=AsyncMock)
    async def test_exception_returns_empty(self, mock_fetch):
        mock_fetch.side_effect = Exception("Network error")

        result = PipelineResult(month=7, day=20)
        out = await _fetch_stage(MagicMock(), 7, 20, result)

        assert out == []
        assert result.fetched == 0
        assert result.success is False
        assert len(result.errors) == 1
        assert "Wikipedia" in result.errors[0]

    @pytest.mark.asyncio
    @patch("app.services.pipeline.fetch_events", new_callable=AsyncMock)
    async def test_empty_response(self, mock_fetch):
        mock_fetch.return_value = []

        result = PipelineResult(month=1, day=1)
        out = await _fetch_stage(MagicMock(), 1, 1, result)

        assert out == []
        assert result.fetched == 0


# ── Stage 2: Dedup ───────────────────────────────────────────────

class TestDedupStage:
    @pytest.mark.asyncio
    @patch("app.services.pipeline.get_events_by_date", new_callable=AsyncMock)
    async def test_filters_duplicates(self, mock_get):
        existing = [_make_event(year=1969, title="Moon Landing")]
        mock_get.return_value = existing

        events = [
            _make_event(year=1969, title="Moon Landing"),
            _make_event(year=1776, title="Independence"),
        ]

        result = PipelineResult(month=7, day=20)
        new = await _dedup_stage(MagicMock(), events, 7, 20, result)

        assert len(new) == 1
        assert new[0].title == "Independence"
        assert result.skipped == 1

    @pytest.mark.asyncio
    @patch("app.services.pipeline.get_events_by_date", new_callable=AsyncMock)
    async def test_no_duplicates(self, mock_get):
        mock_get.return_value = []

        events = [_make_event(), _make_event(year=1776, title="Independence")]

        result = PipelineResult(month=7, day=20)
        new = await _dedup_stage(MagicMock(), events, 7, 20, result)

        assert len(new) == 2
        assert result.skipped == 0

    @pytest.mark.asyncio
    @patch("app.services.pipeline.get_events_by_date", new_callable=AsyncMock)
    async def test_all_duplicates(self, mock_get):
        existing = [_make_event()]
        mock_get.return_value = existing

        events = [_make_event()]

        result = PipelineResult(month=7, day=20)
        new = await _dedup_stage(MagicMock(), events, 7, 20, result)

        assert len(new) == 0
        assert result.skipped == 1

    @pytest.mark.asyncio
    @patch("app.services.pipeline.get_events_by_date", new_callable=AsyncMock)
    async def test_db_error_returns_all_events(self, mock_get):
        """If dedup fails, pipeline should continue with all events."""
        mock_get.side_effect = Exception("DB connection error")

        events = [_make_event()]

        result = PipelineResult(month=7, day=20)
        new = await _dedup_stage(MagicMock(), events, 7, 20, result)

        # Returns all events so upsert can handle conflicts.
        assert len(new) == 1
        assert len(result.errors) == 1


# ── Stage 3: Enrich ──────────────────────────────────────────────

class TestEnrichStage:
    @pytest.mark.asyncio
    @patch("app.services.pipeline.summarize_event", new_callable=AsyncMock)
    async def test_all_succeed(self, mock_summarize):
        mock_summarize.return_value = "A great AI summary"

        events = [_make_event(), _make_event(year=1776, title="Independence")]

        result = PipelineResult(month=7, day=20)
        out = await _enrich_stage(events, _mock_settings(), result)

        assert len(out) == 2
        assert all(e.ai_summary == "A great AI summary" for e in out)
        assert result.enriched == 2
        assert result.failed == 0

    @pytest.mark.asyncio
    @patch("app.services.pipeline.summarize_event", new_callable=AsyncMock)
    async def test_fallback_detected(self, mock_summarize):
        """Events that got the fallback summary are counted as failed."""
        event = _make_event(month=7, day=20, year=1969)
        # Fallback format matches _fallback_summary in gemini.py
        mock_summarize.return_value = (
            "On 7/20/1969, Apollo 11 lands on the Moon. "
            "This event is associated with Moon Landing."
        )

        result = PipelineResult(month=7, day=20)
        await _enrich_stage([event], _mock_settings(), result)

        assert result.enriched == 0
        assert result.failed == 1

    @pytest.mark.asyncio
    @patch("app.services.pipeline.summarize_event", new_callable=AsyncMock)
    async def test_mixed_success_and_fallback(self, mock_summarize):
        e1 = _make_event(year=1969, month=7, day=20)
        e2 = _make_event(year=1776, title="Independence", month=7, day=20)

        mock_summarize.side_effect = [
            "Great AI summary",
            "On 7/20/1776, Apollo 11 lands on the Moon. This event is associated with Independence.",
        ]

        result = PipelineResult(month=7, day=20)
        await _enrich_stage([e1, e2], _mock_settings(), result)

        assert result.enriched == 1
        assert result.failed == 1


# ── Stage 4: Store ───────────────────────────────────────────────

class TestStoreStage:
    @pytest.mark.asyncio
    @patch("app.services.pipeline.upsert_events", new_callable=AsyncMock)
    async def test_success(self, mock_upsert):
        mock_upsert.return_value = 2

        events = [_make_event(), _make_event(year=1776, title="Independence")]

        result = PipelineResult(month=7, day=20)
        await _store_stage(MagicMock(), events, result)

        assert result.stored == 2
        assert result.success is True

    @pytest.mark.asyncio
    @patch("app.services.pipeline.upsert_events", new_callable=AsyncMock)
    async def test_partial_store(self, mock_upsert):
        mock_upsert.return_value = 1  # Only 1 of 2 stored

        events = [_make_event(), _make_event(year=1776, title="Independence")]

        result = PipelineResult(month=7, day=20)
        await _store_stage(MagicMock(), events, result)

        assert result.stored == 1
        assert len(result.errors) == 1
        assert "Partial" in result.errors[0]

    @pytest.mark.asyncio
    async def test_empty_list_skips_db(self):
        result = PipelineResult(month=7, day=20)
        await _store_stage(MagicMock(), [], result)

        assert result.stored == 0

    @pytest.mark.asyncio
    @patch("app.services.pipeline.upsert_events", new_callable=AsyncMock)
    async def test_exception(self, mock_upsert):
        mock_upsert.side_effect = Exception("DB timeout")

        result = PipelineResult(month=7, day=20)
        await _store_stage(MagicMock(), [_make_event()], result)

        assert result.stored == 0
        assert result.success is False
        assert len(result.errors) == 1


# ── Full pipeline integration tests ─────────────────────────────

class TestRunPipeline:
    @pytest.mark.asyncio
    @patch("app.services.pipeline.upsert_events", new_callable=AsyncMock)
    @patch("app.services.pipeline.summarize_event", new_callable=AsyncMock)
    @patch("app.services.pipeline.get_events_by_date", new_callable=AsyncMock)
    @patch("app.services.pipeline.fetch_events", new_callable=AsyncMock)
    async def test_full_happy_path(self, mock_fetch, mock_get, mock_summarize, mock_upsert):
        events = [_make_event(), _make_event(year=1776, title="Independence")]
        mock_fetch.return_value = events
        mock_get.return_value = []  # No existing events.
        mock_summarize.return_value = "AI generated summary"
        mock_upsert.return_value = 2

        result = await run_pipeline(
            MagicMock(), MagicMock(), 7, 20, settings=_mock_settings(),
        )

        assert result.success is True
        assert result.fetched == 2
        assert result.skipped == 0
        assert result.enriched == 2
        assert result.stored == 2
        assert result.errors == []

    @pytest.mark.asyncio
    @patch("app.services.pipeline.upsert_events", new_callable=AsyncMock)
    @patch("app.services.pipeline.summarize_event", new_callable=AsyncMock)
    @patch("app.services.pipeline.get_events_by_date", new_callable=AsyncMock)
    @patch("app.services.pipeline.fetch_events", new_callable=AsyncMock)
    async def test_with_duplicates(self, mock_fetch, mock_get, mock_summarize, mock_upsert):
        events = [
            _make_event(year=1969, title="Moon Landing"),
            _make_event(year=1776, title="Independence"),
        ]
        mock_fetch.return_value = events
        mock_get.return_value = [_make_event(year=1969, title="Moon Landing")]
        mock_summarize.return_value = "Summary"
        mock_upsert.return_value = 1

        result = await run_pipeline(
            MagicMock(), MagicMock(), 7, 20, settings=_mock_settings(),
        )

        assert result.fetched == 2
        assert result.skipped == 1
        assert result.enriched == 1
        assert result.stored == 1
        # Gemini was only called for the new event.
        assert mock_summarize.call_count == 1

    @pytest.mark.asyncio
    @patch("app.services.pipeline.fetch_events", new_callable=AsyncMock)
    async def test_fetch_failure_aborts_early(self, mock_fetch):
        mock_fetch.side_effect = Exception("Wikipedia is down")

        result = await run_pipeline(
            MagicMock(), MagicMock(), 7, 20, settings=_mock_settings(),
        )

        assert result.success is False
        assert result.fetched == 0
        assert result.stored == 0

    @pytest.mark.asyncio
    @patch("app.services.pipeline.get_events_by_date", new_callable=AsyncMock)
    @patch("app.services.pipeline.fetch_events", new_callable=AsyncMock)
    async def test_all_duplicates_skips_ai_and_store(self, mock_fetch, mock_get):
        event = _make_event()
        mock_fetch.return_value = [event]
        mock_get.return_value = [event]

        result = await run_pipeline(
            MagicMock(), MagicMock(), 7, 20, settings=_mock_settings(),
        )

        assert result.fetched == 1
        assert result.skipped == 1
        assert result.enriched == 0
        assert result.stored == 0
        assert result.success is True

    @pytest.mark.asyncio
    @patch("app.services.pipeline.upsert_events", new_callable=AsyncMock)
    @patch("app.services.pipeline.summarize_event", new_callable=AsyncMock)
    @patch("app.services.pipeline.get_events_by_date", new_callable=AsyncMock)
    @patch("app.services.pipeline.fetch_events", new_callable=AsyncMock)
    async def test_store_failure_marks_not_success(
        self, mock_fetch, mock_get, mock_summarize, mock_upsert,
    ):
        mock_fetch.return_value = [_make_event()]
        mock_get.return_value = []
        mock_summarize.return_value = "Summary"
        mock_upsert.side_effect = Exception("DB exploded")

        result = await run_pipeline(
            MagicMock(), MagicMock(), 7, 20, settings=_mock_settings(),
        )

        assert result.success is False
        assert result.fetched == 1
        assert result.enriched == 1
        assert result.stored == 0


# ── PipelineResult model tests ───────────────────────────────────

class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult(month=1, day=1)
        assert r.fetched == 0
        assert r.skipped == 0
        assert r.enriched == 0
        assert r.failed == 0
        assert r.stored == 0
        assert r.errors == []
        assert r.success is True

    def test_serialization(self):
        r = PipelineResult(
            month=7, day=20, fetched=50, skipped=10,
            enriched=38, failed=2, stored=40, success=True,
        )
        d = r.model_dump()
        assert d["fetched"] == 50
        assert d["stored"] == 40
        assert isinstance(d["errors"], list)
