"""
Unit tests for the Supabase database abstraction layer.

All Supabase calls are mocked — no real database connection required.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.db.supabase import (
    _event_to_row,
    _row_to_event,
    upsert_event,
    upsert_events,
    get_events_by_date,
    event_exists,
    delete_events_by_date,
    TABLE,
)
from app.models.event import HistoricalEvent


# ── Fixtures ─────────────────────────────────────────────────────

def _make_event(**overrides) -> HistoricalEvent:
    defaults = dict(
        year=1969,
        title="Moon Landing",
        description="Apollo 11 lands on the Moon.",
        wikipedia_url="https://en.wikipedia.org/wiki/Apollo_11",
        ai_summary="A giant leap for mankind.",
        month=7,
        day=20,
    )
    defaults.update(overrides)
    return HistoricalEvent(**defaults)


SAMPLE_ROW = {
    "id": "abc-123",
    "year": 1969,
    "month": 7,
    "day": 20,
    "title": "Moon Landing",
    "description": "Apollo 11 lands on the Moon.",
    "wikipedia_url": "https://en.wikipedia.org/wiki/Apollo_11",
    "ai_summary": "A giant leap for mankind.",
    "created_at": "2026-04-11T00:00:00Z",
    "updated_at": "2026-04-11T00:00:00Z",
}


def _mock_supabase_client(response_data=None, count=None, raise_exc=None):
    """
    Create a mock Supabase client.

    The mock supports chained calls like:
        client.table("x").upsert(data).execute()
        client.table("x").select("*").eq("a", 1).order("b").execute()
        client.table("x").delete().eq("a", 1).execute()
    """
    client = MagicMock()
    chain = MagicMock()

    # All chainable methods return the same mock.
    chain.upsert.return_value = chain
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.delete.return_value = chain

    # .execute() returns the result.
    if raise_exc:
        chain.execute.side_effect = raise_exc
    else:
        result = MagicMock()
        result.data = response_data
        result.count = count
        chain.execute.return_value = result

    client.table.return_value = chain
    return client


# ── Unit tests: conversion helpers ───────────────────────────────

class TestEventToRow:
    def test_converts_all_fields(self):
        event = _make_event()
        row = _event_to_row(event)
        assert row == {
            "year": 1969,
            "month": 7,
            "day": 20,
            "title": "Moon Landing",
            "description": "Apollo 11 lands on the Moon.",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Apollo_11",
            "ai_summary": "A giant leap for mankind.",
        }

    def test_empty_optional_fields(self):
        event = _make_event(wikipedia_url="", ai_summary="")
        row = _event_to_row(event)
        assert row["wikipedia_url"] == ""
        assert row["ai_summary"] == ""

    def test_does_not_include_id(self):
        """Row dicts should not include 'id' — the DB generates it."""
        row = _event_to_row(_make_event())
        assert "id" not in row


class TestRowToEvent:
    def test_converts_all_fields(self):
        event = _row_to_event(SAMPLE_ROW)
        assert event.year == 1969
        assert event.title == "Moon Landing"
        assert event.description == "Apollo 11 lands on the Moon."
        assert event.month == 7
        assert event.day == 20

    def test_missing_optional_fields_default(self):
        row = {**SAMPLE_ROW}
        del row["wikipedia_url"]
        del row["ai_summary"]
        event = _row_to_event(row)
        assert event.wikipedia_url == ""
        assert event.ai_summary == ""


# ── Unit tests: upsert_event ────────────────────────────────────

class TestUpsertEvent:
    @pytest.mark.asyncio
    async def test_success(self):
        client = _mock_supabase_client(response_data=[SAMPLE_ROW])
        event = _make_event()
        result = await upsert_event(client, event)
        assert result is not None
        assert result["title"] == "Moon Landing"
        client.table.assert_called_with(TABLE)

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        client = _mock_supabase_client(response_data=[])
        result = await upsert_event(client, _make_event())
        assert result is None

    @pytest.mark.asyncio
    async def test_none_data_returns_none(self):
        client = _mock_supabase_client(response_data=None)
        result = await upsert_event(client, _make_event())
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        client = _mock_supabase_client(raise_exc=Exception("DB down"))
        result = await upsert_event(client, _make_event())
        assert result is None


# ── Unit tests: upsert_events (batch) ───────────────────────────

class TestUpsertEvents:
    @pytest.mark.asyncio
    async def test_batch_success(self):
        rows = [SAMPLE_ROW, {**SAMPLE_ROW, "year": 1776, "title": "Independence"}]
        client = _mock_supabase_client(response_data=rows)
        events = [_make_event(), _make_event(year=1776, title="Independence")]
        count = await upsert_events(client, events)
        assert count == 2

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self):
        client = _mock_supabase_client()
        count = await upsert_events(client, [])
        assert count == 0
        # Should not call the DB at all.
        client.table.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_returns_zero(self):
        client = _mock_supabase_client(raise_exc=Exception("timeout"))
        count = await upsert_events(client, [_make_event()])
        assert count == 0


# ── Unit tests: get_events_by_date ───────────────────────────────

class TestGetEventsByDate:
    @pytest.mark.asyncio
    async def test_returns_events(self):
        client = _mock_supabase_client(response_data=[SAMPLE_ROW])
        events = await get_events_by_date(client, 7, 20)
        assert len(events) == 1
        assert events[0].year == 1969

    @pytest.mark.asyncio
    async def test_with_year_filter(self):
        client = _mock_supabase_client(response_data=[SAMPLE_ROW])
        events = await get_events_by_date(client, 7, 20, year=1969)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_empty_result(self):
        client = _mock_supabase_client(response_data=[])
        events = await get_events_by_date(client, 1, 1)
        assert events == []

    @pytest.mark.asyncio
    async def test_none_data_returns_empty(self):
        client = _mock_supabase_client(response_data=None)
        events = await get_events_by_date(client, 1, 1)
        assert events == []

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        client = _mock_supabase_client(raise_exc=Exception("connection error"))
        events = await get_events_by_date(client, 1, 1)
        assert events == []


# ── Unit tests: event_exists ─────────────────────────────────────

class TestEventExists:
    @pytest.mark.asyncio
    async def test_exists(self):
        client = _mock_supabase_client(count=1)
        result = await event_exists(client, 1969, 7, 20, "Moon Landing")
        assert result is True

    @pytest.mark.asyncio
    async def test_not_exists(self):
        client = _mock_supabase_client(count=0)
        result = await event_exists(client, 1969, 7, 20, "Fake Event")
        assert result is False

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        client = _mock_supabase_client(raise_exc=Exception("network"))
        result = await event_exists(client, 1969, 7, 20, "Moon Landing")
        assert result is False


# ── Unit tests: delete_events_by_date ────────────────────────────

class TestDeleteEventsByDate:
    @pytest.mark.asyncio
    async def test_deletes_rows(self):
        client = _mock_supabase_client(response_data=[SAMPLE_ROW])
        count = await delete_events_by_date(client, 7, 20)
        assert count == 1

    @pytest.mark.asyncio
    async def test_no_rows_to_delete(self):
        client = _mock_supabase_client(response_data=[])
        count = await delete_events_by_date(client, 1, 1)
        assert count == 0

    @pytest.mark.asyncio
    async def test_exception_returns_zero(self):
        client = _mock_supabase_client(raise_exc=Exception("permission denied"))
        count = await delete_events_by_date(client, 7, 20)
        assert count == 0
