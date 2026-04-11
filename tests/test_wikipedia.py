"""
Unit tests for the Wikipedia service.

All HTTP calls are mocked — no network access required.
"""

from __future__ import annotations

import pytest
import httpx

from app.services.wikipedia import fetch_events, _strip_html, _parse_event
from app.models.event import HistoricalEvent


# ── Fixtures ─────────────────────────────────────────────────────

SAMPLE_API_RESPONSE = {
    "events": [
        {
            "year": 1912,
            "text": "The <b>Titanic</b> sets sail from Southampton.",
            "pages": [
                {
                    "titles": {"normalized": "RMS Titanic"},
                    "title": "RMS_Titanic",
                    "content_urls": {
                        "desktop": {
                            "page": "https://en.wikipedia.org/wiki/RMS_Titanic"
                        }
                    },
                }
            ],
        },
        {
            "year": 1961,
            "text": "Yuri Gagarin becomes the first human in space.",
            "pages": [
                {
                    "titles": {"normalized": "Yuri Gagarin"},
                    "title": "Yuri_Gagarin",
                    "content_urls": {
                        "desktop": {
                            "page": "https://en.wikipedia.org/wiki/Yuri_Gagarin"
                        }
                    },
                }
            ],
        },
    ]
}


def _mock_transport(response_json: dict, status_code: int = 200) -> httpx.MockTransport:
    """Create a mock transport that returns a fixed JSON response."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=response_json)
    return httpx.MockTransport(handler)


# ── Unit tests: helper functions ─────────────────────────────────

class TestStripHtml:
    def test_removes_bold_tags(self):
        assert _strip_html("The <b>Titanic</b> sinks") == "The Titanic sinks"

    def test_removes_span_tags(self):
        assert _strip_html('<span class="x">hello</span>') == "hello"

    def test_plain_text_unchanged(self):
        assert _strip_html("no tags here") == "no tags here"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_strips_whitespace(self):
        assert _strip_html("  <b>hi</b>  ") == "hi"


class TestParseEvent:
    def test_valid_event(self):
        raw = SAMPLE_API_RESPONSE["events"][0]
        result = _parse_event(raw, 4, 10)
        assert result is not None
        assert result.year == 1912
        assert "Titanic" in result.text
        assert len(result.pages) == 1
        assert result.pages[0].title == "RMS Titanic"

    def test_missing_year_returns_none(self):
        raw = {"text": "Something happened"}
        assert _parse_event(raw, 1, 1) is None

    def test_missing_text_returns_none(self):
        raw = {"year": 2000}
        assert _parse_event(raw, 1, 1) is None

    def test_empty_text_returns_none(self):
        raw = {"year": 2000, "text": ""}
        assert _parse_event(raw, 1, 1) is None

    def test_event_without_pages(self):
        raw = {"year": 1500, "text": "An event with no pages"}
        result = _parse_event(raw, 3, 15)
        assert result is not None
        assert result.pages == []

    def test_page_falls_back_to_title_field(self):
        """If titles.normalized is missing, fall back to the title field."""
        raw = {
            "year": 1800,
            "text": "Something",
            "pages": [{"title": "Some_Page", "content_urls": {}}],
        }
        result = _parse_event(raw, 6, 1)
        assert result is not None
        assert result.pages[0].title == "Some Page"


# ── Integration-style tests: fetch_events with mocked HTTP ──────

class TestFetchEvents:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        transport = _mock_transport(SAMPLE_API_RESPONSE)
        async with httpx.AsyncClient(transport=transport) as client:
            events = await fetch_events(client, 4, 10)

        assert len(events) == 2
        assert all(isinstance(e, HistoricalEvent) for e in events)
        # Should be sorted by year ascending.
        assert events[0].year == 1912
        assert events[1].year == 1961
        # Check fields populated correctly.
        assert events[0].title == "RMS Titanic"
        assert events[0].month == 4
        assert events[0].day == 10
        assert "wikipedia.org" in events[0].wikipedia_url

    @pytest.mark.asyncio
    async def test_empty_events_list(self):
        transport = _mock_transport({"events": []})
        async with httpx.AsyncClient(transport=transport) as client:
            events = await fetch_events(client, 1, 1)

        assert events == []

    @pytest.mark.asyncio
    async def test_missing_events_key(self):
        """API returns JSON without an 'events' key — should return empty list."""
        transport = _mock_transport({"something_else": []})
        async with httpx.AsyncClient(transport=transport) as client:
            events = await fetch_events(client, 1, 1)

        assert events == []

    @pytest.mark.asyncio
    async def test_skips_malformed_events(self):
        """Events missing year or text should be silently skipped."""
        data = {
            "events": [
                {"year": 1999, "text": "Valid event"},
                {"text": "No year"},
                {"year": 2000},
                {"year": 2001, "text": ""},
            ]
        }
        transport = _mock_transport(data)
        async with httpx.AsyncClient(transport=transport) as client:
            events = await fetch_events(client, 7, 4)

        assert len(events) == 1
        assert events[0].year == 1999

    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        """4xx/5xx responses should propagate as HTTPStatusError."""
        transport = _mock_transport({"error": "not found"}, status_code=404)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_events(client, 13, 1)

    @pytest.mark.asyncio
    async def test_event_without_pages_uses_text_as_title(self):
        data = {
            "events": [
                {"year": 1776, "text": "A very important thing happened in history"},
            ]
        }
        transport = _mock_transport(data)
        async with httpx.AsyncClient(transport=transport) as client:
            events = await fetch_events(client, 7, 4)

        assert len(events) == 1
        assert events[0].title == "A very important thing happened in history"
        assert events[0].wikipedia_url == ""

    @pytest.mark.asyncio
    async def test_html_stripped_from_text(self):
        data = {
            "events": [
                {
                    "year": 2000,
                    "text": "The <b>bold</b> and the <i>italic</i>",
                    "pages": [],
                }
            ]
        }
        transport = _mock_transport(data)
        async with httpx.AsyncClient(transport=transport) as client:
            events = await fetch_events(client, 1, 1)

        assert events[0].description == "The bold and the italic"

    @pytest.mark.asyncio
    async def test_results_sorted_by_year(self):
        data = {
            "events": [
                {"year": 2020, "text": "Recent event"},
                {"year": 1066, "text": "Old event"},
                {"year": 1776, "text": "Middle event"},
            ]
        }
        transport = _mock_transport(data)
        async with httpx.AsyncClient(transport=transport) as client:
            events = await fetch_events(client, 10, 14)

        years = [e.year for e in events]
        assert years == [1066, 1776, 2020]
