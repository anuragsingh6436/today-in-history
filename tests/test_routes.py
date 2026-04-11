"""
Unit tests for REST API endpoints.

Supabase and pipeline calls are mocked — tests run without
any external services.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.event import HistoricalEvent
from app.models.pipeline import PipelineResult


# ── Fixtures ─────────────────────────────────────────────────────

client = TestClient(app)


def _make_event(year: int = 1969, title: str = "Moon Landing", **kw) -> HistoricalEvent:
    defaults = dict(
        year=year,
        title=title,
        description="Apollo 11 lands on the Moon.",
        wikipedia_url="https://en.wikipedia.org/wiki/Apollo_11",
        ai_summary="A giant leap for mankind.",
        month=7,
        day=20,
    )
    defaults.update(kw)
    return HistoricalEvent(**defaults)


SAMPLE_EVENTS = [
    _make_event(year=1776, title="Independence"),
    _make_event(year=1969, title="Moon Landing"),
]


# ── GET /api/events/today ────────────────────────────────────────

class TestGetTodayEvents:
    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.get_events_by_date", new_callable=AsyncMock)
    def test_returns_events(self, mock_get, mock_client):
        mock_get.return_value = SAMPLE_EVENTS
        mock_client.return_value = MagicMock()

        resp = client.get("/api/events/today")
        assert resp.status_code == 200

        data = resp.json()
        today = date.today()
        assert data["month"] == today.month
        assert data["day"] == today.day
        assert data["total"] == 2
        assert len(data["events"]) == 2
        assert data["skip"] == 0
        assert data["limit"] == 20

    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.get_events_by_date", new_callable=AsyncMock)
    def test_empty_result(self, mock_get, mock_client):
        mock_get.return_value = []
        mock_client.return_value = MagicMock()

        resp = client.get("/api/events/today")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        assert resp.json()["events"] == []

    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.get_events_by_date", new_callable=AsyncMock)
    def test_pagination(self, mock_get, mock_client):
        mock_get.return_value = SAMPLE_EVENTS
        mock_client.return_value = MagicMock()

        resp = client.get("/api/events/today?skip=1&limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["skip"] == 1
        assert data["limit"] == 1
        assert len(data["events"]) == 1
        assert data["events"][0]["title"] == "Moon Landing"

    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.get_events_by_date", new_callable=AsyncMock)
    def test_year_filter(self, mock_get, mock_client):
        mock_get.return_value = [SAMPLE_EVENTS[1]]
        mock_client.return_value = MagicMock()

        resp = client.get("/api/events/today?year=1969")
        assert resp.status_code == 200
        assert resp.json()["year"] == 1969

    def test_invalid_skip(self):
        resp = client.get("/api/events/today?skip=-1")
        assert resp.status_code == 422

    def test_invalid_limit_zero(self):
        resp = client.get("/api/events/today?limit=0")
        assert resp.status_code == 422

    def test_invalid_limit_too_high(self):
        resp = client.get("/api/events/today?limit=200")
        assert resp.status_code == 422


# ── GET /api/events/{month}/{day} ────────────────────────────────

class TestGetEventsByDate:
    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.get_events_by_date", new_callable=AsyncMock)
    def test_valid_date(self, mock_get, mock_client):
        mock_get.return_value = SAMPLE_EVENTS
        mock_client.return_value = MagicMock()

        resp = client.get("/api/events/7/20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["month"] == 7
        assert data["day"] == 20
        assert data["total"] == 2

    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.get_events_by_date", new_callable=AsyncMock)
    def test_with_year_filter(self, mock_get, mock_client):
        mock_get.return_value = [SAMPLE_EVENTS[0]]
        mock_client.return_value = MagicMock()

        resp = client.get("/api/events/7/4?year=1776")
        assert resp.status_code == 200
        assert resp.json()["year"] == 1776

    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.get_events_by_date", new_callable=AsyncMock)
    def test_pagination(self, mock_get, mock_client):
        mock_get.return_value = SAMPLE_EVENTS
        mock_client.return_value = MagicMock()

        resp = client.get("/api/events/7/20?skip=0&limit=1")
        data = resp.json()
        assert len(data["events"]) == 1
        assert data["total"] == 2

    def test_invalid_month_zero(self):
        resp = client.get("/api/events/0/15")
        assert resp.status_code == 422

    def test_invalid_month_13(self):
        resp = client.get("/api/events/13/1")
        assert resp.status_code == 422

    def test_invalid_day_zero(self):
        resp = client.get("/api/events/1/0")
        assert resp.status_code == 422

    def test_invalid_day_32(self):
        resp = client.get("/api/events/1/32")
        assert resp.status_code == 422

    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.get_events_by_date", new_callable=AsyncMock)
    def test_skip_beyond_total(self, mock_get, mock_client):
        mock_get.return_value = SAMPLE_EVENTS
        mock_client.return_value = MagicMock()

        resp = client.get("/api/events/7/20?skip=100")
        data = resp.json()
        assert data["total"] == 2
        assert data["events"] == []


# ── POST /api/events/trigger/{month}/{day} ───────────────────────

class TestTriggerPipeline:
    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.run_pipeline", new_callable=AsyncMock)
    @patch("app.main.http_client", new_callable=lambda: lambda: MagicMock())
    def test_success(self, mock_http, mock_pipeline, mock_client):
        mock_client.return_value = MagicMock()
        mock_pipeline.return_value = PipelineResult(
            month=7, day=20, fetched=50, skipped=10,
            enriched=38, failed=2, stored=40,
        )

        resp = client.post("/api/events/trigger/7/20")

        assert resp.status_code == 200
        data = resp.json()
        assert "completed" in data["message"]
        assert data["result"]["fetched"] == 50
        assert data["result"]["stored"] == 40

    @patch("app.routes.events.get_supabase_client")
    @patch("app.routes.events.run_pipeline", new_callable=AsyncMock)
    def test_pipeline_with_errors(self, mock_pipeline, mock_client):
        mock_client.return_value = MagicMock()
        mock_pipeline.return_value = PipelineResult(
            month=7, day=20, fetched=50, stored=0,
            success=False, errors=["Store failed"],
        )

        with patch("app.main.http_client", MagicMock()):
            resp = client.post("/api/events/trigger/7/20")

        assert resp.status_code == 200
        data = resp.json()
        assert "errors" in data["message"]

    def test_invalid_month(self):
        resp = client.post("/api/events/trigger/0/1")
        assert resp.status_code == 422

    def test_invalid_day(self):
        resp = client.post("/api/events/trigger/1/0")
        assert resp.status_code == 422


# ── Health check (still works) ───────────────────────────────────

class TestHealthCheck:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
