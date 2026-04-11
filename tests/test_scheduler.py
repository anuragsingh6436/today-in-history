"""
Unit tests for the scheduler system.

APScheduler is tested via its public API — no monkey-patching internals.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.pipeline import PipelineResult
from app.scheduler.jobs import (
    JOB_ID,
    create_scheduler,
    daily_pipeline_job,
)


# ── Fixtures ─────────────────────────────────────────────────────

def _mock_settings(**overrides):
    defaults = dict(
        scheduler_hour_utc=6,
        scheduler_minute_utc=30,
        scheduler_enabled=True,
        gemini_api_key="fake",
        gemini_model="gemini-2.0-flash",
        gemini_max_retries=1,
        gemini_base_delay=0.01,
        supabase_url="https://test.supabase.co",
        supabase_key="test-key",
    )
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# ── Tests: create_scheduler ──────────────────────────────────────

class TestCreateScheduler:
    def test_creates_scheduler_with_job(self):
        settings = _mock_settings()
        scheduler = create_scheduler(MagicMock(), MagicMock(), settings)

        job = scheduler.get_job(JOB_ID)
        assert job is not None
        assert job.name == "Daily History Pipeline"
        assert job.max_instances == 1

    def test_job_trigger_matches_config(self):
        settings = _mock_settings(scheduler_hour_utc=14, scheduler_minute_utc=45)
        scheduler = create_scheduler(MagicMock(), MagicMock(), settings)

        job = scheduler.get_job(JOB_ID)
        trigger = job.trigger

        assert str(trigger) == "cron[hour='14', minute='45']"

    def test_misfire_grace_time_set(self):
        settings = _mock_settings()
        scheduler = create_scheduler(MagicMock(), MagicMock(), settings)
        job = scheduler.get_job(JOB_ID)
        assert job.misfire_grace_time == 3600

    def test_coalesce_enabled(self):
        settings = _mock_settings()
        scheduler = create_scheduler(MagicMock(), MagicMock(), settings)
        job = scheduler.get_job(JOB_ID)
        assert job.coalesce is True

    def test_job_id_is_constant(self):
        assert JOB_ID == "daily_history_pipeline"

    def test_different_hour_configs(self):
        """Job trigger reflects the settings, not hardcoded values."""
        for hour in [0, 12, 23]:
            settings = _mock_settings(scheduler_hour_utc=hour, scheduler_minute_utc=0)
            scheduler = create_scheduler(MagicMock(), MagicMock(), settings)
            job = scheduler.get_job(JOB_ID)
            assert str(job.trigger) == f"cron[hour='{hour}', minute='0']"


# ── Tests: daily_pipeline_job ────────────────────────────────────

class TestDailyPipelineJob:
    @pytest.mark.asyncio
    @patch("app.services.pipeline.run_pipeline", new_callable=AsyncMock)
    async def test_runs_pipeline_for_today(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult(
            month=4, day=11, fetched=50, stored=40, success=True,
        )
        settings = _mock_settings()

        await daily_pipeline_job(MagicMock(), MagicMock(), settings)

        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args.kwargs
        today = date.today()
        assert call_kwargs["month"] == today.month
        assert call_kwargs["day"] == today.day

    @pytest.mark.asyncio
    @patch("app.services.pipeline.run_pipeline", new_callable=AsyncMock)
    async def test_success_does_not_raise(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult(
            month=4, day=11, fetched=50, stored=40, success=True,
        )

        # Should not raise.
        await daily_pipeline_job(MagicMock(), MagicMock(), _mock_settings())

    @pytest.mark.asyncio
    @patch("app.services.pipeline.run_pipeline", new_callable=AsyncMock)
    async def test_errors_do_not_raise(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult(
            month=4, day=11, fetched=50, stored=0,
            success=False, errors=["DB exploded"],
        )

        # Should not raise — errors are logged, not thrown.
        await daily_pipeline_job(MagicMock(), MagicMock(), _mock_settings())

    @pytest.mark.asyncio
    @patch("app.services.pipeline.run_pipeline", new_callable=AsyncMock)
    async def test_passes_settings_to_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult(month=1, day=1, success=True)
        settings = _mock_settings()

        await daily_pipeline_job(MagicMock(), MagicMock(), settings)

        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["settings"] is settings

    @pytest.mark.asyncio
    @patch("app.services.pipeline.run_pipeline", new_callable=AsyncMock)
    async def test_passes_http_and_db_clients(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult(month=1, day=1, success=True)
        http = MagicMock()
        db = MagicMock()

        await daily_pipeline_job(http, db, _mock_settings())

        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["http_client"] is http
        assert call_kwargs["db_client"] is db


# ── Tests: scheduler lifecycle ───────────────────────────────────

class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_start_runs_within_event_loop(self):
        """Scheduler starts successfully when an event loop is running."""
        settings = _mock_settings()
        scheduler = create_scheduler(MagicMock(), MagicMock(), settings)

        scheduler.start()
        assert scheduler.running is True

        # Clean up.
        scheduler.shutdown()

    def test_job_count_is_one(self):
        """Only one job is registered."""
        settings = _mock_settings()
        scheduler = create_scheduler(MagicMock(), MagicMock(), settings)
        assert len(scheduler.get_jobs()) == 1
