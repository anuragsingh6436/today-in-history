"""
Unit tests for the Gemini summarization service.

All Gemini API calls are mocked — no API key or network access required.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.models.event import HistoricalEvent
from app.services.gemini import (
    _fallback_summary,
    _generate_with_retry,
    _is_retryable,
    summarize_event,
    summarize_events,
    generate_daily_digest,
)
from app.services.prompts import (
    BATCH_DIGEST_PROMPT,
    PromptConfig,
    SummaryStyle,
    STYLES,
    get_prompt_config,
)


# ── Fixtures ─────────────────────────────────────────────────────

def _make_event(**overrides) -> HistoricalEvent:
    defaults = dict(
        year=1969,
        title="Moon Landing",
        description="Apollo 11 lands on the Moon.",
        wikipedia_url="https://en.wikipedia.org/wiki/Apollo_11",
        month=7,
        day=20,
    )
    defaults.update(overrides)
    return HistoricalEvent(**defaults)


def _mock_settings(**overrides):
    """Create a mock Settings object with sensible defaults."""
    defaults = dict(
        gemini_api_key="fake-key",
        gemini_model="gemini-2.0-flash",
        gemini_summary_style="detailed",
        gemini_max_retries=3,
        gemini_base_delay=0.01,
        supabase_url="https://fake.supabase.co",
        supabase_key="fake-key",
    )
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


_DEFAULT_PROMPT_CONFIG = get_prompt_config(SummaryStyle.DETAILED)


def _mock_genai_client(response_text: str = "A great summary."):
    """Create a mock genai.Client whose generate_content returns fixed text."""
    client = MagicMock()
    response = MagicMock()
    response.text = response_text
    client.models.generate_content.return_value = response
    return client


def _mock_genai_client_error(exception: Exception):
    """Create a mock genai.Client that always raises."""
    client = MagicMock()
    client.models.generate_content.side_effect = exception
    return client


# ── Unit tests: _is_retryable ────────────────────────────────────

class TestIsRetryable:
    def test_rate_limit_429(self):
        assert _is_retryable(Exception("429 Too Many Requests"))

    def test_server_error_500(self):
        assert _is_retryable(Exception("500 Internal Server Error"))

    def test_server_error_503(self):
        assert _is_retryable(Exception("503 Service Unavailable"))

    def test_overloaded(self):
        assert _is_retryable(Exception("Model is overloaded"))

    def test_quota_exceeded(self):
        assert _is_retryable(Exception("quota exceeded"))

    def test_auth_error_not_retryable(self):
        assert not _is_retryable(Exception("401 Unauthorized"))

    def test_invalid_argument_not_retryable(self):
        assert not _is_retryable(Exception("400 Bad Request"))

    def test_generic_error_not_retryable(self):
        assert not _is_retryable(ValueError("something else"))


# ── Unit tests: _fallback_summary ────────────────────────────────

class TestFallbackSummary:
    def test_contains_event_details(self):
        event = _make_event()
        result = _fallback_summary(event)
        assert "1969" in result
        assert "Moon Landing" in result
        assert "Apollo 11" in result
        assert "7/20/1969" in result


# ── Unit tests: _generate_with_retry ─────────────────────────────

class TestGenerateWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        client = _mock_genai_client("Generated summary")
        result = await _generate_with_retry(
            client, "gemini-2.0-flash", "test prompt",
            prompt_config=_DEFAULT_PROMPT_CONFIG,
            max_retries=3, base_delay=0.01,
        )
        assert result == "Generated summary"
        assert client.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self):
        client = MagicMock()
        response_ok = MagicMock()
        response_ok.text = "Success after retry"
        client.models.generate_content.side_effect = [
            Exception("429 Too Many Requests"),
            response_ok,
        ]
        result = await _generate_with_retry(
            client, "gemini-2.0-flash", "test",
            prompt_config=_DEFAULT_PROMPT_CONFIG,
            max_retries=3, base_delay=0.01,
        )
        assert result == "Success after retry"
        assert client.models.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_500_then_succeeds(self):
        client = MagicMock()
        response_ok = MagicMock()
        response_ok.text = "Recovered"
        client.models.generate_content.side_effect = [
            Exception("500 Internal Server Error"),
            Exception("503 Service Unavailable"),
            response_ok,
        ]
        result = await _generate_with_retry(
            client, "gemini-2.0-flash", "test",
            prompt_config=_DEFAULT_PROMPT_CONFIG,
            max_retries=3, base_delay=0.01,
        )
        assert result == "Recovered"
        assert client.models.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries_returns_none(self):
        client = _mock_genai_client_error(Exception("503 Service Unavailable"))
        result = await _generate_with_retry(
            client, "gemini-2.0-flash", "test",
            prompt_config=_DEFAULT_PROMPT_CONFIG,
            max_retries=2, base_delay=0.01,
        )
        assert result is None
        assert client.models.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_returns_none_immediately(self):
        client = _mock_genai_client_error(Exception("401 Unauthorized"))
        result = await _generate_with_retry(
            client, "gemini-2.0-flash", "test",
            prompt_config=_DEFAULT_PROMPT_CONFIG,
            max_retries=3, base_delay=0.01,
        )
        assert result is None
        assert client.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_response_retries(self):
        client = MagicMock()
        empty_resp = MagicMock()
        empty_resp.text = ""
        good_resp = MagicMock()
        good_resp.text = "Got it"
        client.models.generate_content.side_effect = [empty_resp, good_resp]
        result = await _generate_with_retry(
            client, "gemini-2.0-flash", "test",
            prompt_config=_DEFAULT_PROMPT_CONFIG,
            max_retries=3, base_delay=0.01,
        )
        assert result == "Got it"

    @pytest.mark.asyncio
    async def test_uses_prompt_config_temperature(self):
        """Verify that PromptConfig temperature is passed to Gemini."""
        client = _mock_genai_client("ok")
        custom_config = PromptConfig(template="", temperature=0.3, max_output_tokens=50)
        await _generate_with_retry(
            client, "gemini-2.0-flash", "test",
            prompt_config=custom_config,
            max_retries=1, base_delay=0.01,
        )
        call_kwargs = client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.temperature == 0.3
        assert config.max_output_tokens == 50


# ── Integration-style tests: summarize_event ─────────────────────

class TestSummarizeEvent:
    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_returns_ai_summary(self, mock_client_cls):
        client_inst = _mock_genai_client("An amazing hook about the Moon!")
        mock_client_cls.return_value = client_inst

        event = _make_event()
        result = await summarize_event(event, settings=_mock_settings())
        assert result == "An amazing hook about the Moon!"

    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_fallback_on_total_failure(self, mock_client_cls):
        client_inst = _mock_genai_client_error(Exception("401 Unauthorized"))
        mock_client_cls.return_value = client_inst

        event = _make_event()
        result = await summarize_event(event, settings=_mock_settings(gemini_max_retries=1))
        assert "Moon Landing" in result
        assert "1969" in result

    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_explicit_style_short(self, mock_client_cls):
        client_inst = _mock_genai_client("Short summary")
        mock_client_cls.return_value = client_inst

        event = _make_event()
        result = await summarize_event(
            event, settings=_mock_settings(), style=SummaryStyle.SHORT,
        )
        assert result == "Short summary"

    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_explicit_style_reel(self, mock_client_cls):
        client_inst = _mock_genai_client("Reel script")
        mock_client_cls.return_value = client_inst

        event = _make_event()
        result = await summarize_event(
            event, settings=_mock_settings(), style=SummaryStyle.REEL,
        )
        assert result == "Reel script"

    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_style_from_settings(self, mock_client_cls):
        """When no explicit style, uses settings.gemini_summary_style."""
        client_inst = _mock_genai_client("From settings")
        mock_client_cls.return_value = client_inst

        event = _make_event()
        result = await summarize_event(
            event, settings=_mock_settings(gemini_summary_style="short"),
        )
        assert result == "From settings"


# ── Integration-style tests: summarize_events (batch) ────────────

class TestSummarizeEvents:
    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_enriches_all_events(self, mock_client_cls):
        client_inst = _mock_genai_client("Summary text")
        mock_client_cls.return_value = client_inst

        events = [_make_event(year=1969), _make_event(year=1776, title="Independence")]
        result = await summarize_events(events, settings=_mock_settings())

        assert len(result) == 2
        assert all(e.ai_summary == "Summary text" for e in result)

    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_batch_with_explicit_style(self, mock_client_cls):
        client_inst = _mock_genai_client("Reel batch")
        mock_client_cls.return_value = client_inst

        events = [_make_event()]
        result = await summarize_events(
            events, settings=_mock_settings(), style=SummaryStyle.REEL,
        )
        assert result[0].ai_summary == "Reel batch"


# ── Integration-style tests: generate_daily_digest ───────────────

class TestGenerateDailyDigest:
    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_returns_digest(self, mock_client_cls):
        client_inst = _mock_genai_client("Your daily history digest!")
        mock_client_cls.return_value = client_inst

        events = [_make_event(), _make_event(year=1776)]
        result = await generate_daily_digest(events, 7, 20, settings=_mock_settings())
        assert result == "Your daily history digest!"

    @pytest.mark.asyncio
    @patch("app.services.gemini.genai.Client")
    async def test_fallback_on_failure(self, mock_client_cls):
        client_inst = _mock_genai_client_error(Exception("401 Unauthorized"))
        mock_client_cls.return_value = client_inst

        events = [_make_event(), _make_event(year=1776)]
        result = await generate_daily_digest(
            events, 7, 20, settings=_mock_settings(gemini_max_retries=1),
        )
        assert "2 events" in result
        assert "7/20" in result


# ── Prompt registry tests ────────────────────────────────────────

class TestPromptRegistry:
    def test_all_styles_registered(self):
        for style in SummaryStyle:
            assert style in STYLES

    def test_get_prompt_config_returns_correct_type(self):
        for style in SummaryStyle:
            config = get_prompt_config(style)
            assert isinstance(config, PromptConfig)

    def test_short_has_lower_tokens(self):
        short = get_prompt_config(SummaryStyle.SHORT)
        detailed = get_prompt_config(SummaryStyle.DETAILED)
        assert short.max_output_tokens < detailed.max_output_tokens

    def test_reel_has_highest_temperature(self):
        reel = get_prompt_config(SummaryStyle.REEL)
        short = get_prompt_config(SummaryStyle.SHORT)
        detailed = get_prompt_config(SummaryStyle.DETAILED)
        assert reel.temperature >= short.temperature
        assert reel.temperature >= detailed.temperature

    def test_all_templates_have_required_placeholders(self):
        """Every style template must accept the standard event fields."""
        for style in SummaryStyle:
            config = get_prompt_config(style)
            result = config.template.format(
                month=7, day=20, year=1969,
                title="Moon Landing",
                description="Apollo 11 lands on the Moon.",
            )
            assert "1969" in result
            assert "Moon Landing" in result

    def test_batch_digest_prompt_has_placeholders(self):
        result = BATCH_DIGEST_PROMPT.format(
            month=7, day=20,
            events_text="- 1969: Moon Landing — Apollo 11",
        )
        assert "7/20" in result
        assert "Moon Landing" in result

    def test_summary_style_enum_values(self):
        assert SummaryStyle.SHORT.value == "short"
        assert SummaryStyle.DETAILED.value == "detailed"
        assert SummaryStyle.REEL.value == "reel"

    def test_summary_style_from_string(self):
        """Styles can be constructed from string values (API param)."""
        assert SummaryStyle("short") == SummaryStyle.SHORT
        assert SummaryStyle("detailed") == SummaryStyle.DETAILED
        assert SummaryStyle("reel") == SummaryStyle.REEL

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError):
            SummaryStyle("nonexistent")
