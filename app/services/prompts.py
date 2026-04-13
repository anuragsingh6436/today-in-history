"""
Prompt templates for AI summarization.

Each summary style is a PromptConfig — a self-contained bundle of
prompt template, generation temperature, and max output tokens.

To add a new style:
  1. Define the prompt string below.
  2. Add a PromptConfig entry to STYLES.
  3. Add the key to SummaryStyle enum.
  That's it — no other files need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SummaryStyle(str, Enum):
    """Available summary styles — used as API param and config value."""
    SHORT = "short"
    DETAILED = "detailed"
    REEL = "reel"


@dataclass(frozen=True)
class PromptConfig:
    """Everything Gemini needs to generate a summary in a given style."""
    template: str
    temperature: float
    max_output_tokens: int


# ── Prompt templates ─────────────────────────────────────────────

_SHORT_TEMPLATE = """\
You are a history expert writing for a mobile app.

Write a 2-3 sentence summary of this event, then classify it.

RULES:
- Maximum 50 words for the summary.
- No headings, no labels, no bullet points, no emojis.
- Just flowing prose.
- Do NOT start with "On this day" or "In [year]".

Respond in EXACTLY this JSON format (no markdown, no code blocks):
{{"summary": "your summary here", "category": "one of: War & Conflict, Politics & Government, Science & Technology, Arts & Culture, Sports, Exploration & Discovery, Religion, Disasters, Society", "region": "one of: India, Europe, Americas, Asia, Middle East, Africa, Global"}}

EVENT:
- Date: {month}/{day}/{year}
- Title: {title}
- Description: {description}
"""

_DETAILED_TEMPLATE = """\
You are a world-class history storyteller writing for a "Today in History" app.

Write an engaging 3-paragraph summary of this historical event, then classify it.

Paragraph 1: A single punchy sentence that sparks curiosity.
Paragraph 2: 2-3 sentences covering what happened, who was involved.
Paragraph 3: 1-2 sentences on the lasting impact.

CRITICAL RULES:
- Total length: 80-120 words for the summary.
- Tone: vivid, accessible, slightly dramatic.
- NO headings, NO labels like "HOOK:" — just clean flowing text.
- Separate paragraphs with a blank line.
- Do NOT start with "On this day" or "In [year]".

Respond in EXACTLY this JSON format (no markdown, no code blocks):
{{"summary": "paragraph1\\n\\nparagraph2\\n\\nparagraph3", "category": "one of: War & Conflict, Politics & Government, Science & Technology, Arts & Culture, Sports, Exploration & Discovery, Religion, Disasters, Society", "region": "one of: India, Europe, Americas, Asia, Middle East, Africa, Global"}}

EVENT:
- Date: {month}/{day}/{year}
- Title: {title}
- Description: {description}
"""

_REEL_TEMPLATE = """\
You are a viral history content creator writing a script for a 60-second reel.

Write a script in three parts, then classify the event.

Part 1: A jaw-dropping one-liner.
Part 2: Short punchy sentences telling the story.
Part 3: Surprising aftermath. End with a shareable line.

CRITICAL RULES:
- Total length: 100-150 words.
- NO labels like "HOOK:" or "STORY:" — just clean flowing text.
- Separate the three parts with blank lines.
- No emojis, no hashtags.

Respond in EXACTLY this JSON format (no markdown, no code blocks):
{{"summary": "part1\\n\\npart2\\n\\npart3", "category": "one of: War & Conflict, Politics & Government, Science & Technology, Arts & Culture, Sports, Exploration & Discovery, Religion, Disasters, Society", "region": "one of: India, Europe, Americas, Asia, Middle East, Africa, Global"}}

EVENT:
- Date: {month}/{day}/{year}
- Title: {title}
- Description: {description}
"""

_BATCH_DIGEST_TEMPLATE = """\
You are a world-class history storyteller writing a daily "Today in History" digest.

Given these historical events from {month}/{day}, pick the 5 most fascinating ones \
and write a short, engaging digest. For each event write a brief paragraph with \
a curiosity hook, what happened, and why it matters.

Keep the total digest under 500 words. Make it feel like a morning newsletter \
people look forward to reading. No labels or headings — just flowing prose.

EVENTS:
{events_text}
"""


# ── Style registry ───────────────────────────────────────────────

STYLES: dict[SummaryStyle, PromptConfig] = {
    SummaryStyle.SHORT: PromptConfig(
        template=_SHORT_TEMPLATE,
        temperature=0.7,
        max_output_tokens=200,
    ),
    SummaryStyle.DETAILED: PromptConfig(
        template=_DETAILED_TEMPLATE,
        temperature=0.8,
        max_output_tokens=400,
    ),
    SummaryStyle.REEL: PromptConfig(
        template=_REEL_TEMPLATE,
        temperature=0.9,
        max_output_tokens=500,
    ),
}

BATCH_DIGEST_PROMPT = _BATCH_DIGEST_TEMPLATE


def get_prompt_config(style: SummaryStyle) -> PromptConfig:
    """Look up the PromptConfig for a given style."""
    return STYLES[style]
