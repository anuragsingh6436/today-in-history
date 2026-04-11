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
You are a history expert writing for a mobile widget.

Write a 2-3 sentence summary of this event. First sentence is a \
curiosity hook, remaining sentences cover what happened and why \
it still matters. Be vivid but ultra-concise.

RULES:
- Maximum 50 words.
- No headings, no bullet points, no emojis.
- Do NOT start with "On this day" or "In [year]".

EVENT:
- Date: {month}/{day}/{year}
- Title: {title}
- Description: {description}
"""

_DETAILED_TEMPLATE = """\
You are a world-class history storyteller writing for a "Today in History" app.

Given the following historical event, write an engaging summary in EXACTLY this format:

HOOK: A single punchy sentence that sparks curiosity — make the reader NEED to know more.

WHAT HAPPENED: 2-3 sentences covering what happened, who was involved, and the context.

WHY IT MATTERS: 1-2 sentences on the lasting impact — how this event shaped the world we live in today.

RULES:
- Total length: 80-120 words.
- Tone: vivid, accessible, slightly dramatic — like a great podcast host.
- No bullet points or markdown — just flowing text under each heading.
- Do NOT start with "On this day" or "In [year]".
- Use present tense for the hook to create immediacy.

EVENT:
- Date: {month}/{day}/{year}
- Title: {title}
- Description: {description}
"""

_REEL_TEMPLATE = """\
You are a viral history content creator writing a script for a \
60-second social media reel / TikTok.

Write a script for this historical event. Use this structure:

HOOK (first 3 seconds): A jaw-dropping one-liner that stops the scroll. \
Use "you" or a bold claim.

STORY (next 40 seconds): Tell the story in short, punchy sentences. \
Build tension. Use simple words. One idea per sentence.

TWIST/PAYOFF (last 15 seconds): Reveal the surprising aftermath or \
a little-known fact. End with a line that makes the viewer want to \
share.

RULES:
- Total length: 100-150 words.
- Write in second person ("you") where possible.
- Short sentences. Sentence fragments are OK.
- No emojis, no hashtags.
- Do NOT start with "Did you know" — be more creative.

EVENT:
- Date: {month}/{day}/{year}
- Title: {title}
- Description: {description}
"""

_BATCH_DIGEST_TEMPLATE = """\
You are a world-class history storyteller writing a daily "Today in History" digest.

Given these historical events from {month}/{day}, pick the 5 most fascinating ones \
and write a short, engaging digest. For each event:

A one-line curiosity hook.
One sentence on what happened.
One sentence on why it matters.

Keep the total digest under 500 words. Make it feel like a morning newsletter \
people look forward to reading.

EVENTS:
{events_text}
"""


# ── Style registry ───────────────────────────────────────────────
# Maps SummaryStyle → PromptConfig.  Add new styles here.

STYLES: dict[SummaryStyle, PromptConfig] = {
    SummaryStyle.SHORT: PromptConfig(
        template=_SHORT_TEMPLATE,
        temperature=0.7,
        max_output_tokens=120,
    ),
    SummaryStyle.DETAILED: PromptConfig(
        template=_DETAILED_TEMPLATE,
        temperature=0.8,
        max_output_tokens=300,
    ),
    SummaryStyle.REEL: PromptConfig(
        template=_REEL_TEMPLATE,
        temperature=0.9,
        max_output_tokens=400,
    ),
}

# Batch digest doesn't vary by style — it's always the same format.
BATCH_DIGEST_PROMPT = _BATCH_DIGEST_TEMPLATE


def get_prompt_config(style: SummaryStyle) -> PromptConfig:
    """Look up the PromptConfig for a given style."""
    return STYLES[style]
