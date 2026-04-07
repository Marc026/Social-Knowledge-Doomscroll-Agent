"""
agent/analyzer.py
=================
LLM-powered analysis layer using the Anthropic API (Claude).

Three distinct analysis tasks are handled here:

  1. Per-post analysis
     For each scraped Post, the LLM returns:
       - sentiment   : "bullish" | "bearish" | "neutral" + confidence score
       - topics      : 2-5 short keyword tags (e.g. ["AI", "chips", "earnings"])
       - summary     : 1-2 sentence distillation of the post's core claim

  2. Corpus-level insight generation
     Given a batch of already-analysed Posts, the LLM synthesises:
       - overall_sentiment      : dominant mood across the corpus
       - dominant_topics        : top recurring themes
       - trend_summary          : 2-3 sentence narrative
       - actionable_insights    : 2-4 concrete, urgency-ranked business actions
       - sentiment_breakdown    : bullish/bearish/neutral counts
       - notable_posts          : up to 3 standout posts with reasoning

  3. Trend delta comparison
     Compares two successive insight snapshots to surface what's emerging,
     what's fading, and how overall sentiment has shifted.

Design decisions
----------------
- All LLM outputs are requested as JSON objects so they can be parsed and stored
  structurally rather than as freeform text.
- _safe_json() strips markdown fences and finds the first valid JSON object/array,
  making the parser resilient to minor LLM formatting deviations.
- API calls are sequential (not concurrent) to respect Anthropic's rate limits.
  At 25 posts/run the total latency is ~30-60 s — acceptable for an hourly job.
- The model constant is centralised so ops can swap to haiku for cost reduction
  with a single-line change.

Author : Marc Lapira
Project: Doomscroll Agent — Binox 2026 Take-Home (G2)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import anthropic

from agent.scraper import Post

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _client() -> anthropic.Anthropic:
    """
    Return a configured Anthropic client.

    The API key is read from the ANTHROPIC_API_KEY environment variable, which
    should be set in .env (development) or as a container/CI secret (production).
    Raises EnvironmentError early with a clear message rather than letting the
    first API call fail with a cryptic 401.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example → .env and fill in your key."
        )
    return anthropic.Anthropic(api_key=api_key)


# Model selection
# ---------------
# claude-opus-4-5   : highest quality, higher cost — good for prototyping / demos
# claude-haiku-4-5  : ~70x cheaper, slightly less nuanced — recommended for production
#
# Cost estimate at claude-opus-4-5 (as of mid-2025):
#   25 posts × 3 subreddits = 75 posts/run
#   ~600 tokens input per post + ~1,500 tokens for corpus insight ≈ 47K tokens/run
#   At ~$15/M input tokens: ≈ $0.70/run, $16.80/day at hourly cadence
#
# Swap to claude-haiku-4-5 for production to bring this to ~$0.01/run.
MODEL = "claude-opus-4-5"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_json(text: str) -> dict | list | None:
    """
    Extract and parse the first valid JSON object or array from `text`.

    LLMs occasionally wrap their JSON in markdown code fences (```json ... ```)
    even when instructed not to, or add a brief preamble sentence. This function
    handles both cases gracefully by:
      1. Stripping markdown fences
      2. Scanning for the first `{` or `[` character and the last matching closer
      3. Attempting json.loads() on that substring

    Returns None if no valid JSON can be found — callers should handle this as
    a degraded-mode fallback (e.g. defaulting sentiment to "neutral").
    """
    # Strip markdown code fences — covers ```json, ```JSON, ``` etc.
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # Try to find the outermost JSON structure: object first, then array
    for start_ch, end_ch in [("{", "}"), ("[", "]")]:
        start = text.find(start_ch)
        end   = text.rfind(end_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass    # try the other bracket pair

    logger.debug("_safe_json could not find valid JSON in: %s", text[:200])
    return None


# ---------------------------------------------------------------------------
# 1. Per-post analysis
# ---------------------------------------------------------------------------

# System prompt for per-post analysis.
# Written to be terse and precise — we want JSON output, not conversational prose.
# The "Do not include any text outside the JSON object" instruction is critical
# for reliable parsing; Claude respects it consistently.
_POST_ANALYSIS_SYSTEM = """\
You are a financial-market and technology trend analyst.
Given a Reddit post (title + body + top comments from the specified subreddit),
return ONLY a JSON object with exactly these fields:
{
  "sentiment": "<bullish|bearish|neutral>",
  "confidence": <float 0.0-1.0>,
  "topics": ["<tag1>", "<tag2>"],
  "summary": "<1-2 sentence summary of the post's core claim or discussion>"
}
Rules:
- sentiment: "bullish" = positive/optimistic, "bearish" = negative/pessimistic, "neutral" = mixed/informational
- topics: 2-5 short tags (1-3 words each), e.g. ["Fed policy", "rate hike", "inflation"]
- summary: focus on the claim, not the format (do NOT start with "The post says...")
- Do not include any text outside the JSON object."""


def analyse_post(post: Post) -> Post:
    """
    Run LLM sentiment/topic/summary analysis on a single Post, mutating it in place.

    The post's sentiment, topics, and summary fields are populated by this call.
    On any API or parsing error, sentiment defaults to "neutral" so downstream
    processes can continue without a hard stop.

    Returns the same post object (mutated) for chaining convenience.

    Note: this function is synchronous (no async) because Anthropic's Python SDK
    uses sync calls by default, and making 25 sequential sync calls inside an
    async pipeline is fine — they block the event loop thread but there's no
    concurrent I/O to interleave with at this point.
    """
    client = _client()

    # Build the user message: structured enough for the LLM to parse context,
    # but concise enough to keep token usage low.
    prompt = (
        f"Title: {post.title}\n\n"
        f"Body:\n{post.body or '(no body text)'}\n\n"
        f"Top comments:\n"
        + (
            "\n".join(f"- {c}" for c in post.top_comments[:5])
            or "(none)"
        )
        + f"\n\nSubreddit context: r/{post.subreddit}"
    )

    try:
        resp = client.messages.create(
            model      = MODEL,
            max_tokens = 512,           # analysis output is always small
            system     = _POST_ANALYSIS_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw    = resp.content[0].text
        parsed = _safe_json(raw)

        if isinstance(parsed, dict):
            # Only write fields that are present — let defaults stand for missing ones
            post.sentiment = parsed.get("sentiment", "neutral")
            post.topics    = parsed.get("topics", [])
            post.summary   = parsed.get("summary", "")
            logger.debug(
                "Analysed post %s → sentiment=%s topics=%s",
                post.post_id, post.sentiment, post.topics,
            )
        else:
            # LLM returned something we can't parse — degrade gracefully
            logger.warning(
                "Unexpected LLM response for post %s: %s …",
                post.post_id, raw[:200],
            )
            post.sentiment = "neutral"

    except anthropic.APIError as exc:
        # Network / auth / quota error — log it and continue with defaults
        logger.error(
            "Anthropic API error analysing post %s: %s", post.post_id, exc
        )
        post.sentiment = "neutral"

    return post


def analyse_posts(posts: list[Post]) -> list[Post]:
    """
    Analyse a batch of Posts sequentially, logging progress.

    Sequential (not concurrent) to avoid hitting Anthropic's rate limits.
    At ~1-2 s per call, 25 posts takes ~30-50 s — well within the hourly schedule.

    Returns the same list (mutated in place) for chaining convenience.
    """
    logger.info("Starting LLM analysis on %d posts …", len(posts))
    for i, post in enumerate(posts, start=1):
        logger.info(
            "  [%d/%d] Analysing: %s …", i, len(posts), post.title[:70]
        )
        analyse_post(post)
    logger.info("LLM analysis complete for %d posts", len(posts))
    return posts


# ---------------------------------------------------------------------------
# 2. Corpus-level insight generation
# ---------------------------------------------------------------------------

# System prompt for corpus-level insight generation.
# The LLM receives a compact JSON array of post summaries (not full text) to
# stay within a reasonable context budget while still having enough signal.
_INSIGHT_SYSTEM = """\
You are a senior market intelligence analyst synthesising Reddit community signals.
You will receive a JSON array of analysed Reddit posts (compact summaries).
Return ONLY a JSON object with exactly these fields:
{
  "overall_sentiment": "<bullish|bearish|neutral>",
  "dominant_topics": ["<tag>", ...],
  "trend_summary": "<2-3 sentences describing the dominant narrative in the corpus>",
  "actionable_insights": [
    {
      "insight": "<one concrete, specific insight a business or investor could act on>",
      "evidence": "<which posts, topics, or patterns support this>",
      "urgency": "<high|medium|low>"
    }
  ],
  "notable_posts": [
    {"post_id": "...", "reason": "<why this post stands out>"}
  ],
  "sentiment_breakdown": {
    "bullish": <int>,
    "bearish": <int>,
    "neutral": <int>
  }
}
Rules:
- dominant_topics: top 5 recurring topics across all posts
- actionable_insights: 2-4 items, ranked by urgency (high first)
- notable_posts: up to 3 posts with unusually high engagement or signal value
- Do not include any text outside the JSON object."""


def generate_insights(posts: list[Post]) -> dict[str, Any]:
    """
    Generate corpus-level actionable insights from a batch of analysed Posts.

    Rather than sending full post text, we build compact summary objects
    (title, sentiment, topics, score, comment count) to keep the prompt
    within a predictable token budget regardless of how many posts there are.

    The insight output is the main "product" of the pipeline — what gets posted
    to Slack, persisted to insights_log.jsonl, and compared against prior runs.

    Returns a dict matching the schema in _INSIGHT_SYSTEM, or {"error": "..."}
    if the API call fails entirely.
    """
    if not posts:
        # Callers should guard against this, but we return a meaningful error
        # rather than crashing so the pipeline can continue gracefully.
        return {"error": "No posts provided to generate_insights()"}

    # Build compact representation — avoids sending full body/comment text
    # which would inflate token usage for no additional analytical value here.
    post_summaries = [
        {
            "post_id":      p.post_id,
            "subreddit":    p.subreddit,
            "title":        p.title,
            "score":        p.score,
            "num_comments": p.num_comments,
            "sentiment":    p.sentiment or "unknown",
            "topics":       p.topics,
            # Use LLM summary if available, fall back to the raw title
            "summary":      p.summary or p.title,
        }
        for p in posts
    ]

    client = _client()
    prompt = json.dumps(post_summaries, ensure_ascii=False)

    logger.info(
        "Generating corpus-level insights from %d posts …", len(posts)
    )

    try:
        resp = client.messages.create(
            model      = MODEL,
            max_tokens = 1500,      # insight output can be verbose — give it room
            system     = _INSIGHT_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw    = resp.content[0].text
        parsed = _safe_json(raw)

        if isinstance(parsed, dict):
            logger.info(
                "Insight generation complete — overall_sentiment: %s  topics: %s",
                parsed.get("overall_sentiment"),
                parsed.get("dominant_topics", [])[:3],
            )
            return parsed
        else:
            logger.warning(
                "Unexpected insight response (first 300 chars): %s", raw[:300]
            )
            return {"raw_response": raw}

    except anthropic.APIError as exc:
        logger.error("Anthropic API error generating insights: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# 3. Trend delta — compare successive snapshots
# ---------------------------------------------------------------------------

# System prompt for trend comparison.
# This is a cheap, low-token call (~200 tokens in, ~150 out) that runs once per
# pipeline run to surface what's changing between runs — the "doomscroll diff".
_TREND_SYSTEM = """\
You are a trend analyst comparing two consecutive market intelligence snapshots.
Given 'current' and 'previous' insight reports, return ONLY a JSON object:
{
  "sentiment_shift": "<improved|worsened|stable>",
  "new_topics": ["<topics emerging in current that were absent in previous>"],
  "fading_topics": ["<topics prominent in previous but absent in current>"],
  "key_change": "<1-2 sentences describing the single most significant shift>"
}
Do not include any text outside the JSON object."""


def compare_snapshots(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare two insight snapshots and return a structured trend delta.

    Called once per pipeline run after generate_insights() produces the current
    snapshot. The previous snapshot is loaded from insights_log.jsonl by memory.py.

    On failure returns {"error": "..."} — the pipeline logs this but continues
    because a missing trend delta is cosmetic, not functional.
    """
    client = _client()

    # Send both snapshots as a single JSON payload so the model has full context
    prompt = json.dumps(
        {"current": current, "previous": previous},
        ensure_ascii=False,
    )

    try:
        resp = client.messages.create(
            model      = MODEL,
            max_tokens = 512,   # trend delta is always a small object
            system     = _TREND_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw    = resp.content[0].text
        parsed = _safe_json(raw)

        if isinstance(parsed, dict):
            logger.info(
                "Trend delta: shift=%s  key_change=%s",
                parsed.get("sentiment_shift"),
                parsed.get("key_change", "")[:80],
            )
            return parsed
        else:
            return {"raw": raw}

    except anthropic.APIError as exc:
        logger.error("Anthropic API error comparing snapshots: %s", exc)
        return {"error": str(exc)}
