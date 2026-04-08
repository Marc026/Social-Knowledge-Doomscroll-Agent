"""
tests/test_analyzer.py
======================
Unit tests for the LLM analysis layer (agent/analyzer.py).

All Anthropic API calls are mocked — no API key or network access required.
Tests validate that:
  - _safe_json() correctly extracts JSON from various response formats
  - analyse_post() correctly populates Post fields from LLM output
  - analyse_post() degrades gracefully on bad LLM responses
  - analyse_posts() processes all posts in a batch
  - generate_insights() returns the expected structure
  - generate_insights() handles the empty-posts edge case
  - compare_snapshots() parses trend delta correctly

Run with:
  pytest tests/test_analyzer.py -v
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent.scraper import Post
from agent import analyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_post(
    post_id: str   = "p1",
    title: str     = "NVDA up 15% on earnings beat",
    subreddit: str = "investing",
) -> Post:
    """Factory for Post objects with sensible defaults for analysis tests."""
    return Post(
        post_id      = post_id,
        platform     = "reddit",
        subreddit    = subreddit,
        title        = title,
        body         = "Revenue exceeded analyst consensus by a wide margin.",
        author       = "redditor1",
        url          = f"https://reddit.com/r/{subreddit}/comments/{post_id}",
        score        = 2000,
        num_comments = 150,
        created_utc  = datetime(2024, 6, 1, tzinfo=timezone.utc),
        top_comments = ["Incredible results!", "Buying more at open."],
    )


def _mock_client(response_text: str) -> MagicMock:
    """
    Return a mock Anthropic client whose messages.create() returns `response_text`
    as the first content block's text.

    This mirrors the actual Anthropic SDK response structure:
      resp.content[0].text  → the LLM's output string
    """
    content_block         = MagicMock()
    content_block.text    = response_text
    message               = MagicMock()
    message.content       = [content_block]
    client                = MagicMock()
    client.messages.create.return_value = message
    return client


# ---------------------------------------------------------------------------
# _safe_json tests
# ---------------------------------------------------------------------------

class TestSafeJson:
    """
    _safe_json is the critical parsing bridge between LLM text output and
    structured Python dicts. It must be robust to formatting variations.
    """

    def test_parses_plain_json_object(self):
        result = analyzer._safe_json('{"sentiment": "bullish", "confidence": 0.9}')
        assert result == {"sentiment": "bullish", "confidence": 0.9}

    def test_parses_json_array(self):
        result = analyzer._safe_json('[{"a": 1}, {"b": 2}]')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_strips_markdown_fences_json_tagged(self):
        """
        Some Claude responses wrap JSON in ```json ... ``` despite system prompt
        instructions. _safe_json must handle this.
        """
        result = analyzer._safe_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_strips_plain_markdown_fences(self):
        """Also handles ``` without the 'json' language tag."""
        result = analyzer._safe_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_handles_preamble_text(self):
        """
        LLM sometimes adds a sentence before the JSON object. We should extract
        the JSON portion correctly.
        """
        result = analyzer._safe_json('Here is the analysis:\n{"sentiment": "neutral"}')
        assert isinstance(result, dict)
        assert result["sentiment"] == "neutral"

    def test_returns_none_on_pure_text(self):
        """Non-JSON text should return None, not raise an exception."""
        result = analyzer._safe_json("This is just plain text with no JSON at all.")
        assert result is None

    def test_returns_none_on_empty_string(self):
        result = analyzer._safe_json("")
        assert result is None

    def test_returns_none_on_malformed_json(self):
        """Partial/malformed JSON should return None gracefully."""
        result = analyzer._safe_json('{"unclosed": "object"')
        assert result is None

    def test_handles_nested_objects(self):
        """Nested JSON structures should parse without issues."""
        nested = '{"outer": {"inner": [1, 2, 3]}}'
        result = analyzer._safe_json(nested)
        assert result["outer"]["inner"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# analyse_post() tests
# ---------------------------------------------------------------------------

class TestAnalysePost:
    """Tests for per-post LLM analysis."""

    @patch("agent.analyzer._client")
    def test_sets_sentiment_from_llm_output(self, mock_client_fn):
        """
        The LLM's sentiment value should be written onto the Post object.
        This is the core output of per-post analysis.
        """
        payload = json.dumps({
            "sentiment":  "bullish",
            "confidence": 0.92,
            "topics":     ["earnings", "semiconductors", "AI"],
            "summary":    "NVDA beat Q2 earnings by 15% on AI chip demand.",
        })
        mock_client_fn.return_value = _mock_client(payload)

        post   = _make_post()
        result = analyzer.analyse_post(post)

        assert result.sentiment == "bullish"
        assert "earnings" in result.topics
        assert result.summary is not None
        assert len(result.summary) > 0

    @patch("agent.analyzer._client")
    def test_sets_all_three_analysis_fields(self, mock_client_fn):
        """sentiment, topics, and summary must all be populated by analyse_post()."""
        payload = json.dumps({
            "sentiment":  "bearish",
            "confidence": 0.75,
            "topics":     ["recession", "rate hike"],
            "summary":    "Fed signals further rate hikes; markets react negatively.",
        })
        mock_client_fn.return_value = _mock_client(payload)

        post = _make_post()
        analyzer.analyse_post(post)

        assert post.sentiment == "bearish"
        assert "recession" in post.topics
        assert "Fed" in post.summary

    @patch("agent.analyzer._client")
    def test_defaults_to_neutral_on_bad_json(self, mock_client_fn):
        """
        If the LLM returns something that can't be parsed as JSON, the post
        should get sentiment="neutral" rather than raising an exception.
        This is the graceful-degradation path.
        """
        mock_client_fn.return_value = _mock_client(
            "I'm sorry, I couldn't analyse this post."
        )
        post = _make_post()
        analyzer.analyse_post(post)
        # Should not raise; should default
        assert post.sentiment == "neutral"

    @patch("agent.analyzer._client")
    def test_defaults_to_neutral_on_api_error(self, mock_client_fn):
        """
        An Anthropic APIError (rate limit, network error, etc.) should not
        crash the pipeline — the post gets default values.
        """
        import anthropic as anthropic_module
        client_mock = MagicMock()
        client_mock.messages.create.side_effect = anthropic_module.APIConnectionError(
            request=MagicMock()
        )
        mock_client_fn.return_value = client_mock

        post = _make_post()
        analyzer.analyse_post(post)
        assert post.sentiment == "neutral"

    @patch("agent.analyzer._client")
    def test_returns_same_post_object(self, mock_client_fn):
        """
        analyse_post() mutates in place AND returns the same object.
        This supports both mutation patterns and chaining.
        """
        payload = json.dumps({
            "sentiment": "neutral", "confidence": 0.5,
            "topics": [], "summary": "Mixed signals.",
        })
        mock_client_fn.return_value = _mock_client(payload)

        post   = _make_post()
        result = analyzer.analyse_post(post)
        assert result is post   # same object, not a copy

    @patch("agent.analyzer._client")
    def test_handles_missing_topics_key(self, mock_client_fn):
        """
        If the LLM omits 'topics' from the JSON (rare but possible), the post's
        topics should default to an empty list rather than raising a KeyError.
        """
        payload = json.dumps({
            "sentiment": "bullish",
            "confidence": 0.8,
            # 'topics' intentionally omitted
            "summary": "Strong earnings.",
        })
        mock_client_fn.return_value = _mock_client(payload)

        post = _make_post()
        analyzer.analyse_post(post)
        assert post.topics == []    # default, not KeyError


# ---------------------------------------------------------------------------
# analyse_posts() tests
# ---------------------------------------------------------------------------

class TestAnalysePosts:
    """Tests for batch analysis of multiple posts."""

    @patch("agent.analyzer._client")
    def test_processes_all_posts_in_batch(self, mock_client_fn):
        """Every post in the batch must be analysed."""
        payload = json.dumps({
            "sentiment": "bearish", "confidence": 0.7,
            "topics": ["recession"], "summary": "Markets down.",
        })
        mock_client_fn.return_value = _mock_client(payload)

        posts   = [_make_post(post_id=f"p{i}") for i in range(4)]
        results = analyzer.analyse_posts(posts)

        assert len(results) == 4
        assert all(p.sentiment == "bearish" for p in results)

    @patch("agent.analyzer._client")
    def test_returns_same_list_object(self, mock_client_fn):
        """analyse_posts() should return the same list (mutated), not a copy."""
        payload = json.dumps({
            "sentiment": "neutral", "confidence": 0.5,
            "topics": [], "summary": "n/a",
        })
        mock_client_fn.return_value = _mock_client(payload)

        posts  = [_make_post()]
        result = analyzer.analyse_posts(posts)
        assert result is posts

    @patch("agent.analyzer._client")
    def test_empty_batch_returns_empty_list(self, mock_client_fn):
        """An empty input list should produce an empty output, no API calls made."""
        result = analyzer.analyse_posts([])
        mock_client_fn.assert_not_called()
        assert result == []


# ---------------------------------------------------------------------------
# generate_insights() tests
# ---------------------------------------------------------------------------

class TestGenerateInsights:
    """Tests for corpus-level insight generation."""

    def _analysed_posts(self, count: int = 5) -> list[Post]:
        """Return a list of posts pre-populated with analysis fields."""
        posts = [_make_post(post_id=f"ins_{i}") for i in range(count)]
        for p in posts:
            p.sentiment = "bullish"
            p.topics    = ["AI", "semiconductors"]
            p.summary   = "Positive earnings signal."
        return posts

    @patch("agent.analyzer._client")
    def test_returns_dict_with_required_keys(self, mock_client_fn):
        """The insight dict must contain all keys used by downstream consumers."""
        payload = json.dumps({
            "overall_sentiment": "bullish",
            "dominant_topics":   ["AI", "chips"],
            "trend_summary":     "AI stocks surging on strong earnings season.",
            "actionable_insights": [
                {
                    "insight":  "Consider semiconductor exposure before Q3 reports.",
                    "evidence": "r/investing: 12 posts with strong bullish sentiment on NVDA/AMD.",
                    "urgency":  "high",
                }
            ],
            "notable_posts":        [],
            "sentiment_breakdown": {"bullish": 4, "bearish": 1, "neutral": 0},
        })
        mock_client_fn.return_value = _mock_client(payload)

        result = analyzer.generate_insights(self._analysed_posts())

        assert result["overall_sentiment"] == "bullish"
        assert len(result["actionable_insights"]) >= 1
        assert "evidence" in result["actionable_insights"][0]
        assert "urgency" in result["actionable_insights"][0]

    def test_empty_posts_returns_error_dict(self):
        """
        Calling generate_insights() with an empty list is a caller error.
        We return {"error": "..."} rather than raising so the pipeline
        can log and continue.
        """
        result = analyzer.generate_insights([])
        assert "error" in result
        # Error message should be informative
        assert len(result["error"]) > 0

    @patch("agent.analyzer._client")
    def test_graceful_on_unparseable_llm_response(self, mock_client_fn):
        """If the LLM returns something we can't parse, return the raw text."""
        mock_client_fn.return_value = _mock_client(
            "I cannot generate insights at this time."
        )
        result = analyzer.generate_insights(self._analysed_posts())
        # Should not raise; should include a 'raw_response' key
        assert "raw_response" in result or "error" in result


# ---------------------------------------------------------------------------
# compare_snapshots() tests
# ---------------------------------------------------------------------------

class TestCompareSnapshots:
    """Tests for trend delta generation between two insight snapshots."""

    @patch("agent.analyzer._client")
    def test_returns_expected_fields(self, mock_client_fn):
        """Trend delta must include sentiment_shift, new/fading topics, and key_change."""
        payload = json.dumps({
            "sentiment_shift": "improved",
            "new_topics":      ["AI agents", "open source LLMs"],
            "fading_topics":   ["recession fears", "rate hike"],
            "key_change":      "Market sentiment shifted from bearish to bullish on AI optimism.",
        })
        mock_client_fn.return_value = _mock_client(payload)

        result = analyzer.compare_snapshots(
            current  = {"overall_sentiment": "bullish",  "dominant_topics": ["AI"]},
            previous = {"overall_sentiment": "bearish",  "dominant_topics": ["recession"]},
        )

        assert result["sentiment_shift"] == "improved"
        assert "AI agents" in result["new_topics"]
        assert "rate hike" in result["fading_topics"]
        assert len(result["key_change"]) > 10

    @patch("agent.analyzer._client")
    def test_stable_sentiment_shift(self, mock_client_fn):
        """When sentiment hasn't changed materially, shift should be 'stable'."""
        payload = json.dumps({
            "sentiment_shift": "stable",
            "new_topics":      ["meme stocks"],
            "fading_topics":   [],
            "key_change":      "Sentiment remained broadly neutral with minor topic drift.",
        })
        mock_client_fn.return_value = _mock_client(payload)

        result = analyzer.compare_snapshots(
            current  = {"overall_sentiment": "neutral"},
            previous = {"overall_sentiment": "neutral"},
        )
        assert result["sentiment_shift"] == "stable"
