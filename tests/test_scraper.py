"""
tests/test_scraper.py
=====================
Unit tests for the scraper module (agent/scraper.py).

All tests are pure Python — no network calls, no browser, no API key required.
The Post dataclass and its methods are tested directly.

Test coverage:
  - Post.full_text() construction
  - Post.to_dict() serialisation (keys, types, round-trip safety)
  - Post default field values

Run with:
  pytest tests/test_scraper.py -v
"""

from datetime import datetime, timezone

import pytest

from agent.scraper import Post


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_post(**overrides) -> Post:
    """
    Construct a Post with sensible defaults for testing.

    Using a factory function rather than a fixture means each test gets an
    independent object (no shared state) and overrides are easy to specify.
    """
    defaults = dict(
        post_id      = "abc123",
        platform     = "reddit",
        subreddit    = "investing",
        title        = "Tesla beats earnings expectations",
        body         = "Revenue up 20% year-over-year, beating analyst consensus.",
        author       = "market_watcher",
        url          = "https://reddit.com/r/investing/comments/abc123",
        score        = 1500,
        num_comments = 300,
        created_utc  = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return Post(**defaults)


# ---------------------------------------------------------------------------
# Post.full_text() tests
# ---------------------------------------------------------------------------

class TestFullText:
    def test_includes_title(self):
        """Title must always be present — it's the primary content signal."""
        p = make_post(title="Big earnings news")
        assert "Big earnings news" in p.full_text()

    def test_includes_body_when_present(self):
        """Body text adds depth for self-posts (not link posts)."""
        p = make_post(body="Detailed analysis here")
        assert "Detailed analysis here" in p.full_text()

    def test_body_omitted_when_empty(self):
        """
        Link posts have no body text — full_text() should still work
        and not include a blank line from an empty string.
        """
        p = make_post(body="")
        text = p.full_text()
        assert "Tesla beats earnings" in text
        # No double-newline from an empty body entry
        assert "\n\n" not in text

    def test_includes_top_comments(self):
        """Comments provide community-reaction signal for sentiment analysis."""
        p = make_post(top_comments=["Very bullish!", "Finally some good news."])
        text = p.full_text()
        assert "Very bullish!" in text
        assert "Finally some good news." in text

    def test_caps_comments_at_five(self):
        """
        We cap at 5 comments to keep LLM token usage predictable.
        The 6th comment should NOT appear in full_text().
        """
        comments = [f"Comment number {i}" for i in range(10)]
        p = make_post(top_comments=comments)
        text = p.full_text()
        assert "Comment number 4" in text      # 5th comment (index 4) — should be present
        assert "Comment number 5" not in text  # 6th comment — should be excluded

    def test_no_comments_still_works(self):
        """
        Posts without comments (e.g. from Playwright fallback) should produce
        a valid full_text() string.
        """
        p = make_post(top_comments=[])
        text = p.full_text()
        assert len(text) > 0
        assert p.title in text


# ---------------------------------------------------------------------------
# Post.to_dict() tests
# ---------------------------------------------------------------------------

class TestToDict:
    REQUIRED_KEYS = [
        "post_id", "platform", "subreddit", "title", "body",
        "author", "url", "score", "num_comments", "created_utc",
        "fetched_at", "flair", "top_comments", "sentiment", "topics", "summary",
    ]

    def test_contains_all_required_keys(self):
        """
        All downstream consumers (JSONL log, ChromaDB metadata, API responses)
        depend on a stable dict schema. If a key is missing, things break silently.
        """
        d = make_post().to_dict()
        for key in self.REQUIRED_KEYS:
            assert key in d, f"Missing required key in to_dict(): '{key}'"

    def test_created_utc_is_iso_string(self):
        """
        Datetime fields must be ISO-8601 strings (not datetime objects) so they
        serialise cleanly to JSON without a custom encoder.
        """
        d = make_post().to_dict()
        iso_str = d["created_utc"]
        assert isinstance(iso_str, str)
        # Should be round-trip parseable
        parsed = datetime.fromisoformat(iso_str)
        assert parsed.year == 2024

    def test_fetched_at_is_iso_string(self):
        """fetched_at follows the same contract as created_utc."""
        d = make_post().to_dict()
        iso_str = d["fetched_at"]
        assert isinstance(iso_str, str)
        datetime.fromisoformat(iso_str)  # raises ValueError if invalid

    def test_score_is_int(self):
        """Score must be an int — ChromaDB metadata requires scalar types."""
        d = make_post(score=2500).to_dict()
        assert isinstance(d["score"], int)
        assert d["score"] == 2500

    def test_top_comments_is_list(self):
        """top_comments must serialise as a JSON array, not a tuple."""
        d = make_post(top_comments=["a", "b"]).to_dict()
        assert isinstance(d["top_comments"], list)
        assert d["top_comments"] == ["a", "b"]

    def test_analysis_fields_default_to_none_and_empty(self):
        """
        Analysis fields are None/[] before analyzer.py runs.
        Downstream code should handle None gracefully.
        """
        d = make_post().to_dict()
        assert d["sentiment"] is None
        assert d["topics"] == []
        assert d["summary"] is None

    def test_analysis_fields_preserved_when_set(self):
        """Once the LLM has run, analysis fields should survive serialisation."""
        p = make_post()
        p.sentiment = "bullish"
        p.topics    = ["earnings", "EV", "Tesla"]
        p.summary   = "Tesla beats Q2 expectations significantly."
        d = p.to_dict()
        assert d["sentiment"] == "bullish"
        assert "earnings" in d["topics"]
        assert "Tesla" in d["summary"]


# ---------------------------------------------------------------------------
# Post default field values
# ---------------------------------------------------------------------------

class TestPostDefaults:
    def test_flair_defaults_to_none(self):
        """Un-flaired posts are common — flair=None must be the default."""
        assert make_post().flair is None

    def test_top_comments_defaults_to_empty_list(self):
        """New Post with no comments should have an empty list, not None."""
        assert make_post().top_comments == []

    def test_sentiment_defaults_to_none(self):
        """Sentinel value for 'not yet analysed'."""
        assert make_post().sentiment is None

    def test_topics_defaults_to_empty_list(self):
        assert make_post().topics == []

    def test_summary_defaults_to_none(self):
        assert make_post().summary is None

    def test_fetched_at_is_utc_aware(self):
        """
        fetched_at must be timezone-aware so comparisons with created_utc
        (also UTC-aware) don't raise a TypeError.
        """
        p = make_post()
        assert p.fetched_at.tzinfo is not None

    def test_platform_field(self):
        """Platform string identifies the source for multi-platform future use."""
        p = make_post()
        assert p.platform == "reddit"
