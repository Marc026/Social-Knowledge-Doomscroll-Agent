"""
tests/test_memory.py
====================
Unit tests for the dual-layer memory module (agent/memory.py).

All tests redirect DATA_DIR to a fresh temporary directory (via monkeypatch)
so no real data is read or written. The tests exercise only the JSONL log layer
because chromadb and sentence-transformers are not installed in the test
environment (they're ~500 MB combined). Vector store tests are marked with
`pytest.mark.skipif` guards where appropriate.

Test coverage:
  - store() writes correct JSONL records
  - store() deduplicates correctly (idempotent on repeated calls)
  - load_posts() returns stored posts, respects subreddit/limit filters
  - store_insight() + load_latest_insight() round-trip
  - stats() returns expected structure and correct counts
  - _load_existing_ids() correctly reads existing IDs

Run with:
  pytest tests/test_memory.py -v
"""

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.scraper import Post


# ---------------------------------------------------------------------------
# Shared fixture — redirect DATA_DIR to a temp directory for each test
# ---------------------------------------------------------------------------

@pytest.fixture
def mem(tmp_path, monkeypatch):
    """
    Fixture that:
    1. Sets DATA_DIR env var to a fresh tmp_path for this test
    2. Reloads agent.memory so it picks up the new DATA_DIR
    3. Resets module-level global caches (_chroma_client, _collection, _embedder)
       to avoid state bleed between tests
    4. Returns the reloaded memory module

    Using monkeypatch ensures the env var is restored after each test,
    and importlib.reload() ensures the module-level Path constants are
    recalculated from the patched env var.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    import agent.memory as memory_module
    importlib.reload(memory_module)

    # Reset cached singletons so each test starts clean
    memory_module._chroma_client = None
    memory_module._collection    = None
    memory_module._embedder      = None

    return memory_module


# ---------------------------------------------------------------------------
# Post factory
# ---------------------------------------------------------------------------

def _make_post(
    post_id: str   = "test_001",
    subreddit: str = "investing",
    sentiment: str = "bullish",
) -> Post:
    """
    Factory for test Post objects.

    Creates a fully populated post including analysis fields so tests
    that store and reload posts can verify analysis data survives the
    JSONL round-trip.
    """
    p = Post(
        post_id      = post_id,
        platform     = "reddit",
        subreddit    = subreddit,
        title        = f"Test post: {post_id}",
        body         = "Some body text for testing purposes.",
        author       = "test_user",
        url          = f"https://reddit.com/r/{subreddit}/{post_id}",
        score        = 500,
        num_comments = 50,
        created_utc  = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc),
    )
    # Pre-populate analysis fields to test full round-trip persistence
    p.sentiment = sentiment
    p.topics    = ["stocks", "earnings"]
    p.summary   = f"Summary for post {post_id}."
    return p


# ---------------------------------------------------------------------------
# store() tests — JSONL log layer
# ---------------------------------------------------------------------------

class TestStore:
    def test_creates_jsonl_file(self, mem, tmp_path):
        """store() must create posts_log.jsonl on first call."""
        mem.store([_make_post()])
        log_file = tmp_path / "posts_log.jsonl"
        assert log_file.exists(), "posts_log.jsonl was not created"

    def test_writes_one_line_per_post(self, mem, tmp_path):
        """Each post must produce exactly one non-empty line in the JSONL file."""
        posts = [_make_post(f"p{i}") for i in range(5)]
        mem.store(posts)
        log_file = tmp_path / "posts_log.jsonl"
        lines = [l for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 5

    def test_each_line_is_valid_json(self, mem, tmp_path):
        """Every line in the JSONL file must be parseable as a JSON object."""
        mem.store([_make_post("json_check")])
        log_file = tmp_path / "posts_log.jsonl"
        for line in log_file.read_text().splitlines():
            if line.strip():
                data = json.loads(line)   # raises json.JSONDecodeError if invalid
                assert "post_id" in data

    def test_returns_count_of_new_posts(self, mem):
        """store() return value must equal the number of posts actually written."""
        posts = [_make_post(f"cnt{i}") for i in range(3)]
        count = mem.store(posts)
        assert count == 3

    def test_deduplicates_on_repeated_call(self, mem):
        """
        Calling store() twice with the same posts must NOT create duplicate records.
        The second call should return 0 (no new posts stored).
        """
        post    = _make_post("dup_post")
        first   = mem.store([post])
        second  = mem.store([post])
        assert first  == 1, "First store should write 1 post"
        assert second == 0, "Second store with same post_id should write 0 (duplicate)"

    def test_stores_only_new_posts_in_mixed_batch(self, mem):
        """
        A batch containing both new and existing posts should only write the new ones.
        """
        existing = _make_post("existing_p")
        mem.store([existing])                                           # write once

        new_post = _make_post("new_p")
        count    = mem.store([existing, new_post])                      # mixed batch
        assert count == 1, "Only the new post should be stored"

    def test_persists_analysis_fields(self, mem, tmp_path):
        """sentiment, topics, and summary must survive the JSONL round-trip."""
        p = _make_post("analysis_test")
        p.sentiment = "bearish"
        p.topics    = ["recession", "rate hike"]
        p.summary   = "Fed signals more hikes ahead."
        mem.store([p])

        # Read back raw from JSONL
        log_file = tmp_path / "posts_log.jsonl"
        data = json.loads(log_file.read_text().strip())
        assert data["sentiment"] == "bearish"
        assert "recession" in data["topics"]
        assert "Fed" in data["summary"]


# ---------------------------------------------------------------------------
# load_posts() tests
# ---------------------------------------------------------------------------

class TestLoadPosts:
    def test_returns_all_stored_posts_by_default(self, mem):
        """Without filters, load_posts() should return everything in the log."""
        posts = [_make_post(f"load_{i}") for i in range(5)]
        mem.store(posts)
        loaded = mem.load_posts()
        assert len(loaded) == 5

    def test_filters_by_subreddit(self, mem):
        """subreddit filter must exclude posts from other communities."""
        p_inv  = _make_post("inv_1", subreddit="investing")
        p_tech = _make_post("tech_1", subreddit="technology")
        p_wsb  = _make_post("wsb_1",  subreddit="wallstreetbets")
        mem.store([p_inv, p_tech, p_wsb])

        loaded = mem.load_posts(subreddit="technology")
        assert len(loaded) == 1
        assert loaded[0].subreddit == "technology"
        assert loaded[0].post_id == "tech_1"

    def test_respects_limit(self, mem):
        """
        load_posts(limit=N) must return at most N posts, ordered most-recent first.
        """
        posts = [_make_post(f"lim_{i}") for i in range(10)]
        mem.store(posts)
        loaded = mem.load_posts(limit=3)
        assert len(loaded) == 3

    def test_returns_empty_list_when_log_missing(self, mem, tmp_path):
        """If no log file exists yet, load_posts() should return [] not raise."""
        log_file = tmp_path / "posts_log.jsonl"
        assert not log_file.exists()
        loaded = mem.load_posts()
        assert loaded == []

    def test_reconstructs_post_fields_correctly(self, mem):
        """
        All Post fields written by store() must be correctly reconstructed by load_posts().
        This validates the full JSONL round-trip.
        """
        original = _make_post("round_trip")
        original.flair = "Discussion"
        original.top_comments = ["Comment A", "Comment B"]
        mem.store([original])

        loaded = mem.load_posts()
        assert len(loaded) == 1
        p = loaded[0]
        assert p.post_id      == "round_trip"
        assert p.subreddit    == "investing"
        assert p.sentiment    == "bullish"
        assert p.flair        == "Discussion"
        assert "Comment A" in p.top_comments
        assert p.created_utc.tzinfo is not None    # timezone-aware after reload

    def test_sorted_most_recent_first(self, mem):
        """
        load_posts() should return posts ordered by created_utc descending
        (most recent first) — callers rely on this for insight generation.
        """
        # Create posts with different timestamps
        p_old = _make_post("old_post")
        p_old.created_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)

        p_new = _make_post("new_post")
        p_new.created_utc = datetime(2024, 6, 1, tzinfo=timezone.utc)

        mem.store([p_old, p_new])
        loaded = mem.load_posts()

        assert loaded[0].post_id == "new_post"    # most recent first
        assert loaded[1].post_id == "old_post"


# ---------------------------------------------------------------------------
# store_insight() + load_latest_insight() tests
# ---------------------------------------------------------------------------

class TestInsightLog:
    def test_store_and_reload_insight(self, mem):
        """Insight data must survive the JSONL round-trip via the insights log."""
        insight = {
            "overall_sentiment": "bullish",
            "dominant_topics":   ["AI", "semiconductors"],
            "actionable_insights": [
                {"insight": "Buy NVDA", "evidence": "12 posts", "urgency": "high"}
            ],
        }
        mem.store_insight(insight, subreddits=["investing", "technology"])

        latest = mem.load_latest_insight()
        assert latest is not None
        assert latest["insight"]["overall_sentiment"] == "bullish"
        assert "investing" in latest["subreddits"]
        assert "timestamp" in latest   # timestamp must be written

    def test_load_latest_returns_none_when_no_file(self, mem):
        """First run — no insights log exists yet — should return None."""
        result = mem.load_latest_insight()
        assert result is None

    def test_load_latest_returns_most_recent_when_multiple(self, mem):
        """
        When multiple insight snapshots exist, load_latest_insight() must
        return the LAST one written (most recent run).
        """
        mem.store_insight({"overall_sentiment": "bearish"}, ["investing"])
        mem.store_insight({"overall_sentiment": "bullish"}, ["investing"])

        latest = mem.load_latest_insight()
        # Should be the second (most recent) write
        assert latest["insight"]["overall_sentiment"] == "bullish"

    def test_timestamp_is_iso_string(self, mem):
        """Timestamp in the insight log must be parseable as an ISO datetime."""
        mem.store_insight({"overall_sentiment": "neutral"}, ["technology"])
        latest = mem.load_latest_insight()
        ts = datetime.fromisoformat(latest["timestamp"])
        assert ts.tzinfo is not None    # must be timezone-aware


# ---------------------------------------------------------------------------
# stats() tests
# ---------------------------------------------------------------------------

class TestStats:
    def test_returns_required_keys(self, mem):
        """
        stats() return value must contain all keys used by the --stats CLI flag
        and any monitoring dashboards.
        """
        s = mem.stats()
        required = [
            "json_log_posts", "chroma_embeddings",
            "insight_snapshots", "log_path", "chroma_path",
        ]
        for key in required:
            assert key in s, f"Missing key in stats(): '{key}'"

    def test_correct_post_count(self, mem):
        """json_log_posts must reflect exactly how many posts are stored."""
        mem.store([_make_post(f"stat_{i}") for i in range(4)])
        s = mem.stats()
        assert s["json_log_posts"] == 4

    def test_correct_insight_count(self, mem):
        """insight_snapshots must reflect exactly how many insight runs are stored."""
        mem.store_insight({"overall_sentiment": "neutral"}, ["technology"])
        mem.store_insight({"overall_sentiment": "bullish"}, ["investing"])
        s = mem.stats()
        assert s["insight_snapshots"] == 2

    def test_zero_counts_when_empty(self, mem):
        """On a fresh installation with no data, all counts should be 0."""
        s = mem.stats()
        assert s["json_log_posts"]    == 0
        assert s["insight_snapshots"] == 0
        # chroma_embeddings may be 0 or None depending on whether chromadb is installed
        assert s.get("chroma_embeddings", 0) == 0


# ---------------------------------------------------------------------------
# _load_existing_ids() tests
# ---------------------------------------------------------------------------

class TestLoadExistingIds:
    def test_returns_empty_set_when_no_file(self, mem):
        """_load_existing_ids() must return an empty set on a fresh installation."""
        ids = mem._load_existing_ids()
        assert ids == set()

    def test_returns_correct_ids_after_store(self, mem):
        """After storing posts, their IDs must appear in _load_existing_ids()."""
        posts = [_make_post(f"id_test_{i}") for i in range(3)]
        mem.store(posts)
        ids = mem._load_existing_ids()
        for p in posts:
            assert p.post_id in ids

    def test_ignores_corrupt_lines(self, mem, tmp_path):
        """
        If a line in the JSONL log is corrupt (e.g. partial write on crash),
        _load_existing_ids() should skip it gracefully rather than raising.
        """
        log_file = tmp_path / "posts_log.jsonl"
        # Write one valid line and one corrupt line
        log_file.write_text(
            '{"post_id": "good_post", "title": "ok"}\n'
            'THIS IS NOT JSON AT ALL\n'
        )
        ids = mem._load_existing_ids()
        assert "good_post" in ids       # valid line read correctly
        assert len(ids) == 1            # corrupt line skipped, not raised
