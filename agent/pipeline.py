"""
agent/pipeline.py
=================
Main orchestration layer for the Doomscroll Agent.

This module is the integration point that ties together the four subsystems:
  scraper.py  → analyzer.py  → memory.py  → insight output

Pipeline steps (per run)
------------------------
  1. Scrape   — fetch posts from all configured subreddits concurrently
  2. Dedup    — filter out post_ids already present in the JSONL log
  3. Analyse  — run per-post LLM analysis (sentiment, topics, summary) on new posts only
  4. Store    — persist new posts to both JSONL log and ChromaDB
  5. Insights — generate corpus-level actionable insights from the last 100 posts
  6. Delta    — compare current insights against the previous snapshot
  7. Persist  — save insight snapshot to insights_log.jsonl

Two execution modes
-------------------
  run_once(config)       — execute steps 1-7 once and return a RunResult
  run_scheduled(config)  — call run_once() in an infinite loop at a fixed interval

PipelineConfig and RunResult are dataclasses rather than dicts so callers get
autocomplete, type checking, and clear field documentation.

Author : Marc Lapira
Project: Doomscroll Agent — Binox 2026 Take-Home (G2)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent import analyzer, memory
from agent.scraper import Post, scrape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All tuneable parameters for a pipeline run, collected in one place.

    Keeping configuration in a dataclass (rather than passing kwargs through
    multiple function calls) makes it easy to serialise for logging, compare
    runs, and override in tests.

    Attributes
    ----------
    subreddits              : communities to monitor (no "r/" prefix)
    sort                    : Reddit listing sort order
    posts_per_sub           : max posts fetched per subreddit per run
    time_filter             : time window for sort="top"
    fetch_comments          : whether to pull top comments for each post
    use_playwright_fallback : allow Playwright headless browser as API fallback
    skip_analysis           : store raw posts without LLM analysis (cost = $0)
    interval_seconds        : sleep time between runs in scheduled mode
    """

    subreddits: list[str] = field(
        default_factory=lambda: ["wallstreetbets", "investing", "technology"]
    )
    sort: str             = "hot"
    posts_per_sub: int    = 20
    time_filter: str      = "day"
    fetch_comments: bool  = True
    use_playwright_fallback: bool = True
    skip_analysis: bool   = False
    interval_seconds: int = 3600    # 1 hour — sensible default for hourly Cron job


# ---------------------------------------------------------------------------
# RunResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Captures everything that happened during a single pipeline run.

    Returned by run_once() so callers (main.py, tests) can inspect outcomes
    without needing to parse log output.

    Attributes
    ----------
    run_id        : unique identifier for this run (timestamp-based)
    started_at    : UTC datetime when the run began
    finished_at   : UTC datetime when the run completed (None if still running)
    subreddits    : which subreddits were scraped
    posts_fetched : total posts returned by all scrapers (before dedup)
    posts_stored  : new posts actually written to storage (after dedup)
    insights      : corpus-level insight dict from generate_insights()
    trend_delta   : delta dict from compare_snapshots(), or empty dict
    errors        : list of non-fatal error messages encountered during the run
    """

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    subreddits: list[str]        = field(default_factory=list)
    posts_fetched: int           = 0
    posts_stored: int            = 0
    insights: dict[str, Any]     = field(default_factory=dict)
    trend_delta: dict[str, Any]  = field(default_factory=dict)
    errors: list[str]            = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock time for the completed run, in seconds."""
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0


# ---------------------------------------------------------------------------
# Single pipeline run
# ---------------------------------------------------------------------------

async def run_once(config: PipelineConfig | None = None) -> RunResult:
    """
    Execute one full pipeline pass and return a RunResult.

    This is an async function because the scraping step uses async HTTP calls
    and we want to scrape multiple subreddits concurrently. The LLM analysis
    and storage steps are synchronous and run on the event-loop thread —
    they're fast enough that this isn't a problem at our scale.

    Error handling philosophy
    -------------------------
    Non-fatal errors (individual subreddit scrape fails, LLM analysis error for
    one post) are caught, logged, and added to RunResult.errors. The pipeline
    continues with whatever data it has. Only truly unrecoverable errors
    (e.g. all subreddits return 0 posts) cause an early return.

    This "degrade gracefully" approach means the Slack notification for a run
    still fires even if one subreddit was temporarily unreachable.
    """
    cfg = config or PipelineConfig()

    # Build a unique run ID from the current Unix timestamp
    run_id = f"run_{int(time.time())}"

    result = RunResult(
        run_id     = run_id,
        started_at = datetime.now(timezone.utc),
        subreddits = cfg.subreddits,
    )

    logger.info("=" * 60)
    logger.info("Pipeline run %s starting", run_id)
    logger.info("Subreddits : %s", cfg.subreddits)
    logger.info("Sort       : %s  |  limit: %d/sub", cfg.sort, cfg.posts_per_sub)
    logger.info("Analysis   : %s", "SKIP" if cfg.skip_analysis else "ENABLED")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Scrape all subreddits concurrently
    #
    # asyncio.gather() fires all coroutines simultaneously and waits for all
    # to complete. With 3 subreddits and ~2 s per scrape, this takes ~2 s
    # total rather than ~6 s sequential.
    #
    # return_exceptions=True means one failed scrape doesn't cancel the rest.
    # ------------------------------------------------------------------
    scrape_tasks = [
        scrape(
            subreddit               = sub,
            sort                    = cfg.sort,
            limit                   = cfg.posts_per_sub,
            time_filter             = cfg.time_filter,
            fetch_comments          = cfg.fetch_comments,
            use_playwright_fallback = cfg.use_playwright_fallback,
        )
        for sub in cfg.subreddits
    ]

    logger.info("Scraping %d subreddits concurrently …", len(cfg.subreddits))
    scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    # Collect all posts; log per-subreddit counts and errors
    all_posts: list[Post] = []
    for sub, result_or_exc in zip(cfg.subreddits, scrape_results):
        if isinstance(result_or_exc, Exception):
            msg = f"Scrape failed for r/{sub}: {result_or_exc}"
            logger.error(msg)
            result.errors.append(msg)
        else:
            posts_for_sub: list[Post] = result_or_exc
            logger.info("  r/%-25s → %d posts fetched", sub, len(posts_for_sub))
            all_posts.extend(posts_for_sub)

    result.posts_fetched = len(all_posts)
    logger.info("Total posts fetched across all subreddits: %d", result.posts_fetched)

    # Early exit if every subreddit failed — nothing useful to do
    if not all_posts:
        logger.warning("No posts fetched from any subreddit — aborting this run")
        result.finished_at = datetime.now(timezone.utc)
        return result

    # ------------------------------------------------------------------
    # Step 2: Deduplicate against the JSONL log
    #
    # We only analyse and store posts we haven't seen before. This avoids:
    #   a) wasting LLM API budget re-analysing the same content
    #   b) creating duplicate embeddings in ChromaDB
    #   c) inflating insight counts with stale posts
    # ------------------------------------------------------------------
    existing_ids = memory._load_existing_ids()
    new_posts    = [p for p in all_posts if p.post_id not in existing_ids]

    logger.info(
        "Deduplication: %d fetched | %d already stored | %d new",
        len(all_posts),
        len(all_posts) - len(new_posts),
        len(new_posts),
    )

    # ------------------------------------------------------------------
    # Step 3: LLM Analysis (only on new posts)
    #
    # Each post gets a per-post analyse_post() call (sequential, not concurrent)
    # to respect Anthropic rate limits. The analysis writes sentiment, topics,
    # and summary directly onto the Post objects before they're stored.
    # ------------------------------------------------------------------
    if new_posts and not cfg.skip_analysis:
        logger.info("Running LLM analysis on %d new posts …", len(new_posts))
        try:
            analyzer.analyse_posts(new_posts)
        except Exception as exc:
            # Analysis failure is non-fatal — store raw posts without sentiment
            msg = f"LLM analysis failed: {exc}"
            logger.error(msg)
            result.errors.append(msg)
    elif cfg.skip_analysis:
        logger.info("Analysis skipped (--skip-analysis flag set)")

    # ------------------------------------------------------------------
    # Step 4: Store to memory
    #
    # store() handles its own deduplication as a safety net, but by this
    # point new_posts already excludes duplicates from Step 2.
    # ------------------------------------------------------------------
    stored = memory.store(new_posts)
    result.posts_stored = stored
    logger.info("Stored %d new posts to memory", stored)

    # ------------------------------------------------------------------
    # Step 5: Generate corpus-level insights
    #
    # We load the last 100 posts across all subreddits (not just this run's
    # new posts) to give the LLM a meaningful corpus to work with. A single
    # run might add only 3-5 truly new posts; the insight generation is much
    # richer with historical context included.
    # ------------------------------------------------------------------
    recent_posts = memory.load_posts(limit=100)

    if not cfg.skip_analysis and recent_posts:
        logger.info(
            "Generating corpus insights from %d recent posts …", len(recent_posts)
        )
        try:
            insights = analyzer.generate_insights(recent_posts)
            result.insights = insights
        except Exception as exc:
            msg = f"Insight generation failed: {exc}"
            logger.error(msg)
            result.errors.append(msg)
    else:
        logger.info("Corpus insight generation skipped")
        result.insights = {}

    # ------------------------------------------------------------------
    # Step 6: Trend delta vs previous snapshot
    #
    # Load the most recent entry from insights_log.jsonl and compare it
    # against this run's fresh insights. This is only meaningful if we
    # have both a current and a previous snapshot to compare.
    # ------------------------------------------------------------------
    if result.insights:
        prev_snapshot = memory.load_latest_insight()

        if prev_snapshot and "insight" in prev_snapshot:
            logger.info("Comparing against previous snapshot …")
            try:
                result.trend_delta = analyzer.compare_snapshots(
                    current  = result.insights,
                    previous = prev_snapshot["insight"],
                )
            except Exception as exc:
                logger.warning("Trend comparison failed (non-fatal): %s", exc)
        else:
            logger.info(
                "No previous snapshot found — skipping trend delta (first run?)"
            )

        # ------------------------------------------------------------------
        # Step 7: Persist the current insight snapshot
        #
        # Written AFTER the trend comparison so we compare current against
        # the truly previous run (not the one we just generated).
        # ------------------------------------------------------------------
        memory.store_insight(result.insights, cfg.subreddits)

    # Stamp completion time and print a clean summary to the log
    result.finished_at = datetime.now(timezone.utc)
    logger.info(
        "Pipeline run %s complete in %.1fs — %d new posts stored",
        run_id, result.elapsed_seconds, result.posts_stored,
    )
    _log_run_summary(result)
    return result


# ---------------------------------------------------------------------------
# Scheduled runner
# ---------------------------------------------------------------------------

async def run_scheduled(config: PipelineConfig | None = None) -> None:
    """
    Run the pipeline repeatedly on a fixed interval.

    Designed to run as a long-lived process (e.g. inside Docker via
    docker-compose), where n8n's Execute Command node fires the scheduler
    rather than the built-in Cron trigger. You can also use this directly
    without n8n by starting the container in scheduled mode.

    Error handling: an unhandled exception in run_once() is caught and logged,
    then the loop sleeps and retries. This prevents a transient API error from
    killing the whole scheduler process.

    Interrupt with Ctrl-C (SIGINT) — asyncio.sleep() is interrupted cleanly.
    """
    cfg = config or PipelineConfig()
    interval = cfg.interval_seconds

    logger.info(
        "Scheduled runner started — interval: %ds (%dm %ds)",
        interval, interval // 60, interval % 60,
    )

    run_count = 0
    while True:
        run_count += 1
        logger.info("--- Scheduled run #%d ---", run_count)

        try:
            await run_once(cfg)
        except Exception as exc:
            # Catch-all so the scheduler loop never dies on an unexpected error
            logger.error(
                "Unhandled error in scheduled run #%d: %s",
                run_count, exc,
                exc_info=True,  # include full traceback in the log
            )

        logger.info("Sleeping %ds until next run …", interval)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_run_summary(result: RunResult) -> None:
    """
    Pretty-print a structured summary of the run to the logger at INFO level.

    Called at the end of run_once() so operators have a human-readable digest
    in the log file without parsing the full debug output.
    """
    sep = "─" * 55
    logger.info(sep)
    logger.info("RUN SUMMARY  [%s]", result.run_id)
    logger.info("  Started   : %s UTC", result.started_at.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("  Elapsed   : %.1fs", result.elapsed_seconds)
    logger.info("  Fetched   : %d posts across %d subreddits",
                result.posts_fetched, len(result.subreddits))
    logger.info("  New stored: %d posts", result.posts_stored)

    ins = result.insights
    if ins and "overall_sentiment" in ins:
        logger.info("  Sentiment : %s", ins["overall_sentiment"].upper())
        bd = ins.get("sentiment_breakdown", {})
        logger.info(
            "  Breakdown : 🟢 bullish=%d  🔴 bearish=%d  ⚪ neutral=%d",
            bd.get("bullish", 0), bd.get("bearish", 0), bd.get("neutral", 0),
        )
        logger.info("  Topics    : %s", ", ".join(ins.get("dominant_topics", [])[:5]))

        # Log each actionable insight with urgency prefix
        for ai in ins.get("actionable_insights", []):
            urgency_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                ai.get("urgency", "").lower(), "⚪"
            )
            logger.info(
                "  %s [%s] %s",
                urgency_icon,
                ai.get("urgency", "?").upper(),
                ai.get("insight", ""),
            )

    if result.trend_delta and "key_change" in result.trend_delta:
        logger.info("  Trend     : %s", result.trend_delta["key_change"])
        emerging = result.trend_delta.get("new_topics", [])
        fading   = result.trend_delta.get("fading_topics", [])
        if emerging:
            logger.info("  Emerging  : %s", ", ".join(emerging))
        if fading:
            logger.info("  Fading    : %s", ", ".join(fading))

    if result.errors:
        logger.warning("  ERRORS (%d):", len(result.errors))
        for err in result.errors:
            logger.warning("    • %s", err)

    logger.info(sep)
