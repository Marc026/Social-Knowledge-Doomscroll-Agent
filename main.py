#!/usr/bin/env python3
"""
main.py
=======
CLI entry point for the Doomscroll Agent.

This script is intentionally thin — it parses arguments, configures logging,
and delegates entirely to the pipeline module. All business logic lives in
the agent/ package.

Usage
-----
  # Single run with defaults (wallstreetbets, investing, technology — hot — 20 posts each)
  python main.py

  # Custom subreddits, sorted by top posts of the day
  python main.py --subreddits stocks cryptocurrency ethereum --sort top --time-filter day

  # Continuous scheduled mode — runs every 30 minutes
  python main.py --schedule --interval 1800

  # Raw scrape only — no LLM calls (useful for testing scraper changes or saving cost)
  python main.py --skip-analysis

  # Print memory statistics
  python main.py --stats

  # Semantic search against stored posts
  python main.py --search "AI chip supply chain shortage"

  # Verbose debug logging
  python main.py -v

Environment
-----------
  Reads .env automatically if python-dotenv is installed.
  Required: ANTHROPIC_API_KEY
  Optional: DATA_DIR, SLACK_WEBHOOK_URL

Author : Marc Lapira
Project: Doomscroll Agent — Binox 2026 Take-Home (G2)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Load .env file if python-dotenv is installed.
# This is a convenience for local development — in production, environment
# variables are injected by Docker / the CI system directly.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass    # python-dotenv is optional; environment may already be set


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool = False) -> None:
    """
    Set up root logger to write to both stdout and a rolling log file.

    Log level:
    - Normal mode  : INFO  — shows pipeline progress, post counts, insights
    - Verbose mode : DEBUG — adds per-post analysis details, HTTP response sizes

    The log file is always written at DEBUG level (regardless of verbose flag)
    so that post-hoc debugging is possible without re-running the pipeline.
    Third-party loggers are quieted to WARNING to reduce noise.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)     # capture everything at the root

    fmt = logging.Formatter(
        fmt     = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # Console handler — respects the verbose flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler — always writes at DEBUG for full diagnostic history
    log_path = Path("data") / "agent.log"
    log_path.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # Quieten verbose third-party loggers that aren't useful in daily operation
    for noisy_logger in (
        "httpx", "httpcore", "urllib3",
        "chromadb", "sentence_transformers",
        "anthropic",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Define and parse all CLI arguments.

    Arguments are grouped conceptually but argparse doesn't support visual
    grouping beyond argument_group — we use comments in the source instead.
    """
    p = argparse.ArgumentParser(
        prog        = "doomscroll-agent",
        description = "Social knowledge extraction pipeline — monitors Reddit for market signals.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
examples:
  python main.py
  python main.py --subreddits wallstreetbets stocks --sort top --limit 50
  python main.py --schedule --interval 3600
  python main.py --stats
  python main.py --search "AI chip supply chain"
        """,
    )

    # --- What to scrape ---
    p.add_argument(
        "--subreddits", nargs="+",
        default=["wallstreetbets", "investing", "technology"],
        metavar="SUB",
        help="Subreddits to monitor — no 'r/' prefix (default: wallstreetbets investing technology)",
    )
    p.add_argument(
        "--sort",
        choices=["hot", "new", "top", "rising"],
        default="hot",
        help="Reddit listing sort (default: hot)",
    )
    p.add_argument(
        "--limit", type=int, default=20,
        metavar="N",
        help="Posts to fetch per subreddit per run (default: 20, max: 100)",
    )
    p.add_argument(
        "--time-filter",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="day",
        dest="time_filter",
        help="Time window for --sort top (default: day)",
    )

    # --- Feature flags ---
    p.add_argument(
        "--no-comments", action="store_true",
        help="Skip fetching top comments — faster, slightly lower analysis quality",
    )
    p.add_argument(
        "--skip-analysis", action="store_true",
        help="Store raw posts without LLM analysis — $0 API cost, no sentiment data",
    )
    p.add_argument(
        "--no-playwright", action="store_true",
        help="Disable Playwright fallback scraper (use in environments without Chromium)",
    )

    # --- Scheduling ---
    p.add_argument(
        "--schedule", action="store_true",
        help="Run continuously at --interval seconds (use with Docker / PM2)",
    )
    p.add_argument(
        "--interval", type=int, default=3600,
        metavar="SECS",
        help="Sleep time between scheduled runs in seconds (default: 3600 = 1 hour)",
    )

    # --- Utility modes ---
    p.add_argument(
        "--stats", action="store_true",
        help="Print memory store statistics as JSON and exit",
    )
    p.add_argument(
        "--search", type=str, metavar="QUERY",
        help="Semantic search against stored posts and exit (requires ChromaDB)",
    )

    # --- Logging ---
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level console output",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utility mode handlers
# ---------------------------------------------------------------------------

def _handle_stats() -> None:
    """Print memory store statistics as formatted JSON and exit."""
    from agent import memory
    stats = memory.stats()
    print(json.dumps(stats, indent=2))


def _handle_search(query: str) -> None:
    """Run a semantic search and print results in a readable format."""
    from agent import memory
    results = memory.semantic_search(query, n_results=10)

    if not results:
        print(
            "No results returned.\n"
            "This could mean:\n"
            "  • The vector store is empty (run a pipeline pass first)\n"
            "  • chromadb / sentence-transformers are not installed\n"
        )
        return

    print(f"\nTop {len(results)} semantic matches for: '{query}'")
    print("─" * 65)

    for i, r in enumerate(results, start=1):
        # Lower distance = more similar (cosine: 0.0 = identical, 2.0 = opposite)
        similarity_pct = max(0, (1 - r["distance"]) * 100)
        sentiment_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(
            r.get("sentiment", ""), "❓"
        )
        print(
            f"\n{i}. {sentiment_icon} [{r.get('sentiment', '?'):8s}]  "
            f"score={r.get('score', 0):>6}  "
            f"sim={similarity_pct:.0f}%  "
            f"r/{r.get('subreddit', '?')}"
        )
        print(f"   {r.get('document_snippet', '')[:120].replace(chr(10), ' ')}")
        print(f"   {r.get('url', '')}")


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    """
    Async entry point — called by asyncio.run() in __main__.

    We use an async main so that pipeline.run_once() (which calls async
    scraping functions) can be awaited directly without extra wrappers.
    """
    args = _parse_args()

    # Ensure the data directory exists before configuring file logging
    Path("data").mkdir(exist_ok=True)
    _configure_logging(args.verbose)

    logger = logging.getLogger("main")
    logger.info("Doomscroll Agent starting up")

    # ------------------------------------------------------------------
    # Utility modes — run and exit immediately
    # ------------------------------------------------------------------

    if args.stats:
        _handle_stats()
        return

    if args.search:
        _handle_search(args.search)
        return

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    from agent.pipeline import PipelineConfig, run_once, run_scheduled

    cfg = PipelineConfig(
        subreddits              = args.subreddits,
        sort                    = args.sort,
        posts_per_sub           = args.limit,
        time_filter             = args.time_filter,
        fetch_comments          = not args.no_comments,
        skip_analysis           = args.skip_analysis,
        use_playwright_fallback = not args.no_playwright,
        interval_seconds        = args.interval,
    )

    if args.schedule:
        # Scheduled mode — loops forever until Ctrl-C / SIGTERM
        logger.info(
            "Starting scheduled mode (interval=%ds)", cfg.interval_seconds
        )
        await run_scheduled(cfg)

    else:
        # Single run — execute once and print a clean summary to stdout
        result = await run_once(cfg)

        # ------------------------------------------------------------------
        # Print actionable insights to stdout in a format suitable for
        # piping, CI summary output, or Slack message formatting
        # ------------------------------------------------------------------
        if result.insights and result.insights.get("actionable_insights"):
            print("\n" + "=" * 60)
            print("ACTIONABLE INSIGHTS")
            print("=" * 60)

            for ai in result.insights["actionable_insights"]:
                urgency = ai.get("urgency", "?").upper()
                icon    = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(urgency, "⚪")
                print(f"\n{icon} [{urgency}] {ai.get('insight', '')}")
                print(f"   Evidence: {ai.get('evidence', '')}")

        # Print trend delta if available
        if result.trend_delta and result.trend_delta.get("key_change"):
            print("\n" + "─" * 60)
            print("TREND DELTA (vs previous snapshot)")
            print("─" * 60)
            print(result.trend_delta["key_change"])

            emerging = result.trend_delta.get("new_topics", [])
            fading   = result.trend_delta.get("fading_topics", [])
            if emerging:
                print(f"  📈 Emerging : {', '.join(emerging)}")
            if fading:
                print(f"  📉 Fading   : {', '.join(fading)}")

        # Exit non-zero if there were errors so CI pipelines can detect failures
        if result.errors:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
