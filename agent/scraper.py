"""
agent/scraper.py
================
Reddit content scraper for the Doomscroll Agent.

Scraping strategy (two-layer, ordered by preference):
  1. Reddit public JSON API  — fast, structured, no auth required for public subs.
     Endpoint: https://www.reddit.com/r/{subreddit}/{sort}.json
     Handles rate-limits with exponential back-off.

  2. Playwright headless browser  — fallback when the API is blocked or returns 0
     results. Navigates old.reddit.com (simpler DOM, no JS hydration required).

Public surface for other modules:
  scrape(subreddit, sort, limit, ...) -> list[Post]   # auto-selects strategy
  Post                                                 # normalised data model

All Reddit-specific logic (URL construction, JSON field names, DOM selectors) is
isolated here so swapping to a different platform only requires a new scraper
module — the rest of the pipeline is platform-agnostic.

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

import httpx
from playwright.async_api import async_playwright

# Module-level logger — inherits the root logger configured in main.py
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Post:
    """
    Platform-agnostic representation of a single social-media post.

    Fields are deliberately flat (no nested objects) so they serialise cleanly
    to JSON and map 1-to-1 with ChromaDB metadata fields (which must be scalar).

    Fields populated by the scraper
    --------------------------------
    post_id       : platform-unique identifier (Reddit: base-36 "id" field)
    platform      : source platform string, e.g. "reddit"
    subreddit     : community name or channel equivalent on other platforms
    title         : post headline
    body          : self-text body; empty string for link posts
    author        : username string; "[deleted]" when account is removed
    url           : permalink to the original post
    score         : net upvotes (Reddit fuzzes this slightly to prevent gaming)
    num_comments  : comment count at time of scrape
    created_utc   : original post timestamp (UTC-aware datetime)
    fetched_at    : when this scrape ran (UTC-aware, defaults to now)
    flair         : subreddit tag/category label, or None
    top_comments  : up to 5 top-level comment bodies (best effort, may be empty)

    Fields populated by analyzer.py
    ---------------------------------
    sentiment     : "bullish" | "bearish" | "neutral"  (set by LLM analysis)
    topics        : list of short topic tags, e.g. ["AI", "chips", "earnings"]
    summary       : 1-2 sentence LLM-generated summary of the post's core claim
    """

    # --- Core identity ---
    post_id: str
    platform: str
    subreddit: str

    # --- Content ---
    title: str
    body: str
    author: str
    url: str

    # --- Engagement signals ---
    score: int
    num_comments: int

    # --- Timestamps ---
    created_utc: datetime
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Optional metadata ---
    flair: str | None = None
    top_comments: list[str] = field(default_factory=list)

    # --- Analysis results (written by analyzer.py, None until analysed) ---
    sentiment: str | None = None
    topics: list[str] = field(default_factory=list)
    summary: str | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def full_text(self) -> str:
        """
        Concatenate all available text into a single string suitable for
        LLM prompting and vector embedding.

        We cap at the first 5 comments to keep token count predictable —
        beyond that, thread replies get noisy and repetitive.
        """
        parts = [self.title]
        if self.body:
            parts.append(self.body)
        # Top comments provide community reaction signal alongside the post itself
        parts.extend(self.top_comments[:5])
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise to a plain dict for JSON log persistence.

        Datetime fields are converted to ISO-8601 strings so they round-trip
        cleanly through json.loads() without a custom decoder.
        """
        return {
            "post_id":      self.post_id,
            "platform":     self.platform,
            "subreddit":    self.subreddit,
            "title":        self.title,
            "body":         self.body,
            "author":       self.author,
            "url":          self.url,
            "score":        self.score,
            "num_comments": self.num_comments,
            "created_utc":  self.created_utc.isoformat(),
            "fetched_at":   self.fetched_at.isoformat(),
            "flair":        self.flair,
            "top_comments": self.top_comments,
            "sentiment":    self.sentiment,
            "topics":       self.topics,
            "summary":      self.summary,
        }


# ---------------------------------------------------------------------------
# Reddit JSON API — primary scraping path
# ---------------------------------------------------------------------------

# Reddit requires a meaningful User-Agent; generic strings get throttled faster.
# Format follows Reddit's recommended convention: <name>/<version> (<contact>)
REDDIT_HEADERS = {
    "User-Agent": "doomscroll-agent/1.0 (research prototype; contact via GitHub)",
}

REDDIT_BASE = "https://www.reddit.com"


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
) -> dict:
    """
    GET `url` with `params`, returning the parsed JSON body.

    Retry policy (4 attempts total):
    - HTTP 429 (rate-limited): exponential back-off starting at 5 s
      Waits: 5 s → 10 s → 20 s → 40 s
    - Network errors (timeout, DNS fail, etc.): back-off starting at 1 s
      Waits: 1 s → 2 s → 4 s
    - Other HTTP 4xx/5xx: raises immediately on the final attempt

    Why exponential back-off instead of a fixed sleep?
    Reddit's API enforces ~60 req/min per user-agent. Sleeping progressively
    lets the quota window reset without hammering the server on every failure.
    """
    for attempt in range(4):
        try:
            response = await client.get(
                url,
                params=params,
                headers=REDDIT_HEADERS,
                timeout=20,     # 20 s covers slow international CDN edges
            )

            if response.status_code == 429:
                # Rate-limited — sleep longer on each successive hit
                wait_seconds = (2 ** attempt) * 5      # 5 → 10 → 20 → 40 s
                logger.warning(
                    "Reddit rate-limited (429); sleeping %ds before retry %d/4",
                    wait_seconds, attempt + 1,
                )
                await asyncio.sleep(wait_seconds)
                continue

            response.raise_for_status()     # raise HTTPStatusError on 4xx/5xx
            return response.json()

        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP %s fetching %s (attempt %d/4)",
                exc.response.status_code, url, attempt + 1,
            )
            if attempt == 3:
                raise   # exhausted retries — propagate to caller

        except httpx.RequestError as exc:
            # Network-level failure: DNS, connection refused, timeout, etc.
            logger.error("Network error on attempt %d/4: %s", attempt + 1, exc)
            if attempt == 3:
                raise

        # Back-off between non-429 retries: 1 s, 2 s, 4 s
        await asyncio.sleep(2 ** attempt)

    # Unreachable — all paths either return or raise — but satisfies type-checker
    return {}


async def _fetch_top_comments(
    client: httpx.AsyncClient,
    subreddit: str,
    post_id: str,
    limit: int = 5,
) -> list[str]:
    """
    Fetch the top `limit` first-level comment bodies for a single post.

    We request depth=1 to avoid pulling the full comment tree — we only need the
    highest-upvoted immediate replies, which carry the most community consensus.

    Returns an empty list on any failure (never raises) so a comment-fetch error
    doesn't abort the broader scrape batch.
    """
    url = f"{REDDIT_BASE}/r/{subreddit}/comments/{post_id}.json"
    try:
        data = await _fetch_json(
            client, url,
            {"limit": limit, "sort": "top", "depth": 1},
        )

        # The comments endpoint returns a 2-element list:
        #   data[0] = post metadata (same structure as the listing endpoint)
        #   data[1] = comment listing (what we want)
        if not isinstance(data, list) or len(data) < 2:
            return []

        comments = []
        for child in data[1].get("data", {}).get("children", []):
            kind = child.get("kind")    # "t1" = comment, "more" = collapsed thread
            body = child.get("data", {}).get("body", "")

            # Skip "more" items, deleted/removed comments
            if kind == "t1" and body and body not in ("[deleted]", "[removed]"):
                comments.append(body.strip())

            if len(comments) >= limit:
                break

        return comments

    except Exception as exc:
        # Non-fatal — keep the post but without comments
        logger.warning(
            "Could not fetch comments for post %s in r/%s: %s",
            post_id, subreddit, exc,
        )
        return []


async def scrape_subreddit_api(
    subreddit: str,
    sort: str = "hot",
    limit: int = 25,
    time_filter: str = "day",
    fetch_comments: bool = True,
) -> list[Post]:
    """
    Fetch posts from a subreddit using Reddit's public JSON endpoint.

    Parameters
    ----------
    subreddit      : subreddit name without the "r/" prefix
    sort           : listing sort — "hot" | "new" | "top" | "rising"
    limit          : number of posts to request (Reddit API hard cap: 100)
    time_filter    : time window for sort="top" only
    fetch_comments : whether to concurrently fetch top comments per post

    Returns
    -------
    list[Post] sorted by score descending (highest-engagement first).

    Concurrency note
    ----------------
    Comment fetches are issued concurrently via asyncio.gather(), so a batch of
    25 posts finishes in ~2 s rather than ~25 s if done sequentially. Results are
    returned in task-submission order so we can safely zip() them with raw_posts.
    """
    url = f"{REDDIT_BASE}/r/{subreddit}/{sort}.json"
    params: dict[str, Any] = {
        "limit":    min(limit, 100),    # enforce Reddit's per-request cap
        "raw_json": 1,                   # disable HTML-entity encoding in text fields
    }
    if sort == "top":
        params["t"] = time_filter       # "t" param is ignored for other sort modes

    posts: list[Post] = []

    # Re-use a single httpx client for connection pooling across all requests
    async with httpx.AsyncClient(follow_redirects=True) as client:
        logger.info(
            "Fetching r/%s [sort=%s] via Reddit JSON API …", subreddit, sort
        )
        data = await _fetch_json(client, url, params)

        # Reddit wraps the post listing in data → children
        children = data.get("data", {}).get("children", [])
        logger.info(
            "API returned %d raw post objects for r/%s", len(children), subreddit
        )

        # First pass: build Post objects and queue comment fetch tasks
        comment_tasks = []
        raw_posts: list[Post] = []

        for child in children:
            d = child.get("data", {})

            # Skip mod-stickied posts (rules, megathreads, weekly discussion threads)
            if d.get("stickied"):
                continue

            post_id = d.get("id", "")
            created = datetime.fromtimestamp(
                d.get("created_utc", 0), tz=timezone.utc
            )

            p = Post(
                post_id      = post_id,
                platform     = "reddit",
                subreddit    = subreddit,
                title        = d.get("title", "").strip(),
                # selftext is None for link posts — normalise to empty string
                body         = (d.get("selftext") or "").strip()[:2000],
                author       = d.get("author", "[deleted]"),
                url          = f"https://www.reddit.com{d.get('permalink', '')}",
                score        = d.get("score", 0),
                num_comments = d.get("num_comments", 0),
                created_utc  = created,
                flair        = d.get("link_flair_text"),   # None for un-flaired posts
            )
            raw_posts.append(p)

            # Queue a comment-fetch coroutine for this post if requested
            if fetch_comments and post_id:
                comment_tasks.append(
                    _fetch_top_comments(client, subreddit, post_id, limit=5)
                )

        # Second pass: fire all comment requests concurrently
        if comment_tasks:
            logger.info(
                "Fetching comments for %d posts concurrently …", len(comment_tasks)
            )
            # return_exceptions=True prevents one 429 from cancelling all other tasks
            results = await asyncio.gather(*comment_tasks, return_exceptions=True)

            # Zip results back onto their matching posts (guaranteed same order)
            for post, result in zip(raw_posts, results):
                if isinstance(result, list):
                    post.top_comments = result
                # Exception branches: _fetch_top_comments already logged; keep empty list

        posts = raw_posts

    # Sort highest-score first — downstream analysis focuses on most-engaged content
    posts.sort(key=lambda p: p.score, reverse=True)
    logger.info("Returning %d posts from r/%s", len(posts), subreddit)
    return posts


# ---------------------------------------------------------------------------
# Playwright headless browser — fallback scraping path
# ---------------------------------------------------------------------------

async def scrape_subreddit_playwright(
    subreddit: str,
    limit: int = 20,
) -> list[Post]:
    """
    Scrape a subreddit listing using a Playwright-controlled Chromium browser.

    This path is triggered when the JSON API is rate-limited, returns 0 results,
    or is otherwise unavailable. It's also the template for scraping platforms
    that don't offer a public structured API (e.g. TikTok, Instagram).

    Why old.reddit.com?
    - Server-rendered HTML — no JS hydration wait required
    - Much simpler DOM structure than the React-based new.reddit.com
    - Smaller page payload → faster load

    Trade-offs vs the API path:
    - No concurrent comment fetching (added complexity for a fallback code path)
    - Score parsing may break if Reddit redesigns old.reddit's markup
    - ~5 s wall-time vs ~1 s for the API
    """
    logger.info("Playwright fallback activated for r/%s", subreddit)
    url = f"https://old.reddit.com/r/{subreddit}/"
    posts: list[Post] = []

    async with async_playwright() as pw:
        # Launch headless Chromium — runs fine in Docker and CI without a display
        browser = await pw.chromium.launch(headless=True)

        # Set a realistic User-Agent + viewport to pass basic bot-detection checks
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        try:
            # Navigate and wait for the DOM to be usable (not necessarily all assets)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            # On old.reddit, each post is a <div class="thing link"> element
            await page.wait_for_selector("div.thing", timeout=15_000)
            entries = await page.query_selector_all("div.thing.link")
            logger.info(
                "Playwright found %d post elements in r/%s", len(entries), subreddit
            )

            for entry in entries[:limit]:
                try:
                    # data-fullname is the Reddit "type_id" string, e.g. "t3_abc123"
                    fullname = await entry.get_attribute("data-fullname") or ""
                    post_id  = fullname.replace("t3_", "")

                    # Title text and permalink are on the <a class="title"> element
                    title_el = await entry.query_selector("a.title")
                    title    = (await title_el.inner_text()).strip() if title_el else ""
                    href     = await title_el.get_attribute("href") if title_el else ""

                    # Score lives in div.score.unvoted before the user votes
                    score_el  = await entry.query_selector("div.score.unvoted")
                    score_txt = (await score_el.inner_text()).strip() if score_el else "0"
                    try:
                        # Reddit abbreviates large scores, e.g. "12.3k" → 12300
                        if "k" in score_txt.lower():
                            score = int(float(score_txt.lower().replace("k", "")) * 1000)
                        else:
                            score = int(score_txt)
                    except ValueError:
                        score = 0   # "•" is shown when score is hidden

                    # Timestamp is stored in a <time datetime="ISO-string"> element
                    time_el  = await entry.query_selector("time")
                    ts_str   = await time_el.get_attribute("datetime") if time_el else None
                    created  = (
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts_str
                        else datetime.now(timezone.utc)
                    )

                    # Author username in an <a class="author"> anchor
                    author_el = await entry.query_selector("a.author")
                    author    = (
                        (await author_el.inner_text()).strip() if author_el else "[deleted]"
                    )

                    # Make sure the URL is absolute (old Reddit uses relative permalinks)
                    link = (
                        href if href and href.startswith("http")
                        else f"https://www.reddit.com{href}"
                    )

                    posts.append(Post(
                        # Generate a fallback ID if data-fullname is missing
                        post_id      = post_id or f"pw_{int(time.time())}_{len(posts)}",
                        platform     = "reddit",
                        subreddit    = subreddit,
                        title        = title,
                        body         = "",          # body text not available in listing view
                        author       = author,
                        url          = link,
                        score        = score,
                        num_comments = 0,           # not reliably extractable from old.reddit DOM
                        created_utc  = created,
                    ))

                except Exception as exc:
                    # Skip malformed elements — don't let one bad post crash the batch
                    logger.debug("Skipping a Playwright post element: %s", exc)

        except Exception as exc:
            logger.error("Playwright scrape failed for r/%s: %s", subreddit, exc)

        finally:
            # Always close the browser to release the Chromium subprocess
            await browser.close()

    logger.info(
        "Playwright returned %d posts from r/%s", len(posts), subreddit
    )
    return posts


# ---------------------------------------------------------------------------
# Public entry point — auto-selects the best available strategy
# ---------------------------------------------------------------------------

async def scrape(
    subreddit: str,
    sort: str = "hot",
    limit: int = 25,
    time_filter: str = "day",
    fetch_comments: bool = True,
    use_playwright_fallback: bool = True,
) -> list[Post]:
    """
    Scrape posts from a subreddit, automatically choosing the best strategy.

    Strategy selection order:
      1. Reddit JSON API  (fast, structured, preferred)
      2. Playwright headless browser  (fallback if API fails or returns nothing)

    This is the only function that external modules (pipeline.py) should call.
    Internal helpers are prefixed with `_` or named `scrape_subreddit_*` to
    signal they're implementation details not intended for direct use.

    Parameters
    ----------
    subreddit               : subreddit name, no "r/" prefix (e.g. "wallstreetbets")
    sort                    : "hot" | "new" | "top" | "rising"
    limit                   : max posts to return
    time_filter             : for sort="top" only — "hour"|"day"|"week"|"month"|"year"|"all"
    fetch_comments          : whether to fetch top comments alongside each post
    use_playwright_fallback : set False in environments without a Chromium binary

    Returns
    -------
    list[Post], possibly empty if both strategies fail — never raises.
    """
    # Strategy 1: Reddit JSON API
    try:
        posts = await scrape_subreddit_api(
            subreddit       = subreddit,
            sort            = sort,
            limit           = limit,
            time_filter     = time_filter,
            fetch_comments  = fetch_comments,
        )
        if posts:
            return posts
        logger.warning(
            "API returned 0 posts for r/%s — triggering Playwright fallback", subreddit
        )
    except Exception as exc:
        logger.warning(
            "API scrape failed for r/%s (%s) — triggering Playwright fallback",
            subreddit, exc,
        )

    # Strategy 2: Playwright headless browser
    if use_playwright_fallback:
        return await scrape_subreddit_playwright(subreddit=subreddit, limit=limit)

    # Both strategies either failed or were disabled
    logger.error(
        "All scrape strategies failed for r/%s — returning empty list", subreddit
    )
    return []
