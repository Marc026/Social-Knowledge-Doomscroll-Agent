# Self-Assessment

## What I built
A production-ready social knowledge extraction pipeline that monitors Reddit to surface market and technology trends. The agent scrapes posts, runs LLM-powered analysis (sentiment, topic extraction, summarisation), stores results in a dual-layer memory system (ChromaDB vector store + append-only JSONL log), generates actionable insights from the aggregated corpus, and compares successive snapshots to detect trend deltas.

---

## What went well

**Architecture clarity** — Separating concerns into `scraper → analyzer → memory → pipeline` made each module independently testable and replaceable. Swapping Reddit for another platform (e.g. a Hacker News API or Twitter/X scraper) requires changing only `scraper.py`.

**Dual-memory design** — The JSONL log gives human-readable, replayable history with zero dependencies; ChromaDB adds semantic search without needing a cloud service. Either layer degrades gracefully if the dependency is missing.

**Scraping resilience** — Exponential-backoff retry on the Reddit JSON API, with Playwright as a headless-browser fallback, means the agent handles rate limits and DOM-rendered pages without manual intervention.

**Cost awareness** — Analysis only runs on *new* posts (deduplication gate before LLM call). `--skip-analysis` and model selection (haiku vs. sonnet) are documented escape hatches for cost control.

---

## Trade-offs made

| Decision | Why | What I gave up |
|---|---|---|
| Reddit as platform | Public JSON API — no OAuth, no scraping arms-race | TikTok/Instagram would be higher-signal for consumer trends but require complex bot evasion |
| `all-MiniLM-L6-v2` for embeddings | Runs locally, zero cost, 384-dim vectors are fast | A larger model (e.g. `text-embedding-3-large`) would give better semantic precision |
| Sequential LLM analysis | Avoids hitting Anthropic rate limits | Slower than batched/parallel calls — acceptable for 20–50 posts/run |
| JSONL over a full DB | Zero-dependency, trivially inspectable | No efficient indexed queries by date range or subreddit without loading the full file |
| Single-file ChromaDB | Portable, no server | Doesn't scale beyond ~500K embeddings on a single machine |

---

## What I would add with more time

1. **Multi-platform support** — Add `scraper_hn.py` (Hacker News Algolia API) and `scraper_twitter.py` (Nitter mirrors or official API) behind the same `scrape()` interface.
2. **Automatic topic clustering** — Use HDBSCAN on ChromaDB embeddings to surface emergent themes without hand-crafted prompts.
3. **Alerting thresholds** — Let users define rules like "alert when bearish sentiment > 60% in r/investing for 3 consecutive runs" and emit to Slack/email.
4. **Dashboard** — A small Streamlit/Gradio front-end showing the JSONL log, insight history, and a live semantic-search box.
5. **Evaluation harness** — Annotate a gold set of 50 posts with human sentiment labels and run against the LLM classifier to measure accuracy and calibrate confidence thresholds.

---

## Business impact reasoning

**Use case**: A fund manager or startup founder monitoring Reddit for early signals in their sector (e.g. r/wallstreetbets for retail sentiment, r/technology for product reception) currently does this manually — scrolling and summarising by hand.

**Value delivered**:
- Continuous monitoring without human fatigue
- Structured, timestamped sentiment history enables backtesting ("did high bearish sentiment in r/investing predict SPY drawdown?")
- Actionable insights with urgency ratings let an analyst triage what actually needs attention

**Cost estimate (Anthropic API)**:
- 25 posts × 3 subreddits = 75 posts/run
- ~600 tokens per post analysis + ~1,500 tokens for corpus insight = ~47K tokens/run
- At `claude-haiku-4-5` pricing (~$0.25/M input): ≈ $0.012 per run / ≈ $0.29/day at hourly schedule
- At `claude-opus-4-5` (default in this repo): ≈ $0.70/run — swap to haiku for production economics
