# 🕳️ Doomscroll Agent

> **G2 Take-Home — Social Knowledge "Doomscroll" Agent**  
> Binox 2026 Graduate Assessment

An AI agent that continuously monitors Reddit to extract market insights, trends, and sentiment — turning endless scrolling into structured, actionable intelligence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        n8n Scheduler                           │
│              (hourly cron → Execute Command node)              │
└───────────────────────────┬─────────────────────────────────────┘
                            │  python main.py
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      pipeline.py                               │
│              (orchestrates all steps)                          │
└────┬──────────────┬──────────────┬────────────────────────┬────┘
     │              │              │                        │
     ▼              ▼              ▼                        ▼
┌─────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐
│scraper  │  │analyzer  │  │   memory     │  │  insight output    │
│         │  │          │  │              │  │                    │
│Reddit   │  │Anthropic │  │ Layer 1:     │  │ corpus sentiment   │
│JSON API │  │Claude    │  │ JSONL log    │  │ dominant topics    │
│   +     │  │          │  │              │  │ actionable items   │
│Playwright│ │sentiment │  │ Layer 2:     │  │ trend delta vs     │
│fallback │  │topics    │  │ ChromaDB     │  │ prev snapshot      │
│         │  │summary   │  │ + MiniLM     │  │                    │
│         │  │insights  │  │ embeddings   │  │                    │
└─────────┘  └──────────┘  └──────────────┘  └────────────────────┘
```

**Data flow per run:**
1. Scrape N posts from each configured subreddit (concurrent)
2. Deduplicate against existing JSONL log
3. LLM-analyse new posts only (sentiment + topics + summary)
4. Store to both memory layers
5. Load last 100 posts → generate corpus-level actionable insights
6. Compare against previous insight snapshot → trend delta
7. Persist insight snapshot; print summary to stdout

---

## Quick Start

### Prerequisites
- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone & install

```bash
git clone https://github.com/Marc026/doomscroll-agent.git
cd doomscroll-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium         # headless browser for fallback scraping
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run a single pass

```bash
python main.py
```

Expected output:

```
2024-06-01 10:00:01  INFO      main — Pipeline run run_1717228801 starting
2024-06-01 10:00:01  INFO      agent.scraper — Fetching r/wallstreetbets [hot] via Reddit JSON API …
...
============================================================
ACTIONABLE INSIGHTS
============================================================

[HIGH] Retail traders are rotating out of tech into energy names ahead of Fed announcement
  Evidence: 15 posts in r/wallstreetbets mention oil/gas; sentiment shifted bearish on NVDA

[MEDIUM] Emerging consensus that AI chip supply constraints will persist through Q3
  Evidence: Top posts in r/technology and r/investing both reference TSMC yield issues

────────────────────────────────────────────────────────────
TREND DELTA (vs previous snapshot)
────────────────────────────────────────────────────────────
Overall sentiment improved from bearish to neutral; AI and semiconductors are emerging topics
  Emerging : AI chips, Fed policy
  Fading   : meme stocks, crypto
```

---

## CLI Reference

```bash
# Monitor specific subreddits with "top" posts of the day
python main.py --subreddits wallstreetbets stocks crypto --sort top --time-filter day

# Skip LLM analysis (just scrape and store — fastest, no API cost)
python main.py --skip-analysis

# Scheduled mode — runs every 30 minutes
python main.py --schedule --interval 1800

# Show memory statistics
python main.py --stats

# Semantic search against stored posts
python main.py --search "AI chip shortage supply chain"

# Verbose logging + disable Playwright fallback
python main.py -v --no-playwright
```

---

## Docker

```bash
# Build and run a single pass
docker build -t doomscroll-agent .
docker run --env-file .env doomscroll-agent

# Full stack: agent (scheduled) + n8n UI
docker-compose up -d

# n8n dashboard → http://localhost:5678  (admin / changeme)
# Import n8n/workflow.json via Settings → Import workflow
```

---

## n8n Workflow

The workflow in `n8n/workflow.json` provides:

| Node | Purpose |
|---|---|
| Hourly Cron | Triggers the pipeline every 60 minutes |
| Run Pipeline | `Execute Command` → `python main.py ...` |
| Check for Error | Routes on non-zero exit code |
| Get Memory Stats | Runs `--stats` on success |
| Slack Success Summary | Posts truncated stdout to a Slack channel |
| Slack Error Alert | Posts stderr on failure |
| Semantic Search (disabled) | Manual trigger for ad-hoc queries |

**To import:** n8n UI → Settings → Import workflow → paste `n8n/workflow.json`.  
Set the `SLACK_WEBHOOK_URL` environment variable in docker-compose or n8n credentials.

---

## Project Structure

```
doomscroll-agent/
├── agent/
│   ├── scraper.py      # Reddit JSON API + Playwright fallback
│   ├── analyzer.py     # Anthropic Claude: sentiment / topics / insights
│   ├── memory.py       # ChromaDB vector store + JSONL log
│   └── pipeline.py     # Orchestrator: ties all steps together
├── n8n/
│   └── workflow.json   # n8n workflow for scheduling
├── tests/
│   ├── test_scraper.py
│   ├── test_analyzer.py
│   └── test_memory.py
├── data/               # Created at runtime (gitignored)
│   ├── posts_log.jsonl
│   ├── insights_log.jsonl
│   └── chroma/
├── main.py             # CLI entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── self_assessment.md
└── .env.example
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests mock external dependencies (Anthropic API, network) — no API key needed.

---

## Memory Architecture

| Layer | Technology | Purpose | Survives restart? |
|---|---|---|---|
| JSONL log | Plain file | Full post history, human-readable, replayable | ✅ |
| Vector store | ChromaDB + MiniLM | Semantic search, similarity, dedup at scale | ✅ |
| Insight log | Plain file | Timestamped corpus-level snapshots | ✅ |

Both layers use `DATA_DIR` (default: `./data`). Mount as a Docker volume for persistence.

---

## Cost Optimisation

| Model | Cost/run (75 posts) | Use case |
|---|---|---|
| `claude-haiku-4-5` | ~$0.01 | Production / high-frequency |
| `claude-sonnet-4-6` | ~$0.15 | Balanced quality + cost |
| `claude-opus-4-6` | ~$0.70 | Deep analysis / prototyping |

Change the `MODEL` constant in `agent/analyzer.py`. Use `--skip-analysis` to pay $0 during scrape-only runs.

---

## Demonstrated Success Criteria

| Criterion | How it's met |
|---|---|
| Fetch and parse content from 1 platform | Reddit JSON API + Playwright fallback in `scraper.py` |
| Store insights with timestamps and basic categorization | JSONL log (every post has `created_utc`, `fetched_at`, `sentiment`, `topics`) + ChromaDB |
| Demonstrate one actionable insight from aggregated data | `generate_insights()` in `analyzer.py` returns 2-4 urgency-ranked insights per run |

---

## License

MIT
