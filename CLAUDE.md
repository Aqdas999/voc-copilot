# Myntra Voice-of-Customer Copilot

## What this is
An AI copilot that turns messy Myntra customer feedback (Play Store reviews, Reddit threads,
synthetic support tickets) into ranked pain-point themes with counts, trends, verbatim
source-linked quotes, churn signals, and RICE-prioritized roadmap recommendations.
Focus areas: M-Now (quick fashion delivery), returns/refunds, size & fit.

This is a PM portfolio project. The owner must be able to explain every file in an
interview, so keep code simple, readable, and commented. Prefer clarity over cleverness.

## Core principle
Every insight must cite its sources. No claim without evidence. Quotes must be exact
substrings of the source text — never paraphrased or invented.

## Folder structure
```
voc-copilot/
  CLAUDE.md
  README.md
  requirements.txt
  .env.example          # ANTHROPIC_API_KEY=, REDDIT_CLIENT_ID=, REDDIT_CLIENT_SECRET=
  .gitignore            # must include .env, __pycache__/, .venv/
  data/
    raw/                # scraper output, one CSV per source
    processed/          # cleaned + merged feedback, extractions, themes
    golden/             # hand-labeled eval set (golden_set.csv)
  src/
    ingest/
      scrape_playstore.py
      scrape_reddit.py
      synthetic_tickets.py
    clean.py            # merge sources, dedupe, drop junk, detect language
    extract.py          # LLM per-item extraction -> JSON
    cluster.py          # group into themes, name them, count, trend
    insights.py         # insights + quotes + RICE roadmap recs
  evals/
    rubric.md
    run_evals.py        # citation validity, quote fidelity, churn P/R, LLM-judge
    results/            # one JSON per prompt version (v1, v2, v3...)
  app/
    streamlit_app.py
  .github/workflows/
    daily.yml           # stretch: daily refresh
```

## Unified feedback schema (data/processed/feedback.csv)
| column   | notes |
|----------|-------|
| id       | PS-0001 (Play Store), RD-0001 (Reddit), SYN-0001 (synthetic ticket) |
| source   | playstore / reddit / synthetic_ticket |
| date     | ISO date |
| rating   | 1-5 for Play Store, empty otherwise |
| text     | original text, unmodified (Hinglish kept as-is) |
| url      | link to source where available |
| language | en / hinglish / hi / other |

Do NOT store usernames or personal data. Synthetic tickets must always be labeled synthetic.

## Extraction output (data/processed/extractions.jsonl), one per feedback item
```json
{"id": "PS-0042", "pain_point": "...", "theme_hint": "refund delay",
 "sentiment": "negative", "severity": 1-3, "churn_signal": true,
 "evidence_span": "exact substring of the original text"}
```

## Conventions
- Python 3.10+, pandas, anthropic SDK, streamlit. Keep dependencies minimal.
- API key read from .env via python-dotenv. Never hardcode or print keys.
- Batch LLM calls (about 20 items per call). Cache results to disk so reruns are cheap.
- Version prompts: store them as constants named PROMPT_V1, PROMPT_V2... and record
  which version produced each eval result.
- Every script runnable standalone: `python -m src.extract` etc.
- After writing a file, briefly explain what it does in plain language.
- Commit after each working step with a clear message.
