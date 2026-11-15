# Myntra Voice-of-Customer Copilot

An AI copilot that turns messy Myntra customer feedback — Play Store reviews, Reddit threads,
and clearly-labeled synthetic support tickets — into ranked pain-point themes with counts,
trends, verbatim source-linked quotes, churn signals, and RICE-prioritized roadmap recommendations.

**Focus areas:** M-Now (quick fashion delivery), returns/refunds, size & fit.

**Core principle:** every insight cites its sources. Quotes are exact substrings of the
original feedback — never paraphrased or invented.

## Pipeline

```
ingest (Play Store, synthetic, Reddit-optional) -> clean -> extract (LLM) -> cluster -> insights -> Streamlit app
                                                        \-> evals (citation validity, quote fidelity, churn P/R)
```

| Step | Script | Output |
|------|--------|--------|
| Ingest | `src/ingest/*.py` | `data/raw/*.csv` |
| Clean & merge | `src/clean.py` | `data/processed/feedback.csv` |
| Extract | `src/extract.py` | `data/processed/extractions.jsonl` |
| Cluster | `src/cluster.py` | themes with counts + trends |
| Insights | `src/insights.py` | quotes + RICE roadmap |
| Evaluate | `evals/run_evals.py` | `evals/results/<prompt_version>.json` |

### Data sources

| Source | Script | Items | Status |
|--------|--------|-------|--------|
| Play Store (India) | `scrape_playstore.py` | 2,000 (1,600 newest English + 400 Hindi-tagged, last 12 months) | ✅ collected |
| Synthetic support tickets | `synthetic_tickets.py` | 20 hand-written, labeled `SYN-` | ✅ generated |
| Reddit | `scrape_reddit.py` | ~150 target | ⏸️ skipped for now |

**Why Reddit is skipped:** Reddit blocks logged-out scripts from its public JSON endpoints
(HTTP 403, tested Nov 2026), and the official API needs an approved app key. The PRAW-based
scraper is ready; add `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` to `.env` to enable it.
The rest of the pipeline works without it.

## LLM providers (zero budget)

The project runs entirely on **free-tier** LLM APIs. Every LLM call goes through one
function, `call_llm(prompt)` in `src/llm.py`:

- **Primary:** [Groq](https://console.groq.com) (`openai/gpt-oss-120b`, an open-weight model) via its OpenAI-compatible API.
- **Backup:** [Google Gemini](https://aistudio.google.com) Flash, used automatically if Groq fails.
- Switch the primary provider with `LLM_PROVIDER=groq|gemini` in `.env`.
- 429 rate-limit errors are retried with exponential backoff (2s, 4s, 8s…), and results
  are cached to disk so reruns don't use up free quota.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then add your free GROQ_API_KEY and GEMINI_API_KEY
python -m src.llm               # smoke test: should print "OK"
```

Each script runs standalone, e.g. `python -m src.extract`.

## Privacy

No usernames or personal data are stored. Synthetic tickets are always labeled `synthetic_ticket`.

_Status: work in progress._
