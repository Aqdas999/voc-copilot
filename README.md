# Myntra Voice-of-Customer Copilot

An AI copilot that turns messy Myntra customer feedback — Play Store reviews, Reddit threads,
and clearly-labeled synthetic support tickets — into ranked pain-point themes with counts,
trends, verbatim source-linked quotes, churn signals, and RICE-prioritized roadmap recommendations.

**Focus areas:** M-Now (quick fashion delivery), returns/refunds, size & fit.

**Core principle:** every insight cites its sources. Quotes are exact substrings of the
original feedback — never paraphrased or invented.

## Pipeline

```
ingest (Play Store, App Store, synthetic) -> clean -> extract (LLM) -> cluster -> insights -> Streamlit app
                                                        \-> evals (citation validity, quote fidelity, churn P/R)
```

| Step | Script | Output |
|------|--------|--------|
| Ingest | `src/ingest/*.py` | `data/raw/*.csv` |
| Clean & merge | `src/clean.py` | `data/processed/feedback.csv` |
| Extract | `src/extract.py` | `data/processed/extractions_v3_complaints.jsonl` (1–3★ + SYN) |
| Cluster | `src/cluster.py` | `theme_stats_v3_complaints.csv`, `theme_weekly_v3_complaints.csv` |
| Insights | `src/insights.py` | `data/processed/insights.json` (sub-issues, judged quotes, one sentence per theme) |
| Evaluate | `evals/run_evals.py`, `src/insights.py` | `evals/results/<prompt_version>.json`, `evals/results/insights_v2.json` |

### Evaluation

The eval rubric is maintained in the project PRD, not in this repo. What the code checks:

- **Extractions** (`evals/run_evals.py`, against the human-labelled golden and holdout sets):
  theme agreement, churn precision/recall, severity within ±1, quote fidelity.
- **Insights** (`src/insights.py`): count accuracy (every number in a sentence is recomputed
  from the data), citation validity (every cited ID exists, belongs to the theme, and its quote
  is an exact substring), a quote-relevance judge (quotes scoring below 4/5 or mainly about
  another theme are dropped), and an LLM judge scoring each insight 1–5 for specificity,
  grounding and actionability.
- **Refund-delay cross-check** (`src/refund_crosscheck.py`, no LLM): regex keyword counts
  (cancel / not delivered, return / pickup, "credited" but not received) saved to
  `data/processed/refund_crosscheck.csv` and shown on the dashboard as "keyword count" next to
  the LLM sub-issues. Counts overlap and are a sanity check, not a classification.

### Data sources

| Source | Script | Raw items | After cleaning | IDs |
|--------|--------|-----------|----------------|-----|
| Play Store (India) | `scrape_playstore.py` | 5,000 (4,600 newest English + 400 Hindi-tagged, last 12 months) | 2,599 | `PS-` |
| App Store (India) | `scrape_appstore.py` | 500 (Apple's public RSS feed maximum) | 420 | `AS-` |
| Synthetic support tickets | `synthetic_tickets.py` | 20 hand-written | 20 | `SYN-` |
| Reddit | `scrape_reddit.py` | — | — | `RD-` (skipped) |
| **Total** | | **5,520** | **3,039** | |

Snapshot collected 15 Nov 2026. Cleaning drops items under 3 words and duplicates.

## Limitations

- **No Reddit data.** Reddit blocks logged-out scripts from its public JSON endpoints
  (reddit.com returns HTTP 403; old.reddit.com redirects to login — both retested Nov 2026),
  and the official API needs an app key. The PRAW-based scraper is ready: add
  `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` to `.env` to enable it.
- **Short time window for English reviews.** Myntra gets ~400 English Play Store reviews a
  day, so the newest 5,000 cover only ~2 weeks (2–14 Nov 2026); App Store covers ~1 week.
  Trends are week-over-week at best. Hindi reviews span a year because they are rare.
- **Only 1–3★ reviews (plus synthetic tickets) were run through the LLM.** Groq's free tier
  allows only a limited number of tokens per day, so the 4–5★ Play Store and App Store reviews were not
  extracted. Complaints hidden inside high-star reviews (e.g. "love the app, but my refund
  took 3 weeks", 4★) are therefore not counted. All theme shares are
  **% of complaints (1–3★)**, not % of all reviews.
- **Not a representative sample of all users.** Short reviews ("good app") are mostly praise
  and are dropped, so the cleaned set skews negative (Play Store mean 3.60 vs. Myntra's
  public 4.59). Counts describe *what people complain about*, not overall satisfaction.
- **App Store reviews have no per-review link.** Apple's feed gives none, so their `url`
  points to the app's review page.
- **Synthetic tickets are fake** (20, hand-written) and always labeled `synthetic_ticket`.
- **Reviews can name individuals** (e.g. staff). Text is kept unmodified for quote fidelity;
  names should be masked in anything displayed.

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
