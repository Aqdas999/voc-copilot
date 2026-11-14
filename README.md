# Myntra Voice-of-Customer Copilot

An AI copilot that turns messy Myntra customer feedback — Play Store reviews, Reddit threads,
and clearly-labeled synthetic support tickets — into ranked pain-point themes with counts,
trends, verbatim source-linked quotes, churn signals, and RICE-prioritized roadmap recommendations.

**Focus areas:** M-Now (quick fashion delivery), returns/refunds, size & fit.

**Core principle:** every insight cites its sources. Quotes are exact substrings of the
original feedback — never paraphrased or invented.

## Pipeline

```
ingest (Play Store, Reddit, synthetic) -> clean -> extract (LLM) -> cluster -> insights -> Streamlit app
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

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then fill in your API keys
```

Each script runs standalone, e.g. `python -m src.extract`.

## Privacy

No usernames or personal data are stored. Synthetic tickets are always labeled `synthetic_ticket`.

_Status: work in progress._
