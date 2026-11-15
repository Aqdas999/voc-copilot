"""Use the LLM to extract a structured pain point from every feedback item.

For each item we ask for:
  id, pain_point (short free text), theme (exactly one name from themes.md),
  sentiment, severity (1-3, null for "no complaint"), churn_signal (Y/N),
  evidence_span (an EXACT substring of the original text — our citation).

How it works:
  1. Load data/processed/feedback.csv (or only the golden-set ids with --golden).
  2. Skip items already in the cache for this prompt version (reruns are free).
  3. Send the rest in batches of ~20 to call_llm(), parse the JSON reply,
     retry a batch if the reply is broken or missing items.
  4. Append each good result to the cache, then write the output file.

The prompt is versioned (PROMPT_V1, PROMPT_V2...). Changing the prompt = new version,
so cached results and eval scores are always tied to the exact prompt that made them.

Run:
  python -m src.extract --golden --version v1     # 100 golden items only
  python -m src.extract --version v1              # all items
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src import llm

FEEDBACK_PATH = Path("data/processed/feedback.csv")
GOLDEN_PATH = Path("data/golden/golden_set.csv")
THEMES_PATH = Path("data/golden/themes.md")
CACHE_DIR = Path("data/processed/cache")
OUTPUT_DIR = Path("data/processed")

BATCH_SIZE = 20
MAX_BATCH_ATTEMPTS = 3   # re-ask if the JSON is broken or items are missing


# ---------------------------------------------------------------------------
# Themes: read from themes.md so the prompt and the human labels share one list
# ---------------------------------------------------------------------------
def load_themes() -> list[tuple[str, str]]:
    """Return [(theme name, description)] from the table in themes.md, plus the 2 special labels."""
    themes = []
    for line in THEMES_PATH.read_text(encoding="utf-8").splitlines():
        # table rows look like: | 1 | `delivery delay` | Late, stuck in transit... | ~246 (25%) |
        match = re.match(r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\|", line)
        if match:
            description = match.group(2).replace("**", "").strip()
            themes.append((match.group(1), description))
    themes.append(("no complaint", "Praise or neutral; no problem described"))
    themes.append(("other", "A real complaint that fits none of the themes above"))
    return themes


THEMES = load_themes()
THEME_NAMES = {name for name, _ in THEMES}


# ---------------------------------------------------------------------------
# Prompts — never edit a released version; add PROMPT_V2 instead.
# ---------------------------------------------------------------------------
PROMPT_V1 = """You analyse customer feedback about Myntra, an Indian fashion shopping app.
Feedback may be in English, Hindi, or Hinglish (Hindi written in English letters).

For EACH item below, return one JSON object with these fields:
- "id": copy the item's id exactly.
- "pain_point": the main problem in under 12 words of plain English ("" if no complaint).
- "theme": exactly ONE name from this list — the MAIN thing the customer is upset about:
{theme_list}
  If customer support is mentioned alongside another problem, pick the other problem.
- "sentiment": "positive", "neutral", "negative", or "mixed".
- "severity": 1, 2 or 3 (null if theme is "no complaint"):
    1 = minor annoyance, no money or order lost
    2 = real problem with an order (late delivery, wrong size, return hassle)
    3 = money lost or stuck, repeated failures, fraud/fake claims, or churn signal is "Y"
- "churn_signal": "Y" only if the customer says or clearly implies they will stop using
  Myntra, switch to a competitor, or tells others not to use it. Plain anger alone is "N".
- "evidence_span": the shortest phrase (3-20 words) that proves the theme, COPIED
  CHARACTER-FOR-CHARACTER from the item's text: same spelling, typos, capitals, emoji,
  language and script. Do not translate, fix, shorten with "...", or cross a line break.

Return ONLY a JSON array of {n} objects, one per item, no other text.

Items:
{items_json}
"""

PROMPTS = {"v1": PROMPT_V1}


def build_prompt(version: str, batch: pd.DataFrame) -> str:
    theme_list = "\n".join(f'    - "{name}": {desc}' for name, desc in THEMES)
    items = [{"id": row.id, "text": row.text} for row in batch.itertuples()]
    return PROMPTS[version].format(
        theme_list=theme_list,
        n=len(items),
        items_json=json.dumps(items, ensure_ascii=False, indent=1),
    )


# ---------------------------------------------------------------------------
# Parsing the model's reply
# ---------------------------------------------------------------------------
def parse_reply(reply: str) -> list[dict]:
    """Pull the JSON array out of the reply (models sometimes wrap it in ```json fences)."""
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in reply")
    data = json.loads(reply[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("reply is not a list")
    return data


def normalise(record: dict) -> dict:
    """Coerce fields into the expected types. Nothing here edits evidence_span."""
    severity = record.get("severity")
    try:
        severity = int(severity) if severity not in (None, "", "null") else None
    except (TypeError, ValueError):
        severity = None
    churn = str(record.get("churn_signal", "N")).strip().upper()[:1]
    return {
        "id": str(record.get("id", "")).strip(),
        "pain_point": str(record.get("pain_point") or "").strip(),
        "theme": str(record.get("theme") or "").strip(),
        "sentiment": str(record.get("sentiment") or "").strip().lower(),
        "severity": severity,
        "churn_signal": "Y" if churn == "Y" else "N",
        "evidence_span": record.get("evidence_span") or "",
    }


# ---------------------------------------------------------------------------
# Cache: one JSONL file per prompt version, keyed by id
# ---------------------------------------------------------------------------
def cache_path(version: str) -> Path:
    return CACHE_DIR / f"extract_{version}.jsonl"


def load_cache(version: str) -> dict[str, dict]:
    path = cache_path(version)
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r for r in records}


def append_cache(version: str, records: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path(version).open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def extract_batch(version: str, batch: pd.DataFrame) -> list[dict]:
    """Send one batch; retry if the reply is unusable or missing ids. Returns good records."""
    wanted = set(batch["id"])
    got: dict[str, dict] = {}
    for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
        todo = batch[~batch["id"].isin(got)]          # only re-ask for what's still missing
        try:
            reply = llm.call_llm(build_prompt(version, todo))
            for raw in parse_reply(reply):
                rec = normalise(raw)
                if rec["id"] in wanted:
                    rec["prompt_version"] = version
                    rec["model"] = llm.LAST_MODEL
                    got[rec["id"]] = rec
        except (ValueError, json.JSONDecodeError, RuntimeError) as error:
            print(f"    attempt {attempt}: {type(error).__name__}: {str(error)[:120]}")
        if wanted <= set(got):
            break
        print(f"    attempt {attempt}: {len(wanted - set(got))} item(s) missing, retrying")
    return list(got.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="LLM pain-point extraction")
    parser.add_argument("--version", default="v1", choices=sorted(PROMPTS))
    parser.add_argument("--golden", action="store_true", help="only the golden-set ids")
    args = parser.parse_args()

    df = pd.read_csv(FEEDBACK_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.golden:
        golden_ids = pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False,
                                 encoding="utf-8-sig")["id"]
        df = df[df["id"].isin(golden_ids)]

    cache = load_cache(args.version)
    todo = df[~df["id"].isin(cache)]
    print(f"Prompt {args.version}: {len(df)} items, {len(df) - len(todo)} cached, "
          f"{len(todo)} to extract")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[start:start + BATCH_SIZE]
        print(f"  batch {start // BATCH_SIZE + 1}/{-(-len(todo) // BATCH_SIZE)} "
              f"({len(batch)} items)")
        records = extract_batch(args.version, batch)
        append_cache(args.version, records)       # saved immediately: a crash loses one batch at most
        cache.update({r["id"]: r for r in records})

    # Write the output for exactly the requested ids, in feedback order.
    results = [cache[i] for i in df["id"] if i in cache]
    suffix = "_golden" if args.golden else ""
    out_path = OUTPUT_DIR / f"extractions_{args.version}{suffix}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    missing = len(df) - len(results)
    bad_themes = sum(r["theme"] not in THEME_NAMES for r in results)
    print(f"Saved {len(results)} extractions to {out_path}"
          + (f" | {missing} still missing (rerun to retry)" if missing else "")
          + (f" | {bad_themes} with a theme not in themes.md" if bad_themes else ""))


if __name__ == "__main__":
    main()
