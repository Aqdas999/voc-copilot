"""Scrape Myntra's iPhone App Store reviews (India) from Apple's public RSS feed.

What it does:
  1. Reads Apple's customer-reviews feed for the India store, newest first.
  2. Pages through every available page (Apple serves at most 10 pages x 50 = ~500).
  3. Saves data/raw/appstore.csv in the project's unified schema, IDs AS-0001 onward.

No API key needed: the feed is public. Uses only Python's standard library to fetch it.

Privacy: the feed includes the reviewer's nickname — we never save it.

Run:  python -m src.ingest.scrape_appstore
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

APP_ID = "907394059"   # confirmed: "Myntra - Fashion Shopping App" by Myntra Designs Pvt Ltd
COUNTRY = "in"         # India store
MAX_PAGES = 10         # Apple's limit for this feed
PAUSE_SECONDS = 2      # polite pause between pages

FEED_URL = ("https://itunes.apple.com/{country}/rss/customerreviews/"
            "page={page}/id={app_id}/sortby=mostrecent/json")
APP_URL = f"https://apps.apple.com/{COUNTRY}/app/id{APP_ID}?see-all=reviews"
USER_AGENT = "voc-copilot-research/0.1 (educational customer-feedback project)"

OUTPUT_PATH = Path("data/raw/appstore.csv")


def fetch_page(page: int) -> list[dict]:
    """Return the review entries on one feed page ([] when there are no more)."""
    url = FEED_URL.format(country=COUNTRY, page=page, app_id=APP_ID)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError) as error:
        # Past the last page Apple returns an error or non-JSON body: treat as "no more".
        print(f"  page {page}: no data ({type(error).__name__}), stopping")
        return []

    entries = data.get("feed", {}).get("entry", [])
    # With a single review Apple returns a dict instead of a list.
    return entries if isinstance(entries, list) else [entries]


def to_row(entry: dict) -> dict:
    """Keep only what we need. The 'author' field (nickname) is dropped here."""
    title = entry["title"]["label"].strip()
    body = entry["content"]["label"].strip()
    return {
        "review_id": entry["id"]["label"],
        "date": entry["updated"]["label"][:10],           # '2026-11-13T11:19:56-07:00' -> ISO date
        "rating": int(entry["im:rating"]["label"]),
        # Same convention as Reddit: title + body, both exactly as written.
        "text": f"{title}\n\n{body}" if title else body,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    rows = []
    for page in range(1, MAX_PAGES + 1):
        entries = fetch_page(page)
        if not entries:
            break
        rows += [to_row(e) for e in entries]
        print(f"  page {page}: {len(entries)} reviews")
        time.sleep(PAUSE_SECONDS)

    df = pd.DataFrame(rows).drop_duplicates(subset="review_id")
    df = df.sort_values("date", ascending=False, kind="stable").reset_index(drop=True)

    # Unified schema from CLAUDE.md
    df["id"] = [f"AS-{i:04d}" for i in range(1, len(df) + 1)]
    df["source"] = "appstore"
    # Apple has no per-review link; this opens the app's review page in the India store.
    df["url"] = APP_URL
    df["language"] = ""   # detected later in src/clean.py
    df = df[["id", "source", "date", "rating", "text", "url", "language"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} App Store reviews to {OUTPUT_PATH} "
          f"({df['date'].min()} to {df['date'].max()})")


if __name__ == "__main__":
    main()
