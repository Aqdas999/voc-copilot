"""Scrape the newest Myntra reviews from the Google Play Store (India).

What it does:
  1. Fetches the newest reviews written in English and in Hindi, from India.
  2. Keeps every Hindi review from the last 12 months (capped at 400), then fills the
     rest of the ~2,000 with the newest English reviews.
  3. Saves data/raw/playstore.csv in the project's unified schema.

Why the Hindi window: English reviews arrive at ~2,000/week but Hindi ones are rare
(~400/year). Taking "the 2,000 newest overall" would leave almost no Hindi at all,
so Hindi gets a longer time window to stay represented.

Privacy: we never keep usernames, profile pictures, or anything else about the reviewer.
Only the review text, star rating, date, and a link to the review are saved.

Run:  python -m src.ingest.scrape_playstore
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, app, reviews

APP_ID = "com.myntra.android"   # confirmed: "Myntra - Fashion Shopping App" by Myntra
COUNTRY = "in"                  # India
TARGET_TOTAL = 2000             # how many reviews to keep in the end
HINDI_WINDOW_DAYS = 365         # keep Hindi reviews from the last 12 months...
HINDI_MAX = 400                 # ...but at most this many

OUTPUT_PATH = Path("data/raw/playstore.csv")


def fetch_newest(lang: str, count: int) -> list[dict]:
    """Return up to `count` newest reviews for one language.

    The library pages through results for us (about 100-200 per request).
    """
    print(f"Fetching up to {count} newest '{lang}' reviews from Play Store ({COUNTRY})...")
    results, _next_page_token = reviews(
        APP_ID,
        lang=lang,
        country=COUNTRY,
        sort=Sort.NEWEST,
        count=count,
    )
    print(f"  got {len(results)}")
    return results


def to_rows(raw_reviews: list[dict], lang: str) -> list[dict]:
    """Keep ONLY the fields we need. Usernames and avatars are dropped here."""
    rows = []
    for r in raw_reviews:
        rows.append({
            "review_id": r["reviewId"],    # used only for dedupe + the link, then dropped
            "date": r["at"],
            "rating": r["score"],
            "text": r["content"],          # original text, unmodified
            # Play's 'lang' filter is only a first guess (Hinglish often shows up under 'en').
            # src/clean.py re-detects language properly later.
            "language": lang,
        })
    return rows


def main() -> None:
    # Hindi text crashes the default Windows console encoding, so print as UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Confirm we're scraping the right app before collecting anything.
    info = app(APP_ID, lang="en", country=COUNTRY)
    print(f"App: {info['title']} by {info['developer']} ({info['installs']} installs)")

    # 2. Hindi: newest reviews, keep only the last 12 months, cap at HINDI_MAX.
    hindi = pd.DataFrame(to_rows(fetch_newest("hi", TARGET_TOTAL), "hi"))
    cutoff = datetime.now() - timedelta(days=HINDI_WINDOW_DAYS)
    hindi = hindi[hindi["date"] >= cutoff].sort_values("date", ascending=False).head(HINDI_MAX)
    print(f"  kept {len(hindi)} Hindi reviews from the last {HINDI_WINDOW_DAYS} days")
    time.sleep(2)  # be polite to Google between requests

    # 3. English: newest reviews, skipping any already in the Hindi set,
    #    just enough to bring the total to TARGET_TOTAL.
    english = pd.DataFrame(to_rows(fetch_newest("en", TARGET_TOTAL), "en"))
    english = english[~english["review_id"].isin(hindi["review_id"])]
    english = english.sort_values("date", ascending=False).head(TARGET_TOTAL - len(hindi))
    print(f"  kept {len(english)} newest English reviews")

    # 4. Combine, newest first.
    df = pd.concat([english, hindi]).drop_duplicates(subset="review_id")
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # 5. Build the unified schema from CLAUDE.md.
    df["id"] = [f"PS-{i:04d}" for i in range(1, len(df) + 1)]
    df["source"] = "playstore"
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)   # ISO date, e.g. 2026-11-12
    df["url"] = (f"https://play.google.com/store/apps/details?id={APP_ID}&reviewId="
                 + df["review_id"])
    df = df[["id", "source", "date", "rating", "text", "url", "language"]]

    # 6. Save. utf-8-sig so Hindi text also displays correctly if opened in Excel.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} reviews to {OUTPUT_PATH} "
          f"({df['date'].min()} to {df['date'].max()})")


if __name__ == "__main__":
    main()
