"""Collect ~150 Reddit posts and comments about Myntra pain points.

Why PRAW (Reddit's official API) and not the public .json URLs:
  Reddit now blocks logged-out scripts from its public JSON endpoints (HTTP 403 /
  redirect to login, tested Nov 2026). The official API with a free app key works.

What it does:
  1. Searches r/india, r/bangalore, r/mumbai, r/delhi and all of Reddit for
     Myntra + (return / refund / M-Now / size / delivery).
  2. Keeps posts that actually mention Myntra, plus their comments that mention
     Myntra or one of those topics.
  3. Saves data/raw/reddit.csv in the project's unified schema, IDs RD-0001 onward.

Privacy: Reddit usernames are never read or saved.

Setup: create a free "script" app at https://www.reddit.com/prefs/apps and put
REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.

Run:  python -m src.ingest.scrape_reddit
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()

SUBREDDITS = ["india", "bangalore", "mumbai", "delhi", "all"]  # "all" = site-wide search
QUERIES = [
    "myntra return",
    "myntra refund",
    'myntra "m-now" OR myntra mnow',
    "myntra size",
    "myntra delivery",
]
TOPIC_WORDS = ["return", "refund", "m-now", "mnow", "m now", "size", "fit", "delivery", "deliver"]

TARGET_TOTAL = 150          # posts + comments to keep
POSTS_PER_SEARCH = 25       # results per (subreddit, query) search
COMMENTS_PER_POST = 5       # at most this many matching comments per post
PAUSE_SECONDS = 1.0         # polite pause between API requests (PRAW also rate-limits itself)

# Generic, honest user-agent — no personal details.
USER_AGENT = "voc-copilot-research/0.1 (educational customer-feedback project)"

OUTPUT_PATH = Path("data/raw/reddit.csv")


def connect() -> praw.Reddit:
    """Log in to the Reddit API in read-only mode using keys from .env."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        sys.exit("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET missing in .env. "
                 "Create a free 'script' app at https://www.reddit.com/prefs/apps")
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                         user_agent=USER_AGENT)
    reddit.read_only = True
    return reddit


def is_usable(text: str) -> bool:
    """Skip deleted/removed/empty content."""
    return bool(text) and text.strip() not in ("[deleted]", "[removed]")


def mentions_topic(text: str) -> bool:
    low = text.lower()
    return "myntra" in low or any(word in low for word in TOPIC_WORDS)


def to_iso_date(created_utc: float) -> str:
    return datetime.fromtimestamp(created_utc, tz=timezone.utc).date().isoformat()


def collect(reddit: praw.Reddit) -> list[dict]:
    """Search every (subreddit, query) pair and gather matching posts + comments."""
    rows, seen_posts = [], set()

    for sub in SUBREDDITS:
        for query in QUERIES:
            if len(rows) >= TARGET_TOTAL:
                return rows
            print(f"Searching r/{sub} for: {query}")
            results = reddit.subreddit(sub).search(query, sort="relevance",
                                                   time_filter="year", limit=POSTS_PER_SEARCH)
            for post in results:
                if post.id in seen_posts:
                    continue
                seen_posts.add(post.id)

                # Post text = title + body, kept exactly as written.
                text = post.title if not post.selftext else f"{post.title}\n\n{post.selftext}"
                if "myntra" not in text.lower() or not is_usable(text):
                    continue  # search can return loosely related posts; keep real mentions only
                rows.append({"date": to_iso_date(post.created_utc), "text": text,
                             "url": f"https://www.reddit.com{post.permalink}"})

                # Comments on this post that talk about Myntra or our topics.
                post.comments.replace_more(limit=0)  # skip "load more comments" stubs
                kept = 0
                for comment in post.comments.list():
                    if kept >= COMMENTS_PER_POST or len(rows) >= TARGET_TOTAL:
                        break
                    body = comment.body
                    if is_usable(body) and mentions_topic(body):
                        rows.append({"date": to_iso_date(comment.created_utc), "text": body,
                                     "url": f"https://www.reddit.com{comment.permalink}"})
                        kept += 1

                if len(rows) >= TARGET_TOTAL:
                    return rows
                time.sleep(PAUSE_SECONDS)
            time.sleep(PAUSE_SECONDS)
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Hindi/emoji-safe printing on Windows
    rows = collect(connect())

    df = pd.DataFrame(rows).drop_duplicates(subset="url")
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # Unified schema from CLAUDE.md
    df["id"] = [f"RD-{i:04d}" for i in range(1, len(df) + 1)]
    df["source"] = "reddit"
    df["rating"] = ""     # Reddit has no star rating
    df["language"] = ""   # detected later in src/clean.py
    df = df[["id", "source", "date", "rating", "text", "url", "language"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} Reddit items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
