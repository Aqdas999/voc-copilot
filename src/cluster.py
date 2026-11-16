"""Turn per-item extractions into theme-level numbers: counts, churn, rating, trend.

"Clustering" here is simple on purpose: the LLM already put every item into exactly one
theme from data/golden/themes.md, so a theme = all items with that label. This script
just counts and compares them.

Input:   data/processed/extractions_<version>_complaints.jsonl   (from src.extract --complaints)
         data/processed/feedback.csv                             (rating, date)
Scope:   the COMPLAINT SET only = 1-3★ Play Store + 1-3★ App Store + synthetic tickets.
         4-5★ reviews were not extracted (free-tier token limits), so every percentage
         here is "% of complaints (1-3★)", never "% of all reviews".

Outputs (data/processed/):
  theme_stats_<version>_complaints.csv    one row per theme
  theme_weekly_<version>_complaints.csv   items per theme per week

Run:  python -m src.cluster --version v3
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

FEEDBACK_PATH = Path("data/processed/feedback.csv")
OUTPUT_DIR = Path("data/processed")
PCT_LABEL = "% of complaints (1-3★)"   # the only denominator used in this file

# Weeks with fewer items than this are lumped into one "earlier" bucket. Our English
# reviews only cover ~2 weeks; older items are rare Hindi reviews spread over a year,
# and a week with 2 items would make week-over-week change meaningless.
MIN_ITEMS_PER_WEEK = 50


def load(version: str) -> pd.DataFrame:
    """Extractions joined with rating + date from feedback.csv."""
    path = OUTPUT_DIR / f"extractions_{version}_complaints.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found. Run: python -m src.extract --complaints --version {version}")
    ext = pd.DataFrame([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l])
    fb = pd.read_csv(FEEDBACK_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = ext.merge(fb[["id", "source", "rating", "date"]], on="id", how="left")
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")   # synthetic -> NaN
    df["date"] = pd.to_datetime(df["date"])
    return df


def theme_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per theme: count, share, churn-Y, average rating, nulled quotes."""
    g = df.groupby("theme")
    table = pd.DataFrame({
        "items": g.size(),
        PCT_LABEL: (100 * g.size() / len(df)).round(1),
        "churn_Y": g["churn_signal"].apply(lambda s: int((s == "Y").sum())),
        # mean over rated items only: synthetic tickets have no star rating
        "avg_rating (1-3★ only)": g["rating_num"].mean().round(2),
        "n_rated": g["rating_num"].count(),
        "quotes_nulled": g["quote_valid"].apply(lambda s: int((s == False).sum())),  # noqa: E712
    })
    return table.sort_values("items", ascending=False)


def weekly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Items per theme per week (Mon-Sun), sparse old weeks lumped together, plus WoW change."""
    week_start = df["date"].dt.to_period("W-SUN").dt.start_time
    sizes = week_start.value_counts()
    full_weeks = sorted(sizes[sizes >= MIN_ITEMS_PER_WEEK].index)
    first = full_weeks[0]

    def label(ts):
        if ts < first:
            return f"before {first:%d %b}"
        return f"wk {ts:%d %b}–{ts + pd.Timedelta(days=6):%d %b}"

    df = df.assign(week=week_start.map(label))
    order = [f"before {first:%d %b}"] + [label(w) for w in full_weeks]
    pivot = (df.pivot_table(index="theme", columns="week", values="id",
                            aggfunc="count", fill_value=0)
               .reindex(columns=[c for c in order if c in set(df["week"])]))
    if len(full_weeks) >= 2:
        last, prev = label(full_weeks[-1]), label(full_weeks[-2])
        pivot["WoW change"] = pivot[last] - pivot[prev]
        pivot["WoW %"] = (100 * pivot["WoW change"] / pivot[prev].where(pivot[prev] > 0)).round(0)
    return pivot.sort_values(pivot.columns[-3] if len(full_weeks) >= 2 else pivot.columns[-1],
                             ascending=False)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3")
    args = parser.parse_args()

    df = load(args.version)
    stats = theme_table(df)
    weekly = weekly_table(df)

    stats.to_csv(OUTPUT_DIR / f"theme_stats_{args.version}_complaints.csv", encoding="utf-8-sig")
    weekly.to_csv(OUTPUT_DIR / f"theme_weekly_{args.version}_complaints.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"Complaint set (1-3★ Play Store + App Store, plus synthetic tickets): {len(df)} items, "
          f"prompt {args.version}")
    print(f"  by source: {df['source'].value_counts().to_dict()}")
    print(f"  Percentages are {PCT_LABEL}, NOT % of all reviews (4-5★ were not extracted).\n")
    print(stats.to_string())
    print(f"\nTotal churn-Y: {int((df['churn_signal'] == 'Y').sum())}  |  "
          f"Quotes nulled by the guard: {int((df['quote_valid'] == False).sum())} of {len(df)}")  # noqa: E712
    print(f"\nWeek-over-week items per theme (weeks run Mon-Sun; data ends "
          f"{df['date'].max():%d %b %Y}, so the latest week may be partial):\n")
    print(weekly.to_string())


if __name__ == "__main__":
    main()
