"""Draw a 30-item HOLDOUT set: fresh items the prompts were never tuned on.

Why: PROMPT_V2 and V3 were written by looking at golden-set misses. Even with invented
examples, repeatedly fixing what the golden set shows can "overfit" to those 100 items.
A holdout the prompts have never seen tells us whether the improvements are real.

Sampling plan (stratified, seed 7, excludes every golden-set id):
  Play Store 1-3 stars   18   (12 English + 4 Hinglish + 2 Hindi)
  App Store  1-3 stars    4
  Play Store 4-5 stars    3
  App Store  4-5 stars    2
  Synthetic tickets       3   (only 5 are left outside the golden set)
  ----------------------------
  total                  30   (25 complaints-likely, 6+ non-English guaranteed)

Refuses to overwrite an existing holdout.csv (it may hold your labels). --force to redraw.

Run:  python -m evals.make_holdout
"""

import sys
from pathlib import Path

import pandas as pd

SEED = 7
FEEDBACK_PATH = Path("data/processed/feedback.csv")
GOLDEN_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("data/golden/holdout.csv")

PLAN = [
    ("Play Store 1-3 English",  "playstore",        "1-3", "en",       12),
    ("Play Store 1-3 Hinglish", "playstore",        "1-3", "hinglish",  4),
    ("Play Store 1-3 Hindi",    "playstore",        "1-3", "hi",        2),
    ("App Store 1-3",           "appstore",         "1-3", None,        4),
    ("Play Store 4-5",          "playstore",        "4-5", None,        3),
    ("App Store 4-5",           "appstore",         "4-5", None,        2),
    ("Synthetic tickets",       "synthetic_ticket", "",    None,        3),
]
MIN_NON_ENGLISH = 5

COLUMNS = ["id", "source", "language", "rating", "text",
           "theme", "churn_signal", "severity", "ai_confidence", "ai_reason", "notes", "verified"]


def rating_band(rating: str) -> str:
    if rating == "":
        return ""
    return "1-3" if int(rating) <= 3 else "4-5"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if OUTPUT_PATH.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUTPUT_PATH} already exists and may contain your labels. Use --force to redraw.")

    df = pd.read_csv(FEEDBACK_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    golden_ids = set(pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False,
                                 encoding="utf-8-sig")["id"])
    df = df[~df["id"].isin(golden_ids)]          # never reuse a golden item
    df["band"] = df["rating"].apply(rating_band)

    samples = []
    for description, source, band, language, n in PLAN:
        group = df[(df["source"] == source) & (df["band"] == band)]
        if language:
            group = group[group["language"] == language]
        samples.append(group.sample(n=n, random_state=SEED))
        print(f"{description:24s} picked {n:3d} of {len(group):5d}")

    holdout = pd.concat(samples)
    assert len(holdout) == 30 and holdout["id"].is_unique
    assert not set(holdout["id"]) & golden_ids
    assert (holdout["language"] != "en").sum() >= MIN_NON_ENGLISH

    for col in ["theme", "churn_signal", "severity", "ai_confidence", "ai_reason", "notes", "verified"]:
        holdout[col] = ""
    holdout = holdout[COLUMNS].sort_values(["source", "id"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(holdout)} items to {OUTPUT_PATH}")
    print("Languages:", holdout["language"].value_counts().to_dict())


if __name__ == "__main__":
    main()
