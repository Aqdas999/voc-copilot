"""Draw the 100-item golden set used to evaluate the pipeline.

The golden set is a sample that a human labels by hand (theme, churn signal, severity).
We then compare the LLM's extractions against these human labels.

Sampling plan (stratified = a fixed number from each group, chosen at random):
  Play Store 1-3 stars   55   (45 English + 7 Hinglish + 3 Hindi)
  Play Store 4-5 stars   10
  App Store  1-3 stars   12
  App Store  4-5 stars    8
  Synthetic tickets      15
  ----------------------------
  total                 100

Why this mix: most pain points live in 1-3 star reviews, so they get most of the sample,
but a few 4-5 star items check the model doesn't invent complaints in happy reviews.
Hinglish/Hindi are forced into the Play Store 1-3 star group so there are always
at least 10 non-English items (they'd otherwise be too rare to test).

Reproducible: a fixed random seed means the same 100 items every run.
Safe: refuses to overwrite an existing golden set (it may contain your hand labels).
Pass --force only if you really want to redraw it.

Run:  python -m evals.make_golden_set
"""

import sys
from pathlib import Path

import pandas as pd

SEED = 42
INPUT_PATH = Path("data/processed/feedback.csv")
OUTPUT_PATH = Path("data/golden/golden_set.csv")

# (description, source, rating band, language or None for any, how many)
PLAN = [
    ("Play Store 1-3★ English",  "playstore",        "1-3", "en",       45),
    ("Play Store 1-3★ Hinglish", "playstore",        "1-3", "hinglish",  7),
    ("Play Store 1-3★ Hindi",    "playstore",        "1-3", "hi",        3),
    ("Play Store 4-5★",          "playstore",        "4-5", None,       10),
    ("App Store 1-3★",           "appstore",         "1-3", None,       12),
    ("App Store 4-5★",           "appstore",         "4-5", None,        8),
    ("Synthetic tickets",        "synthetic_ticket", "",    None,       15),
]
MIN_NON_ENGLISH = 10

# Label columns are left empty for the human labeler.
COLUMNS = ["id", "source", "language", "rating", "text",
           "theme", "churn_signal", "severity", "notes", "verified"]


def rating_band(rating: str) -> str:
    """'1'-'3' -> '1-3', '4'-'5' -> '4-5', empty (no rating) -> ''."""
    if rating == "":
        return ""
    return "1-3" if int(rating) <= 3 else "4-5"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # so ★ prints on the Windows console
    if OUTPUT_PATH.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUTPUT_PATH} already exists and may contain your labels. "
                 "Not overwriting. Use --force to redraw.")

    df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["band"] = df["rating"].apply(rating_band)

    samples = []
    for description, source, band, language, n in PLAN:
        group = df[(df["source"] == source) & (df["band"] == band)]
        if language:
            group = group[group["language"] == language]
        picked = group.sample(n=n, random_state=SEED)
        print(f"{description:26s} picked {n:3d} of {len(group):5d}")
        samples.append(picked)

    golden = pd.concat(samples)
    non_english = (golden["language"] != "en").sum()
    assert len(golden) == 100 and golden["id"].is_unique
    assert non_english >= MIN_NON_ENGLISH, f"only {non_english} non-English items"

    for col in ["theme", "churn_signal", "severity", "notes", "verified"]:
        golden[col] = ""
    golden = golden[COLUMNS].sort_values(["source", "id"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(golden)} items to {OUTPUT_PATH}")
    print("Languages:", golden["language"].value_counts().to_dict())


if __name__ == "__main__":
    main()
