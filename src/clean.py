"""Merge all raw feedback into one clean file: data/processed/feedback.csv

Steps:
  1. Load every CSV in data/raw/ (Play Store, synthetic tickets, Reddit if present).
  2. Drop items with fewer than 3 words ("good", "nice app") — too short to contain a pain point.
  3. Remove duplicates (same text, ignoring case/spacing/punctuation), keeping the first copy.
  4. Detect language: en / hinglish / hi / other.
  5. Save in the unified schema from CLAUDE.md and print counts.

The `text` column is NEVER modified. Any lowercasing/normalising happens on temporary
copies used only for comparison, so later quotes are exact substrings of what users wrote.

Run:  python -m src.clean
"""

import re
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/feedback.csv")
COLUMNS = ["id", "source", "date", "rating", "text", "url", "language"]
MIN_WORDS = 3

# Common Hindi words as people type them in English letters (Hinglish), including
# common shorthand/spellings ("h" for hai, "bhut" for bahut).
# Deliberately leaves out words that are also normal English ("main", "to", "me", "so", "do"),
# and "aap" — in these reviews it's usually a typo for "app", not Hindi "you".
HINGLISH_WORDS = {
    "hai", "h", "hain", "ho", "hu", "hoon", "hota", "hoti", "hote", "tha", "thi",
    "nahi", "nahin", "nhi", "mat", "kya", "kyun", "kyu", "kyon", "kaise", "kab", "kaha", "kahan",
    "bahut", "bohot", "bhut", "bhutt", "bahot", "bhot",
    "accha", "acha", "achha", "achcha", "aacha", "achi", "acchi", "achhi",
    "bekar", "bekaar", "bakwas", "bakwaas", "ghatiya", "paisa", "paise", "paiso",
    "mera", "meri", "mere", "mujhe", "maine", "humne", "hum", "apna", "apni",
    "yah", "yeh", "ye", "naya", "nayi", "sundar", "sabse", "sbse", "itna", "jyada", "zyada",
    "milti", "milta", "sasta", "sasti", "ekdum", "mast",
    "kar", "karo", "karna", "karte", "karta", "kiya", "kiye", "diya", "diye", "liya", "liye",
    "gaya", "gayi", "gaye", "raha", "rahi", "rahe", "wala", "wali", "wale",
    "aur", "lekin", "par", "pe", "ke", "ki", "ka", "ko", "se", "mein", "mai",
    "ab", "abhi", "tak", "bhi", "sab", "koi", "kuch", "jab", "toh", "yaar", "bhai",
    "ghante", "baar", "dikh", "chahiye", "jaldi", "bilkul", "sahi", "galat", "hisaab",
}
HINGLISH_WORDS |= {"bakwass", "badhiya", "bdhiya", "rha", "rhi", "dikha"}
HINGLISH_MIN_MATCHES = 2   # need at least 2 different Hindi words to call it Hinglish...

# ...unless one of these appears: they are never English words, so one is enough.
# ("tha", "se", "ki" are NOT here: they show up as English typos, e.g. "tha best app".)
STRONG_HINGLISH_WORDS = {
    "hai", "hain", "nahi", "nahin", "nhi", "mujhe", "bahut", "bhut", "bhutt", "bohot",
    "kyun", "chahiye", "bakwas", "bakwaas", "bakwass", "bekar", "bekaar", "ghatiya",
    "accha", "acha", "achha", "achcha", "aacha", "badhiya", "bdhiya",
}


def load_raw() -> pd.DataFrame:
    """Read every raw CSV as plain text (dtype=str) so nothing gets reinterpreted.

    keep_default_na=False stops pandas turning a review that literally says "NA"
    or "null" into an empty value.
    """
    files = sorted(RAW_DIR.glob("*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        print(f"Loaded {len(df):5d} rows from {f.name}")
        frames.append(df[COLUMNS])
    return pd.concat(frames, ignore_index=True)


def word_count(text: str) -> int:
    """Count tokens that contain at least one letter (any script).

    So emoji-only or punctuation-only tokens like '👍👍' or '!!!' don't count as words,
    but Hindi words in Devanagari do.
    """
    return sum(any(ch.isalpha() for ch in token) for token in text.split())


def dedupe_key(text: str) -> str:
    """Comparison-only version of the text: lowercase, letters/digits only, single spaces."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def detect_language(text: str) -> str:
    """Rule-based language detection, tuned for Indian app reviews.

    - mostly Devanagari (Hindi script)          -> 'hi'
    - some Devanagari mixed with English letters -> 'hinglish' (code-mixed)
    - mostly other non-Latin letters             -> 'other' (e.g. Tamil, Bengali script)
    - English letters with 2+ Hindi words,
      or 1 unmistakable one like "hai"/"nahi"    -> 'hinglish'
    - otherwise                                  -> 'en'

    We use simple rules instead of a library: off-the-shelf detectors don't know
    Hinglish and are unreliable on short texts.
    """
    def is_devanagari(ch: str) -> bool:
        return "ऀ" <= ch <= "ॿ"

    # Hindi vowel signs (ि, ी, ु...) are not "letters" to Python's isalpha(),
    # so count every Devanagari character explicitly.
    letters = [ch for ch in text if ch.isalpha() or is_devanagari(ch)]
    if not letters:
        return "other"

    devanagari = sum(is_devanagari(ch) for ch in letters)
    latin = sum(ch.isascii() for ch in letters)

    if devanagari / len(letters) >= 0.5:
        return "hi"
    if devanagari > 0:
        return "hinglish"
    if latin / len(letters) < 0.5:
        return "other"

    words = set(re.findall(r"[a-z]+", text.lower()))
    if len(words & HINGLISH_WORDS) >= HINGLISH_MIN_MATCHES or words & STRONG_HINGLISH_WORDS:
        return "hinglish"
    return "en"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    df = load_raw()
    before = df["source"].value_counts()

    # 1. Drop empty and very short items.
    df = df[df["text"].str.strip() != ""]
    df = df[df["text"].apply(word_count) >= MIN_WORDS]
    after_short = df["source"].value_counts()

    # 2. Remove duplicates. Same ID twice = same item; same text = repeated review.
    df = df.drop_duplicates(subset="id")
    df = df[~df["text"].apply(dedupe_key).duplicated(keep="first")]
    after_dedupe = df["source"].value_counts()

    # 3. Detect language (replaces the rough guess from the scrapers).
    df["language"] = df["text"].apply(detect_language)

    # 4. Save.
    df = df[COLUMNS].reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 5. Report.
    report = pd.DataFrame({
        "raw": before,
        f"after <{MIN_WORDS}-word drop": after_short,
        "after dedupe (final)": after_dedupe,
    }).fillna(0).astype(int)
    report.loc["TOTAL"] = report.sum()
    print("\nItems per source:")
    print(report.to_string())

    lang = pd.crosstab(df["source"], df["language"], margins=True, margins_name="TOTAL")
    print("\nLanguage per source (final):")
    print(lang.to_string())

    print(f"\nSaved {len(df)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
