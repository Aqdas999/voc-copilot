"""Deterministic cross-check of the refund-delay sub-issues. Keyword rules only, NO LLM calls.

Why: the LLM split refund delay into sub-issues but left 38% as "other / unclear". This
script counts, with plain regular expressions anyone can read, how many refund-delay items
mention each of three situations. Counts can overlap (one review can mention a cancellation
AND a "credited" message) and a review can match none.

  cancel_or_undelivered    the order was cancelled or never delivered
                           (cancel / cancelled / cancellation, "cancel ho gaya", रद्द, कैंसल,
                            not delivered, undelivered, "deliver nahi hua", डिलीवर नहीं ...)
  return_or_pickup         a product return or pickup (return, pickup, रिटर्न, "wapas kiya")
                           - money phrases like "paise return karo" / "पैसे वापस" do NOT count
  credited_not_received    the app says completed / credited / processed, but the money
                           is not there ("credited likh dete h ... aata ni h")
                           - "refund not credited" alone does NOT count: that is just a delay

Also, for the roadmap's R1 reach: how many ORDER CANCELLED complaints say the refund/money
has not come back (refund/money word near a negation) -> cancelled_refund_crosscheck.csv.

Output: data/processed/refund_crosscheck.csv (one row per refund-delay item, a 0/1 column per
rule) and a printed summary. src/insights.py adds the counts to insights.json, labelled
"keyword count", so the dashboard can show them next to the LLM sub-issues.

Run:  python -m src.refund_crosscheck
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

EXTRACTIONS_PATH = Path("data/processed/extractions_v3_complaints.jsonl")
FEEDBACK_PATH = Path("data/processed/feedback.csv")
OUTPUT_PATH = Path("data/processed/refund_crosscheck.csv")
THEME = "refund delay"
EXAMPLES = 2

# --- Rule 1: cancelled or never delivered --------------------------------------------
CANCEL = re.compile(
    r"cancel|cancle|cancal|कैंसल|कैंसिल|रद्द|\bradd\b"
    r"|not\s+(been\s+)?delivered|undelivered|never\s+delivered|non[-\s]?delivery"
    r"|did\s*n[o']?t\s+deliver|deliver(y|ed)?\s+(hi\s+)?(nahi|nhi|nahin)"
    r"|(order|parcel|product)\s+(hi\s+)?(nahi|nhi)\s+(aaya|aya|mila)"
    r"|डिलीवर(ी)?\s+नहीं|never\s+(arrived|came)"
    r"|(order|parcel|product)\s+(was\s+|is\s+)?(not|never)\s+(rec(ei|ie|i)ved|arrived)"
    r"|ordered\b[^.]{0,60}\bnot\s+rec(ei|ie|i)ved",       # "ordered ... but not recived it"
    re.IGNORECASE)

# --- Rule 2: product return or pickup ----------------------------------------------
# First remove phrases where "return"/"wapas" means MONEY back, then look for return words.
MONEY_BACK = re.compile(
    r"(paise?|paisa|pese|pesse|money|amount|payment|पैसे|पैसा|रुपये)\s+(bhi\s+|ka\s+)?"
    r"(return|wapas|vapas|वापस)"
    r"|(return|refund)\w*\s+(the\s+|my\s+|mera\s+|mere\s+)?(paise?|paisa|money|amount|payment)",
    re.IGNORECASE)
RETURN = re.compile(r"return|pick\s*-?\s*up|picked|रिटर्न|पिकअप|wapas|vapas|वापस",
                    re.IGNORECASE)

# --- Rule 3: "completed / credited / processed" but not received --------------------
# A status claim (with negated forms like "not credited" removed first) ...
NEGATED_STATUS = re.compile(
    r"(not|n't|nahi|nhi|never)\s+(been\s+|yet\s+|be\s+)?(credited|processed|completed)",
    re.IGNORECASE)
STATUS_CLAIM = re.compile(
    r"complete[d]?|credit(ed)?|processed|क्रेडिट|प्रोसेस|पूरा\s+हो", re.IGNORECASE)
# ... AND a statement that the money did not arrive.
NOT_RECEIVED = re.compile(
    r"not\s+(been\s+)?(rec(ei|ie|i)ved|reflect\w*|credited|got)"
    r"|have\s*n[o']?t\s+(rec(ei|ie|i)ved|got)|did\s*n[o']?t\s+(rec(ei|ie|i)ve|get)"
    r"|not\s+in\s+my\s+(account|bank)"
    r"|(nahi|nhi|ni|nahin)\s+(aaya|aya|aata|mila|hua)|aata\s+(ni|nhi|nahi)"
    r"|kuch\s+(nhi|nahi)\s+(hota|aaya)|नहीं\s+(आया|मिला|आए)|still\s+(not|waiting)",
    re.IGNORECASE)


def flags(text: str) -> dict:
    """Apply the three rules to one review. Returns 0/1 per rule."""
    no_money = MONEY_BACK.sub(" ", text)
    no_negation = NEGATED_STATUS.sub(" ", text)
    return {
        "cancel_or_undelivered": int(bool(CANCEL.search(text))),
        "return_or_pickup": int(bool(RETURN.search(no_money))),
        "credited_not_received": int(bool(STATUS_CLAIM.search(no_negation))
                                     and bool(NOT_RECEIVED.search(text))),
    }


# --- Rule 4 (for the roadmap's reach): ORDER CANCELLED items whose money did not come back ---
# A refund/money word within 60 characters of a negation or "pending/stuck".
# "refunded" on its own means the money DID arrive, so it only counts as "not refunded".
MONEY_WORD = re.compile(
    r"refund(?!ed\b)\w*|money|paise?|paisa|pese|amount|payment|₹|रिफंड|पैसे|पैसा"
    r"|not\s+(been\s+)?refunded", re.IGNORECASE)
NEGATION = re.compile(
    r"\b(not|no|never|didn.?t|doesn.?t|haven.?t|hasn.?t|won.?t|without|yet\s+to|pending|stuck"
    r"|nahi|nhi|nahin|ni|na)\b|ना|नहीं", re.IGNORECASE)
WINDOW = 60
CANCEL_THEME = "order cancelled"
CANCEL_OUTPUT_PATH = Path("data/processed/cancelled_refund_crosscheck.csv")


REFUNDED_BEFORE = re.compile(r"refunded\s+(my|the|your|our)?\s*$", re.IGNORECASE)


def money_not_back(text: str) -> bool:
    for m in MONEY_WORD.finditer(text):
        if REFUNDED_BEFORE.search(text[max(0, m.start() - 25):m.start()]):
            continue                      # "refunded my money" = the money came back
        if NEGATION.search(text[max(0, m.start() - WINDOW):m.end() + WINDOW]):
            return True
    return False


def cancelled_refund_missing() -> pd.DataFrame:
    """Order-cancelled items with a 0/1 flag: says the refund/money has not come back."""
    ext = pd.DataFrame([json.loads(l) for l in EXTRACTIONS_PATH.read_text(encoding="utf-8").splitlines() if l])
    fb = pd.read_csv(FEEDBACK_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    items = ext[ext["theme"] == CANCEL_THEME][["id"]].merge(fb[["id", "source", "language", "text"]], on="id")
    items["refund_not_received"] = items["text"].map(money_not_back).astype(int)
    return items


RULES = ["cancel_or_undelivered", "return_or_pickup", "credited_not_received"]
LABELS = {
    "cancel_or_undelivered": "mentions cancel / not delivered",
    "return_or_pickup": "mentions return / pickup",
    "credited_not_received": "says completed/credited/processed but not received",
}


def crosscheck() -> tuple[pd.DataFrame, dict]:
    """Per-item flags for the refund-delay theme, plus a summary for insights.json."""
    ext = pd.DataFrame([json.loads(l) for l in EXTRACTIONS_PATH.read_text(encoding="utf-8").splitlines() if l])
    fb = pd.read_csv(FEEDBACK_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    items = (ext[ext["theme"] == THEME][["id"]]
             .merge(fb[["id", "source", "language", "date", "text"]], on="id"))
    items = pd.concat([items, pd.DataFrame([flags(t) for t in items["text"]], index=items.index)], axis=1)
    items["none_of_these"] = (items[RULES].sum(axis=1) == 0).astype(int)

    summary = {
        "label": "keyword count",
        "method": "regex rules in src/refund_crosscheck.py; no LLM. Counts overlap.",
        "theme_items": len(items),
        "counts": [{"rule": r, "label": LABELS[r], "keyword_count": int(items[r].sum()),
                    "example_ids": items.loc[items[r] == 1, "id"].head(EXAMPLES).tolist()}
                   for r in RULES],
        "none_of_these": int(items["none_of_these"].sum()),
    }
    return items, summary


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    items, summary = crosscheck()
    out = items.drop(columns=["text"])           # text stays in feedback.csv; join on id
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Refund delay: {summary['theme_items']} items (keyword count, overlapping)\n")
    for c in summary["counts"]:
        print(f"  {c['label']:52s} {c['keyword_count']:3d}   e.g. {', '.join(c['example_ids'])}")
    print(f"  {'none of these':52s} {summary['none_of_these']:3d}")
    print(f"\nSaved {OUTPUT_PATH}")

    cancelled = cancelled_refund_missing()
    cancelled.drop(columns=["text"]).to_csv(CANCEL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    hits = cancelled[cancelled["refund_not_received"] == 1]
    print(f"\nOrder cancelled: {len(hits)} of {len(cancelled)} say the refund/money has not come back "
          f"(keyword count): {', '.join(hits['id'])}")
    print(f"Saved {CANCEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
