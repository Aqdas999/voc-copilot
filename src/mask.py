"""Mask person names (e.g. delivery agents) in text shown on the dashboard.

Rule-based, so it is predictable and explainable; it will not catch every name.
Patterns covered (English, Hinglish, Hindi):
  "named Rahul", "name is Rahul", "Mr. Rahul", "Mrs Sharma"
  "delivery boy Rahul", "agent Rahul", "executive Priya"   (capitalised word after a role)
  "jiska naam rushikesh tha", "naam Rahul hai"             (Hinglish, any case)
  "जिसका नाम राहुल था"                                     (Hindi)
Each name becomes "[name]". The source text in data/ is never changed; masking is display-only.
"""

import re

# Capitalised words that follow "agent"/"boy" etc. but are not names.
NOT_NAMES = {
    "The", "And", "Is", "Was", "Has", "Had", "Did", "Didn", "Does", "Not", "Never", "Of", "To",
    "In", "On", "At", "For", "From", "With", "I", "My", "Me", "We", "He", "She", "They", "It",
    "Who", "Which", "That", "This", "Refused", "Called", "Came", "Said", "Says", "Told", "Asked",
    "Cancelled", "Canceled", "Marked", "Myntra", "Team", "Service", "Support", "Name", "Please",
    "Very", "Also", "But", "Or", "So", "Then", "Even", "Still", "Just", "Again", "Today",
    "Centre", "Center", "Hub", "Partner", "Office", "Facility", "Company", "Branch", "Side",
}

ROLE = r"(?:delivery\s+)?(?:agent|boy|executive|guy|rider|associate|bhaiya|person|courier)"

PATTERNS = [
    # named X / name is X / called X  (1-2 capitalised words)
    (re.compile(r"\b(named|name\s+is|name\s+was)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
     r"\1 [name]"),
    # Mr / Mrs / Ms / Shri / Smt + name
    (re.compile(r"\b(Mr|Mrs|Ms|Miss|Shri|Smt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"), r"\1 [name]"),
    # Hinglish: naam <word> tha/hai/thi  (any case; "naam per" etc. excluded by the verb)
    (re.compile(r"\b(naam|name)\s+(?!per\b|pe\b|par\b|se\b|ka\b|ki\b|ke\b)\w+(\s+\w+)?\s+(tha|thi|hai|h)\b",
                re.IGNORECASE), r"\1 [name] \3"),
    # Hindi: नाम <word> था/है/थी
    (re.compile(r"नाम\s+\S+\s+(था|थी|है)"), r"नाम [name] \1"),
]
# (?![\w'’]) stops "Agent Won't" being read as a name ("Won" followed by an apostrophe)
ROLE_NAME = re.compile(rf"\b({ROLE})(\s*[(,:-]?\s*)([A-Z][a-z]{{2,}})(?![\w'’])", re.IGNORECASE)


def mask_names(text: str | None) -> str | None:
    """Return text with person names replaced by [name]. None stays None."""
    if not text:
        return text
    for pattern, replacement in PATTERNS:
        text = pattern.sub(replacement, text)

    def role_name(m: re.Match) -> str:
        word = m.group(3)
        if word in NOT_NAMES or not word[0].isupper():
            return m.group(0)
        return f"{m.group(1)}{m.group(2)}[name]"
    return ROLE_NAME.sub(role_name, text)


if __name__ == "__main__":
    # Quick self-test: python -m src.mask
    tests = {
        "jiska naam rushikesh tha usne": "jiska naam [name] tha usne",
        "the delivery boy Rahul was rude": "the delivery boy [name] was rude",
        "delivery agent didn't come": "delivery agent didn't come",
        "the agent named Suresh Kumar refused": "the agent named [name] refused",
        "Mr. Sharma from support": "Mr [name] from support",
        "exchange ke naam per dhokha": "exchange ke naam per dhokha",
        "जिसका नाम राहुल था": "जिसका नाम [name] था",
        "The Delivery Agent didn't reached": "The Delivery Agent didn't reached",
        "Agent Won’t let you type": "Agent Won’t let you type",
        "stuck in the courier Centre": "stuck in the courier Centre",
    }
    for given, expected in tests.items():
        got = mask_names(given)
        print("OK  " if got == expected else "FAIL", given, "->", got)
