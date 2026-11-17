"""Turn theme stats into PM-ready insights: sub-issues, verified quotes, one sentence each.

For each of the top 6 themes in the complaint set (1-3★ + synthetic tickets):
  1. Sub-issues: the LLM reads the pain_point summaries and names 2-4 sub-issues, then
     assigns every item to one of them. The COUNTS are computed in code from those
     assignments (the LLM never reports a number).
  2. Quotes: 3-5 evidence spans, picked in code. Only spans that passed the quote guard
     (exact substrings of the original text), chosen to mix sources and languages.
  3. Insight sentence in the rubric's "good output" format (evals/rubric.md).
     The LLM writes the WORDING ONLY, as a template with {placeholders} and no digits.
     Code fills in every number and ID. If the template breaks the rules, we fall back to a
     fixed template, so the sentence is never blocked by a bad LLM reply.
  4. Checks: every number in the final sentence is re-computed from the raw data and
     compared (count accuracy), and every cited ID must exist, belong to the theme and
     have an exact-substring quote (citation validity).

LLM replies are cached by prompt hash in data/processed/cache/, so reruns cost no quota.

Run:  python -m src.insights            # writes data/processed/insights.json
"""

import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src import llm
from src.cluster import load

VERSION = "v3"                 # which extraction prompt's output we build on
TOP_N_THEMES = 6
ASSIGN_CHUNK = 100             # pain points per assignment call (keeps us under Groq's 8K TPM)
NAMING_SAMPLE = 120            # pain points shown to the LLM when naming sub-issues
MAX_QUOTES, MIN_QUOTES = 5, 3
CITED_IDS = 3                  # IDs cited inside the sentence itself
SKIP_THEMES = {"no complaint", "other"}
PCT_LABEL = "of complaints (1–3★)"

CACHE_PATH = Path("data/processed/cache/insights_llm.jsonl")
OUTPUT_PATH = Path("data/processed/insights.json")
FEEDBACK_PATH = Path("data/processed/feedback.csv")
ID_PATTERN = re.compile(r"\b(?:PS|AS|SYN|RD)-\d{4}\b")

# ---------------------------------------------------------------------------
# Prompts (versioned; changing one = new version name)
# ---------------------------------------------------------------------------
SUBISSUE_PROMPT_V1 = """You are analysing customer complaints about Myntra, an Indian fashion
e-commerce app. Every complaint below was labelled with the theme "{theme}".
Each line is a short summary of one complaint.

Group them into 2 to 4 distinct, specific sub-issues a product manager could act on.
Rules:
- Each name is 3-8 words of plain English and contains NO digits.
- Be specific ("refund stuck after return pickup"), not generic ("refund problems").
- Sub-issues must not overlap. Cover the most common problems first.
Return ONLY a JSON list of strings, like ["first sub-issue", "second sub-issue"].

Complaints:
{lines}
"""

ASSIGN_PROMPT_V1 = """Assign each numbered complaint (theme: "{theme}") to exactly one sub-issue.

Sub-issues:
{options}

Complaints:
{lines}

Return ONLY a JSON list of exactly {n} integers: the sub-issue number for complaint 1,
complaint 2, ... complaint {n}, in order. Use {other} if none fits.
"""

INSIGHT_PROMPT_V1 = """You write one-sentence insights for a product manager at Myntra.
For each theme below, write ONE sentence as a TEMPLATE. Numbers and IDs are filled in later
by code, so you must use these placeholders, each exactly once, written with curly braces:

  {{theme}}        theme name, e.g. "Refund delay"
  {{pct}}          its share, already includes the words "% of complaints (1–3★)"
  {{n}}            number of complaints in the theme
  {{total}}        number of complaints overall
  {{trend}}        one of: rising / falling / roughly flat
  {{wk_prev}}      share of the earlier week's complaints, e.g. "7.9%"
  {{wk_last}}      share of the latest week's complaints
  {{top_sub}}      name of the biggest sub-issue
  {{top_sub_n}}    number of complaints in that sub-issue
  {{churn_n}}      number of complainers who say they will leave or warn others off
  {{churn_pct}}    that number as a % of the theme
  {{ids}}          bracketed list of source IDs, put at the very end of the sentence

Rules:
- Use every placeholder exactly once. Write NO digits at all; never invent numbers.
- Use {{trend}} as given; do not add your own direction words like "up" or "down".
- Plain, direct business English, one sentence, at most 60 words.
- Shape to follow: size and share, then the weekly direction, then the biggest
  sub-issue, then the churn signal, then {{ids}}.

Themes (with the top sub-issue name, only to help your wording):
{themes}

Return ONLY a JSON object mapping each theme name exactly as given to its template sentence.
"""

PLACEHOLDERS = ["theme", "pct", "n", "total", "trend", "wk_prev", "wk_last",
                "top_sub", "top_sub_n", "churn_n", "churn_pct", "ids"]
FALLBACK_TEMPLATE = ("{theme} is {pct} ({n} of {total}), {trend} from {wk_prev} to {wk_last} "
                     "of weekly complaints; the biggest sub-issue is \"{top_sub}\" ({top_sub_n} "
                     "complaints), and {churn_n} complainers ({churn_pct}) say they will leave "
                     "or warn others off {ids}.")


# ---------------------------------------------------------------------------
# LLM with a disk cache keyed by the exact prompt
# ---------------------------------------------------------------------------
def _cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    rows = [json.loads(l) for l in CACHE_PATH.read_text(encoding="utf-8").splitlines() if l]
    return {r["key"]: r for r in rows}


def ask_json(prompt: str, is_valid, attempts: int = 3) -> tuple[object | None, str | None]:
    """Call the LLM (or the cache), parse JSON, keep it only if is_valid(parsed) is True."""
    key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    cached = _cache().get(key)
    if cached:
        return cached["parsed"], cached["model"]
    for attempt in range(1, attempts + 1):
        try:
            reply = llm.call_llm(prompt)
            start = min([i for i in (reply.find("["), reply.find("{")) if i >= 0], default=-1)
            end = max(reply.rfind("]"), reply.rfind("}"))
            parsed = json.loads(reply[start:end + 1])
            if is_valid(parsed):
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with CACHE_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "parsed": parsed, "model": llm.LAST_MODEL},
                                       ensure_ascii=False) + "\n")
                return parsed, llm.LAST_MODEL
            print(f"    attempt {attempt}: reply did not pass validation")
        except (ValueError, RuntimeError) as error:
            print(f"    attempt {attempt}: {type(error).__name__}: {str(error)[:100]}")
    return None, None


# ---------------------------------------------------------------------------
# 1. Sub-issues
# ---------------------------------------------------------------------------
def sub_issues(theme: str, items: pd.DataFrame, models: set) -> tuple[list[dict], pd.Series]:
    """Name 2-4 sub-issues, assign every item, return [{name, count, pct_of_theme}] + mapping."""
    points = items["pain_point"].where(items["pain_point"].str.strip() != "", items["evidence_span"])
    points = points.fillna("").str.replace(r"\s+", " ", regex=True).str.slice(0, 160)

    sample = points.sample(n=min(NAMING_SAMPLE, len(points)), random_state=42)
    names, model = ask_json(
        SUBISSUE_PROMPT_V1.format(theme=theme, lines="\n".join(f"- {p}" for p in sample)),
        lambda x: isinstance(x, list) and 2 <= len(x) <= 4
                  and all(isinstance(s, str) and s.strip() and not re.search(r"\d", s) for s in x))
    if names is None:
        sys.exit(f"Could not get sub-issue names for {theme}; rerun later (cached parts are kept).")
    models.add(model)
    other = len(names) + 1
    options = "\n".join(f"{i}. {s}" for i, s in enumerate(names, 1)) + f"\n{other}. none of these"

    def assign(chunk: pd.Series, attempts: int = 3) -> list[int]:
        """Sub-issue number per item. If the model can't return a list of exactly the right
        length, split the chunk in half and try each half (short lists are easier to count)."""
        prompt = ASSIGN_PROMPT_V1.format(
            theme=theme, options=options, n=len(chunk), other=other,
            lines="\n".join(f"{i}. {p}" for i, p in enumerate(chunk, 1)))
        picks, model = ask_json(prompt, lambda x: isinstance(x, list) and len(x) == len(chunk)
                                and all(isinstance(v, int) and 1 <= v <= other for v in x),
                                attempts=attempts)
        if picks is not None:
            models.add(model)
            return picks
        if len(chunk) <= 10:
            sys.exit(f"Could not assign items for {theme}; rerun later (cached parts are kept).")
        half = len(chunk) // 2
        print(f"    splitting {len(chunk)} items into {half} + {len(chunk) - half}")
        return assign(chunk.iloc[:half], 2) + assign(chunk.iloc[half:], 2)

    labels = []
    for start in range(0, len(points), ASSIGN_CHUNK):
        # cached chunks return instantly; a new full chunk gets 2 tries before splitting
        picks = assign(points.iloc[start:start + ASSIGN_CHUNK], attempts=2)
        labels += [names[v - 1] if v <= len(names) else "other / unclear" for v in picks]

    mapping = pd.Series(labels, index=items["id"].values)
    counts = mapping.value_counts()
    result = [{"name": name, "count": int(counts.get(name, 0)),
               "pct_of_theme": round(100 * counts.get(name, 0) / len(items), 1)}
              for name in names + ["other / unclear"] if counts.get(name, 0) > 0]
    result.sort(key=lambda s: (s["name"] == "other / unclear", -s["count"]))
    return result, mapping


# ---------------------------------------------------------------------------
# 2. Quotes: verified spans only, mixing sources, languages and sub-issues
# ---------------------------------------------------------------------------
def pick_quotes(items: pd.DataFrame, mapping: pd.Series, texts: dict, urls: dict) -> list[dict]:
    pool = items[(items["quote_valid"] == True)  # noqa: E712 - passed the extract.py guard
                 & items["evidence_span"].str.len().between(25, 220)].copy()
    pool = pool[[span in texts[i] for i, span in zip(pool["id"], pool["evidence_span"])]]
    pool["sub_issue"] = pool["id"].map(mapping)
    pool = pool.sort_values("id")               # deterministic starting order

    chosen, sources, languages, subs = [], set(), set(), set()
    while len(chosen) < MAX_QUOTES and len(pool):
        # Greedy: prefer whatever adds a NEW source / language / sub-issue; then strong churn/severity.
        score = (2 * ~pool["source"].isin(sources) + 2 * ~pool["language"].isin(languages)
                 + 1 * ~pool["sub_issue"].isin(subs) + 0.5 * (pool["churn_signal"] == "Y")
                 + 0.1 * pd.to_numeric(pool["severity"], errors="coerce").fillna(0))
        row = pool.loc[score.idxmax()]
        chosen.append(row)
        sources.add(row["source"]); languages.add(row["language"]); subs.add(row["sub_issue"])
        pool = pool.drop(index=row.name)

    return [{"id": r["id"], "source": r["source"], "synthetic": r["source"] == "synthetic_ticket",
             "language": r["language"], "rating": r["rating"] or None, "date": r["date"].date().isoformat(),
             "sub_issue": r["sub_issue"], "quote": r["evidence_span"], "url": urls.get(r["id"]) or None}
            for r in chosen]


# ---------------------------------------------------------------------------
# 3. Facts (all numbers) and the sentence
# ---------------------------------------------------------------------------
def facts_for(theme: str, df: pd.DataFrame, weeks: list, subs: list, quotes: list) -> dict:
    items = df[df["theme"] == theme]
    wk = df["week"]
    prev_share = 100 * (items["week"] == weeks[0]).sum() / (wk == weeks[0]).sum()
    last_share = 100 * (items["week"] == weeks[1]).sum() / (wk == weeks[1]).sum()
    change = last_share - prev_share
    churn = int((items["churn_signal"] == "Y").sum())
    # cite real reviews first; synthetic tickets only if there aren't enough
    cite = sorted(quotes, key=lambda q: q["synthetic"])[:CITED_IDS]
    return {
        "n": len(items), "total": len(df), "pct": round(100 * len(items) / len(df), 1),
        "wk_prev_pct": round(prev_share, 1), "wk_last_pct": round(last_share, 1),
        "trend": "roughly flat" if abs(change) < 0.5 else ("rising" if change > 0 else "falling"),
        "top_sub": subs[0]["name"], "top_sub_n": subs[0]["count"],
        "churn_n": churn, "churn_pct": round(100 * churn / len(items), 1),
        "cited_ids": [q["id"] for q in cite],
    }


def fill(template: str, theme: str, f: dict) -> str:
    return template.format(
        theme=theme[0].upper() + theme[1:], pct=f"{f['pct']}% {PCT_LABEL}", n=f["n"], total=f["total"],
        trend=f["trend"], wk_prev=f"{f['wk_prev_pct']}%", wk_last=f"{f['wk_last_pct']}%",
        top_sub=f["top_sub"], top_sub_n=f["top_sub_n"], churn_n=f["churn_n"],
        churn_pct=f"{f['churn_pct']}%", ids="[" + ", ".join(f["cited_ids"]) + "]")


def template_ok(t) -> bool:
    """Every placeholder exactly once, nothing else in braces, and no digits from the LLM."""
    if not isinstance(t, str) or re.search(r"\d", t):
        return False
    found = re.findall(r"\{(\w+)\}", t)
    return sorted(found) == sorted(PLACEHOLDERS) and t.count("{") == len(PLACEHOLDERS)


# ---------------------------------------------------------------------------
# 4. Checks: recompute every number from scratch, validate every cited ID
# ---------------------------------------------------------------------------
def check(sentence: str, theme: str, quotes: list, raw: pd.DataFrame, texts: dict, weeks: list) -> dict:
    """Independent re-count from the raw joined data (not from facts_for), then compare."""
    problems = []
    items = raw[raw["theme"] == theme]
    allowed = {
        str(len(items)), str(len(raw)), f"{100 * len(items) / len(raw):.1f}",
        str(int((items["churn_signal"] == "Y").sum())),
        f"{100 * (items['churn_signal'] == 'Y').sum() / len(items):.1f}",
    }
    for w in weeks:
        allowed.add(f"{100 * (items['week'] == w).sum() / (raw['week'] == w).sum():.1f}")
    # sub-issue count: recount from the saved per-item mapping
    allowed |= {str(c) for c in items["sub_issue"].value_counts().tolist()}

    stripped = ID_PATTERN.sub("", sentence).replace("1–3★", "")
    for number in re.findall(r"\d+(?:\.\d+)?", stripped):
        if number not in allowed:
            problems.append(f"number {number} does not match the data")

    quote_by_id = {q["id"]: q for q in quotes}
    cited = ID_PATTERN.findall(sentence)
    if len(cited) < MIN_QUOTES:
        problems.append(f"only {len(cited)} IDs cited")
    for cid in cited:
        if cid not in texts:
            problems.append(f"{cid} not in feedback.csv")
        elif cid not in set(items["id"]):
            problems.append(f"{cid} is not in theme {theme}")
        elif cid not in quote_by_id or quote_by_id[cid]["quote"] not in texts[cid]:
            problems.append(f"{cid} has no exact-substring quote")
    for q in quotes:  # every listed quote, not just the cited ones
        if q["quote"] not in texts[q["id"]]:
            problems.append(f"quote for {q['id']} is not an exact substring")

    return {"count_accuracy": not any(p.startswith("number") for p in problems),
            "citation_validity": not any(not p.startswith("number") for p in problems),
            "problems": problems}


# ---------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = load(VERSION)                                   # complaint set + rating + date
    fb = pd.read_csv(FEEDBACK_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    texts, urls = dict(zip(fb["id"], fb["text"])), dict(zip(fb["id"], fb["url"]))
    df = df.merge(fb[["id", "language"]], on="id", how="left")

    # The two full weeks (same rule as cluster.py: weeks with 50+ items).
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    sizes = df["week"].value_counts()
    weeks = sorted(sizes[sizes >= 50].index)[-2:]

    themes = [t for t in df["theme"].value_counts().index if t not in SKIP_THEMES][:TOP_N_THEMES]
    models, results = set(), []
    df["sub_issue"] = None
    for rank, theme in enumerate(themes, 1):
        print(f"[{rank}/{len(themes)}] {theme}")
        items = df[df["theme"] == theme]
        subs, mapping = sub_issues(theme, items, models)
        df.loc[items.index, "sub_issue"] = items["id"].map(mapping).values
        quotes = pick_quotes(df[df["theme"] == theme], mapping, texts, urls)
        results.append({"rank": rank, "theme": theme, "sub_issues": subs, "quotes": quotes,
                        "facts": facts_for(theme, df, weeks, subs, quotes)})

    # One wording call for all themes; the LLM sees names and words only, no numbers.
    listing = "\n".join(f'- "{r["theme"]}" (trend: {r["facts"]["trend"]}; '
                        f'top sub-issue: {r["facts"]["top_sub"]})' for r in results)
    templates, model = ask_json(
        INSIGHT_PROMPT_V1.format(themes=listing),
        lambda x: isinstance(x, dict) and set(themes) <= set(x))
    if model:
        models.add(model)
    for r in results:
        t = (templates or {}).get(r["theme"])
        r["template_source"] = "llm" if template_ok(t) else "fallback"
        r["insight_template"] = t if template_ok(t) else FALLBACK_TEMPLATE
        r["insight"] = fill(r["insight_template"], r["theme"], r["facts"])
        r["checks"] = check(r["insight"], r["theme"], r["quotes"], df, texts, weeks)

    out = {
        "generated": date.today().isoformat(),
        "scope": "complaint set: 1-3★ Play Store + App Store reviews, plus synthetic tickets "
                 "(4-5★ reviews were not extracted). All shares are % of complaints (1–3★).",
        "extraction_prompt": VERSION,
        "insight_prompts": {"sub_issues": "SUBISSUE_PROMPT_V1", "assign": "ASSIGN_PROMPT_V1",
                            "wording": "INSIGHT_PROMPT_V1"},
        "model": sorted(m for m in models if m),
        "total_complaints": len(df),
        "weeks_compared": [f"{w:%Y-%m-%d}" for w in weeks],
        "themes": results,
    }
    OUTPUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in results:
        c = r["checks"]
        print(f"\n{r['rank']}. {r['insight']}")
        print(f"   sub-issues: " + "; ".join(f"{s['name']} ({s['count']})" for s in r["sub_issues"]))
        print(f"   wording: {r['template_source']} | count accuracy: {'PASS' if c['count_accuracy'] else 'FAIL'}"
              f" | citation validity: {'PASS' if c['citation_validity'] else 'FAIL'}"
              + (f" | {c['problems']}" if c["problems"] else ""))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
