"""Turn theme stats into PM-ready insights: sub-issues, judged quotes, one sentence each.

For each of the top 6 themes in the complaint set (1-3★ + synthetic tickets):
  1. Sub-issues: the LLM names 2-4 short sub-issues from the pain_point summaries, then
     assigns every item to one. COUNTS are computed in code from those assignments.
     If "other / unclear" holds over 25% of a theme, those items are re-split once with
     stricter instructions.
  2. Quotes: candidates are evidence spans that passed the extract.py quote guard (exact
     substrings). An LLM judge rates each candidate 1-5 for "does this support the theme's
     claim" and says which theme it is mainly about. We drop scores below 4 and quotes
     mainly about another theme, then pick the best 3-5, mixing sources and languages.
  3. Sentence: a FIXED template filled entirely in code, so no number comes from the LLM:
       "{Theme}: {n} complaints ({pct}% of complaints), share up/down/flat from {a}% to {b}%
        week-over-week. Top issue: {sub-issue} ({k}). {c} users ({r}%) say they'll leave or
        warn others. [IDs]"
     If the top sub-issues are within 2 of each other: "Top issues: A (k1), B (k2), C (k3)."
     The overall total (e.g. 982) appears once, in the dashboard header, not in each sentence.
  4. Checks: every number in the sentence is re-computed from the raw data (count accuracy);
     every cited ID must exist, belong to the theme and have an exact-substring quote
     (citation validity).
  5. Insight-quality eval: an LLM judge scores each sentence 1-5 for specificity, grounding
     and actionability -> evals/results/insights_<version>.json. (The full rubric lives in
     the project PRD; see README.)

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

from src import llm, refund_crosscheck
from src.cluster import load
from src.extract import THEME_NAMES

INSIGHTS_VERSION = "v2"        # version of this whole insight recipe (prompts + rules)
EXTRACT_VERSION = "v3"         # which extraction prompt's output we build on
TOP_N_THEMES = 6
ASSIGN_CHUNK = 50              # pain points per assignment call (long lists get miscounted)
NAMING_SAMPLE = 120            # pain points shown to the LLM when naming sub-issues
MAX_SUBISSUE_WORDS = 6
OTHER_RESPLIT_SHARE = 0.25     # re-split when "other / unclear" is above this share
TIE_GAP = 2                    # top sub-issues within this many items are reported together
JUDGE_CANDIDATES = 15          # quotes sent to the judge per round
MIN_QUOTE_SCORE = 4
MAX_QUOTES, MIN_QUOTES = 5, 3
CITED_IDS = 3                  # IDs cited inside the sentence itself
SKIP_THEMES = {"no complaint", "other"}
OTHER = "other / unclear"

CACHE_PATH = Path("data/processed/cache/insights_llm.jsonl")
OUTPUT_PATH = Path("data/processed/insights.json")
EVAL_PATH = Path(f"evals/results/insights_{INSIGHTS_VERSION}.json")
FEEDBACK_PATH = Path("data/processed/feedback.csv")
ID_PATTERN = re.compile(r"\b(?:PS|AS|SYN|RD)-\d{4}\b")

# ---------------------------------------------------------------------------
# Prompts (versioned; changing one = new version name)
# ---------------------------------------------------------------------------
SUBISSUE_PROMPT_V2 = """You are analysing customer complaints about Myntra, an Indian fashion
e-commerce app. Every complaint below was labelled with the theme "{theme}".
Each line is a short summary of one complaint.

Group them into 2 to 4 distinct, specific sub-issues a product manager could act on.
Rules:
- Each name is lowercase, at most 6 words, and contains NO digits.
- Be specific ("refund stuck after return pickup"), not generic ("refund problems").
- A sub-issue must be about "{theme}" itself. Do NOT create a sub-issue that is really one
  of these other themes: {other_themes}.
- Sub-issues must not overlap. Cover the most common problems first.
Return ONLY a JSON list of strings, like ["first sub-issue", "second sub-issue"].

Complaints:
{lines}
"""

RESPLIT_PROMPT_V1 = """These complaints (theme: "{theme}") did not fit any of these sub-issues:
{existing}

Find 1 to 3 NEW sub-issues that cover as many of them as possible.
Rules:
- Each name is lowercase, at most 6 words, NO digits, and names a concrete failure
  (what went wrong, at which step), e.g. "pickup rescheduled without agent visit".
- Must not repeat or overlap the existing sub-issues above.
- Must be about "{theme}" itself, not about: {other_themes}.
- Ignore complaints that are only angry with no concrete detail.
Return ONLY a JSON list of strings.

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

QUOTE_JUDGE_PROMPT_V1 = """You check evidence quotes for a product insight about Myntra.

Theme: "{theme}"
Claim: customers complain about {theme}, mainly: {subs}.

All theme names: {all_themes}

For each numbered quote, give:
- "score": 1-5, how well the quote ALONE supports the claim (5 = clear, specific evidence of
  this theme; 3 = related but vague or mostly about something else; 1 = does not support it)
- "main_theme": the ONE theme name (from the list) the quote is mainly about

Quotes:
{lines}

Return ONLY a JSON list with one object per quote, in order:
[{{"n": 1, "score": 4, "main_theme": "..."}}, ...]
"""

INSIGHT_JUDGE_PROMPT_V1 = """You are reviewing product insights written for a Myntra product manager.
Score each insight 1-5 on three criteria:
- "specificity": names a concrete problem with numbers, not a vague complaint (5 = precise)
- "grounding": the quotes clearly back up the sentence (5 = every part is supported)
- "actionability": a PM would know what to investigate or fix next (5 = obvious next step)
Also give a one-sentence "reason" mentioning the weakest point.

{insights}

Return ONLY a JSON object mapping each theme name exactly as given to
{{"specificity": n, "grounding": n, "actionability": n, "reason": "..."}}.
"""


# ---------------------------------------------------------------------------
# LLM with a disk cache keyed by the exact prompt
# ---------------------------------------------------------------------------
def _cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    rows = [json.loads(l) for l in CACHE_PATH.read_text(encoding="utf-8").splitlines() if l]
    return {r["key"]: r for r in rows}


MODELS_USED: set = set()


def ask_json(prompt: str, is_valid, attempts: int = 3):
    """Call the LLM (or the cache), parse JSON, keep it only if is_valid(parsed) is True."""
    key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    cached = _cache().get(key)
    if cached:
        MODELS_USED.add(cached["model"])
        return cached["parsed"]
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
                MODELS_USED.add(llm.LAST_MODEL)
                return parsed
            print(f"    attempt {attempt}: reply did not pass validation: "
                  f"{json.dumps(parsed, ensure_ascii=False)[:200]}")
        except (ValueError, RuntimeError) as error:
            print(f"    attempt {attempt}: {type(error).__name__}: {str(error)[:100]}")
    return None


def name_list(x, low: int, high: int) -> bool:
    """A JSON list of low..high non-empty strings without digits (length checked separately)."""
    return isinstance(x, list) and low <= len(x) <= high and all(
        isinstance(s, str) and s.strip() and not re.search(r"\d", s) for s in x)


def ask_names(prompt: str, low: int, high: int) -> list[str] | None:
    """Ask for sub-issue names. Names over 6 words are sent back once or twice to be
    shortened (the model often overshoots by one word), instead of being thrown away."""
    names = ask_json(prompt, lambda x: name_list(x, low, high))
    for _ in range(2):
        if names is None:
            return None
        too_long = [n for n in names if len(n.split()) > MAX_SUBISSUE_WORDS]
        if not too_long:
            return [n.strip().lower() for n in names]
        print(f"    shortening: {too_long}")
        names = ask_json(prompt + f"\nYour previous answer was {json.dumps(names)}. These names are "
                         f"longer than {MAX_SUBISSUE_WORDS} words: {json.dumps(too_long)}. Shorten "
                         f"them to at most {MAX_SUBISSUE_WORDS} words, keep the rest, and answer again.",
                         lambda x: name_list(x, low, high))
    if names and all(len(n.split()) <= MAX_SUBISSUE_WORDS for n in names):
        return [n.strip().lower() for n in names]
    return None


# ---------------------------------------------------------------------------
# 1. Sub-issues
# ---------------------------------------------------------------------------
def assign(theme: str, names: list[str], points: pd.Series, attempts: int = 2) -> list[str]:
    """Sub-issue name per item. If the model can't return a list of exactly the right
    length, split the chunk in half and try each half (short lists are easier to count)."""
    other_no = len(names) + 1
    options = "\n".join(f"{i}. {s}" for i, s in enumerate(names, 1)) + f"\n{other_no}. none of these"
    labels = []
    for start in range(0, len(points), ASSIGN_CHUNK):
        labels += _assign_chunk(theme, names, options, other_no,
                                points.iloc[start:start + ASSIGN_CHUNK], attempts)
    return labels


def _assign_chunk(theme, names, options, other_no, chunk, attempts) -> list[str]:
    prompt = ASSIGN_PROMPT_V1.format(theme=theme, options=options, n=len(chunk), other=other_no,
                                     lines="\n".join(f"{i}. {p}" for i, p in enumerate(chunk, 1)))
    picks = ask_json(prompt, lambda x: isinstance(x, list) and len(x) == len(chunk)
                     and all(isinstance(v, int) and 1 <= v <= other_no for v in x), attempts)
    if picks is not None:
        return [names[v - 1] if v <= len(names) else OTHER for v in picks]
    if len(chunk) <= 10:
        sys.exit(f"Could not assign items for {theme}; rerun later (cached parts are kept).")
    half = len(chunk) // 2
    print(f"    splitting {len(chunk)} items into {half} + {len(chunk) - half}")
    return (_assign_chunk(theme, names, options, other_no, chunk.iloc[:half], 2)
            + _assign_chunk(theme, names, options, other_no, chunk.iloc[half:], 2))


def sub_issues(theme: str, items: pd.DataFrame, other_themes: str) -> tuple[list[dict], pd.Series, bool]:
    """Name sub-issues, assign items, re-split 'other' if too big. Returns list, mapping, resplit?"""
    points = items["pain_point"].where(items["pain_point"].str.strip() != "", items["evidence_span"])
    points = points.fillna("").str.replace(r"\s+", " ", regex=True).str.slice(0, 160)
    points.index = items["id"].values

    sample = points.sample(n=min(NAMING_SAMPLE, len(points)), random_state=42)
    names = ask_names(SUBISSUE_PROMPT_V2.format(
        theme=theme, other_themes=other_themes, lines="\n".join(f"- {p}" for p in sample)), 2, 4)
    if names is None:
        sys.exit(f"Could not get sub-issue names for {theme}; rerun later (cached parts are kept).")
    mapping = pd.Series(assign(theme, names, points), index=points.index)

    resplit = False
    leftover = points[mapping == OTHER]
    if len(leftover) / len(points) > OTHER_RESPLIT_SHARE:
        print(f"    '{OTHER}' is {len(leftover)}/{len(points)} -> re-splitting")
        extra = ask_names(RESPLIT_PROMPT_V1.format(
            theme=theme, other_themes=other_themes, existing="\n".join(f"- {n}" for n in names),
            lines="\n".join(f"- {p}" for p in leftover)), 1, 3)
        if extra:
            extra = [n for n in extra if n not in names]
            mapping.loc[leftover.index] = assign(theme, extra, leftover)
            names += extra
            resplit = True

    counts = mapping.value_counts()
    result = [{"name": n, "count": int(counts.get(n, 0)),
               "pct_of_theme": round(100 * counts.get(n, 0) / len(items), 1)}
              for n in names + [OTHER] if counts.get(n, 0) > 0]
    result.sort(key=lambda s: (s["name"] == OTHER, -s["count"]))
    return result, mapping, resplit


# ---------------------------------------------------------------------------
# 2. Quotes: guard-verified spans, judged for relevance, mixed sources/languages
# ---------------------------------------------------------------------------
def _diverse_order(pool: pd.DataFrame, k: int, weight=None) -> list:
    """Greedy pick of k rows that adds new sources / languages / sub-issues first."""
    pool = pool.copy()
    chosen, sources, languages, subs = [], set(), set(), set()
    while len(chosen) < k and len(pool):
        score = (2 * ~pool["source"].isin(sources) + 2 * ~pool["language"].isin(languages)
                 + 1 * ~pool["sub_issue"].isin(subs) + 0.5 * (pool["churn_signal"] == "Y")
                 + 0.1 * pd.to_numeric(pool["severity"], errors="coerce").fillna(0))
        if weight is not None:
            score = score + pool["id"].map(weight)
        row = pool.loc[score.idxmax()]
        chosen.append(row)
        sources.add(row["source"]); languages.add(row["language"]); subs.add(row["sub_issue"])
        pool = pool.drop(index=row.name)
    return chosen


def pick_quotes(theme: str, items: pd.DataFrame, subs: list, texts: dict, urls: dict) -> tuple[list, list]:
    pool = items[(items["quote_valid"] == True)  # noqa: E712 - passed the extract.py guard
                 & items["evidence_span"].str.len().between(25, 220)].copy()
    pool = pool[[span in texts[i] for i, span in zip(pool["id"], pool["evidence_span"])]]
    pool = pool.sort_values("id")
    sub_names = ", ".join(s["name"] for s in subs if s["name"] != OTHER)
    all_themes = ", ".join(sorted(THEME_NAMES))

    judged, kept = [], {}
    while len(kept) < MIN_QUOTES and len(pool):
        batch = _diverse_order(pool, JUDGE_CANDIDATES)
        pool = pool.drop(index=[r.name for r in batch])
        lines = "\n".join(f'{i}. "{r["evidence_span"]}"' for i, r in enumerate(batch, 1))
        verdicts = ask_json(QUOTE_JUDGE_PROMPT_V1.format(
            theme=theme, subs=sub_names, all_themes=all_themes, lines=lines),
            lambda x: isinstance(x, list) and len(x) == len(batch) and all(
                isinstance(v, dict) and isinstance(v.get("score"), int) and 1 <= v["score"] <= 5
                and v.get("main_theme") in THEME_NAMES for v in x))
        if verdicts is None:
            sys.exit(f"Quote judge failed for {theme}; rerun later (cached parts are kept).")
        for r, v in zip(batch, verdicts):
            ok = v["score"] >= MIN_QUOTE_SCORE and v["main_theme"] == theme
            judged.append({"id": r["id"], "score": v["score"], "main_theme": v["main_theme"],
                           "kept": ok, "quote": r["evidence_span"]})
            if ok:
                kept[r["id"]] = r
        if len(batch) < JUDGE_CANDIDATES:
            break

    survivors = pd.DataFrame(kept.values())
    if survivors.empty:
        return [], judged
    score_of = {j["id"]: j["score"] for j in judged}
    # best first: judge score dominates, diversity breaks ties
    chosen = _diverse_order(survivors, MAX_QUOTES, weight={i: 3 * s for i, s in score_of.items()})
    quotes = [{"id": r["id"], "source": r["source"], "synthetic": r["source"] == "synthetic_ticket",
               "language": r["language"], "rating": r["rating"] or None,
               "date": r["date"].date().isoformat(), "sub_issue": r["sub_issue"],
               "judge_score": score_of[r["id"]], "quote": r["evidence_span"],
               "url": urls.get(r["id"]) or None} for r in chosen]
    return quotes, judged


# ---------------------------------------------------------------------------
# 3. Facts (every number) and the fixed-template sentence
# ---------------------------------------------------------------------------
def facts_for(theme: str, df: pd.DataFrame, weeks: list, subs: list, quotes: list) -> dict:
    items = df[df["theme"] == theme]
    prev = 100 * (items["week"] == weeks[0]).sum() / (df["week"] == weeks[0]).sum()
    last = 100 * (items["week"] == weeks[1]).sum() / (df["week"] == weeks[1]).sum()
    churn = int((items["churn_signal"] == "Y").sum())
    real = [s for s in subs if s["name"] != OTHER]
    top = [s for s in real if real[0]["count"] - s["count"] <= TIE_GAP]
    cite = sorted(quotes, key=lambda q: (q["synthetic"], -q["judge_score"]))[:CITED_IDS]
    return {
        "n": len(items), "pct": round(100 * len(items) / len(df), 1),
        "wk_prev_pct": round(prev, 1), "wk_last_pct": round(last, 1),
        "direction": "flat" if abs(last - prev) < 0.5 else ("up" if last > prev else "down"),
        "top_issues": [{"name": s["name"], "count": s["count"]} for s in top],
        "churn_n": churn, "churn_pct": round(100 * churn / len(items), 1),
        "cited_ids": [q["id"] for q in cite],
    }


def sentence(theme: str, f: dict) -> str:
    top = f["top_issues"]
    if len(top) == 1:
        top_part = f"Top issue: {top[0]['name']} ({top[0]['count']})."
    else:
        top_part = "Top issues: " + ", ".join(f"{t['name']} ({t['count']})" for t in top) + "."
    return (f"{theme[0].upper() + theme[1:]}: {f['n']} complaints ({f['pct']}% of complaints), "
            f"share {f['direction']} from {f['wk_prev_pct']}% to {f['wk_last_pct']}% week-over-week. "
            f"{top_part} {f['churn_n']} users ({f['churn_pct']}%) say they'll leave or warn others. "
            f"[{', '.join(f['cited_ids'])}]")


# ---------------------------------------------------------------------------
# 4. Checks: recompute every number from scratch, validate every cited ID
# ---------------------------------------------------------------------------
def check(text: str, theme: str, quotes: list, raw: pd.DataFrame, texts: dict, weeks: list) -> dict:
    """Independent re-count from the raw joined data (not from facts_for), then compare."""
    problems = []
    items = raw[raw["theme"] == theme]
    y = (items["churn_signal"] == "Y").sum()
    allowed = {str(len(items)), f"{100 * len(items) / len(raw):.1f}", str(int(y)),
               f"{100 * y / len(items):.1f}"}
    for w in weeks:
        allowed.add(f"{100 * (items['week'] == w).sum() / (raw['week'] == w).sum():.1f}")
    allowed |= {str(c) for c in items["sub_issue"].value_counts().tolist()}

    for number in re.findall(r"\d+(?:\.\d+)?", ID_PATTERN.sub("", text)):
        if number not in allowed:
            problems.append(f"number {number} does not match the data")

    by_id = {q["id"]: q for q in quotes}
    cited = ID_PATTERN.findall(text)
    if len(cited) < MIN_QUOTES:
        problems.append(f"only {len(cited)} IDs cited")
    for cid in cited:
        if cid not in texts:
            problems.append(f"{cid} not in feedback.csv")
        elif cid not in set(items["id"]):
            problems.append(f"{cid} is not in theme {theme}")
        elif cid not in by_id or by_id[cid]["quote"] not in texts[cid]:
            problems.append(f"{cid} has no exact-substring quote")
    for q in quotes:
        if q["quote"] not in texts[q["id"]]:
            problems.append(f"quote for {q['id']} is not an exact substring")
    return {"count_accuracy": not any(p.startswith("number") for p in problems),
            "citation_validity": not any(not p.startswith("number") for p in problems),
            "problems": problems}


# ---------------------------------------------------------------------------
# 5. Insight-quality judge
# ---------------------------------------------------------------------------
def judge_insights(results: list) -> dict | None:
    blocks = []
    for r in results:
        quotes = "\n".join(f'   - [{q["id"]}] "{q["quote"]}"' for q in r["quotes"])
        blocks.append(f'Theme "{r["theme"]}"\n  Insight: {r["insight"]}\n  Quotes:\n{quotes}')
    names = {r["theme"] for r in results}

    def valid(x):
        return isinstance(x, dict) and names <= set(x) and all(
            isinstance(x[t], dict) and all(isinstance(x[t].get(k), int) and 1 <= x[t][k] <= 5
                                           for k in ("specificity", "grounding", "actionability"))
            for t in names)
    return ask_json(INSIGHT_JUDGE_PROMPT_V1.format(insights="\n\n".join(blocks)), valid)


# ---------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = load(EXTRACT_VERSION)                           # complaint set + rating + date
    fb = pd.read_csv(FEEDBACK_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    texts, urls = dict(zip(fb["id"], fb["text"])), dict(zip(fb["id"], fb["url"]))
    df = df.merge(fb[["id", "language"]], on="id", how="left")

    # The two full weeks (same rule as cluster.py: weeks with 50+ items).
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    sizes = df["week"].value_counts()
    weeks = sorted(sizes[sizes >= 50].index)[-2:]

    themes = [t for t in df["theme"].value_counts().index if t not in SKIP_THEMES][:TOP_N_THEMES]
    df["sub_issue"] = None
    results = []
    for rank, theme in enumerate(themes, 1):
        print(f"[{rank}/{len(themes)}] {theme}")
        others = ", ".join(sorted(THEME_NAMES - {theme, "no complaint", "other"}))
        items = df[df["theme"] == theme]
        subs, mapping, resplit = sub_issues(theme, items, others)
        df.loc[items.index, "sub_issue"] = items["id"].map(mapping).values
        quotes, judged = pick_quotes(theme, df[df["theme"] == theme], subs, texts, urls)
        facts = facts_for(theme, df, weeks, subs, quotes)
        text = sentence(theme, facts)
        results.append({"rank": rank, "theme": theme, "insight": text, "facts": facts,
                        "sub_issues": subs, "other_resplit": resplit, "quotes": quotes,
                        "quote_judgements": judged,
                        "checks": check(text, theme, quotes, df, texts, weeks)})

    # Deterministic keyword cross-check for refund delay (no LLM), shown on the dashboard
    # next to the LLM sub-issues and labelled "keyword count".
    for r in results:
        if r["theme"] == refund_crosscheck.THEME:
            r["keyword_counts"] = refund_crosscheck.crosscheck()[1]

    total = len(df)
    header = (f"{total} complaints analysed: 1–3★ Play Store + App Store reviews and synthetic "
              f"support tickets, {df['date'].min():%d %b %Y} – {df['date'].max():%d %b %Y}. "
              f"Shares are % of these complaints; week-over-week compares the weeks of "
              f"{weeks[0]:%d %b} and {weeks[1]:%d %b}. 4–5★ reviews were not analysed.")
    out = {
        "insights_version": INSIGHTS_VERSION, "generated": date.today().isoformat(),
        "dashboard_header": header, "total_complaints": total,
        "extraction_prompt": EXTRACT_VERSION,
        "prompts": ["SUBISSUE_PROMPT_V2", "RESPLIT_PROMPT_V1", "ASSIGN_PROMPT_V1",
                    "QUOTE_JUDGE_PROMPT_V1"],
        "model": sorted(m for m in MODELS_USED if m),
        "weeks_compared": [f"{w:%Y-%m-%d}" for w in weeks],
        "themes": results,
    }
    OUTPUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Insight-quality eval ---
    scores = judge_insights(results)
    if scores is None:
        print("Insight judge failed; insights.json is saved, rerun to retry the judge.")
    else:
        rows = [{"theme": r["theme"], "insight": r["insight"],
                 **{k: scores[r["theme"]][k] for k in ("specificity", "grounding", "actionability")},
                 "reason": scores[r["theme"]].get("reason", "")} for r in results]
        means = {k: round(sum(x[k] for x in rows) / len(rows), 2)
                 for k in ("specificity", "grounding", "actionability")}
        EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_PATH.write_text(json.dumps({
            "insights_version": INSIGHTS_VERSION, "judge_prompt": "INSIGHT_JUDGE_PROMPT_V1",
            "model": sorted(m for m in MODELS_USED if m), "run_date": date.today().isoformat(),
            "scale": "1-5", "means": means, "scores": rows,
            "automatic_checks": {r["theme"]: r["checks"] for r in results},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{header}")
    for r in results:
        c = r["checks"]
        print(f"\n{r['rank']}. {r['insight']}")
        print("   sub-issues: " + "; ".join(f"{s['name']} ({s['count']})" for s in r["sub_issues"])
              + ("  [other re-split]" if r["other_resplit"] else ""))
        dropped = [f"{j['id']}({j['score']},{j['main_theme']})" for j in r["quote_judgements"] if not j["kept"]]
        print(f"   quotes kept {len(r['quotes'])}, dropped {len(dropped)}: {', '.join(dropped)}")
        print(f"   count accuracy: {'PASS' if c['count_accuracy'] else 'FAIL'} | citation validity: "
              f"{'PASS' if c['citation_validity'] else 'FAIL'}" + (f" | {c['problems']}" if c["problems"] else ""))
        if scores:
            s = scores[r["theme"]]
            print(f"   judge: specificity {s['specificity']}, grounding {s['grounding']}, "
                  f"actionability {s['actionability']} - {s.get('reason', '')}")
    print(f"\nSaved {OUTPUT_PATH}" + (f" and {EVAL_PATH}" if scores else ""))


if __name__ == "__main__":
    main()
