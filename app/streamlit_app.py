"""Myntra Voice-of-Customer Copilot: Streamlit dashboard.

Reads precomputed results only (no LLM calls), except in the "Try it" tab:
  data/processed/feedback.csv                    cleaned feedback (rating, date, url, language)
  data/processed/extractions_v3_complaints.jsonl per-item theme, churn, verified quote
  data/processed/insights.json                   top-6 insights, sub-issues, judged quotes
  data/roadmap.yaml                              RICE recommendations (written by the PM)
  evals/results/*.json                           eval scores (golden vs holdout, v1-v3)
Person names in displayed text are masked with [name] (src/mask.py).

Run from the project root:   streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                      # src/ modules use paths relative to the project root
sys.path.insert(0, str(ROOT))

from src.mask import mask_names  # noqa: E402

PROCESSED = Path("data/processed")
ROADMAP_PATH = Path("data/roadmap.yaml")
RESULTS_DIR = Path("evals/results")
TRY_IT_LIMIT = 20                   # cap so the free LLM quota lasts
SOURCE_NAMES = {"playstore": "Play Store", "appstore": "App Store", "synthetic_ticket": "SYN (synthetic)"}
PLAYSTORE_PUBLIC_RATING = 4.59      # Myntra's Play Store listing rating when scraped (Nov 2026)
FLAT_PTS = 0.5                      # share changes smaller than this are shown as flat

st.set_page_config(page_title="Myntra VoC Copilot", page_icon="🛍️", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached: read once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_feedback() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "feedback.csv", dtype=str, keep_default_na=False,
                       encoding="utf-8-sig")


@st.cache_data
def load_complaints() -> pd.DataFrame:
    """Extractions for the complaint set joined with feedback metadata, plus a week label."""
    rows = [json.loads(l) for l in (PROCESSED / "extractions_v3_complaints.jsonl")
            .read_text(encoding="utf-8").splitlines() if l]
    df = pd.DataFrame(rows).merge(
        load_feedback()[["id", "source", "language", "rating", "date", "url", "text"]], on="id")
    df["date"] = pd.to_datetime(df["date"])
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    # Weeks with 50+ items get their own label; older sparse items share one bucket
    start = df["date"].dt.to_period("W-SUN").dt.start_time
    sizes = start.value_counts()
    full = sorted(sizes[sizes >= 50].index)
    df["week_start"] = start
    df["week"] = start.map(lambda s: f"{s:%d}–{s + pd.Timedelta(days=6):%d %b}" if s in full
                           else f"before {full[0]:%d %b}")
    return df


@st.cache_data
def load_insights() -> dict:
    return json.loads((PROCESSED / "insights.json").read_text(encoding="utf-8"))


def full_weeks(df: pd.DataFrame) -> list:
    sizes = df["week_start"].value_counts()
    return sorted(sizes[sizes >= 50].index)


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------
def source_label(source: str) -> str:
    return SOURCE_NAMES.get(source, source)


def quote_card(qid, source, language, quote, url=None, rating=None, date=None, extra=""):
    """One quote with its metadata. Synthetic tickets are always labelled."""
    meta = [f"**{qid}**", source_label(source), language]
    if rating:
        meta.append(f"{rating}★")
    if date:
        meta.append(str(date)[:10])
    if url:
        # Play Store links point at the review; Apple's feed gives only the app's reviews page
        label = "App Store reviews page (no per-review link)" if source == "appstore" else "open review"
        meta.append(f"[{label}]({url})")
    if extra:
        meta.append(extra)
    badge = " :orange-badge[SYNTHETIC]" if source == "synthetic_ticket" else ""
    st.markdown(f"> {mask_names(quote)}\n\n" + " · ".join(meta) + badge)


def arrow(change: float) -> str:
    if pd.isna(change):
        return "–"
    if abs(change) < FLAT_PTS:
        return f"▶ flat ({change:+.1f} pts)"
    return f"{'▲' if change > 0 else '▼'} {change:+.1f} pts"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
complaints = load_complaints()
insights = load_insights()
insight_by_theme = {t["theme"]: t for t in insights["themes"]}
feedback = load_feedback()

st.title("Myntra Voice-of-Customer Copilot")
st.markdown(insights["dashboard_header"])
ps_mean = pd.to_numeric(feedback.loc[feedback["source"] == "playstore", "rating"]).mean()
st.warning(
    "**Read with care.**  \n"
    "• **Sample is 1–3★ only**: 4–5★ reviews were not analysed, so complaints inside "
    "high-star reviews are missed and shares are % of complaints, not of all reviews.  \n"
    f"• **The cleaned Play Store mean ({ps_mean:.2f}★) is not Myntra's real rating** "
    f"({PLAYSTORE_PUBLIC_RATING}★ on the store): short praise reviews were dropped in cleaning.  \n"
    "• **About 2 weeks of data** (older items are a few Hindi reviews), so week-over-week "
    "changes are directional, not trends.  \n"
    "• **Synthetic support tickets** (IDs `SYN-…`) are invented examples and are always "
    "labelled SYNTHETIC.", icon="⚠️")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    sources = sorted(complaints["source"].unique(), key=list(SOURCE_NAMES).index)
    pick_sources = st.multiselect("Source", sources, default=sources, format_func=source_label)
    languages = sorted(complaints["language"].unique())
    pick_languages = st.multiselect("Language", languages, default=languages)
    weeks = list(dict.fromkeys(complaints.sort_values("week_start")["week"]))
    pick_weeks = st.multiselect("Week", weeks, default=weeks)
    themes_all = complaints["theme"].value_counts().index.tolist()
    pick_themes = st.multiselect("Theme", themes_all, default=themes_all)
    st.caption("Insight sentences, sub-issues and judged quotes were computed on all "
               f"{insights['total_complaints']} complaints; the filters change the table, "
               "chart and churn alerts.")

base = complaints[complaints["source"].isin(pick_sources)
                  & complaints["language"].isin(pick_languages)]
filtered = base[base["week"].isin(pick_weeks)]          # denominator for % (all themes)
shown = filtered[filtered["theme"].isin(pick_themes)]

tab_themes, tab_churn, tab_roadmap, tab_how, tab_try = st.tabs(
    ["Themes", "Churn alerts", "Roadmap", "How it works / Evals", "Try it"])

# ---------------------------------------------------------------------------
# Tab 1: Themes
# ---------------------------------------------------------------------------
with tab_themes:
    if shown.empty:
        st.info("No complaints match the filters.")
    else:
        # Week-over-week share change uses the two full weeks (ignores the Week filter).
        w1, w2 = full_weeks(complaints)[-2:]
        share = {w: base[base["week_start"] == w]["theme"].value_counts(normalize=True) * 100
                 for w in (w1, w2)}
        g = filtered.groupby("theme")
        table = pd.DataFrame({
            "complaints": g.size(),
            "% of complaints": (100 * g.size() / len(filtered)).round(1),
            "churn rate %": (100 * g["churn_signal"].apply(lambda s: (s == "Y").mean())).round(1),
        })
        change = share[w2].reindex(table.index).fillna(0) - share[w1].reindex(table.index).fillna(0)
        table["share change WoW"] = change.map(arrow)
        table = table[table.index.isin(pick_themes)]

        sort_churn = st.toggle("Sort by churn rate", value=False)
        table = table.sort_values("churn rate %" if sort_churn else "complaints", ascending=False)

        st.subheader(f"Themes: {len(shown)} of {len(filtered)} complaints shown")
        chart = alt.Chart(table.reset_index()).mark_bar().encode(
            x=alt.X("% of complaints:Q", title="% of complaints (1–3★, filtered)"),
            y=alt.Y("theme:N", sort="-x", title=None),
            tooltip=["theme", "complaints", "% of complaints", "churn rate %"],
        ).properties(height=28 * len(table) + 20)
        st.altair_chart(chart, use_container_width=True)

        st.caption(f"Click a row for details. Share change compares the weeks of {w1:%d %b} "
                   f"and {w2:%d %b} (% of that week's complaints, source/language filters applied).")
        event = st.dataframe(table, use_container_width=True, on_select="rerun",
                             # selection is by row position, so reset it when the sort changes
                             selection_mode="single-row", key=f"theme_table_{sort_churn}",
                             height=35 * (len(table) + 1) + 3)        # all rows, no inner scroll
        rows = event.selection.rows if event and event.selection else []
        # No click yet: use ?theme=... from the URL if given (shareable link), else the top row
        linked = st.query_params.get("theme")
        theme = (table.index[rows[0]] if rows
                 else linked if linked in table.index else table.index[0])
        if not rows:
            st.caption(f"Showing **{theme}**. Click a row to switch.")

        # ---- Theme detail ----
        st.divider()
        st.subheader(f"{theme[0].upper() + theme[1:]}")
        info = insight_by_theme.get(theme)
        if info:
            st.info(info["insight"])
            sub = pd.DataFrame(info["sub_issues"]).rename(columns={
                "name": "sub-issue", "count": "complaints", "pct_of_theme": "% of theme"})
            if "keyword_counts" in info:
                left, right = st.columns(2)
                with left:
                    st.markdown("**Sub-issues (LLM)**")
                    st.dataframe(sub, hide_index=True, use_container_width=True)
                with right:
                    kc = info["keyword_counts"]
                    st.markdown("**keyword count (cross-check)**")
                    short = {"cancel_or_undelivered": "cancel / not delivered",
                             "return_or_pickup": "return / pickup",
                             "credited_not_received": "says credited, not received"}
                    st.dataframe(pd.DataFrame([{"mentions": short.get(c["rule"], c["label"]),
                                                "keyword count": c["keyword_count"],
                                                "e.g.": ", ".join(c["example_ids"])}
                                               for c in kc["counts"]]
                                              + [{"mentions": "none of these",
                                                  "keyword count": kc["none_of_these"], "e.g.": ""}]),
                                 hide_index=True, use_container_width=True)
                    st.caption("Regex rules in src/refund_crosscheck.py, no LLM. One review can "
                               f"match several rules, so counts overlap and don't sum to {kc['theme_items']}.")
            else:
                st.markdown("**Sub-issues (LLM)**")
                st.dataframe(sub, hide_index=True, use_container_width=True)

            st.markdown(f"**Verified quotes** ({len(info['quotes'])}; exact substrings of the source, "
                        "rated 4–5/5 for relevance by the quote judge)")
            for q in info["quotes"]:
                quote_card(q["id"], q["source"], q["language"], q["quote"], q.get("url"),
                           q.get("rating"), q.get("date"), extra=f"judge {q['judge_score']}/5")
        else:
            st.caption("No insight sentence for this theme (insights cover the top 6 themes). "
                       "Quotes below are guard-verified but not reviewed by the quote judge.")
            pool = shown[(shown["theme"] == theme) & (shown["quote_valid"] == True)]  # noqa: E712
            pool = pool.sort_values(["churn_signal", "severity"], ascending=False)
            # one quote per source first (mix), then fill up to 5
            picks = pd.concat([pool.drop_duplicates("source"), pool]).drop_duplicates("id").head(5)
            for r in picks.itertuples():
                quote_card(r.id, r.source, r.language, r.evidence_span, r.url or None,
                           r.rating or None, r.date.date())

# ---------------------------------------------------------------------------
# Tab 2: Churn alerts
# ---------------------------------------------------------------------------
with tab_churn:
    churn = shown[shown["churn_signal"] == "Y"]
    st.subheader(f"Churn alerts: {len(churn)} complaints say they'll leave or warn others off")
    st.caption("Churn = states they will stop using Myntra, tells others not to use it, or "
               "recommends a competitor (anger alone doesn't count). Filters apply.")
    for theme, group in sorted(churn.groupby("theme"), key=lambda x: -len(x[1])):
        with st.expander(f"{theme}: {len(group)}", expanded=len(group) >= 15):
            for r in group.sort_values("date", ascending=False).itertuples():
                if isinstance(r.evidence_span, str) and r.quote_valid:
                    quote_card(r.id, r.source, r.language, r.evidence_span, r.url or None,
                               r.rating or None, r.date.date())
                else:
                    st.markdown(f"*No verified quote. LLM summary:* {mask_names(r.pain_point)}  \n"
                                f"**{r.id}** · {source_label(r.source)} · {r.language}")

# ---------------------------------------------------------------------------
# Tab 3: Roadmap
# ---------------------------------------------------------------------------
with tab_roadmap:
    st.subheader("Roadmap (RICE)")
    if not ROADMAP_PATH.exists():
        st.info(f"`{ROADMAP_PATH}` not found yet. Add it and refresh. Expected shape:")
        st.code("""meta:
  status: draft
  caveat: Effort values are PM estimates.
recommendations:
  - id: R1
    title: Refund tracker for every refund
    description: What to build.
    themes: [refund delay, order cancelled]
    reach: 118         # complaints the fix addresses
    impact: 3          # 1 (low) .. 3 (high)
    confidence: 0.8    # 0-1
    effort: 4          # person-weeks
    evidence:          # claims (IDs like PS-0154 inside a claim get their quote shown)
      - Refund delay has the highest churn rate (19.6%).
    key_metric: Refund-related complaints
watch:                 # same fields, plus why_watch
not_building:
  - title: M-Now-specific fixes
    reason: Why not now.""", language="yaml")
    else:
        import re
        import yaml
        roadmap = yaml.safe_load(ROADMAP_PATH.read_text(encoding="utf-8")) or {}
        meta = roadmap.get("meta", {})
        known_themes = set(complaints["theme"])
        by_id = complaints.set_index("id")
        id_pattern = re.compile(r"\b(?:PS|AS|SYN|RD)-\d{4}\b")

        def with_rice(items: list) -> list:
            """RICE = Reach x Impact x Confidence / Effort, computed here (not read from the file)."""
            for r in items:
                numbers = all(isinstance(r.get(k), (int, float))
                              for k in ("reach", "impact", "confidence", "effort"))
                if numbers and r["effort"] > 0:
                    r["rice_score"] = round(r["reach"] * r["impact"] * r["confidence"] / r["effort"], 1)
                r["themes"] = r.get("themes") or ([r["theme"]] if r.get("theme") else [])
            return items

        def rice_table(items: list) -> None:
            rows = [{"id": r.get("id", ""), "title": r.get("title", ""), "themes": ", ".join(r["themes"]),
                     "reach": r.get("reach"), "impact": r.get("impact"),
                     "confidence": r.get("confidence"), "effort (wks)": r.get("effort"),
                     "RICE score": r.get("rice_score")} for r in items]
            st.dataframe(pd.DataFrame(rows).sort_values("RICE score", ascending=False),
                         hide_index=True, use_container_width=True)

        def details(r: dict) -> None:
            """Description, evidence claims, cited IDs (checked), quotes for the themes, metric."""
            if r.get("description"):
                st.markdown(r["description"])
            if r.get("reach_note"):
                st.caption(f"Reach: {r['reach_note']}")
            for t in r["themes"]:
                if t not in known_themes:
                    st.markdown(f":red[Theme **{t}** is not a theme in the data]")
            st.markdown("**Evidence**")
            for claim in r.get("evidence", []):
                st.markdown(f"- {claim}")
                for qid in id_pattern.findall(str(claim)):      # IDs cited inside a claim
                    if qid in by_id.index and isinstance(by_id.at[qid, "evidence_span"], str):
                        x = by_id.loc[qid]
                        quote_card(qid, x["source"], x["language"], x["evidence_span"],
                                   x["url"] or None, x["rating"] or None, x["date"].date())
                    else:
                        st.markdown(f":red[**{qid}**: not found in the complaint set "
                                    "or has no verified quote]")
            quotes = [q for t in r["themes"] if t in insight_by_theme
                      for q in insight_by_theme[t]["quotes"][:2]]
            if quotes:
                st.markdown("**What customers say** (verified quotes from the theme insights)")
                for q in quotes:
                    quote_card(q["id"], q["source"], q["language"], q["quote"], q.get("url"),
                               q.get("rating"), q.get("date"))
            if r.get("why_watch"):
                st.markdown(f"**Why watch, not build:** {r['why_watch']}")
            if r.get("key_metric"):
                st.markdown(f"**Key metric:** {r['key_metric']}")

        if meta:
            st.info(" · ".join(f"**{k}:** {meta[k]}" for k in ("status", "updated", "scope")
                               if meta.get(k))
                    + (f"  \n⚠️ {meta['caveat']}" if meta.get("caveat") else ""))
        st.caption("RICE = Reach × Impact × Confidence ÷ Effort, computed by the dashboard.")

        recs = with_rice(roadmap.get("recommendations", []))
        if recs:
            st.markdown("### Recommendations")
            rice_table(recs)
            for r in sorted(recs, key=lambda r: -(r.get("rice_score") or 0)):
                with st.expander(f"{r.get('id', '')} {r.get('title', '')}: RICE {r.get('rice_score')}"):
                    details(r)

        watch = with_rice(roadmap.get("watch", []))
        if watch:
            st.markdown("### Watch list")
            rice_table(watch)
            for r in watch:
                with st.expander(f"{r.get('id', '')} {r.get('title', '')}: RICE {r.get('rice_score')}"):
                    details(r)

        st.markdown("### Not building (and why)")
        nb = roadmap.get("not_building", [])
        if nb:
            st.dataframe(pd.DataFrame(nb), hide_index=True, use_container_width=True)
        else:
            st.caption("None listed.")

# ---------------------------------------------------------------------------
# Tab 4: How it works / Evals
# ---------------------------------------------------------------------------
with tab_how:
    st.subheader("Pipeline")
    st.code("""Play Store (5,000) ─┐
App Store (500)  ───┼─> clean.py ──> feedback.csv (3,039: dedupe, <3 words dropped, language)
Synthetic (20)   ───┘                      │
                                           ▼
             extract.py (LLM, PROMPT_V3, batches of 20, cached)
             theme · churn · severity · evidence_span ──> quote guard (exact substring or null)
                                           │   1–3★ + synthetic only (982 complaints)
                                           ▼
             cluster.py ──> counts, % of complaints, weekly share
                                           ▼
             insights.py ──> sub-issues (LLM names, code counts) · quote judge (≥4/5)
                             fixed-template sentence (numbers from code) · count + citation checks
                             refund_crosscheck.py (regex keyword counts, no LLM)
                                           ▼
             this dashboard (no LLM calls except "Try it")""", language=None)

    st.subheader("Extraction evals: golden (100, tuned on) vs holdout (30, never seen)")
    rows = []
    for v in ("v1", "v2", "v3"):
        for label_set, name in (("golden", f"{v}.json"), ("holdout", f"{v}_holdout.json")):
            path = RESULTS_DIR / name
            if path.exists():
                m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
                rows.append({"prompt": v, "set": label_set,
                             "theme agreement %": m["theme_agreement_pct"],
                             "churn precision %": m["churn_precision_pct"],
                             "churn recall %": m["churn_recall_pct"],
                             "severity ±1 %": m["severity_within_1_pct"],
                             "quote fidelity %": m["quote_fidelity_pct"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Holdout is the honest number: the prompts were improved by looking at golden-set "
               "misses. Churn has few positives (14 golden, 5 holdout), so those % move in big steps.")

    judge_path = RESULTS_DIR / "insights_v2.json"
    if judge_path.exists():
        j = json.loads(judge_path.read_text(encoding="utf-8"))
        st.subheader("Insight evals")
        checks = j["automatic_checks"]
        st.markdown(f"Automatic checks: count accuracy "
                    f"{sum(c['count_accuracy'] for c in checks.values())}/{len(checks)}, citation "
                    f"validity {sum(c['citation_validity'] for c in checks.values())}/{len(checks)}. "
                    f"LLM judge (1–5): specificity {j['means']['specificity']}, grounding "
                    f"{j['means']['grounding']}, actionability {j['means']['actionability']}.")
        st.dataframe(pd.DataFrame(j["scores"])[["theme", "specificity", "grounding",
                                                "actionability", "reason"]],
                     hide_index=True, use_container_width=True)

    st.subheader("Limitations")
    readme = Path("README.md").read_text(encoding="utf-8")
    section = readme.split("## Limitations", 1)[-1].split("\n## ", 1)[0]
    st.markdown(section)

# ---------------------------------------------------------------------------
# Tab 5: Try it (the only place that calls the LLM)
# ---------------------------------------------------------------------------
with tab_try:
    st.subheader("Try it: run the v3 extraction on your own reviews")
    st.caption(f"One review per line, up to {TRY_IT_LIMIT}. This calls the free-tier LLM "
               "(Groq), so it can take a minute, longer if the daily limit is reached. "
               "Nothing is saved.")
    pasted = st.text_area("Reviews", height=200, placeholder=(
        "Refund not received 2 weeks after pickup, never ordering from Myntra again\n"
        "Delivery boy marked it delivered but I never got the parcel"))
    lines = [l.strip() for l in pasted.splitlines() if l.strip()]
    if len(lines) > TRY_IT_LIMIT:
        st.warning(f"{len(lines)} reviews pasted; only the first {TRY_IT_LIMIT} will be used.")
        lines = lines[:TRY_IT_LIMIT]

    if st.button(f"Extract {len(lines)} review(s)", disabled=not lines, type="primary"):
        from src import extract   # imported here so other tabs never touch the LLM code
        batch = pd.DataFrame({"id": [f"TRY-{i:04d}" for i in range(1, len(lines) + 1)], "text": lines})
        with st.spinner("Calling the LLM..."):
            try:
                records = extract.extract_batch("v3", batch)
            except Exception as error:      # show the problem instead of crashing the page
                st.error(f"Extraction failed: {type(error).__name__}: {error}")
                records = []
        text_by_id = dict(zip(batch["id"], batch["text"]))
        st.session_state["try_results"] = [extract.guard_quote(r, text_by_id[r["id"]]) for r in records]
        st.session_state["try_missing"] = len(batch) - len(records)

    results = st.session_state.get("try_results")
    if results:
        res = pd.DataFrame(results)
        if st.session_state.get("try_missing"):
            st.warning(f"{st.session_state['try_missing']} review(s) got no answer; try again.")
        c1, c2 = st.columns(2)
        c1.metric("Reviews extracted", len(res))
        c2.metric("Churn signals", int((res["churn_signal"] == "Y").sum()))
        st.markdown("**Themes**")
        st.dataframe(res["theme"].value_counts().rename("reviews"), use_container_width=True)
        st.markdown("**Per review**")
        st.dataframe(pd.DataFrame({
            "id": res["id"], "theme": res["theme"], "churn": res["churn_signal"],
            "severity": res["severity"], "pain point": res["pain_point"].map(mask_names),
            "quote (verified)": res["evidence_span"].map(lambda q: mask_names(q) if q else
                                                         "— rejected by the quote guard"),
        }), hide_index=True, use_container_width=True)
