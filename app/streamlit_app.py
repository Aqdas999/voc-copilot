"""Myntra Voice-of-Customer Copilot: Streamlit dashboard.

Reads precomputed results only (no LLM calls), except in the "Try it" tab:
  data/processed/feedback.csv                    cleaned feedback (rating, date, url, language)
  data/processed/extractions_v3_complaints.jsonl per-item theme, churn, verified quote
  data/processed/insights.json                   top-6 insights, sub-issues, judged quotes
  data/roadmap.yaml                              RICE recommendations (written by the PM)
  evals/results/*.json                           eval scores (golden vs holdout, v1-v3)
Person names in displayed text are masked with [name] (src/mask.py).
Every number on the page is computed here from those files; nothing is typed in by hand.

Run from the project root:   streamlit run app/streamlit_app.py
"""

import json
import os
import re
import sys
from functools import partial
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                      # src/ modules use paths relative to the project root
sys.path.insert(0, str(ROOT))

from src.mask import mask_names  # noqa: E402

PROCESSED = Path("data/processed")
ROADMAP_PATH = Path("data/roadmap.yaml")
RESULTS_DIR = Path("evals/results")
TRY_IT_LIMIT = 20                   # cap so the free LLM quota lasts
TRY_IT_RUNS_PER_SESSION = 3         # LLM runs allowed per browser session (public demo)
SOURCE_NAMES = {"playstore": "Play Store", "appstore": "App Store",
                "synthetic_ticket": "Synthetic tickets"}
PLAYSTORE_PUBLIC_RATING = 4.59      # Myntra's Play Store listing rating when scraped (Nov 2026)
FLAT_PTS = 0.5                      # share changes smaller than this are shown as flat
MIN_THEME_SIZE = 20                 # headline "where customers leave" ignores tiny themes

# "no complaint" is a label the AI gives to 1-3★ reviews that describe no actual problem.
# Renamed for display only; the data files keep the original label.
NOT_COMPLAINT = "not a complaint (no issue described)"
UNRANKED = ("other", NOT_COMPLAINT)  # catch-all buckets: left out of the churn ranking

# Phrases where a customer says they will leave or warns others off (en / Hinglish / Hindi).
# Used only to pick WHICH sentence of a review to show; the churn flag itself comes from the AI.
LEAVING = re.compile(
    r"uninstall|un-install|never (?:again|order|buy|use|shop|purchase)|won'?t (?:order|buy|use|shop)"
    r"|will not (?:order|buy|use|shop)|not (?:going to|gonna) (?:order|buy|use|shop|trust)"
    r"|(?:don'?t|do not|never) (?:buy|order|use|shop|purchase|download)"
    r"|switch(?:ing|ed)? to|stop(?:ped)? using|delet(?:e|ed|ing) (?:the |this |my )?(?:app|account)"
    r"|better to order from|order from (?:some )?other|go (?:for|to) (?:amazon|flipkart|ajio|meesho|nykaa)"
    r"|i quit|done with myntra|last order|leaving myntra"
    r"|kabhi (?:bhi )?(?:nahi|nhi|na)|dobara (?:nahi|nhi)|mat (?:lena|lo|kharido|karo|order|mangao|mangwao)"
    r"|uninstall kar|delete kar|कभी नहीं|कभी भी नहीं|दोबारा नहीं|मत (?:खरीद|लेना|मंगा)|अनइंस्टॉल|डिलीट",
    re.IGNORECASE)

st.set_page_config(page_title="Myntra VoC Copilot", page_icon="🛍️", layout="wide")


# ---------------------------------------------------------------------------
# API keys: Streamlit Cloud secrets first, then the local .env file
# ---------------------------------------------------------------------------
def read_secret(name: str) -> str | None:
    """Value from st.secrets (Streamlit Community Cloud), else from .env / the environment."""
    try:
        value = st.secrets.get(name)
    except Exception:               # no secrets.toml at all (normal when running locally)
        value = None
    return str(value) if value else os.getenv(name)


from dotenv import load_dotenv  # noqa: E402

load_dotenv()
# src/llm.py reads keys with os.getenv, so copy any secrets into the environment
# before it is imported (only the Try it tab imports it).
for _name in ("GROQ_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
    _value = read_secret(_name)
    if _value:
        os.environ[_name] = _value
HAS_LLM_KEY = bool(os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY"))


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
    df["theme"] = df["theme"].replace({"no complaint": NOT_COMPLAINT})
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


def load_roadmap() -> dict:
    if not ROADMAP_PATH.exists():
        return {}
    return yaml.safe_load(ROADMAP_PATH.read_text(encoding="utf-8")) or {}


def with_rice(items: list) -> list:
    """RICE = Reach x Impact x Confidence / Effort, computed here (not read from the file)."""
    for r in items:
        numbers = all(isinstance(r.get(k), (int, float))
                      for k in ("reach", "impact", "confidence", "effort"))
        if numbers and r["effort"] > 0:
            r["rice_score"] = round(r["reach"] * r["impact"] * r["confidence"] / r["effort"], 1)
        r["themes"] = r.get("themes") or ([r["theme"]] if r.get("theme") else [])
    return items


def full_weeks(df: pd.DataFrame) -> list:
    sizes = df["week_start"].value_counts()
    return sorted(sizes[sizes >= 50].index)


def churn_pct(s: pd.Series) -> float:
    return 100 * (s == "Y").mean()


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


def card(label: str, value: str, note: str) -> None:
    """A bordered headline card: small label, big value, one-line note."""
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"#### {value}")
        st.caption(note)


def arrow(change: float) -> str:
    if pd.isna(change):
        return "–"
    if abs(change) < FLAT_PTS:
        return f"▶ flat ({change:+.1f} pts)"
    return f"{'▲' if change > 0 else '▼'} {change:+.1f} pts"


def leaving_sentence(text: str) -> str | None:
    """The first sentence of the review that says they'll leave, or None.

    Sentences are cut from the original text, so the result is still an exact substring.
    """
    for sentence in re.split(r"(?<=[.!?।])\s+|\n+", text or ""):
        sentence = sentence.strip()
        if sentence and LEAVING.search(sentence):
            return sentence
    return None


def ordered_themes(table: pd.DataFrame, by: str) -> list:
    """Theme order for the chart and table: catch-all buckets always go last.

    by="complaints": biggest first, 'not a complaint' last.
    by="churn rate %": highest churn first, then 'other' and 'not a complaint'.
    """
    last = [t for t in (UNRANKED if by == "churn rate %" else (NOT_COMPLAINT,)) if t in table.index]
    ranked = table.drop(last).sort_values(by, ascending=False).index.tolist()
    return ranked + last


# ---------------------------------------------------------------------------
# Data + headline numbers (all unfiltered, computed from the files)
# ---------------------------------------------------------------------------
complaints = load_complaints()
insights = load_insights()
insight_by_theme = {t["theme"]: t for t in insights["themes"]}
feedback = load_feedback()
roadmap = load_roadmap()
recs = with_rice(roadmap.get("recommendations", []))

overall_churn = churn_pct(complaints["churn_signal"])
by_theme = complaints.groupby("theme").agg(n=("id", "size"), churn=("churn_signal", churn_pct))
named = by_theme.drop(list(UNRANKED), errors="ignore")
top_volume = named["n"].idxmax()
top_churn = named[named["n"] >= MIN_THEME_SIZE]["churn"].idxmax()
w1, w2 = full_weeks(complaints)[-2:]
week_share = {w: 100 * (complaints[complaints["week_start"] == w]["theme"] == top_churn).mean()
              for w in (w1, w2)}
top_rec = max(recs, key=lambda r: r.get("rice_score") or 0) if recs else None

# ---------------------------------------------------------------------------
# Header: headline, three cards, caveats
# ---------------------------------------------------------------------------
st.caption("Myntra Voice-of-Customer Copilot")
st.markdown(f"## {top_volume[0].upper() + top_volume[1:]} is the most common complaint, "
            f"but {top_churn} is where customers leave.")

c1, c2, c3 = st.columns(3)
with c1:
    card(f"{top_churn.capitalize()} churn rate vs overall",
         f"{by_theme.at[top_churn, 'churn']:.1f}% vs {overall_churn:.1f}%",
         f"{by_theme.at[top_churn, 'churn'] / overall_churn:.1f}× the average share of complaints "
         "that say they'll leave or warn others off.")
with c2:
    change = week_share[w2] - week_share[w1]
    card(f"{top_churn.capitalize()} share of complaints, week over week",
         f"{change:+.1f} pts",
         f"{week_share[w1]:.1f}% → {week_share[w2]:.1f}% (weeks of {w1:%d %b} and {w2:%d %b}). "
         "Two weeks only: directional, not a trend.")
with c3:
    if top_rec:
        card(f"Top roadmap item: {top_rec['id']}", f"RICE {top_rec['rice_score']}",
             f"**{top_rec['title']}** (highest of {len(recs)} recommendations).")
    else:
        card("Top roadmap item", "–", f"`{ROADMAP_PATH}` not found.")

ps_mean = pd.to_numeric(feedback.loc[feedback["source"] == "playstore", "rating"]).mean()
with st.expander("⚠️ Read with care: what this data does and doesn't cover"):
    st.markdown(insights["dashboard_header"])
    st.markdown(
        "- **Sample is 1–3★ only**: 4–5★ reviews were not analysed, so complaints inside "
        "high-star reviews are missed and shares are % of complaints, not of all reviews.\n"
        f"- **The cleaned Play Store mean ({ps_mean:.2f}★) is not Myntra's real rating** "
        f"({PLAYSTORE_PUBLIC_RATING}★ on the store): short praise reviews were dropped in cleaning.\n"
        "- **About 2 weeks of data** (plus ~36 older items, mostly Hindi/Hinglish reviews), so "
        "week-over-week changes are directional, not trends.\n"
        "- **Synthetic support tickets** (IDs `SYN-…`) are invented examples and are always "
        "labelled SYNTHETIC.")

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
    st.caption("The headline and cards above use all "
               f"{insights['total_complaints']} complaints, as do insight sentences, sub-issues "
               "and judged quotes. The filters change the chart, table and churn alerts.")

base = complaints[complaints["source"].isin(pick_sources)
                  & complaints["language"].isin(pick_languages)]
filtered = base[base["week"].isin(pick_weeks)]          # denominator for % (all themes)
shown = filtered[filtered["theme"].isin(pick_themes)]

tab_themes, tab_churn, tab_roadmap, tab_how, tab_try = st.tabs(
    ["Themes", "Churn alerts", "Roadmap", "How it works / Evals", "Try it"])


# ---------------------------------------------------------------------------
# Tab 1: Themes
# ---------------------------------------------------------------------------
def pick_from_chart(key: str) -> None:
    """Chart click -> remember the theme (clicking empty space keeps the current one)."""
    event = st.session_state.get(key)
    points = (event.selection.get("pick") if event and event.selection else None) or []
    if points and points[0].get("theme"):
        st.session_state["picked_theme"] = points[0]["theme"]


def pick_from_table(key: str) -> None:
    event = st.session_state.get(key)
    rows = event.selection.rows if event and event.selection else []
    if rows:
        st.session_state["picked_theme"] = st.session_state["table_order"][rows[0]]


def reset_pick() -> None:
    """Switching Volume / Churn rate jumps to the new #1 theme."""
    st.session_state.pop("picked_theme", None)


def theme_detail(theme: str) -> None:
    """Insight, sub-issues, keyword cross-check and verified quotes for one theme."""
    st.subheader(theme[0].upper() + theme[1:])
    info = insight_by_theme.get(theme)
    if not info:
        st.caption("No insight sentence for this theme (insights cover the top 6 themes). "
                   "Quotes below are checked word-for-word but not reviewed by the quote judge.")
        pool = shown[(shown["theme"] == theme) & (shown["quote_valid"] == True)]  # noqa: E712
        pool = pool.sort_values(["churn_signal", "severity"], ascending=False)
        # one quote per source first (mix), then fill up to 5
        picks = pd.concat([pool.drop_duplicates("source"), pool]).drop_duplicates("id").head(5)
        for r in picks.itertuples():
            quote_card(r.id, r.source, r.language, r.evidence_span, r.url or None,
                       r.rating or None, r.date.date())
        return

    st.info(info["insight"])
    st.markdown("**Sub-issues** (named by the AI, counted by code)")
    sub = pd.DataFrame(info["sub_issues"]).rename(columns={
        "name": "sub-issue", "count": "complaints", "pct_of_theme": "% of theme"})
    st.dataframe(sub, hide_index=True, use_container_width=True)
    if "keyword_counts" in info:
        kc = info["keyword_counts"]
        st.markdown("**Keyword cross-check** (simple word matching, no AI)")
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
        st.caption("One review can match several rules, so counts overlap and don't sum "
                   f"to {kc['theme_items']}.")
    st.markdown(f"**Verified quotes** ({len(info['quotes'])}; copied word-for-word from the "
                "review and rated 4–5/5 for relevance by a second AI)")
    for q in info["quotes"]:
        quote_card(q["id"], q["source"], q["language"], q["quote"], q.get("url"),
                   q.get("rating"), q.get("date"), extra=f"judge {q['judge_score']}/5")


with tab_themes:
    if shown.empty:
        st.info("No complaints match the filters.")
    else:
        # Week-over-week share change uses the two full weeks (ignores the Week filter).
        share = {w: base[base["week_start"] == w]["theme"].value_counts(normalize=True) * 100
                 for w in (w1, w2)}
        g = filtered.groupby("theme")
        table = pd.DataFrame({
            "complaints": g.size(),
            "% of complaints": (100 * g.size() / len(filtered)).round(1),
            "churn rate %": g["churn_signal"].apply(churn_pct).round(1),
        })
        change = share[w2].reindex(table.index).fillna(0) - share[w1].reindex(table.index).fillna(0)
        table["share change WoW"] = change.map(arrow)
        table = table[table.index.isin(pick_themes)]
        avg_churn = churn_pct(filtered["churn_signal"])

        mode = st.radio("Rank themes by", ["Volume", "Churn rate"], horizontal=True,
                        key="rank_mode", on_change=reset_pick)
        by = "complaints" if mode == "Volume" else "churn rate %"
        order = ordered_themes(table, by)
        table = table.loc[order]

        # Theme shown in the detail panel: last click, else ?theme= in the URL, else the #1 theme
        picked = st.session_state.get("picked_theme")
        linked = st.query_params.get("theme")
        theme = (picked if picked in table.index
                 else linked if linked in table.index else order[0])

        left, right = st.columns([1, 1], gap="large")
        with left:
            if mode == "Volume":
                st.markdown(f"**% of complaints by theme** ({len(shown)} of {len(filtered)} "
                            "complaints shown)")
                chart_df, x_field = table, "% of complaints"
            else:
                st.markdown("**Churn rate by theme**: % of a theme's complaints that say "
                            "they'll leave or warn others off")
                chart_df, x_field = table.drop([t for t in UNRANKED if t in table.index]), "churn rate %"
            chart_df = chart_df.reset_index()
            sel = alt.selection_point(name="pick", fields=["theme"], on="click")
            y = alt.Y("theme:N", sort=chart_df["theme"].tolist(), title=None,
                      axis=alt.Axis(labelLimit=0))                 # never cut theme names
            bars = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X(f"{x_field}:Q", title=x_field),
                y=y,
                color=alt.condition(alt.datum.theme == theme, alt.value("#ff3f6c"),
                                    alt.value("#8fb8de")),
                tooltip=["theme", "complaints", "% of complaints", "churn rate %"],
            ).add_params(sel)
            labels = alt.Chart(chart_df).mark_text(align="left", dx=3, color="#999").encode(
                x=f"{x_field}:Q", y=y, text=alt.Text(f"{x_field}:Q", format=".1f"))
            layers = bars + labels
            if mode == "Churn rate":
                rule = alt.Chart(pd.DataFrame({"avg": [avg_churn]})).mark_rule(
                    strokeDash=[6, 4], color="#999").encode(x="avg:Q")
                layers = layers + rule
            key = f"theme_chart_{mode}"
            st.altair_chart(layers.properties(height=28 * len(chart_df) + 30),
                            use_container_width=True, key=key, on_select=partial(pick_from_chart, key),
                            selection_mode="pick")
            if mode == "Churn rate":
                skipped = [f"{t} ({table.at[t, 'churn rate %']}%, {table.at[t, 'complaints']} complaints)"
                           for t in UNRANKED if t in table.index]
                st.caption(f"Dashed line: average across all complaints shown ({avg_churn:.1f}%). "
                           + (f"Not ranked (catch-all buckets): {'; '.join(skipped)}." if skipped else ""))
            st.caption("Click a bar (or a table row below) to see that theme on the right.")
        with right:
            theme_detail(theme)

        st.divider()
        st.markdown("**All themes**")
        st.caption(f"Share change compares the weeks of {w1:%d %b} and {w2:%d %b} (% of that week's "
                   "complaints, source/language filters applied).")
        st.session_state["table_order"] = order
        table_key = f"theme_table_{mode}"       # selection is by row position: reset per order
        st.dataframe(table, use_container_width=True, on_select=partial(pick_from_table, table_key),
                     selection_mode="single-row", key=table_key,
                     height=35 * (len(table) + 1) + 3)          # all rows, no inner scroll

# ---------------------------------------------------------------------------
# Tab 2: Churn alerts
# ---------------------------------------------------------------------------
with tab_churn:
    churn = shown[shown["churn_signal"] == "Y"]
    st.subheader(f"Churn alerts: {len(churn)} complaints say they'll leave or warn others off")
    st.caption("Churn = states they will stop using Myntra, tells others not to use it, or "
               "recommends a competitor (anger alone doesn't count). Themes are sorted by churn "
               "rate. Each quote is the sentence where the customer says they'll leave; if no such "
               "sentence was found, the main complaint is shown, labelled *complaint quote*. "
               "Filters apply.")
    if not churn.empty:
        rates = (shown.groupby("theme")["churn_signal"].agg(["size", churn_pct])
                 .rename(columns={"churn_pct": "churn rate %"}))
        rates = rates[rates.index.isin(churn["theme"])]
        for i, theme in enumerate(ordered_themes(rates, "churn rate %")):
            group = churn[churn["theme"] == theme].copy()
            group["leaving"] = group["text"].map(leaving_sentence)
            group = group.sort_values(["date"], ascending=False)
            group = pd.concat([group[group["leaving"].notna()], group[group["leaving"].isna()]])
            title = (f"{theme}: {rates.at[theme, 'churn rate %']:.1f}% churn "
                     f"({len(group)} of {rates.at[theme, 'size']} complaints)")
            with st.expander(title, expanded=i == 0):
                for r in group.itertuples():
                    if isinstance(r.leaving, str):          # (missing values are NaN, not None)
                        quote_card(r.id, r.source, r.language, r.leaving, r.url or None,
                                   r.rating or None, r.date.date())
                    elif isinstance(r.evidence_span, str) and r.quote_valid:
                        quote_card(r.id, r.source, r.language, r.evidence_span, r.url or None,
                                   r.rating or None, r.date.date(), extra=":gray[complaint quote]")
                    else:
                        st.markdown(f"*No verified quote. AI summary:* {mask_names(r.pain_point)}  \n"
                                    f"**{r.id}** · {source_label(r.source)} · {r.language}")

# ---------------------------------------------------------------------------
# Tab 3: Roadmap
# ---------------------------------------------------------------------------
with tab_roadmap:
    st.subheader("Roadmap (RICE)")
    if not roadmap:
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
    reach_note: One short plain-language line.
    reach_detail: How the reach number was counted.
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
        meta = roadmap.get("meta", {})
        known_themes = set(complaints["theme"])
        by_id = complaints.set_index("id")
        id_pattern = re.compile(r"\b(?:PS|AS|SYN|RD)-\d{4}\b")

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
                st.caption(f"Reach ({r.get('reach')}): {r['reach_note']}")
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
        st.caption("RICE = Reach × Impact × Confidence ÷ Effort, computed by the dashboard. "
                   "Reach = number of real (non-synthetic) complaints the fix addresses.")

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

        counted = [r for r in recs + watch if r.get("reach_detail")]
        if counted:
            with st.expander("How reach was counted"):
                for r in counted:
                    st.markdown(f"- **{r.get('id', '')} {r.get('title', '')}** (reach {r.get('reach')}): "
                                f"{r['reach_detail']}")

        st.markdown("### Not building (and why)")
        nb = roadmap.get("not_building", [])
        if nb:
            for item in nb:
                st.markdown(f"- **{item.get('title', '')}**: {str(item.get('reason', '')).strip()}")
        else:
            st.caption("None listed.")

# ---------------------------------------------------------------------------
# Tab 4: How it works / Evals
# ---------------------------------------------------------------------------
with tab_how:
    st.subheader("How it works")
    steps = [
        ("Collect reviews", "5,000 Play Store and 500 App Store reviews, plus 20 hand-written "
                            "synthetic support tickets."),
        ("Clean", "Remove duplicates and reviews under 3 words, detect the language: 3,039 left."),
        ("AI tags each complaint", "For the 982 complaints (1–3★ and synthetic tickets), the AI "
                                   "picks a theme, a churn signal, a severity and a supporting quote."),
        ("Check every quote is exact", "A quote is kept only if it appears word-for-word in the "
                                       "review. Otherwise it is dropped, never reworded."),
        ("Group and count", "Counts, % of complaints and week-over-week shares are computed by "
                            "code, not by the AI."),
        ("Recommend", "Insight sentences use numbers from code, a second AI rates quote "
                      "relevance, and the PM writes the RICE roadmap."),
    ]
    for row_start in (0, 3):                               # two rows of three steps
        for col, i in zip(st.columns(3), range(row_start, row_start + 3)):
            name, text = steps[i]
            with col, st.container(border=True, height="stretch"):
                st.markdown(f"**{i + 1}. {name}** {'→' if i < len(steps) - 1 else ''}")
                st.caption(text)
    st.caption("The dashboard itself makes no AI calls, except in **Try it**.")
    with st.expander("Technical detail (scripts and files)"):
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

    st.subheader("How well does the AI tagging work?")
    st.markdown("I compared the AI's tags with my own on two sets of reviews. "
                "**Golden set** = 100 reviews I hand-labelled to check the AI. "
                "**Holdout** = 30 extra reviews never used while tuning, so it is the honest score. "
                "The prompt was improved three times: **prompt version 1/2/3 (3 = current)**.")
    col_help = {
        "theme agreement %": "The AI picked the same theme I did.",
        "churn precision %": "Of the reviews the AI flagged as 'will leave', how many really were.",
        "churn recall %": "Of the reviews that really say 'will leave', how many the AI caught.",
        "severity ±1 %": "The AI's severity (1–3) is within 1 point of mine.",
        "quote fidelity %": "The AI's quote is copied word-for-word from the review.",
    }
    keys = {"theme agreement %": "theme_agreement_pct", "churn precision %": "churn_precision_pct",
            "churn recall %": "churn_recall_pct", "severity ±1 %": "severity_within_1_pct",
            "quote fidelity %": "quote_fidelity_pct"}
    rows = []
    for n in (1, 2, 3):
        for label_set, name in (("golden set", f"v{n}.json"), ("holdout", f"v{n}_holdout.json")):
            path = RESULTS_DIR / name
            if path.exists():
                m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
                rows.append({"prompt version": f"{n} (current)" if n == 3 else str(n), "set": label_set,
                             **{col: m[k] for col, k in keys.items()}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={col: st.column_config.NumberColumn(col, help=h)
                                for col, h in col_help.items()})
    st.caption(" · ".join(f"**{c.replace(' %', '')}**: {h}" for c, h in col_help.items())
               + "  \nChurn has few positives (14 golden, 5 holdout), so those % move in big steps.")

    judge_path = RESULTS_DIR / "insights_v2.json"
    if judge_path.exists():
        j = json.loads(judge_path.read_text(encoding="utf-8"))
        st.subheader("How good are the insight sentences?")
        checks = j["automatic_checks"]
        st.markdown(f"Code checks: every number matches the data "
                    f"({sum(c['count_accuracy'] for c in checks.values())}/{len(checks)}) and every "
                    f"cited review exists ({sum(c['citation_validity'] for c in checks.values())}/"
                    f"{len(checks)}). A second AI then scored each insight from 1 to 5 on "
                    f"**specificity** (names a concrete problem) {j['means']['specificity']}, "
                    f"**grounding** (backed by the quotes) {j['means']['grounding']}, and "
                    f"**actionability** (suggests a clear next step) {j['means']['actionability']}.")
        st.table(pd.DataFrame(j["scores"])[["theme", "specificity", "grounding",
                                            "actionability", "reason"]].set_index("theme"))

    st.subheader("Limitations")
    readme = Path("README.md").read_text(encoding="utf-8")
    section = readme.split("## Limitations", 1)[-1].split("\n## ", 1)[0]
    st.markdown(section)

# ---------------------------------------------------------------------------
# Tab 5: Try it (the only place that calls the LLM)
# ---------------------------------------------------------------------------
TRY_IT_EXAMPLE = "\n".join([
    "My refund for the cancelled order is still not credited after 12 days. Uninstalling Myntra.",
    "Return pickup 3 baar cancel ho gaya, ab kabhi Myntra se order nahi karungi",
    "साइज़ चार्ट गलत है, M साइज़ बहुत टाइट आया और एक्सचेंज भी नहीं हो रहा",
])


def load_example() -> None:
    st.session_state["try_text"] = TRY_IT_EXAMPLE


with tab_try:
    st.subheader("Try it: run the current prompt (version 3) on your own reviews")
    st.caption(f"One review per line, up to {TRY_IT_LIMIT}. This calls the free-tier AI "
               "(Groq), so it can take a minute, longer if the daily limit is reached. "
               f"Limited to {TRY_IT_RUNS_PER_SESSION} runs per session. Nothing is saved.")
    runs_used = st.session_state.get("try_runs", 0)
    runs_left = TRY_IT_RUNS_PER_SESSION - runs_used
    if not HAS_LLM_KEY:
        st.info("Try it is switched off here because no LLM API key is configured. "
                "Everything else on this page works from precomputed results. To enable it, "
                "add `GROQ_API_KEY` to the app's secrets (or to `.env` when running locally).",
                icon="🔑")
    elif runs_left <= 0:
        st.info(f"You've used all {TRY_IT_RUNS_PER_SESSION} runs for this session, which keeps "
                "the free LLM quota available for other visitors. Thanks for trying it!", icon="✋")
    else:
        st.caption(f"Runs left this session: {runs_left} of {TRY_IT_RUNS_PER_SESSION}.")
    try_enabled = HAS_LLM_KEY and runs_left > 0
    st.button("Load example (English, Hinglish, Hindi)", on_click=load_example,
              disabled=not try_enabled)
    pasted = st.text_area("Reviews", height=160, key="try_text", disabled=not try_enabled,
                          placeholder="Paste reviews here, one per line, or click Load example.")
    lines = [l.strip() for l in pasted.splitlines() if l.strip()]
    if len(lines) > TRY_IT_LIMIT:
        st.warning(f"{len(lines)} reviews pasted; only the first {TRY_IT_LIMIT} will be used.")
        lines = lines[:TRY_IT_LIMIT]

    label = (f"Extract {len(lines)} review{'s' if len(lines) != 1 else ''}" if lines
             else "Extract reviews")
    if st.button(label, disabled=not (lines and try_enabled), type="primary"):
        st.session_state["try_runs"] = runs_used + 1     # counted even if the call fails
        from src import extract   # imported here so other tabs never touch the LLM code
        batch = pd.DataFrame({"id": [f"TRY-{i:04d}" for i in range(1, len(lines) + 1)], "text": lines})
        with st.spinner("Calling the AI..."):
            try:
                records = extract.extract_batch("v3", batch)
            except Exception as error:      # show the problem instead of crashing the page
                st.error(f"Extraction failed: {type(error).__name__}: {error}")
                records = []
        text_by_id = dict(zip(batch["id"], batch["text"]))
        st.session_state["try_results"] = [extract.guard_quote(r, text_by_id[r["id"]]) for r in records]
        st.session_state["try_missing"] = len(batch) - len(records)
        if not records:
            st.stop()               # keep the error message on screen
        st.rerun()                  # redraw so "Runs left" above shows the new count

    results = st.session_state.get("try_results")
    if results:
        res = pd.DataFrame(results)
        res["theme"] = res["theme"].replace({"no complaint": NOT_COMPLAINT})
        if st.session_state.get("try_missing"):
            st.warning(f"{st.session_state['try_missing']} review(s) got no answer; try again.")
        c1, c2 = st.columns(2)
        c1.metric("Reviews extracted", len(res))
        c2.metric("Churn signals", int((res["churn_signal"] == "Y").sum()))
        st.markdown("**Themes**")
        st.dataframe(res["theme"].value_counts().rename("reviews"), use_container_width=True)
        st.markdown("**Per review**")
        # st.table wraps long text, so quotes are readable without sideways scrolling
        st.table(pd.DataFrame({
            "id": res["id"], "theme": res["theme"], "churn": res["churn_signal"],
            "severity": res["severity"].map(lambda s: "–" if pd.isna(s) else str(s)),
            "pain point": res["pain_point"].fillna("").map(mask_names),
            "quote (verified)": res["evidence_span"].map(lambda q: mask_names(q) if q else
                                                         "— rejected by the quote check"),
        }).set_index("id"))
