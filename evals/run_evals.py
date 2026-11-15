"""Score LLM extractions against the human-labelled golden set.

Metrics (golden set = the truth):
  - coverage:        % of golden items that have an extraction at all
  - theme agreement: % where the extracted theme equals the human theme
  - churn precision: of the items the model flagged churn=Y, % that really are Y
  - churn recall:    of the truly churn=Y items, % the model caught
  - severity ±1:     % of complaints where model severity is within 1 of the human's
  - quote fidelity:  % where evidence_span is non-empty AND an exact substring of the
                     original text (our "no invented quotes" guarantee)

Also lists every row where theme or churn disagree, ranked worst first, so prompt
changes can be targeted. Results go to evals/results/<version>.json.

Run:  python -m evals.run_evals --version v1
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

GOLDEN_PATH = Path("data/golden/golden_set.csv")
FEEDBACK_PATH = Path("data/processed/feedback.csv")
RESULTS_DIR = Path("evals/results")


def load_extractions(version: str) -> pd.DataFrame:
    """Prefer the golden-only file; fall back to the full run."""
    for name in (f"extractions_{version}_golden.jsonl", f"extractions_{version}.jsonl"):
        path = Path("data/processed") / name
        if path.exists():
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
            print(f"Using {path}")
            return pd.DataFrame(rows)
    sys.exit(f"No extractions found for {version}. Run: python -m src.extract --golden --version {version}")


def pct(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    golden = pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    texts = pd.read_csv(FEEDBACK_PATH, dtype=str, keep_default_na=False,
                        encoding="utf-8-sig").set_index("id")["text"]
    ext = load_extractions(args.version)

    # One row per golden item, with the model's answer alongside (missing -> NaN).
    df = golden.merge(ext, on="id", how="left", suffixes=("_gold", "_pred"))
    has = df["theme_pred"].notna()
    scored = df[has].copy()

    # --- Theme ---
    scored["theme_ok"] = scored["theme_gold"] == scored["theme_pred"]

    # --- Churn (positive class = Y) ---
    gold_y = scored["churn_signal_gold"] == "Y"
    pred_y = scored["churn_signal_pred"] == "Y"
    tp, fp, fn = int((gold_y & pred_y).sum()), int((~gold_y & pred_y).sum()), int((gold_y & ~pred_y).sum())
    scored["churn_ok"] = gold_y == pred_y

    # --- Severity: only where the human gave one (i.e. a complaint) ---
    sev_gold = pd.to_numeric(scored["severity_gold"], errors="coerce")
    sev_pred = pd.to_numeric(scored["severity_pred"], errors="coerce")
    has_sev = sev_gold.notna()
    sev_diff = (sev_gold - sev_pred).abs()
    within1 = int(((sev_diff <= 1) & has_sev).sum())   # a missing prediction counts as a miss
    exact = int(((sev_diff == 0) & has_sev).sum())

    # --- Quote fidelity: exact substring of the ORIGINAL text, no normalising ---
    def quote_ok(row) -> bool:
        span = row["evidence_span"]
        return isinstance(span, str) and span != "" and span in texts[row["id"]]
    scored["quote_ok"] = scored.apply(quote_ok, axis=1)

    n = len(scored)
    metrics = {
        "coverage_pct": pct(n, len(golden)),
        "theme_agreement_pct": pct(int(scored["theme_ok"].sum()), n),
        "churn_precision_pct": pct(tp, tp + fp),
        "churn_recall_pct": pct(tp, tp + fn),
        "churn_counts": {"tp": tp, "fp": fp, "fn": fn, "gold_Y": int(gold_y.sum()), "pred_Y": int(pred_y.sum())},
        "severity_within_1_pct": pct(within1, int(has_sev.sum())),
        "severity_exact_pct": pct(exact, int(has_sev.sum())),
        "quote_fidelity_pct": pct(int(scored["quote_ok"].sum()), n),
    }

    # --- Confusion list: rows where theme or churn disagree, worst first ---
    # Score: wrong theme 2, wrong churn 2, severity off by 2+ 1, bad quote 1; ties: higher human severity first.
    scored["miss_score"] = (2 * ~scored["theme_ok"] + 2 * ~scored["churn_ok"]
                            + (sev_diff >= 2).fillna(False).astype(int) + ~scored["quote_ok"])
    scored["sev_gold_num"] = sev_gold.fillna(0)
    misses = scored[~scored["theme_ok"] | ~scored["churn_ok"]].sort_values(
        ["miss_score", "sev_gold_num"], ascending=False)
    confusion = [{
        "id": r.id, "miss_score": int(r.miss_score),
        "gold": {"theme": r.theme_gold, "churn": r.churn_signal_gold, "severity": r.severity_gold},
        "pred": {"theme": r.theme_pred, "churn": r.churn_signal_pred,
                 "severity": None if pd.isna(r.severity_pred) else int(r.severity_pred)},
        "evidence_span": r.evidence_span, "quote_ok": bool(r.quote_ok),
        "text": r.text,
    } for r in misses.itertuples()]

    models = sorted(ext["model"].dropna().unique().tolist()) if "model" in ext else []
    result = {
        "prompt_version": args.version,
        "model": models,
        "run_date": date.today().isoformat(),
        "n_golden": len(golden),
        "n_scored": n,
        "metrics": metrics,
        "theme_confusions": scored[~scored["theme_ok"]].groupby(
            ["theme_gold", "theme_pred"]).size().sort_values(ascending=False)
            .rename("count").reset_index().to_dict("records"),
        "misses": confusion,
        "missing_ids": df.loc[~has, "id"].tolist(),
        "bad_quote_ids": scored.loc[~scored["quote_ok"], "id"].tolist(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.version}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Print summary ---
    m = metrics
    print(f"\nPrompt {args.version} | model {', '.join(models)} | scored {n}/{len(golden)}")
    print(f"  Theme agreement   {m['theme_agreement_pct']}%")
    print(f"  Churn precision   {m['churn_precision_pct']}%   recall {m['churn_recall_pct']}%   {m['churn_counts']}")
    print(f"  Severity within 1 {m['severity_within_1_pct']}%   (exact {m['severity_exact_pct']}%)")
    print(f"  Quote fidelity    {m['quote_fidelity_pct']}%")
    print(f"  Rows with theme or churn disagreement: {len(confusion)}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
