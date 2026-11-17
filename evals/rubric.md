# Eval rubric

Two kinds of output are scored: **per-item extractions** (scored automatically by
`evals/run_evals.py` against the human-labelled golden and holdout sets) and
**theme insights** (checked automatically by `src/insights.py`, then read by a human).

## 1. Per-item extractions (`run_evals.py`)

| Metric | Pass bar | How |
|---|---|---|
| Theme agreement | ≥ 80% on holdout | extracted theme == human theme |
| Churn precision / recall | ≥ 80% / ≥ 60% | positive class = churn Y (rule v2 in `data/golden/themes.md`) |
| Severity within ±1 | ≥ 90% | only items the human marked as complaints |
| Quote fidelity | 100% after the guard | `evidence_span` is an exact substring of the original text, else nulled |

Scores are always reported with the prompt version and the label set's md5 fingerprint.

## 2. Theme insights (`insights.py`)

### Good output format

One sentence per theme, in this shape (illustrative: the sub-issue name, its count and the
cited IDs are made up for the example):

> **Refund delay** is **9.4% of complaints (1–3★)** (92 of 982), rising from 7.9% to 11.0%
> of weekly complaints; the biggest sub-issue is "refund not credited after pickup"
> (41 items), and 18 complainers (19.6%) say they will leave or warn others off
> [PS-1862, AS-0193, SYN-0007].

A good insight:

1. **Names the theme and its size**: count and share, with the denominator spelled out as
   "% of complaints (1–3★)". Never "% of all reviews": 4–5★ reviews were not extracted.
2. **Gives the direction**: the share in the previous vs the latest full week.
3. **Names the top sub-issue** with its count.
4. **States the churn signal**: count and % of the theme.
5. **Cites 3 or more IDs**, each pointing to a verified quote (an exact substring of the source).

### Bad output (fails)

- Any number that is not computed from the data (counts, %, trends).
- Any cited ID that is not in the data, or whose quote is not an exact substring.
- Vague claims with no numbers ("many users are unhappy with refunds").
- Paraphrased "quotes", or a share given as % of all reviews.

### Automatic checks (run on every insight before it is saved)

| Check | Rule |
|---|---|
| Count accuracy | every number in the final sentence equals a value computed in code for that theme |
| Citation validity | every cited ID exists in `feedback.csv`, belongs to that theme, and its quote is an exact substring |
| No LLM numbers | the LLM's wording (before numbers are filled in) contains no digits |

### Human review (1–5 each)

- **Actionable**: would a PM know what to investigate next?
- **Faithful**: do the quotes really support the sentence?
- **Clear**: readable in one pass by a non-analyst?
