# Golden set — theme list and labeling guide

Use these names in the `theme` column of `golden_set.csv`. Pick **one primary theme** per
item: the main thing the customer is upset about. Add any secondary issue in `notes`.

The themes come from a keyword skim of the 982 complaints in `feedback.csv`
(1–3★ reviews + synthetic tickets). Counts are rough keyword matches, and one review
can match several, so they add up to more than 100%. They're just a guide, not results.

| # | Theme | What it covers | Skim count |
|---|-------|----------------|-----------:|
| 1 | `delivery delay` | Late, stuck in transit, or never arrived (no blame on the agent) | ~246 (25%) |
| 2 | `order cancelled` | Myntra / seller cancels the order, often without reason | ~174 (18%) |
| 3 | `refund delay` | Refund late, missing, smaller than paid, or sent to Myntra credit instead | ~133 (14%) |
| 4 | `customer support` | Unresponsive, scripted replies, long queues, unhelpful agents — **only when support is the main complaint** | ~441 (45%)* |
| 5 | `return/exchange failed` | Pickup not done, return or exchange rejected, size not available for exchange | ~130 (13%) |
| 6 | `wrong/damaged item` | Wrong product/colour, damaged, used, or missing item in package | ~79 (8%) |
| 7 | `app bugs/payments` | Crashes, payment deducted but no order, coupon not applied, login/OTP | ~106 (11%) |
| 8 | `pricing/fees` | Expensive, extra platform/convenience fees, misleading discounts | ~80 (8%) |
| 9 | `product quality` | Poor fabric, stitching, colour fades, not as pictured | ~50 (5%) |
| 10 | `size/fit mismatch` | Size chart wrong, runs small/large, inconsistent sizes across brands | ~50 (5%) |
| 11 | `fake/counterfeit` | Not original, first copy, fraud/scam accusations about products | ~35–78 (4–8%) |
| 12 | `M-Now late delivery` | Quick-delivery promise (30/60 min) missed or auto-cancelled | ~18 (2%) |
| 13 | `delivery agent behaviour` | Agent misconduct: collects extra cash, rude or abusive, marks order "delivered" without handing it over, blocks the customer's number | ~38 (4%) |

**Delivery delay vs. delivery agent behaviour:** if the order is just late or never arrives,
use `delivery delay`. If the customer blames the agent's conduct (fake "delivered" status,
extra money, rudeness), use `delivery agent behaviour`.

\* "Customer support" is mentioned in almost half of complaints, but usually *alongside*
another problem ("refund not received **and** support doesn't reply"). Label that example
`refund delay` and write "support unresponsive" in `notes`.

**Special labels** (not pain-point themes, but needed for some items):
- `no complaint` — praise or neutral (common in 4–5★ items).
- `other` — a real complaint that fits none of the above. Put its **sub-type** in `notes`:
  `range` (products/styles not stocked), `rewards` (MyCash, spin-and-win, points),
  `listing accuracy` (AI or misleading product photos), `trust (vague)` (fraud claim with
  no specifics), or a new short tag. If several `other` items share a sub-type, that's a
  candidate new theme.

## Other columns

**`churn_signal`** — `Y` / `N`. `Y` only if the customer says or clearly implies they
will stop using Myntra or switch: "uninstalling", "never ordering again", "going back to
Amazon/Flipkart", "telling everyone not to use it". Plain anger alone is `N`.

**`severity`** — 1 to 3:
- `1` minor annoyance, no money or order lost (slow app, UI complaint, small delay)
- `2` real problem with an order (late delivery, wrong size, return hassle)
- `3` money lost or stuck, repeated failures, fraud/fake claims, or churn threatened

**`notes`** — anything useful: secondary theme, why an item was hard to label, mismatched
rating (e.g. 5★ with a complaint).

**`verified`** — put `yes` once you've finished and double-checked that row.

**`ai_confidence` / `ai_reason`** — added by the AI pre-labeling pass. `low` means check
this row carefully: two competing problems, Hinglish/Hindi, vague text, unclear churn
intent, or no theme fits. The untouched AI draft is kept in `golden_set_ai_draft.csv`,
so corrections can be counted by comparing the two files.

**`severity`** is left blank for `no complaint` rows.

## Tips

- Label from the **text**, not the star rating. 3★ reviews are sometimes very angry,
  and 5★ reviews sometimes contain a complaint.
- Excel: if you edit in Excel, save as **CSV UTF-8** so Hindi text and emoji survive.
