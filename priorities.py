"""
priorities.py

"Today's Priority": a ranked triage list across every (store, product) pair
for the currently-selected strategy, so a manager doesn't have to click
through all of them one at a time to find the ones that actually need a
decision today. Meant to be run once per day (the UI caches it by
calendar date) for all Store x Product combinations, producing a single
DataFrame the UI renders as a colored table with a "View" action per row.

Priority tiers (checked in this order -- first match wins, these are not
cumulative):
    High   (red)    -- either (a) assuming today plays out as *expected*
                        (mean demand), this policy's own forward plan still
                        shows a stockout risk tomorrow if demand spikes to
                        the high (P90) case, or (b) a promotion or holiday
                        starts tomorrow (a demand surge is about to hit
                        with only one day of lead time to react).
    Medium (yellow) -- today's safety buffer is *thin* relative to P90
                        demand -- not just an outright deficit (buffer < 0)
                        but anything under SAFETY_BUFFER_RATIO_THRESHOLD
                        (default 20%) of P90 demand. A live, present-day
                        cushion concern rather than a forward-looking one,
                        and deliberately checked after (a), see the note
                        below on why these two must use different demand
                        assumptions to stay distinguishable.
    Low    (green)  -- none of the above; included so the table always
                        shows the whole store network, not just problems.

Why a ratio, not just "buffer < 0": a strict negative-buffer requirement
is an extreme, rare condition -- it only fires when today's order falls
completely short of worst-case demand even counting current stock. In
practice, most pairs are either comfortably stocked (Low) or already
tripped by a High condition first, so a hard zero cutoff leaves Medium
almost empty (confirmed empirically: 0 of 12 pairs landed there in a
synthetic test before this was widened). Framing it as "buffer under X% of
P90 demand" (same style as explain.confidence_level's gap-ratio bands)
catches thin-but-still-technically-covered cases too, giving Medium an
actual, usable range. Tune SAFETY_BUFFER_RATIO_THRESHOLD below if 20% is
too loose or too strict for your data.

Note on why the two demand assumptions differ: if the "stockout risk
tomorrow" check also assumed worst-case (P90) demand *today*, it would
almost always coincide with "safety buffer negative today" (both are the
same fact -- today's order doesn't cover P90 demand -- viewed two ways),
so Medium would rarely if ever be reached; the two checks would collapse
into just High/Low. Using the expected (mean) demand path for today when
projecting tomorrow keeps the two signals meaningfully different: High
means a demand spike tomorrow is a problem even if today is unremarkable;
Medium means today itself is already thin against the worst case, but
tomorrow isn't (yet) independently flagged as risky.

This is a rules-based triage layer sitting on top of the RL policy +
LightGBM forecast (via Recommender.recommend_from_history and
Recommender.plan_ahead, and explain.compute_decision_breakdown for the
safety-buffer math) -- it doesn't replace the Recommendation tab, it just
tells you where to look first. Pairs that error for any reason (e.g. not
enough lookback history at the very start of a series) are silently
skipped rather than failing the whole batch, since this is a best-effort
overview, not a per-pair guarantee.

On "today": this demo's retail_store_inventory.csv is a fixed, static
historical snapshot (731 days), not a live feed that keeps appending new
days -- so the literal last date in the file has no real "tomorrow" row to
check price/discount/holiday context against. To make "stockout risk
tomorrow" / "promotion starts tomorrow" actually checkable against real
data (rather than always silently returning False), "today" here is the
*second-to-last* available date per store/product, so the literal last
date in the file can serve as a real "tomorrow". In a live deployment
with an actual current date and a forward-looking promo/pricing calendar
feed (see the Future plan tab's caveat in inference.plan_ahead), you would
instead use the real current date as "today" and pull tomorrow's calendar
context from that feed, not from this same historical CSV.
"""

from __future__ import annotations

import pandas as pd

from explain import compute_decision_breakdown

PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}
PRIORITY_ICON = {"High": "\U0001F534", "Medium": "\U0001F7E1", "Low": "\U0001F7E2"}

# Medium fires when safety_buffer / demand_p90 falls below this fraction --
# i.e. today's order + current stock clears P90 demand by less than 20%.
# 0 or negative (an outright deficit) always qualifies too, since 0 < 0.20.
# Tune this if Medium comes back too crowded (raise it) or still empty
# (lower it, or check whether your policy is simply very conservative and
# rarely runs a thin buffer at all -- that's a legitimate outcome too).
SAFETY_BUFFER_RATIO_THRESHOLD = 0.20

# A "promotion starting" requires the Discount to rise by at least this many
# percentage points, not just any increase. Discount data is rarely a clean
# step function -- a 1-point day-to-day wiggle isn't a promotion, and
# treating it like one would spuriously push pairs into High on noise alone,
# crowding out Medium/Low the same way the P90-double-counting bug did.
PROMO_DISCOUNT_INCREASE_THRESHOLD = 5


def _today_and_tomorrow(history_df: pd.DataFrame, store_id: str, product_id: str):
    """Returns (today, tomorrow) for this store/product: today is the
    second-to-last available date (so a real "tomorrow" row exists to check
    promo/holiday context against), falling back to the last date with
    tomorrow=None if there's only one day of history at all."""
    g = history_df[(history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)]
    dates = sorted(g["Date"].unique())
    if len(dates) == 0:
        raise ValueError(f"No data for store={store_id}, product={product_id}")
    if len(dates) == 1:
        return dates[-1], None
    return dates[-2], dates[-1]


def _promo_or_holiday_starting(history_df: pd.DataFrame, store_id: str, product_id: str,
                                 today, tomorrow) -> bool:
    """True if a real promotion (Discount rises by at least
    PROMO_DISCOUNT_INCREASE_THRESHOLD percentage points -- a small
    day-to-day wiggle doesn't count) or a holiday/promotion flag turns on
    between `today` and `tomorrow` for this store/product, using the real
    historical record for those dates (see the same caveat as
    inference.plan_ahead: future price/discount/holiday context is assumed
    to be planned in advance and pulled from the record, even though demand
    itself is not). Returns False if `tomorrow` is None or has no row."""
    if tomorrow is None:
        return False
    g = history_df[(history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)]
    row_tomorrow = g[g["Date"] == tomorrow]
    if len(row_tomorrow) == 0:
        return False
    rt = row_tomorrow.iloc[0]
    row_today = g[g["Date"] == today]
    if len(row_today) == 0:
        return bool(rt["Holiday/Promotion"] == 1)
    ry = row_today.iloc[0]
    promo_started = (rt["Discount"] - ry["Discount"]) >= PROMO_DISCOUNT_INCREASE_THRESHOLD
    holiday_started = (rt["Holiday/Promotion"] == 1) and (ry["Holiday/Promotion"] == 0)
    return bool(promo_started or holiday_started)


def compute_todays_priorities(history_df: pd.DataFrame, recommender) -> pd.DataFrame:
    """Runs `recommender`'s policy over every distinct (store, product)
    pair's "today" (see _today_and_tomorrow) and returns a DataFrame with
    columns: store_id, product_id, date, priority ("High"/"Medium"/"Low"),
    reason, order_qty, safety_buffer, safety_buffer_ratio, current_stock,
    demand_p90 -- sorted High-to-Low, then alphabetically within a tier.
    Returns an empty DataFrame (not None) if nothing could be computed."""
    pairs = history_df[["Store ID", "Product ID"]].drop_duplicates().itertuples(index=False)
    rows = []
    for store_id, product_id in pairs:
        try:
            today, tomorrow = _today_and_tomorrow(history_df, store_id, product_id)
            rec = recommender.recommend_from_history(store_id, product_id, today)
            b = compute_decision_breakdown(rec)

            stockout_tomorrow = False
            try:
                # demand_scenario="mean" -- today is simulated under
                # *expected* demand here, deliberately not "p90". Using
                # "p90" for today would make this check redundant with
                # below_safety_buffer (see module docstring): both would
                # just be re-detecting "today's order doesn't cover P90
                # demand," and Medium would never be reached.
                plan = recommender.plan_ahead(store_id, product_id, today, n_days=2, demand_scenario="mean")
                if len(plan) >= 2:
                    stockout_tomorrow = bool(plan.iloc[1]["stockout_risk_units_if_high_demand"] > 0)
            except Exception:
                pass

            promo_starting = _promo_or_holiday_starting(history_df, store_id, product_id, today, tomorrow)
            safety_buffer_ratio = b["safety_buffer"] / max(b["demand_p90"], 1e-6)
            thin_safety_buffer = safety_buffer_ratio < SAFETY_BUFFER_RATIO_THRESHOLD

            if stockout_tomorrow:
                priority, reason = "High", "Stockout risk tomorrow"
            elif promo_starting:
                priority, reason = "High", "Promotion starts tomorrow"
            elif thin_safety_buffer:
                priority, reason = "Medium", "Inventory below safety buffer"
            else:
                priority, reason = "Low", "Stable"

            rows.append({
                "store_id": store_id, "product_id": product_id, "date": today,
                "priority": priority, "reason": reason,
                "order_qty": rec["fulfilled_order_qty"], "safety_buffer": b["safety_buffer"],
                "safety_buffer_ratio": safety_buffer_ratio,
                "current_stock": b["current_stock"], "demand_p90": b["demand_p90"],
            })
        except Exception:
            continue

    df = pd.DataFrame(rows, columns=[
        "store_id", "product_id", "date", "priority", "reason",
        "order_qty", "safety_buffer", "safety_buffer_ratio", "current_stock", "demand_p90",
    ])
    if len(df) == 0:
        return df
    df["_rank"] = df["priority"].map(PRIORITY_RANK)
    df = df.sort_values(["_rank", "store_id", "product_id"]).drop(columns="_rank").reset_index(drop=True)
    return df
