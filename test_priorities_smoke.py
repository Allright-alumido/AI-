"""
test_priorities_smoke.py

Smoke test for priorities.py. Uses a hand-built fake "recommender" object
(not the real Recommender/PPO/LightGBM stack -- that's exercised separately
in test_inference_smoke.py) so each priority tier can be engineered and
checked deterministically. This validates the triage *rules* (what counts
as High/Medium/Low and why) in isolation from RL policy behavior.

Run: python test_priorities_smoke.py
"""

import pandas as pd

from priorities import (
    compute_todays_priorities, _today_and_tomorrow, _promo_or_holiday_starting, PRIORITY_RANK,
    SAFETY_BUFFER_RATIO_THRESHOLD, PROMO_DISCOUNT_INCREASE_THRESHOLD,
)


def make_row(store_id, product_id, date, discount=0, holiday=0):
    return {
        "Store ID": store_id, "Product ID": product_id, "Date": pd.Timestamp(date),
        "Discount": discount, "Holiday/Promotion": holiday,
    }


def make_history():
    """Five store/product pairs, each with two days of history (today =
    second-to-last date, tomorrow = last date, per priorities.py's
    "today" convention). Discount/Holiday flags are engineered so
    _promo_or_holiday_starting fires for exactly the intended pairs."""
    specs = [
        ("S_STOCKOUT", "P1", 0, 0, 0, 0),   # no promo/holiday change
        ("S_PROMO", "P2", 0, 10, 0, 0),     # discount jumps 0 -> 10 tomorrow
        ("S_HOLIDAY", "P3", 0, 0, 0, 1),    # holiday flag turns on tomorrow
        ("S_SAFETY", "P4", 0, 0, 0, 0),     # no promo/holiday change
        ("S_STABLE", "P5", 0, 0, 0, 0),     # no promo/holiday change
    ]
    rows = []
    for store, product, disc_today, disc_tomorrow, hol_today, hol_tomorrow in specs:
        rows.append(make_row(store, product, "2024-06-01", disc_today, hol_today))
        rows.append(make_row(store, product, "2024-06-02", disc_tomorrow, hol_tomorrow))
    return pd.DataFrame(rows)


class FakeRecommender:
    """Returns whatever recommend_from_history/plan_ahead output is
    engineered per pair, so the classification rules in
    compute_todays_priorities can be checked without running the real RL
    policy or LightGBM forecast."""

    def __init__(self, recs_by_pair, plans_by_pair, raise_for=()):
        self.recs_by_pair = recs_by_pair
        self.plans_by_pair = plans_by_pair
        self.raise_for = set(raise_for)

    def recommend_from_history(self, store_id, product_id, date):
        if store_id in self.raise_for:
            raise RuntimeError("simulated failure")
        return self.recs_by_pair[(store_id, product_id)]

    def plan_ahead(self, store_id, product_id, start_date, n_days=2, demand_scenario="p90"):
        return self.plans_by_pair[(store_id, product_id)]


class RecordingRecommender(FakeRecommender):
    """Same as FakeRecommender, but records which demand_scenario each
    plan_ahead() call was made with -- used to regression-test the fix for
    a real bug: an earlier version called plan_ahead with
    demand_scenario="p90" here, which made "stockout risk tomorrow" almost
    always coincide with "safety buffer negative today" (both boil down to
    "today's order doesn't cover P90 demand"), so Medium was essentially
    unreachable -- every pair came back either High or Low, never Yellow."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_ahead_calls = []

    def plan_ahead(self, store_id, product_id, start_date, n_days=2, demand_scenario="p90"):
        self.plan_ahead_calls.append(demand_scenario)
        return super().plan_ahead(store_id, product_id, start_date, n_days, demand_scenario)


def make_rec(order_qty, demand_mean, demand_p90, current_stock):
    return {
        "fulfilled_order_qty": order_qty,
        "forecast_mean": demand_mean,
        "forecast_q90": demand_p90,
        "current_inventory": current_stock,
    }


def make_plan(stockout_risk_tomorrow):
    return pd.DataFrame([
        {"date": pd.Timestamp("2024-06-01"), "stockout_risk_units_if_high_demand": 0.0},
        {"date": pd.Timestamp("2024-06-02"), "stockout_risk_units_if_high_demand": stockout_risk_tomorrow},
    ])


def main():
    history_df = make_history()

    # --- _today_and_tomorrow: second-to-last / last convention --------------
    today, tomorrow = _today_and_tomorrow(history_df, "S_STABLE", "P5")
    assert today == pd.Timestamp("2024-06-01") and tomorrow == pd.Timestamp("2024-06-02")
    single_day_df = history_df[history_df["Store ID"] == "S_STABLE"].iloc[[0]]
    today1, tomorrow1 = _today_and_tomorrow(single_day_df, "S_STABLE", "P5")
    assert today1 == pd.Timestamp("2024-06-01") and tomorrow1 is None
    print("[ok] _today_and_tomorrow: second-to-last/last, and single-day fallback")

    # --- _promo_or_holiday_starting: unit checks -----------------------------
    assert _promo_or_holiday_starting(history_df, "S_PROMO", "P2",
                                       pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")) is True
    assert _promo_or_holiday_starting(history_df, "S_HOLIDAY", "P3",
                                       pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")) is True
    assert _promo_or_holiday_starting(history_df, "S_STABLE", "P5",
                                       pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")) is False
    assert _promo_or_holiday_starting(history_df, "S_STABLE", "P5", pd.Timestamp("2024-06-01"), None) is False
    print("[ok] _promo_or_holiday_starting unit checks")

    # --- compute_todays_priorities: one engineered pair per tier -------------
    recs = {
        # safety_buffer = order_qty - (demand_p90 - current_stock); keep >= 0
        # for every pair except S_SAFETY, which is engineered negative.
        ("S_STOCKOUT", "P1"): make_rec(order_qty=250, demand_mean=100, demand_p90=130, current_stock=200),
        ("S_PROMO", "P2"): make_rec(order_qty=250, demand_mean=100, demand_p90=130, current_stock=200),
        ("S_HOLIDAY", "P3"): make_rec(order_qty=250, demand_mean=100, demand_p90=130, current_stock=200),
        ("S_SAFETY", "P4"): make_rec(order_qty=50, demand_mean=200, demand_p90=260, current_stock=10),
        ("S_STABLE", "P5"): make_rec(order_qty=250, demand_mean=100, demand_p90=130, current_stock=200),
    }
    plans = {
        ("S_STOCKOUT", "P1"): make_plan(stockout_risk_tomorrow=40.0),
        ("S_PROMO", "P2"): make_plan(stockout_risk_tomorrow=0.0),
        ("S_HOLIDAY", "P3"): make_plan(stockout_risk_tomorrow=0.0),
        ("S_SAFETY", "P4"): make_plan(stockout_risk_tomorrow=0.0),
        ("S_STABLE", "P5"): make_plan(stockout_risk_tomorrow=0.0),
    }
    # Sanity-check the engineered safety buffers land where intended before
    # trusting the classification assertions below.
    assert recs[("S_STOCKOUT", "P1")]["fulfilled_order_qty"] - (130 - 200) >= 0
    assert recs[("S_SAFETY", "P4")]["fulfilled_order_qty"] - (260 - 10) < 0

    fake = FakeRecommender(recs, plans)
    result = compute_todays_priorities(history_df, fake).set_index(["store_id", "product_id"])

    assert result.loc[("S_STOCKOUT", "P1"), "priority"] == "High"
    assert result.loc[("S_STOCKOUT", "P1"), "reason"] == "Stockout risk tomorrow"

    assert result.loc[("S_PROMO", "P2"), "priority"] == "High"
    assert result.loc[("S_PROMO", "P2"), "reason"] == "Promotion starts tomorrow"

    assert result.loc[("S_HOLIDAY", "P3"), "priority"] == "High"
    assert result.loc[("S_HOLIDAY", "P3"), "reason"] == "Promotion starts tomorrow"

    assert result.loc[("S_SAFETY", "P4"), "priority"] == "Medium"
    assert result.loc[("S_SAFETY", "P4"), "reason"] == "Inventory below safety buffer"

    assert result.loc[("S_STABLE", "P5"), "priority"] == "Low"
    assert result.loc[("S_STABLE", "P5"), "reason"] == "Stable"
    print("[ok] compute_todays_priorities assigns the correct tier + reason per engineered pair")

    # --- sort order: High -> Medium -> Low -----------------------------------
    ranked = compute_todays_priorities(history_df, fake)
    ranks = [PRIORITY_RANK[p] for p in ranked["priority"]]
    assert ranks == sorted(ranks), "priorities are not sorted High -> Medium -> Low"
    print("[ok] sort order is High -> Medium -> Low")

    # --- a pair whose recommend_from_history raises is skipped, not fatal ----
    flaky = FakeRecommender(recs, plans, raise_for={"S_STABLE"})
    result_flaky = compute_todays_priorities(history_df, flaky)
    assert ("S_STABLE", "P5") not in set(zip(result_flaky["store_id"], result_flaky["product_id"]))
    assert len(result_flaky) == 4
    print("[ok] a pair that errors is skipped rather than failing the whole batch")

    # --- empty input -> empty DataFrame, not an exception --------------------
    empty_result = compute_todays_priorities(history_df.iloc[0:0], fake)
    assert len(empty_result) == 0
    print("[ok] empty history -> empty result, no crash")

    # --- SAFETY_BUFFER_RATIO_THRESHOLD boundary: prove Medium is a real,
    # reachable *band*, not just the single hand-picked S_SAFETY case above.
    # ratio = safety_buffer / demand_p90; the S_STOCKOUT/S_PROMO/S_HOLIDAY/
    # S_STABLE history rows are reused (their promo/holiday/plan-ahead
    # signals are all "quiet" for these three, isolating the buffer check).
    just_below = SAFETY_BUFFER_RATIO_THRESHOLD - 0.01   # e.g. 0.19 -> Medium
    at_threshold = SAFETY_BUFFER_RATIO_THRESHOLD         # e.g. 0.20 -> Low (strict <)
    deeply_negative = -1.0                               # outright deficit -> Medium

    # Solve directly for current_stock so safety_buffer/demand_p90 lands
    # exactly at the intended ratio: safety_buffer = order - (p90 - stock)
    # => stock = safety_buffer - order + p90
    def rec_for_ratio(ratio, order_qty=100.0, demand_mean=100.0, demand_p90=200.0):
        safety_buffer = ratio * demand_p90
        current_stock = safety_buffer - order_qty + demand_p90
        return make_rec(order_qty=order_qty, demand_mean=demand_mean,
                         demand_p90=demand_p90, current_stock=current_stock)

    boundary_plans = {("S_STABLE", "P5"): make_plan(stockout_risk_tomorrow=0.0)}

    for ratio, expected_priority, label in [
        (just_below, "Medium", "just under threshold"),
        (at_threshold, "Low", "exactly at threshold (strict <, so not Medium)"),
        (deeply_negative, "Medium", "deeply negative (outright deficit)"),
        (0.5, "Low", "comfortably above threshold"),
    ]:
        boundary_recs = {("S_STABLE", "P5"): rec_for_ratio(ratio)}
        boundary_fake = FakeRecommender(boundary_recs, boundary_plans)
        r = compute_todays_priorities(history_df.iloc[history_df["Store ID"].eq("S_STABLE").values], boundary_fake)
        got = r.set_index(["store_id", "product_id"]).loc[("S_STABLE", "P5"), "priority"]
        assert got == expected_priority, f"ratio={ratio} ({label}): expected {expected_priority}, got {got}"
    print(f"[ok] SAFETY_BUFFER_RATIO_THRESHOLD ({SAFETY_BUFFER_RATIO_THRESHOLD}) boundary behaves as documented")

    # --- PROMO_DISCOUNT_INCREASE_THRESHOLD boundary: a small day-to-day
    # wiggle must NOT count as "a promotion starting".
    small_wiggle_df = pd.DataFrame([
        make_row("S_WIGGLE", "P9", "2024-06-01", discount=10, holiday=0),
        make_row("S_WIGGLE", "P9", "2024-06-02", discount=10 + PROMO_DISCOUNT_INCREASE_THRESHOLD - 1, holiday=0),
    ])
    real_promo_df = pd.DataFrame([
        make_row("S_WIGGLE", "P9", "2024-06-01", discount=10, holiday=0),
        make_row("S_WIGGLE", "P9", "2024-06-02", discount=10 + PROMO_DISCOUNT_INCREASE_THRESHOLD, holiday=0),
    ])
    assert _promo_or_holiday_starting(small_wiggle_df, "S_WIGGLE", "P9",
                                       pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")) is False
    assert _promo_or_holiday_starting(real_promo_df, "S_WIGGLE", "P9",
                                       pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")) is True
    print(f"[ok] PROMO_DISCOUNT_INCREASE_THRESHOLD ({PROMO_DISCOUNT_INCREASE_THRESHOLD}) "
          f"ignores small wiggles but catches a real jump")

    # --- regression: Medium must actually be reachable -----------------------
    # Guards against the P90-double-counting bug: compute_todays_priorities
    # must call plan_ahead with demand_scenario="mean" (not "p90") when
    # checking tomorrow's stockout risk, otherwise that check silently
    # absorbs every case that should have been Medium.
    recorder = RecordingRecommender(recs, plans)
    result_recorded = compute_todays_priorities(history_df, recorder)
    assert set(recorder.plan_ahead_calls) == {"mean"}, (
        f"plan_ahead was called with {set(recorder.plan_ahead_calls)} -- expected only 'mean'; "
        "calling it with 'p90' collapses Medium into High (see class docstring)"
    )
    assert "Medium" in set(result_recorded["priority"]), "Medium tier is unreachable -- the bug is back"
    print("[ok] regression: plan_ahead is called with demand_scenario='mean', and Medium is reachable")

    print("\nAll priorities smoke-test assertions passed.")


if __name__ == "__main__":
    main()
