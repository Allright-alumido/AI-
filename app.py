"""
app.py

Streamlit UI for the replenishment recommendation system.

Run with:
    streamlit run app.py
(Streamlit runs from a terminal and opens in your browser -- it isn't run
as a Jupyter cell.)

Four tabs:
  1. Recommendation -- opens with "Today's Priority": every store/product
     pair for the selected strategy, triaged Red/Yellow/Green (stockout
     risk tomorrow or a promotion starting, inventory below safety buffer,
     or stable) with a "View" button per row that jumps straight to that
     pair's recommendation below. Then: pick a strategy, store, product,
     and date; see the recommended order quantity, a numeric decision
     breakdown (Demand P90 / Current Stock / Safety Buffer / Expected
     Ending Inventory), a plain-language explanation, a confidence rating,
     a side-by-side comparison of all four strategies for the same
     situation, and a "What changed?" note versus the previous day.
  2. Stress test -- override demand/inventory/price/discount/holiday
     directly and watch the recommendation react live. Bypasses the
     LightGBM forecast call on purpose (see inference.recommend_manual).
  3. Decision vs. history -- side-by-side: what this policy would have
     ordered vs. what was actually ordered historically, over a chosen
     window for one store/product, with Service Level / Profit / Holding
     Cost / Stockout Cost shown for both sides, plus the profit uplift.
  4. Future plan -- a forward-looking, multi-day replenishment plan
     starting from a chosen date, for a horizon you pick. Unlike tab 3,
     this does NOT use real historical demand (the future hasn't happened
     yet) -- it recursively simulates using the LightGBM forecast itself,
     showing both an expected-demand case and a high-demand (P90) case so
     you can see how uncertainty widens the further out you look. If the
     requested horizon runs past the last real date this store/product
     has data for, it keeps going anyway ("extrapolation") using
     assumed/carried-forward price, discount, promo, etc. -- see the
     "Assumptions for dates beyond the data" expander and the
     `Recommender.plan_ahead` docstring in inference.py.

Requires: streamlit, gymnasium, stable-baselines3, torch, lightgbm,
scikit-learn, joblib, pandas, numpy (see requirements.txt), plus trained
models under models/<scenario>/seed_<seed>/ (run train_rl.py first). The
"Policy Comparison" section on the Recommendation tab loads all four
scenarios' models, not just the selected one -- it still works with
whichever subset is trained, showing "Not trained yet" for the rest.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from feature_engineering import prepare_history
from demand_model import DemandModel
from inference import Recommender, SCENARIO_LABELS, LABEL_TO_SCENARIO
from explain import explain_recommendation, compute_decision_breakdown, confidence_level
from priorities import compute_todays_priorities, PRIORITY_ICON
from reward_configs import SCENARIOS
from replenishment_env import MAX_INVENTORY
from streaming_pipeline import StreamingPipeline

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

DATA_PATH = "retail_store_inventory.csv"
DEFAULT_SEED = 0
POLICY_COMPARISON_CSV = "policy_comparison.csv"

st.set_page_config(page_title="Replenishment Recommender", layout="wide")


@st.cache_resource
def load_shared_resources():
    raw = pd.read_csv(DATA_PATH)
    history_df = prepare_history(raw)
    demand_model = DemandModel()
    return history_df, demand_model


@st.cache_resource
def load_recommender(scenario_name: str, seed: int = DEFAULT_SEED):
    history_df, demand_model = load_shared_resources()
    return Recommender(scenario_name, seed=seed, history_df=history_df, demand_model=demand_model)


@st.cache_data
def load_policy_comparison_df():
    """Aggregate backtest stats from evaluate_policies.py, if it's been
    run. Returns None (not an exception) if the file doesn't exist yet --
    every caller here treats that as "just don't show the stats"."""
    if not os.path.exists(POLICY_COMPARISON_CSV):
        return None
    try:
        return pd.read_csv(POLICY_COMPARISON_CSV, index_col=0)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _priorities_cached(scenario_name: str, cache_day: str, _recommender):
    """Cached by (scenario, calendar day) so the full store x product scan
    (recommend_from_history + a 2-day plan_ahead for every pair) only runs
    once per strategy per day, not on every rerun. `_recommender` is
    prefixed with an underscore so Streamlit doesn't try to hash the
    (unhashable) model object -- caching is keyed on scenario_name/cache_day
    instead, which is exactly the granularity we want anyway."""
    history_df, _ = load_shared_resources()
    return compute_todays_priorities(history_df, _recommender)


def sidebar_scenario_picker() -> str:
    st.sidebar.header("Strategy")
    label = st.sidebar.selectbox("Choose a replenishment strategy", list(LABEL_TO_SCENARIO.keys()))
    scenario_name = LABEL_TO_SCENARIO[label]
    st.sidebar.caption(SCENARIOS[scenario_name].description)
    return scenario_name


def management_summary(scenario_name: str) -> str:
    """Short business-facing blurb, using aggregate backtest stats if
    available, else the static scenario description alone."""
    base = SCENARIOS[scenario_name].description
    df = load_policy_comparison_df()
    if df is None or scenario_name not in df.index:
        return base
    row = df.loc[scenario_name]
    return (
        f"{base} Based on backtesting: ~{row['service_level'] * 100:.0f}% service level, "
        f"~{row['avg_end_inventory']:.0f} units average inventory, and "
        f"${row['profit']:,.0f} total profit on the historical evaluation set."
    )


def get_row(history_df, store_id, product_id, date):
    g = history_df[(history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)]
    match = g[g["Date"] == pd.Timestamp(date)]
    return match.iloc[0] if len(match) else None


def previous_date_for(history_df, store_id, product_id, date):
    dates = sorted(pd.Timestamp(d) for d in history_df[
        (history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)
    ]["Date"].unique())
    ts = pd.Timestamp(date)
    if ts not in dates:
        return None
    idx = dates.index(ts)
    return dates[idx - 1] if idx > 0 else None


def what_changed_reasons(rec_today: dict, rec_yesterday: dict, row_today, row_yesterday) -> list[str]:
    reasons = []
    if rec_today["forecast_mean"] > rec_yesterday["forecast_mean"] * 1.10:
        reasons.append("Demand Increased")
    if rec_today["current_inventory"] < rec_yesterday["current_inventory"]:
        reasons.append("Inventory Lower")
    if row_today["Discount"] > row_yesterday["Discount"]:
        reasons.append("Promotion Added")
    if row_today["Holiday/Promotion"] == 1 and row_yesterday["Holiday/Promotion"] == 0:
        reasons.append("Holiday Approaching")
    return reasons


def cascading_selectbox(label, options, key, default_index=0, format_func=None):
    """A selectbox for use in a Store -> Product -> Date style chain, where
    changing an earlier widget can make the current session_state value for
    a later widget invalid (e.g. picking a new Store means the previously
    selected Product may not exist for that store).

    The bug this fixes: passing both `key=` and `index=` to st.selectbox
    only sets the default on first render -- once `key` has a value in
    st.session_state, Streamlit keeps using that stored value and ignores
    `index` on every later rerun. If that stored value is no longer present
    in the freshly computed `options` list (because a parent widget just
    changed), Streamlit has nothing valid to show and the widget appears to
    "snap back"/reset in a way that feels arbitrary to the user -- this is
    what was reported as "it can't choose other store or product, while I
    want to change, it will refresh to the original."

    The fix: validate/repair st.session_state[key] *before* creating the
    widget (reset it to `options[default_index]` only if it's missing or no
    longer valid), then create the widget with `key=` alone, never `index=`.
    Since session_state already holds a valid value at that point, Streamlit
    uses it directly and there's no conflict.
    """
    if not options:
        return None
    if key not in st.session_state or st.session_state[key] not in options:
        safe_index = default_index if -len(options) <= default_index < len(options) else 0
        st.session_state[key] = options[safe_index]
    kwargs = {"key": key}
    if format_func is not None:
        kwargs["format_func"] = format_func
    return st.selectbox(label, options, **kwargs)


def default_future_assumptions(history_df, store_id: str, product_id: str) -> dict:
    """Starting values for the Future plan tab's "assumptions for dates
    beyond the data" inputs -- carried forward from the last real row for
    this store/product, mirroring plan_ahead's own carry-forward default
    in inference.py (holiday_promo isn't included: its sensible default is
    False regardless of the last real day, see plan_ahead's docstring)."""
    g = history_df[(history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)]
    last_row = g.sort_values("Date").iloc[-1]
    return {
        "price": float(last_row["Price"]),
        "discount": float(last_row["Discount"]),
        "competitor_pricing": float(last_row["Competitor Pricing"]),
    }


# ---------------------------------------------------------------------- #
# Today's Priority
# ---------------------------------------------------------------------- #
def todays_priority_section(scenario_name, recommender):
    st.markdown("## Today's Priority")
    st.caption(
        "Every store/product pair for this strategy, ranked by what needs a "
        "decision today -- run once for the whole store network so you don't "
        "have to click through each pair one at a time. Click \"View\" to "
        "load that pair below."
    )

    cache_day = pd.Timestamp.today().strftime("%Y-%m-%d")
    with st.spinner("Scanning all store x product combinations..."):
        priorities_df = _priorities_cached(scenario_name, cache_day, recommender)

    if priorities_df is None or len(priorities_df) == 0:
        st.info("No priority data available yet for this strategy.")
        return

    counts = priorities_df["priority"].value_counts()
    st.caption(
        f"{PRIORITY_ICON['High']} {counts.get('High', 0)} High   "
        f"{PRIORITY_ICON['Medium']} {counts.get('Medium', 0)} Medium   "
        f"{PRIORITY_ICON['Low']} {counts.get('Low', 0)} Low"
    )

    def render_rows(rows_df):
        header = st.columns([1.1, 1.2, 1.2, 2.8, 0.9])
        for col, text in zip(header, ["Priority", "Store", "Product", "Reason", "Action"]):
            col.markdown(f"**{text}**")
        for _, row in rows_df.iterrows():
            c1, c2, c3, c4, c5 = st.columns([1.1, 1.2, 1.2, 2.8, 0.9])
            c1.write(f"{PRIORITY_ICON[row['priority']]} {row['priority']}")
            c2.write(row["store_id"])
            c3.write(row["product_id"])
            c4.write(row["reason"])
            if c5.button("View", key=f"priority_view_{row['store_id']}_{row['product_id']}"):
                st.session_state["rec_store"] = row["store_id"]
                st.session_state["rec_product"] = row["product_id"]
                st.session_state["rec_date"] = row["date"]
                st.session_state["rec_state"] = {
                    "store_id": row["store_id"], "product_id": row["product_id"],
                    "date": row["date"], "scenario": scenario_name,
                }
                st.rerun()

    urgent = priorities_df[priorities_df["priority"].isin(["High", "Medium"])]
    stable = priorities_df[priorities_df["priority"] == "Low"]

    if len(urgent) > 0:
        render_rows(urgent)
    else:
        st.success("Nothing urgent today -- every pair is stable.")

    if len(stable) > 0:
        with st.expander(f"{len(stable)} more (stable)"):
            render_rows(stable)

    st.divider()


# ---------------------------------------------------------------------- #
# Tab 1: Recommendation
# ---------------------------------------------------------------------- #
def tab_recommendation(scenario_name, history_df, recommender):
    todays_priority_section(scenario_name, recommender)

    st.subheader("Get a recommendation")
    st.info(management_summary(scenario_name))

    col1, col2, col3 = st.columns(3)
    store_ids = sorted(history_df["Store ID"].unique())
    with col1:
        store_id = cascading_selectbox("Store", store_ids, key="rec_store")
    products = sorted(history_df[history_df["Store ID"] == store_id]["Product ID"].unique())
    with col2:
        product_id = cascading_selectbox("Product", products, key="rec_product")
    dates = sorted(history_df[
        (history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)
    ]["Date"].unique())
    with col3:
        date = cascading_selectbox(
            "Date", dates, key="rec_date", default_index=len(dates) - 1,
            format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        )

    if st.button("Get recommendation", type="primary"):
        st.session_state["rec_state"] = {
            "store_id": store_id, "product_id": product_id, "date": date, "scenario": scenario_name,
        }

    rs = st.session_state.get("rec_state")
    if not rs:
        return
    # If the sidebar strategy changed since the last click, keep showing
    # results for the strategy that's currently selected.
    rs["scenario"] = scenario_name

    rec = recommender.recommend_from_history(rs["store_id"], rs["product_id"], rs["date"])

    st.metric("Recommended order quantity", f"{rec['fulfilled_order_qty']:.0f} units")
    if rec["capped_by_capacity"]:
        st.warning(
            f"Policy wanted to order {rec['requested_order_qty']:.0f} units, "
            f"capped at warehouse capacity ({MAX_INVENTORY})."
        )

    # --- Decision breakdown ------------------------------------------------
    b = compute_decision_breakdown(rec)
    st.markdown("**Decision breakdown**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demand P90", f"{b['demand_p90']:.0f}")
    c2.metric("Current Stock", f"{b['current_stock']:.0f}")
    c3.metric("Safety Buffer", f"{b['safety_buffer']:.0f}")
    c4.metric("Expected Ending Inventory", f"{b['expected_ending_inventory']:.0f}")
    if b["safety_buffer"] < 0:
        st.caption("Note: this order does not fully cover worst-case (P90) demand.")

    # --- Confidence ---------------------------------------------------------
    conf = confidence_level(rec)
    st.markdown("**Recommendation Confidence**")
    badge_fn = {"green": st.success, "orange": st.warning, "red": st.error}[conf["color"]]
    badge_fn(conf["label"])
    st.progress(conf["score"])
    st.caption(f"Based on the gap between mean and P90 demand forecast ({conf['gap_ratio'] * 100:.0f}% of the mean).")

    # --- Explanation (LLM or template) --------------------------------------
    st.markdown("**Why this recommendation**")
    with st.spinner("Generating explanation..."):
        text, source = explain_recommendation(rec)
    source_label = {
        "anthropic": "AI-generated (Anthropic)", "gemini": "AI-generated (Gemini)",
        "openai": "AI-generated (OpenAI)", "template": "Template-based (no LLM configured)",
    }[source]
    with st.chat_message("assistant"):
        st.caption(source_label)
        st.write(text)

    # --- Policy comparison ---------------------------------------------------
    st.markdown("### Policy Comparison")
    st.caption("How the other strategies would respond to this exact same situation.")
    policy_df = load_policy_comparison_df()
    if policy_df is not None:
        policy_df = policy_df.loc[policy_df.index.isin(SCENARIOS.keys())]

    qty_by_policy = {}
    cols = st.columns(4)
    for col, name in zip(cols, SCENARIOS.keys()):
        with col:
            st.markdown(f"**{SCENARIO_LABELS[name]}**")
            try:
                other = load_recommender(name)
                other_rec = other.recommend_from_history(rs["store_id"], rs["product_id"], rs["date"])
                st.metric("Order qty", f"{other_rec['fulfilled_order_qty']:.0f}")
                qty_by_policy[SCENARIO_LABELS[name]] = other_rec["fulfilled_order_qty"]
            except FileNotFoundError:
                st.warning("Not trained yet")
                qty_by_policy[SCENARIO_LABELS[name]] = None
            if policy_df is not None and name in policy_df.index:
                row = policy_df.loc[name]
                st.caption(f"Service level: {row['service_level'] * 100:.0f}%")
                st.caption(f"Avg inventory: {row['avg_end_inventory']:.0f}")
                st.caption(f"Profit: ${row['profit']:,.0f}")
            else:
                st.caption("Backtest stats unavailable")
            st.caption(SCENARIOS[name].description)

    valid_qty = {k: v for k, v in qty_by_policy.items() if v is not None}
    if len(valid_qty) > 1:
        st.markdown("**Recommended order quantity by strategy**")
        st.bar_chart(pd.Series(valid_qty, name="order_qty"))

    if policy_df is not None and len(policy_df) > 1:
        st.markdown("**Backtested performance by strategy**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.caption("Service level")
            st.bar_chart(policy_df["service_level"])
        with cc2:
            st.caption("Avg ending inventory")
            st.bar_chart(policy_df["avg_end_inventory"])
        with cc3:
            st.caption("Profit")
            st.bar_chart(policy_df["profit"])

    # --- What changed? -------------------------------------------------------
    prev_date = previous_date_for(history_df, rs["store_id"], rs["product_id"], rs["date"])
    if prev_date is not None:
        rec_prev = recommender.recommend_from_history(rs["store_id"], rs["product_id"], prev_date)
        prev_qty = max(rec_prev["fulfilled_order_qty"], 1e-6)
        pct_change = (rec["fulfilled_order_qty"] - rec_prev["fulfilled_order_qty"]) / prev_qty
        if pct_change > 0.20:
            st.markdown("### What changed?")
            st.caption(
                f"Order increased {pct_change * 100:.0f}% versus the previous day "
                f"({rec_prev['fulfilled_order_qty']:.0f} -> {rec['fulfilled_order_qty']:.0f} units)."
            )
            row_today = get_row(history_df, rs["store_id"], rs["product_id"], rs["date"])
            row_prev = get_row(history_df, rs["store_id"], rs["product_id"], prev_date)
            reasons = what_changed_reasons(rec, rec_prev, row_today, row_prev)
            if reasons:
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.caption("No specific driver detected among the tracked factors "
                           "(demand, inventory, promotion, holiday).")


# ---------------------------------------------------------------------- #
# Tab 2: Stress test
# ---------------------------------------------------------------------- #
def tab_stress_test(scenario_name, history_df, recommender):
    st.subheader("Stress test: what if...?")
    st.caption(
        "Override the situation directly and see how the recommendation reacts. "
        "This bypasses the demand-forecasting model -- you're testing the *strategy's* "
        "reaction to a scenario you specify, not asking it to forecast anything."
    )

    col1, col2 = st.columns(2)
    store_ids = sorted(history_df["Store ID"].unique())
    with col1:
        store_id = cascading_selectbox("Store (for product context)", store_ids, key="stress_store")
    products = sorted(history_df[history_df["Store ID"] == store_id]["Product ID"].unique())
    with col2:
        product_id = cascading_selectbox("Product", products, key="stress_product")

    ref = history_df[
        (history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)
    ].iloc[-1]

    st.markdown("**Situation**")
    c1, c2, c3 = st.columns(3)
    with c1:
        current_inventory = st.slider("Current inventory", 0, 500, int(ref["Inventory Level"]))
        forecast_mean = st.slider("Expected demand today", 0, 500, int(ref["Demand Forecast"]))
    with c2:
        price = st.slider("Price ($)", 10.0, 100.0, float(ref["Price"]))
        discount = st.slider("Discount (%)", 0, 20, int(ref["Discount"]))
    with c3:
        holiday_promo = st.checkbox("Holiday / promotion today", bool(ref["Holiday/Promotion"]))
        competitor_pricing = st.slider("Competitor pricing ($)", 10.0, 100.0, float(ref["Competitor Pricing"]))

    rec = recommender.recommend_manual(
        store_id=store_id, product_id=product_id, category=ref["Category"], region=ref["Region"],
        weather=ref["Weather Condition"], seasonality=ref["Seasonality"],
        price=price, discount=discount, competitor_pricing=competitor_pricing,
        holiday_promo=holiday_promo, current_inventory=current_inventory, forecast_mean=forecast_mean,
    )

    st.metric("Recommended order quantity", f"{rec['fulfilled_order_qty']:.0f} units")
    if rec["capped_by_capacity"]:
        st.warning(
            f"Policy wanted to order {rec['requested_order_qty']:.0f} units, "
            f"capped at warehouse capacity ({MAX_INVENTORY})."
        )
    st.caption(f"90th-percentile demand used for this scenario: {rec['forecast_q90']:.0f} units.")


# ---------------------------------------------------------------------- #
# Tab 3: Decision vs. history
# ---------------------------------------------------------------------- #
def tab_comparison(scenario_name, history_df, recommender):
    st.subheader("Decision vs. historical experience")

    col1, col2, col3 = st.columns(3)
    store_ids = sorted(history_df["Store ID"].unique())
    with col1:
        store_id = cascading_selectbox("Store", store_ids, key="cmp_store")
    products = sorted(history_df[history_df["Store ID"] == store_id]["Product ID"].unique())
    with col2:
        product_id = cascading_selectbox("Product", products, key="cmp_product")
    with col3:
        n_days = st.slider("Window length (days)", 30, 180, 90, key="cmp_ndays")

    dates = sorted(history_df[
        (history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)
    ]["Date"].unique())
    selectable_starts = dates[:-n_days] if len(dates) > n_days else dates[:1]
    # select_slider has no `key` here on purpose: with one, the same
    # session-state-vs-options mismatch as the Store/Product selectboxes
    # above would apply whenever Store/Product/window length changes the
    # available start dates. Resetting the value into session_state first
    # (the same fix as cascading_selectbox) keeps this widget's selection
    # valid without needing a key at all.
    if st.session_state.get("cmp_start_date") not in selectable_starts:
        st.session_state["cmp_start_date"] = selectable_starts[0]
    start_date = st.select_slider(
        "Start date", options=selectable_starts, key="cmp_start_date",
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
    )

    if st.button("Compare", type="primary"):
        with st.spinner("Running both trajectories..."):
            result = recommender.compare_to_historical(store_id, product_id, start_date, n_days=n_days)

        uplift = result["profit_uplift"]
        hist_profit = result["historical_kpis"]["profit"]
        pct = (uplift / hist_profit * 100) if hist_profit else 0.0
        st.metric(
            f"Additional profit if \"{SCENARIO_LABELS[scenario_name]}\" had been used",
            f"${uplift:,.0f}", delta=f"{pct:.0f}% vs. historical",
        )

        rl_kpis = result["rl_kpis"]
        hist_kpis = result["historical_kpis"]

        left, right = st.columns(2)
        with left:
            st.markdown(f"### {SCENARIO_LABELS[scenario_name]} (this strategy)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Service Level", f"{rl_kpis['service_level'] * 100:.0f}%")
            k2.metric("Profit", f"${rl_kpis['profit']:,.0f}")
            k3.metric("Holding Cost", f"${rl_kpis['holding_cost']:,.0f}")
            k4.metric("Stockout Cost", f"${rl_kpis['stockout_cost_unweighted']:,.0f}")
            with st.expander("All details"):
                st.dataframe(pd.Series(rl_kpis, name="value"))
            st.line_chart(result["rl_log"].set_index("date")[["end_inventory", "order_qty"]])
        with right:
            st.markdown("### Historical decisions")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Service Level", f"{hist_kpis['service_level'] * 100:.0f}%")
            k2.metric("Profit", f"${hist_kpis['profit']:,.0f}")
            k3.metric("Holding Cost", f"${hist_kpis['holding_cost']:,.0f}")
            k4.metric("Stockout Cost", f"${hist_kpis['stockout_cost_unweighted']:,.0f}")
            with st.expander("All details"):
                st.dataframe(pd.Series(hist_kpis, name="value"))
            st.line_chart(result["historical_log"].set_index("date")[["end_inventory", "order_qty"]])


# ---------------------------------------------------------------------- #
# Tab 4: Future plan
# ---------------------------------------------------------------------- #
def tab_future_plan(scenario_name, history_df, recommender):
    st.subheader("Future replenishment plan")
    st.caption(
        "Simulates forward from a chosen start date using this strategy, since "
        "real future demand isn't known yet -- it recursively advances inventory "
        "using the LightGBM forecast itself. Two lines are shown: an expected-demand "
        "case and a high-demand (90th percentile) case, so you can see how much the "
        "plan's confidence narrows or widens the further out you look."
    )

    col1, col2, col3 = st.columns(3)
    store_ids = sorted(history_df["Store ID"].unique())
    with col1:
        store_id = cascading_selectbox("Store", store_ids, key="plan_store")
    products = sorted(history_df[history_df["Store ID"] == store_id]["Product ID"].unique())
    with col2:
        product_id = cascading_selectbox("Product", products, key="plan_product")
    dates = sorted(history_df[
        (history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)
    ]["Date"].unique())
    with col3:
        start_date = cascading_selectbox(
            "Start date (\"today\")", dates, key="plan_start_date", default_index=len(dates) - 1,
            format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        )

    n_days = st.slider("How many days ahead to simulate", 3, 60, 14, key="plan_ndays")

    # How much of the requested horizon is backed by real data vs. how much
    # will need to be extrapolated, given the current Store/Product/Start
    # date/horizon selection -- used both for the heads-up message below and
    # to decide whether to auto-expand the assumptions expander.
    available_days = len(dates) - dates.index(start_date)
    extrapolated_days = max(0, n_days - available_days)

    # Reset the "assumptions for the future" inputs to this store/product's
    # own last known values whenever the pair changes -- same
    # key-vs-default-value conflict cascading_selectbox fixes above would
    # otherwise apply here too (a slider's `key` keeps its stored value
    # across reruns, so without this it would keep showing the *previous*
    # store/product's discount after switching).
    pair_key = (store_id, product_id)
    if st.session_state.get("plan_future_pair") != pair_key:
        st.session_state["plan_future_pair"] = pair_key
        defaults = default_future_assumptions(history_df, store_id, product_id)
        st.session_state["plan_future_discount"] = min(50, max(0, int(defaults["discount"])))
        st.session_state["plan_future_price"] = float(defaults["price"])
        st.session_state["plan_future_competitor_pricing"] = float(defaults["competitor_pricing"])
        st.session_state["plan_future_holiday"] = False

    if extrapolated_days > 0:
        st.info(
            f"Real data for this store/product runs out after {available_days} of the "
            f"{n_days} requested days. The remaining {extrapolated_days} will be "
            f"**extrapolated** using the assumptions below instead of stopping there."
        )
    with st.expander("Assumptions for dates beyond the data", expanded=(extrapolated_days > 0)):
        st.caption(
            "Only affects days past the real data (see above) -- everything up to that "
            "point still uses this store/product's real price/discount/promo/weather "
            "history. Defaults to whatever was last in effect; override to plan against "
            "a specific future scenario instead."
        )
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            future_discount = st.slider("Discount (%)", 0, 50, key="plan_future_discount")
        with a2:
            future_holiday_promo = st.checkbox("Holiday/promotion running", key="plan_future_holiday")
        with a3:
            future_price = st.number_input("Price ($)", min_value=0.0, key="plan_future_price")
        with a4:
            future_competitor_pricing = st.number_input(
                "Competitor pricing ($)", min_value=0.0, key="plan_future_competitor_pricing"
            )
    future_assumptions = {
        "discount": future_discount,
        "holiday_promo": future_holiday_promo,
        "price": future_price,
        "competitor_pricing": future_competitor_pricing,
    }

    if st.button("Generate plan", type="primary"):
        with st.spinner("Simulating forward..."):
            plan_mean = recommender.plan_ahead(
                store_id, product_id, start_date, n_days=n_days, demand_scenario="mean",
                future_assumptions=future_assumptions,
            )
            plan_p90 = recommender.plan_ahead(
                store_id, product_id, start_date, n_days=n_days, demand_scenario="p90",
                future_assumptions=future_assumptions,
            )

        if plan_mean.attrs.get("uses_extrapolation"):
            st.warning(
                f"{plan_mean.attrs['real_days']} of the {plan_mean.attrs['requested_days']} days "
                f"below are backed by real historical data; the remaining "
                f"{plan_mean.attrs['extrapolated_days']} are extrapolated using the assumptions "
                f"above instead of real price/discount/promo/weather records -- see the "
                f"**Data source** column in the day-by-day table for which is which."
            )

        st.markdown("**Recommended order quantity by day**")
        chart_orders = plan_mean[["date", "recommended_order_qty"]].set_index("date")
        st.bar_chart(chart_orders)

        st.markdown("**Expected ending inventory range**")
        inv_band = pd.DataFrame({
            "date": plan_mean["date"],
            "Ending inventory (expected demand)": plan_mean["expected_ending_inventory"].values,
            "Ending inventory (high-demand case)": plan_p90["expected_ending_inventory"].values,
        }).set_index("date")
        st.line_chart(inv_band)
        st.caption(
            "The gap between these two lines widens the further out you look -- that's "
            "compounding forecast uncertainty, not a bug. Treat the near-term days with "
            "more confidence than the far-out ones, and treat any extrapolated days (see "
            "the Data source column below) with extra caution on top of that, since "
            "they're also assuming rather than using real price/discount/promo context."
        )

        risky_days = plan_mean[plan_mean["stockout_risk_units_if_high_demand"] > 0]
        if len(risky_days) > 0:
            st.warning(
                f"{len(risky_days)} of {len(plan_mean)} days would risk a stockout if "
                f"actual demand ran at the high (P90) end instead of the expected case."
            )

        st.markdown("**Day-by-day plan**")
        st.caption(
            "If the recommended order looks the same for several days in a row, "
            "that's usually the policy settling into a steady reorder-to-target "
            "pattern once inventory stabilizes -- not a bug. Compare against "
            "**Starting inventory** below: if it's also flat and sitting at 0 or "
            "at the warehouse cap, the order is being floor/capacity-limited every "
            "day, which is worth a closer look via the Stress test tab; if "
            "inventory is moving but the order isn't, that's expected convergence."
        )
        display_df = plan_mean[[
            "date", "data_source", "starting_inventory", "recommended_order_qty", "forecast_mean", "forecast_q90",
            "expected_ending_inventory", "stockout_risk_units_if_high_demand",
        ]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df["data_source"] = display_df["data_source"].map(
            {"historical": "Historical", "extrapolated": "Extrapolated (assumed)"}
        )
        display_df = display_df.rename(columns={
            "data_source": "Data source",
            "starting_inventory": "Starting inventory",
            "recommended_order_qty": "Recommended order",
            "forecast_mean": "Demand forecast (mean)",
            "forecast_q90": "Demand forecast (P90)",
            "expected_ending_inventory": "Expected ending inventory",
            "stockout_risk_units_if_high_demand": "Stockout risk (units, if high demand)",
        })
        st.dataframe(display_df, use_container_width=True)


# ---------------------------------------------------------------------- #
# Tab 5: Live Stream Monitor
#
# Wires streaming_pipeline.StreamingPipeline into the UI. Ingestion/
# Processing/Decision/Serving all happen in streaming_pipeline.py /
# streaming_data_generator.py -- this function is presentation only: it
# calls pipeline.tick() and renders whatever comes back. See
# proposal_assets/lambda_architecture.png for how this maps onto a Lambda
# Architecture's Batch vs. Speed layer split.
#
# Interaction model (as scoped): auto-play is primary, manual step is
# always available as a fallback/complement, and a "smart pause" stops
# auto-play automatically the moment any pair shows a stockout this tick,
# so a live demo never silently scrolls past the interesting moment.
# ---------------------------------------------------------------------- #
def _init_stream_pipeline(scenario_name: str, history_df, compare_all: bool, use_case_memory: bool = True):
    names = list(SCENARIOS.keys()) if compare_all else [scenario_name]
    recommenders = {}
    for name in names:
        try:
            recommenders[name] = load_recommender(name)
        except FileNotFoundError:
            continue
    if scenario_name not in recommenders:
        return None
    return StreamingPipeline(recommenders, primary_scenario=scenario_name,
                              history_df=history_df, rng_seed=0, use_case_memory=use_case_memory)


STOCKOUT_ALERT_THRESHOLD = 0.05  # pause only once at least this fraction of pairs stock out this tick


def _check_stream_alerts(results: dict, sim_date: str) -> str | None:
    """Pure function (no Streamlit/session_state touched) so this can be
    unit-tested directly: given one tick's {pair_key: result_dict} plus the
    simulated date, decide whether auto-play should smart-pause, and if so,
    with what message. Returns None if nothing noteworthy happened.

    Gated on STOCKOUT_ALERT_THRESHOLD (fraction of pairs affected, not raw
    count): with dozens/hundreds of (store, product) pairs running at once,
    at least ONE going into stockout on any given tick is common and not,
    by itself, noteworthy -- pausing on every single tick for that would
    make auto-play unusable. Pausing only once a meaningful share of pairs
    are affected keeps the smart-pause reserved for genuinely significant
    moments (e.g. a correlated demand spike or a policy that's stocking out
    broadly), matching what a demo audience would actually want flagged."""
    alert_pairs = [k for k, r in results.items() if r["stockout_units"] > 0]
    if not alert_pairs:
        return None
    fraction = len(alert_pairs) / len(results)
    if fraction < STOCKOUT_ALERT_THRESHOLD:
        return None
    example = alert_pairs[0]
    return (f"{example} on {sim_date} went into stockout "
            f"({len(alert_pairs)} of {len(results)} pairs affected this tick, "
            f"{fraction * 100:.0f}% -- above the {STOCKOUT_ALERT_THRESHOLD * 100:.0f}% alert threshold).")


def _stream_tick_and_check_alerts(pipeline: StreamingPipeline):
    snapshot = pipeline.tick()
    pipeline.write_snapshot(snapshot)

    results = snapshot["results"]
    total_profit = sum(r["profit"] for r in results.values())
    total_stockout_units = sum(r["stockout_units"] for r in results.values())

    st.session_state["stream_history"].append({
        "tick": snapshot["tick"], "sim_date": snapshot["sim_date"],
        "total_profit": total_profit, "total_stockout_units": total_stockout_units,
    })
    st.session_state["stream_latest"] = snapshot

    pause_reason = _check_stream_alerts(results, snapshot["sim_date"])
    if pause_reason:
        st.session_state["stream_autoplay"] = False
        st.session_state["stream_pause_reason"] = pause_reason


def _render_stream_snapshot():
    snapshot = st.session_state.get("stream_latest")
    if snapshot is None:
        st.info("Press ▶ Start auto-play or ⏭ Advance 1 hour to begin the simulation.")
        return

    st.markdown(f"**Tick {snapshot['tick']}** — simulated date **{snapshot['sim_date']}** "
                f"— primary strategy: **{snapshot['scenario_label']}**")

    results = snapshot["results"]
    total_profit = sum(r["profit"] for r in results.values())
    n_stockout = sum(1 for r in results.values() if r["stockout_units"] > 0)
    avg_order = sum(r["recommended_order_qty"] for r in results.values()) / max(len(results), 1)

    k1, k2, k3 = st.columns(3)
    k1.metric("This tick's total profit (all pairs)", f"${total_profit:,.0f}")
    k2.metric("Pairs with a stockout this tick", f"{n_stockout} / {len(results)}")
    k3.metric("Avg recommended order qty", f"{avg_order:.0f}")

    rows = []
    nudge_rows = []
    for pair_key, r in results.items():
        rows.append({
            "Store/Product": pair_key,
            "Order qty": r["recommended_order_qty"],
            "Forecast (mean)": round(r["forecast_mean"], 1),
            "Units sold": round(r["units_sold"], 1),
            "Stockout units": round(r["stockout_units"], 1),
            "Ending inventory": round(r["end_inventory"], 1),
            "Profit": round(r["profit"], 1),
        })
        case_memory = r.get("case_memory") or {}
        if case_memory.get("nudged"):
            nudge_rows.append((pair_key, case_memory["explanation"]))
    df = pd.DataFrame(rows).sort_values("Stockout units", ascending=False)
    st.dataframe(df, use_container_width=True, height=320)

    # --- compare_all: what the OTHER 3 strategies would have ordered this
    # tick, aggregated across all tracked pairs. This is what the "Also load
    # the other 3 strategies for side-by-side comparison" checkbox actually
    # turns on -- streaming_pipeline.py computes it every tick regardless
    # (it's in each result's "other_scenarios" dict), this is just the
    # first place in the UI that displays it. Aggregated rather than shown
    # per-pair because with up to ~100 pairs tracked at once, a per-pair x
    # 4-strategy table would be unreadable; the aggregate total answers the
    # question a demo audience actually asks ("would a different strategy
    # have ordered a lot more/less overall?").
    sample_result = next(iter(results.values()), {})
    other_names = list(sample_result.get("other_scenarios", {}))
    if other_names:
        st.markdown("**All 4 strategies' total order quantity this tick**")
        st.caption("Summed across every tracked store/product pair -- how the other 3 strategies "
                   "would have reacted to this exact same tick, for comparison only (they don't "
                   "drive the simulated inventory forward; only the primary strategy does).")
        totals = {
            f"{snapshot['scenario_label']} (primary)": sum(
                r["recommended_order_qty"] for r in results.values()
            ),
        }
        for name in other_names:
            totals[SCENARIO_LABELS[name]] = sum(
                r["other_scenarios"][name]["order_qty"] for r in results.values()
            )
        st.bar_chart(pd.Series(totals, name="total_order_qty"))

    pipeline = st.session_state.get("stream_pipeline")
    with st.expander(f"🧠 Case Memory ({'on' if getattr(pipeline, 'use_case_memory', False) else 'off'})",
                      expanded=bool(nudge_rows)):
        if pipeline is None or not getattr(pipeline, "use_case_memory", False):
            st.caption("Case Memory is off for this run -- decisions shown are PPO's own, unmodified.")
        elif nudge_rows:
            for pair_key, explanation in nudge_rows:
                st.info(f"**{pair_key}** -- {explanation}")
        else:
            n_logged = sum(pipeline.memory.n_episodes(k) for k in pipeline.pairs) if pipeline else 0
            st.caption(
                f"No nudge this tick -- either not enough similar past cases yet, or PPO's own "
                f"suggestion already agreed with them. Memory so far: {n_logged} logged episodes "
                f"across {len(pipeline.pairs) if pipeline else 0} store/product pairs."
            )
        st.caption(
            "Case Memory is a bounded, evidence-gated case-based memory layered on top of the "
            "frozen PPO policy -- it never retrains the network, only proposes a capped nudge "
            "(±1 order step) when enough similar past ticks consistently did better."
        )

    history = pd.DataFrame(st.session_state["stream_history"])
    if len(history) > 1:
        st.markdown("**Trend across ticks**")
        st.line_chart(history.set_index("tick")[["total_profit", "total_stockout_units"]])


def tab_live_stream(scenario_name: str, history_df):
    st.subheader("Live Stream Monitor (simulated)")
    st.caption(
        "Simulates a large retailer's daily settlement arriving on a compressed timeline "
        "(1 tick = 1 new simulated day, across every store/product pair). Ingestion → "
        "Processing → Decision → Serving runs on the SAME trained models as the other "
        "tabs -- nothing is retrained here. This is a demo simulation, not a production "
        "streaming deployment -- see proposal_assets/lambda_architecture.png for what's real "
        "vs. simulated."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        compare_all = st.checkbox("Also load the other 3 strategies for side-by-side comparison",
                                   key="stream_compare_all")
    with col_b:
        use_case_memory = st.checkbox(
            "🧠 Enable Case Memory (case-based nudge + online feedback)",
            value=True, key="stream_use_case_memory",
            help="When on, PPO's suggested order can get a small, evidence-gated nudge based on "
                 "similar past ticks for the same store/product -- see the Case Memory expander below "
                 "the results table. Turn off to see PPO's raw decisions only. This never retrains "
                 "the PPO network itself.",
        )

    state_key, scenario_key = "stream_pipeline", "stream_pipeline_scenario"
    needs_init = (
        state_key not in st.session_state
        or st.session_state.get(scenario_key) != scenario_name
        or st.session_state.get("stream_compare_all_active") != compare_all
        or st.session_state.get("stream_use_case_memory_active") != use_case_memory
    )
    if needs_init:
        st.session_state[state_key] = _init_stream_pipeline(
            scenario_name, history_df, compare_all, use_case_memory=use_case_memory
        )
        st.session_state[scenario_key] = scenario_name
        st.session_state["stream_compare_all_active"] = compare_all
        st.session_state["stream_use_case_memory_active"] = use_case_memory
        st.session_state["stream_history"] = []
        st.session_state["stream_latest"] = None
        st.session_state.setdefault("stream_autoplay", False)
        st.session_state["stream_pause_reason"] = None

    pipeline = st.session_state[state_key]
    if pipeline is None:
        st.warning(
            f"No trained model for '{SCENARIO_LABELS[scenario_name]}' yet -- train it first "
            f"(python train_rl.py --scenario {scenario_name}) before starting the live stream."
        )
        return

    c1, c2, c3, c4 = st.columns([1.3, 1.5, 1.6, 1.4])
    with c1:
        if st.session_state["stream_autoplay"]:
            if st.button("⏸ Pause"):
                st.session_state["stream_autoplay"] = False
        else:
            if st.button("▶ Start auto-play"):
                st.session_state["stream_autoplay"] = True
                st.session_state["stream_pause_reason"] = None
    with c2:
        manual_step = st.button("⏭ Advance 1 hour")
    with c3:
        interval = st.slider("Seconds per tick (auto-play)", 1, 5, 2, key="stream_interval")
    with c4:
        if st.button("↺ Reset stream"):
            st.session_state.pop(state_key, None)
            st.rerun()

    if st.session_state["stream_autoplay"] and not _HAS_AUTOREFRESH:
        st.info(
            "Auto-play needs the `streamlit-autorefresh` package "
            "(`pip install streamlit-autorefresh`). Falling back to manual-step mode for now."
        )
        st.session_state["stream_autoplay"] = False

    if st.session_state.get("stream_pause_reason"):
        st.warning(
            f"⚠️ Auto-play paused automatically -- {st.session_state['stream_pause_reason']} "
            f"Review it below, then press ▶ Start auto-play to resume."
        )

    should_tick = manual_step
    if st.session_state["stream_autoplay"] and _HAS_AUTOREFRESH:
        st_autorefresh(interval=interval * 1000, key="stream_autorefresh_timer")
        should_tick = True

    if should_tick:
        _stream_tick_and_check_alerts(pipeline)

    _render_stream_snapshot()


# ---------------------------------------------------------------------- #
def main():
    st.title("Replenishment Recommender")
    history_df, _ = load_shared_resources()
    scenario_name = sidebar_scenario_picker()

    try:
        recommender = load_recommender(scenario_name)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Recommendation", "Stress test", "Decision vs. history", "Future plan", "Live stream"]
    )
    with tab1:
        tab_recommendation(scenario_name, history_df, recommender)
    with tab2:
        tab_stress_test(scenario_name, history_df, recommender)
    with tab3:
        tab_comparison(scenario_name, history_df, recommender)
    with tab4:
        tab_future_plan(scenario_name, history_df, recommender)
    with tab5:
        tab_live_stream(scenario_name, history_df)


if __name__ == "__main__":
    main()
