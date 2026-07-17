"""
test_app_smoke.py

Smoke test for app.py's pure-logic helper functions (previous_date_for,
what_changed_reasons, get_row) that does NOT require streamlit,
gymnasium, stable-baselines3, or lightgbm to be installed. Stubs just
enough of each to let `import app` succeed, then calls the helpers
directly against synthetic data. Does not exercise the actual Streamlit
widgets/rendering -- that can only be checked by running the real app.

Run: python test_app_smoke.py
"""

import sys
import types

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Minimal streamlit stand-in -- just enough for `import app` to succeed
# (module-level st.set_page_config call + @st.cache_resource/@st.cache_data
# decorators). Nothing else in app.py runs at import time.
# ---------------------------------------------------------------------------
if "streamlit" not in sys.modules:
    st_stub = types.ModuleType("streamlit")
    st_stub.set_page_config = lambda **kwargs: None
    st_stub.cache_resource = lambda fn=None, **kwargs: (fn if fn else (lambda f: f))
    st_stub.cache_data = lambda fn=None, **kwargs: (fn if fn else (lambda f: f))
    # Plain dict stands in for st.session_state -- supports the `in`/`[]`
    # access cascading_selectbox uses. Real Streamlit also supports
    # attribute access, which isn't needed by anything under test here.
    st_stub.session_state = {}

    def _stub_selectbox(label, options, index=0, key=None, format_func=None):
        # Mirrors real Streamlit's key-vs-index precedence: once `key` has a
        # stored session_state value, that value wins and `index` is only
        # used to seed it the first time -- which is exactly the behavior
        # cascading_selectbox relies on (it pre-seeds/repairs session_state
        # itself before calling st.selectbox).
        if key is not None:
            return st_stub.session_state[key]
        return options[index] if options else None

    st_stub.selectbox = _stub_selectbox
    sys.modules["streamlit"] = st_stub

# ---------------------------------------------------------------------------
# Minimal gymnasium / stable_baselines3 / joblib stand-ins (same as
# test_inference_smoke.py -- inference.py and its dependencies need these
# importable even though we won't exercise the RL machinery here).
# ---------------------------------------------------------------------------
if "gymnasium" not in sys.modules:
    gym_stub = types.ModuleType("gymnasium")

    class _Env:
        metadata = {}

        def reset(self, *, seed=None, options=None):
            pass

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    spaces_stub = types.ModuleType("gymnasium.spaces")
    spaces_stub.Discrete = _Discrete
    spaces_stub.Box = _Box
    gym_stub.Env = _Env
    gym_stub.spaces = spaces_stub
    sys.modules["gymnasium"] = gym_stub
    sys.modules["gymnasium.spaces"] = spaces_stub

if "stable_baselines3" not in sys.modules:
    sb3_stub = types.ModuleType("stable_baselines3")

    class FakePPO:
        @classmethod
        def load(cls, path):
            return cls()

        def predict(self, obs, deterministic=True):
            return np.array([25]), None

    sb3_stub.PPO = FakePPO
    sys.modules["stable_baselines3"] = sb3_stub

    common_stub = types.ModuleType("stable_baselines3.common")
    vecenv_stub = types.ModuleType("stable_baselines3.common.vec_env")

    class FakeDummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]

    class FakeVecNormalize:
        def __init__(self, venv, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True):
            self.venv, self.training, self.norm_reward = venv, training, norm_reward

        @classmethod
        def load(cls, path, venv):
            return cls(venv)

        def normalize_obs(self, obs):
            return obs

    vecenv_stub.DummyVecEnv = FakeDummyVecEnv
    vecenv_stub.VecNormalize = FakeVecNormalize
    common_stub.__path__ = []
    common_stub.vec_env = vecenv_stub
    sys.modules["stable_baselines3.common"] = common_stub
    sys.modules["stable_baselines3.common.vec_env"] = vecenv_stub

if "joblib" not in sys.modules:
    joblib_stub = types.ModuleType("joblib")
    joblib_stub.load = lambda path: None
    joblib_stub.dump = lambda obj, path: None
    sys.modules["joblib"] = joblib_stub

from test_env_smoke import make_synthetic_history  # noqa: E402
from test_inference_smoke import ensure_dummy_model_files  # noqa: E402
import app  # noqa: E402


def main():
    history_df = make_synthetic_history()

    dates = sorted(pd.Timestamp(d) for d in history_df[
        (history_df["Store ID"] == "S001") & (history_df["Product ID"] == "P0001")
    ]["Date"].unique())

    # --- previous_date_for --------------------------------------------------
    mid_date = dates[10]
    prev = app.previous_date_for(history_df, "S001", "P0001", mid_date)
    assert prev == dates[9], f"expected {dates[9]}, got {prev}"
    first = app.previous_date_for(history_df, "S001", "P0001", dates[0])
    assert first is None, "previous_date_for should return None for the first date"
    print("[ok] previous_date_for")

    # --- get_row --------------------------------------------------------------
    row = app.get_row(history_df, "S001", "P0001", mid_date)
    assert row is not None and pd.Timestamp(row["Date"]) == mid_date
    print("[ok] get_row")

    # --- what_changed_reasons ---------------------------------------------
    rec_today = {"forecast_mean": 150.0, "current_inventory": 10.0}
    rec_yesterday = {"forecast_mean": 100.0, "current_inventory": 50.0}
    row_today = pd.Series({"Discount": 15, "Holiday/Promotion": 1})
    row_yesterday = pd.Series({"Discount": 5, "Holiday/Promotion": 0})
    reasons = app.what_changed_reasons(rec_today, rec_yesterday, row_today, row_yesterday)
    assert set(reasons) == {"Demand Increased", "Inventory Lower", "Promotion Added", "Holiday Approaching"}, reasons
    print("[ok] what_changed_reasons (all 4 triggers):", reasons)

    # no-change case should report nothing
    reasons_none = app.what_changed_reasons(
        {"forecast_mean": 100.0, "current_inventory": 50.0},
        {"forecast_mean": 100.0, "current_inventory": 50.0},
        pd.Series({"Discount": 5, "Holiday/Promotion": 0}),
        pd.Series({"Discount": 5, "Holiday/Promotion": 0}),
    )
    assert reasons_none == [], reasons_none
    print("[ok] what_changed_reasons (no triggers):", reasons_none)

    # --- load_policy_comparison_df: never crashes, regardless of whether
    # this environment happens to already have a real policy_comparison.csv
    # on disk (e.g. from having actually run evaluate_policies.py locally)
    # or not. The real contract being tested is "no exception, and if it
    # does return something, it's a DataFrame" -- NOT "the file must be
    # absent," which was too strong an assumption about the caller's
    # working directory.
    result = app.load_policy_comparison_df()
    assert result is None or isinstance(result, pd.DataFrame), (
        f"expected None or a DataFrame, got {type(result)}"
    )
    if result is None:
        print("[ok] load_policy_comparison_df gracefully returns None when csv is absent")
    else:
        print(f"[ok] load_policy_comparison_df loaded a real policy_comparison.csv "
              f"({len(result)} rows) without crashing")

    # --- cascading_selectbox --------------------------------------------------
    # Regression test for the reported bug: "It can't choose other store or
    # product, while I want to change, it will refresh to the original."
    # Root cause was st.selectbox being called with both `key=` and `index=`
    # in the Store -> Product -> Date chains: once `key` has a stored
    # session_state value, Streamlit ignores `index` on every later rerun,
    # and if a parent selection change makes that stored value invalid for
    # the new `options` list, the widget has nothing valid to fall back to.
    # cascading_selectbox fixes this by validating/repairing session_state
    # *before* creating the widget.
    import streamlit as st

    st.session_state.clear()

    # First render, no prior session_state: falls back to default_index (0).
    val = app.cascading_selectbox("Store", ["S001", "S002"], key="t_store")
    assert val == "S001", val

    # A later rerun with a still-valid stored selection must be preserved,
    # not reset -- this is the "I picked S002 and it snapped back" case.
    st.session_state["t_store"] = "S002"
    val2 = app.cascading_selectbox("Store", ["S001", "S002"], key="t_store")
    assert val2 == "S002", f"a valid selection must be preserved, got {val2}"

    # The cascading case: Product was "P002" (valid for the old Store), but
    # the Store just changed and P002 isn't offered for the new one. This
    # must reset to a valid product instead of crashing or silently keeping
    # the now-invalid stale value.
    st.session_state["t_product"] = "P002"
    products_for_new_store = ["P001", "P003"]
    val3 = app.cascading_selectbox("Product", products_for_new_store, key="t_product")
    assert val3 in products_for_new_store, val3
    assert val3 == "P001", val3  # falls back to default_index=0
    print("[ok] cascading_selectbox: preserves a still-valid selection, "
          "resets a now-invalid one instead of crashing or reverting silently")

    # default_index is used to pick "the last date" the first time a Date
    # widget is created for a given store/product (mirrors the Future plan
    # and Recommendation tabs' "default to most recent date" behavior).
    st.session_state.pop("t_date", None)
    dts = ["2024-01-01", "2024-01-02", "2024-01-03"]
    val4 = app.cascading_selectbox("Date", dts, key="t_date", default_index=len(dts) - 1)
    assert val4 == dts[-1], val4
    print("[ok] cascading_selectbox: default_index picks the most recent date on first render")

    # --- default_future_assumptions --------------------------------------
    # Powers the Future plan tab's "assumptions for dates beyond the data"
    # inputs -- should carry forward the *last* real row's price/discount/
    # competitor pricing for the requested store/product, not some other
    # row or an average.
    last_row = history_df[
        (history_df["Store ID"] == "S001") & (history_df["Product ID"] == "P0001")
    ].sort_values("Date").iloc[-1]
    fa = app.default_future_assumptions(history_df, "S001", "P0001")
    assert set(fa) == {"price", "discount", "competitor_pricing"}, fa
    assert abs(fa["price"] - float(last_row["Price"])) < 1e-9
    assert abs(fa["discount"] - float(last_row["Discount"])) < 1e-9
    assert abs(fa["competitor_pricing"] - float(last_row["Competitor Pricing"])) < 1e-9
    print("[ok] default_future_assumptions carries forward the last real row's price/discount/competitor pricing")

    # --- Live Stream tab: _check_stream_alerts (pure logic) -----------------
    no_alert = app._check_stream_alerts(
        {"S001/P0001": {"stockout_units": 0.0}, "S002/P0002": {"stockout_units": 0.0}},
        "2024-06-01",
    )
    assert no_alert is None, no_alert
    with_alert = app._check_stream_alerts(
        {"S001/P0001": {"stockout_units": 3.5}, "S002/P0002": {"stockout_units": 0.0}},
        "2024-06-02",
    )
    assert with_alert is not None and "S001/P0001" in with_alert and "2024-06-02" in with_alert
    print("[ok] _check_stream_alerts: no pause when no stockouts, pause message names the "
          "affected pair when one occurs")

    # --- percentage-threshold gating: with a large pair count (mirrors a
    # real 100-pair run), a handful of stockouts below
    # STOCKOUT_ALERT_THRESHOLD should NOT pause auto-play -- only once a
    # meaningful share of pairs are affected does it pause. Regression test
    # for the reported "auto-play pauses almost every single tick" issue --
    # root cause was pausing on ANY stockout at all, regardless of how many
    # of the (say) 100 simultaneously-tracked pairs it represented.
    n_pairs = 100
    below_threshold = {
        f"pair_{i}": {"stockout_units": 1.0 if i < 2 else 0.0} for i in range(n_pairs)
    }  # 2/100 = 2% < 5% threshold
    assert app._check_stream_alerts(below_threshold, "2024-06-03") is None, (
        "2 of 100 pairs (2%) is below the alert threshold and should NOT pause"
    )
    above_threshold = {
        f"pair_{i}": {"stockout_units": 1.0 if i < 9 else 0.0} for i in range(n_pairs)
    }  # 9/100 = 9% >= 5% threshold
    alert = app._check_stream_alerts(above_threshold, "2024-06-04")
    assert alert is not None and "9 of 100" in alert, alert
    print(f"[ok] _check_stream_alerts: only pauses once stockouts reach the "
          f"{app.STOCKOUT_ALERT_THRESHOLD * 100:.0f}% threshold (2/100 stays silent, 9/100 pauses)")

    # --- Live Stream tab: _init_stream_pipeline ------------------------------
    # Monkeypatch load_shared_resources so Recommender construction uses
    # FakeDemandModel instead of the real DemandModel (which needs real
    # lightgbm .pkl files this stub environment doesn't have) -- same
    # substitution test_inference_smoke.py makes, just applied through
    # app.py's cached loader instead of calling inference.Recommender directly.
    from test_env_smoke import FakeDemandModel
    fake_demand_model = FakeDemandModel()
    app.load_shared_resources = lambda: (history_df, fake_demand_model)
    ensure_dummy_model_files("pure_profit")  # leave the other 3 scenarios untrained on purpose

    pipeline = app._init_stream_pipeline("pure_profit", history_df, compare_all=False)
    assert pipeline is not None
    assert set(pipeline.recommenders) == {"pure_profit"}
    assert pipeline.use_case_memory is True, "use_case_memory should default to True"
    print("[ok] _init_stream_pipeline: builds a pipeline when the primary scenario is trained")

    pipeline_no_case_memory = app._init_stream_pipeline(
        "pure_profit", history_df, compare_all=False, use_case_memory=False
    )
    assert pipeline_no_case_memory is not None
    assert pipeline_no_case_memory.use_case_memory is False
    print("[ok] _init_stream_pipeline: use_case_memory=False is passed through to the pipeline")

    missing = app._init_stream_pipeline("zero_stockout", history_df, compare_all=False)
    assert missing is None, "expected None when the primary scenario has no trained model"
    print("[ok] _init_stream_pipeline: returns None (not a crash) when the primary scenario "
          "isn't trained yet")

    compare_pipeline = app._init_stream_pipeline("pure_profit", history_df, compare_all=True)
    assert compare_pipeline is not None
    assert set(compare_pipeline.recommenders) == {"pure_profit"}, (
        "compare_all should silently skip untrained scenarios, not crash or include them"
    )
    print("[ok] _init_stream_pipeline: compare_all mode gracefully skips untrained scenarios")

    # --- compare_all aggregation data: _render_stream_snapshot()'s "All 4
    # strategies' total order quantity this tick" chart sums
    # recommended_order_qty (primary) and other_scenarios[name]["order_qty"]
    # (the rest) across every tracked pair. That chart itself isn't unit
    # tested here (rendering isn't, per this file's module docstring), but
    # this confirms the snapshot data it reads from is actually populated
    # and well-formed once all 4 scenarios are trained -- regression test
    # for the reported "I checked the box and nothing changed" gap, whose
    # root cause was this data existing but never being displayed anywhere.
    import os
    import shutil
    import inference as _inference_mod
    scenarios_before_cmp = (
        set(os.listdir(_inference_mod.MODELS_DIR)) if os.path.exists(_inference_mod.MODELS_DIR) else set()
    )
    for name in app.SCENARIOS:
        ensure_dummy_model_files(name)
    full_compare_pipeline = app._init_stream_pipeline("pure_profit", history_df, compare_all=True)
    assert set(full_compare_pipeline.recommenders) == set(app.SCENARIOS.keys())
    snap = full_compare_pipeline.tick()
    sample = next(iter(snap["results"].values()))
    other_names = set(sample["other_scenarios"].keys())
    assert other_names == set(app.SCENARIOS.keys()) - {"pure_profit"}
    total_primary = sum(r["recommended_order_qty"] for r in snap["results"].values())
    assert np.isfinite(total_primary)
    for name in other_names:
        total_other = sum(r["other_scenarios"][name]["order_qty"] for r in snap["results"].values())
        assert np.isfinite(total_other)
    print("[ok] compare_all snapshot data: every pair's 'other_scenarios' is populated with all "
          "3 non-primary strategies' order quantities, ready for Tab5's aggregate comparison chart")
    scenarios_after_cmp = set(os.listdir(_inference_mod.MODELS_DIR))
    for name in scenarios_after_cmp - scenarios_before_cmp:
        shutil.rmtree(os.path.join(_inference_mod.MODELS_DIR, name), ignore_errors=True)

    # one real tick through the full Streamlit-facing path
    st.session_state["stream_history"] = []
    st.session_state["stream_latest"] = None
    st.session_state["stream_autoplay"] = True
    st.session_state["stream_pause_reason"] = None
    app._stream_tick_and_check_alerts(pipeline)
    assert len(st.session_state["stream_history"]) == 1
    assert st.session_state["stream_latest"] is not None
    assert st.session_state["stream_latest"]["tick"] == 1
    print("[ok] _stream_tick_and_check_alerts: one tick updates stream_history/stream_latest")

    print("\nAll app.py smoke-test assertions passed.")


if __name__ == "__main__":
    main()
