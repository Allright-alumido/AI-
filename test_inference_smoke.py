"""
test_inference_smoke.py

Smoke test for inference.py (the Recommender class app.py calls) that does
NOT require gymnasium/stable-baselines3/lightgbm/scikit-learn to be
installed. Stubs just enough of `gymnasium` and `stable_baselines3` to
exercise the real observation-building, action-decoding, and capacity-
capping logic in inference.py and replenishment_env.py, using a fake PPO
model and a fake demand model (same style as test_env_smoke.py).

This checks structure and internal consistency (fulfilled <= requested,
capacity cap kicks in correctly, compare_to_historical's two trajectories
run to completion and produce a profit_uplift number) -- it does NOT
validate real trained-model behavior, which can only be checked in an
environment with the real packages installed.

Run: python test_inference_smoke.py
"""

import os
import sys
import types

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Minimal gymnasium stand-in
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

# ---------------------------------------------------------------------------
# Minimal stable_baselines3 stand-in
# ---------------------------------------------------------------------------
if "stable_baselines3" not in sys.modules:
    sb3_stub = types.ModuleType("stable_baselines3")

    class FakePPO:
        FIXED_ACTION = 25  # -> order_qty = 250 (mid-range), for predictable test assertions

        @classmethod
        def load(cls, path):
            return cls()

        def predict(self, obs, deterministic=True):
            return np.array([self.FIXED_ACTION]), None

    sb3_stub.PPO = FakePPO
    sys.modules["stable_baselines3"] = sb3_stub

    common_stub = types.ModuleType("stable_baselines3.common")
    vecenv_stub = types.ModuleType("stable_baselines3.common.vec_env")

    class FakeDummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]

    class FakeVecNormalize:
        def __init__(self, venv, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True):
            self.venv = venv
            self.training = training
            self.norm_reward = norm_reward

        @classmethod
        def load(cls, path, venv):
            return cls(venv)

        def normalize_obs(self, obs):
            return obs  # identity -- fine for structural testing

    vecenv_stub.DummyVecEnv = FakeDummyVecEnv
    vecenv_stub.VecNormalize = FakeVecNormalize

    callbacks_stub = types.ModuleType("stable_baselines3.common.callbacks")

    class FakeEvalCallback:
        def __init__(self, *args, **kwargs):
            pass

    callbacks_stub.EvalCallback = FakeEvalCallback

    common_stub.__path__ = []  # mark as a package so submodule imports resolve
    common_stub.vec_env = vecenv_stub
    common_stub.callbacks = callbacks_stub
    sys.modules["stable_baselines3.common"] = common_stub
    sys.modules["stable_baselines3.common.vec_env"] = vecenv_stub
    sys.modules["stable_baselines3.common.callbacks"] = callbacks_stub


if "joblib" not in sys.modules:
    joblib_stub = types.ModuleType("joblib")
    joblib_stub.load = lambda path: None
    joblib_stub.dump = lambda obj, path: None
    sys.modules["joblib"] = joblib_stub

from test_env_smoke import make_synthetic_history, FakeDemandModel  # noqa: E402
from reward_configs import SCENARIOS  # noqa: E402
import inference  # noqa: E402


def ensure_dummy_model_files(scenario_name: str, seed: int = 0):
    model_dir = os.path.join(inference.MODELS_DIR, scenario_name, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    open(os.path.join(model_dir, "ppo_model.zip"), "a").close()
    open(os.path.join(model_dir, "vecnormalize.pkl"), "a").close()


def main():
    history_df = make_synthetic_history()
    demand_model = FakeDemandModel()
    scenario_name = "pure_profit"
    ensure_dummy_model_files(scenario_name)

    rec_engine = inference.Recommender(
        scenario_name, seed=0, history_df=history_df, demand_model=demand_model
    )

    # --- recommend_from_history ------------------------------------------
    dates = sorted(history_df[
        (history_df["Store ID"] == "S001") & (history_df["Product ID"] == "P0001")
    ]["Date"].unique())
    rec = rec_engine.recommend_from_history("S001", "P0001", dates[50])
    assert rec["fulfilled_order_qty"] <= rec["requested_order_qty"] + 1e-9
    assert set(["scenario", "scenario_label", "requested_order_qty", "fulfilled_order_qty",
                "forecast_mean", "forecast_q90", "current_inventory", "historical_order_qty"]).issubset(rec)
    print("[ok] recommend_from_history:", rec["fulfilled_order_qty"], "units")

    # --- capacity cap kicks in when inventory is already near MAX_INVENTORY
    from replenishment_env import MAX_INVENTORY
    rec_capped = rec_engine.recommend_manual(
        store_id="S001", product_id="P0001", category="Groceries", region="North",
        weather="Sunny", seasonality="Autumn", price=50.0, discount=10, competitor_pricing=50.0,
        holiday_promo=False, current_inventory=MAX_INVENTORY - 5, forecast_mean=100.0,
    )
    assert rec_capped["fulfilled_order_qty"] <= 5 + 1e-9, "capacity cap did not engage"
    assert rec_capped["capped_by_capacity"] is True
    print("[ok] capacity cap engages near MAX_INVENTORY:", rec_capped["fulfilled_order_qty"], "units")

    # --- recommend_manual normal case ------------------------------------
    rec_manual = rec_engine.recommend_manual(
        store_id="S001", product_id="P0001", category="Groceries", region="North",
        weather="Sunny", seasonality="Autumn", price=50.0, discount=10, competitor_pricing=50.0,
        holiday_promo=True, current_inventory=100.0, forecast_mean=120.0,
    )
    assert np.isfinite(rec_manual["fulfilled_order_qty"])
    print("[ok] recommend_manual:", rec_manual["fulfilled_order_qty"], "units")

    # --- compare_to_historical --------------------------------------------
    result = rec_engine.compare_to_historical("S001", "P0001", dates[20], n_days=30)
    assert len(result["rl_log"]) == 30
    assert len(result["historical_log"]) == 30
    assert np.isfinite(result["profit_uplift"])
    print(f"[ok] compare_to_historical: profit_uplift = {result['profit_uplift']:.2f}, "
          f"rl_profit={result['rl_kpis']['profit']:.2f}, hist_profit={result['historical_kpis']['profit']:.2f}")

    # --- plan_ahead: normal case, both demand scenarios --------------------
    plan_mean = rec_engine.plan_ahead("S001", "P0001", dates[20], n_days=14, demand_scenario="mean")
    plan_p90 = rec_engine.plan_ahead("S001", "P0001", dates[20], n_days=14, demand_scenario="p90")
    assert len(plan_mean) == 14 and len(plan_p90) == 14
    assert plan_mean.attrs["truncated"] is False
    expected_cols = {
        "date", "day_offset", "starting_inventory", "recommended_order_qty",
        "requested_order_qty", "capped_by_capacity", "forecast_mean", "forecast_q90",
        "assumed_demand", "expected_ending_inventory", "stockout_risk_units_if_high_demand",
    }
    assert expected_cols.issubset(plan_mean.columns), plan_mean.columns
    assert list(plan_mean["day_offset"]) == list(range(14))
    # day_offset 0's starting inventory should match the actual inventory on dates[20]
    row0 = get_row_for_test(history_df, "S001", "P0001", dates[20])
    assert abs(plan_mean.iloc[0]["starting_inventory"] - row0["Inventory Level"]) < 1e-6
    # inventory carries forward: day N's starting inventory == day N-1's ending inventory
    for i in range(1, 14):
        assert abs(
            plan_mean.iloc[i]["starting_inventory"] - plan_mean.iloc[i - 1]["expected_ending_inventory"]
        ) < 1e-6, f"inventory did not carry forward at day {i}"
    # p90 scenario assumes >= demand than mean scenario each day, so its
    # simulated ending inventory should never run higher than the mean case
    # (it's being drawn down faster or equal)
    assert (plan_p90["assumed_demand"].values >= plan_mean["assumed_demand"].values - 1e-9).all()
    print("[ok] plan_ahead (mean):", plan_mean["recommended_order_qty"].tolist())
    print("[ok] plan_ahead (p90):", plan_p90["recommended_order_qty"].tolist())

    # --- plan_ahead: truncation near the end of available history ----------
    plan_trunc = rec_engine.plan_ahead("S001", "P0001", dates[-3], n_days=14, demand_scenario="mean")
    assert plan_trunc.attrs["truncated"] is True
    assert plan_trunc.attrs["requested_days"] == 14
    assert len(plan_trunc) == plan_trunc.attrs["available_days"]
    assert len(plan_trunc) < 14
    print(f"[ok] plan_ahead truncates near end of history: requested 14, got {len(plan_trunc)}")

    # --- plan_ahead: invalid demand_scenario raises -------------------------
    try:
        rec_engine.plan_ahead("S001", "P0001", dates[20], n_days=5, demand_scenario="bogus")
        raise AssertionError("expected ValueError for invalid demand_scenario")
    except ValueError:
        print("[ok] plan_ahead rejects invalid demand_scenario")

    print("\nAll inference smoke-test assertions passed.")


def get_row_for_test(history_df, store_id, product_id, date):
    g = history_df[(history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)]
    match = g[g["Date"] == pd.Timestamp(date)]
    return match.iloc[0]


if __name__ == "__main__":
    main()
