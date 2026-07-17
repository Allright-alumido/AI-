"""
test_plan_ahead_extrapolation.py

Dedicated tests for the "predict beyond the CSV's last date" feature added
to Recommender.plan_ahead() (inference.py) and
ReplenishmentEnv._build_observation_from_row() (replenishment_env.py).

Before this feature, plan_ahead stopped early ("truncated") once it ran out
of real historical rows for a store/product, because it needed a real row
to pull exogenous context (price, discount, holiday, weather, season,
competitor pricing, plus the LightGBM model's own "Units Ordered"/"Demand
Forecast" exogenous inputs) from. Now it keeps going by constructing a
synthetic row for each day past the real data, using carried-forward/
assumed values (optionally overridden via `future_assumptions`) for
price/discount/etc., calendar fields computed from the actual date, and a
bootstrap of Units Ordered/Demand Forecast from the previous simulated
day's own output. See plan_ahead's docstring in inference.py for the full
rationale.

This file uses a deliberately simple, fully deterministic fake demand
model (ExtrapolationProbeDemandModel) whose forecast is a direct,
predictable function of the "Demand Forecast" and "Discount" exogenous
inputs alone (ignoring rolling sales stats entirely) -- this isolates the
three things that actually need checking here from the rest of the
simulation's dynamics:
  1. The full requested horizon is returned (no more truncation), with
     `data_source` correctly marking which days are historical vs
     extrapolated, and the `.attrs` counts matching.
  2. Extrapolated dates are consecutive calendar days with no gap at the
     real/extrapolated boundary.
  3. `future_assumptions` overrides (e.g. "discount") actually reach the
     synthetic row and change the model's output.
  4. The Units Ordered/Demand Forecast bootstrap actually evolves day to
     day on extrapolated days instead of freezing at the last real value.

Run: python test_plan_ahead_extrapolation.py
"""

import os
import sys
import types

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Same gymnasium / stable_baselines3 / joblib stubs used elsewhere in this
# suite -- a fixed-action fake PPO is fine here since none of these tests
# depend on the *order quantity* the policy chooses, only on the demand
# forecast each day, which our fake demand model computes independently of
# the policy's action.
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
        FIXED_ACTION = 15  # arbitrary but fixed -- irrelevant to what's under test here

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
            self.venv, self.training, self.norm_reward = venv, training, norm_reward

        @classmethod
        def load(cls, path, venv):
            return cls(venv)

        def normalize_obs(self, obs):
            return obs

    vecenv_stub.DummyVecEnv = FakeDummyVecEnv
    vecenv_stub.VecNormalize = FakeVecNormalize

    callbacks_stub = types.ModuleType("stable_baselines3.common.callbacks")

    class FakeEvalCallback:
        def __init__(self, *args, **kwargs):
            pass

    callbacks_stub.EvalCallback = FakeEvalCallback

    common_stub.__path__ = []
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

from test_env_smoke import make_synthetic_history, FakeEncoder  # noqa: E402
import inference  # noqa: E402


class ExtrapolationProbeDemandModel:
    """Deliberately ignores rolling sales stats -- forecast_mean is just
    the exogenous "Demand Forecast" input plus a constant increment plus
    the exogenous "Discount" input. This isolates exactly the two
    exogenous inputs this test suite needs to probe (the future_assumptions
    override path, and the Units Ordered/Demand Forecast bootstrap path)
    from the rest of the simulation's dynamics (inventory, rolling stats,
    the policy's chosen action), which would otherwise make the forecast
    sequence harder to predict exactly."""

    INCREMENT = 10.0

    def __init__(self):
        self.encoders = {
            "Store ID": FakeEncoder(["S001", "S002", "__unseen__"]),
            "Product ID": FakeEncoder(["P0001", "P0002", "__unseen__"]),
            "Category": FakeEncoder(["Groceries", "Toys", "__unseen__"]),
            "Region": FakeEncoder(["North", "South", "__unseen__"]),
            "Weather Condition": FakeEncoder(["Sunny", "Rainy", "__unseen__"]),
            "Seasonality": FakeEncoder(["Autumn", "Summer", "__unseen__"]),
        }

    def predict(self, row, current_inventory, rolling_state):
        mean = float(row["Demand Forecast"]) + self.INCREMENT + float(row["Discount"])
        mean = max(mean, 1.0)
        return mean, mean * 1.3


def ensure_dummy_model_files(scenario_name: str, seed: int = 0):
    model_dir = os.path.join(inference.MODELS_DIR, scenario_name, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    open(os.path.join(model_dir, "ppo_model.zip"), "a").close()
    open(os.path.join(model_dir, "vecnormalize.pkl"), "a").close()


def main():
    # A short 40-day dataset so it's easy to request a horizon that runs
    # well past the end of the real data.
    history_df = make_synthetic_history(n_days=40, seed=7)
    demand_model = ExtrapolationProbeDemandModel()
    scenario_name = "pure_profit"
    ensure_dummy_model_files(scenario_name)

    rec_engine = inference.Recommender(
        scenario_name, seed=0, history_df=history_df, demand_model=demand_model
    )

    key_store, key_product = "S001", "P0001"
    dates = sorted(history_df[
        (history_df["Store ID"] == key_store) & (history_df["Product ID"] == key_product)
    ]["Date"].unique())
    assert len(dates) == 40

    # Start at day index 30 -> 10 real days remain (indices 30..39); request
    # 15 days, so 5 of them must be extrapolated.
    start = dates[30]
    n_days = 15
    plan = rec_engine.plan_ahead(key_store, key_product, start, n_days=n_days, demand_scenario="mean")

    # --- 1. full horizon returned, no truncation, attrs/data_source agree ---
    assert len(plan) == n_days, f"expected {n_days} rows, got {len(plan)}"
    assert plan.attrs["available_days"] == 10
    assert plan.attrs["real_days"] == 10
    assert plan.attrs["extrapolated_days"] == 5
    assert plan.attrs["uses_extrapolation"] is True
    assert list(plan["data_source"]) == ["historical"] * 10 + ["extrapolated"] * 5
    print(f"[ok] plan_ahead returns the full requested horizon ({n_days} days: "
          f"{plan.attrs['real_days']} real + {plan.attrs['extrapolated_days']} extrapolated) "
          "instead of truncating")

    # --- 2. extrapolated dates are consecutive calendar days, no gap -------
    plan_dates = pd.to_datetime(plan["date"]).tolist()
    for i in range(1, len(plan_dates)):
        gap = plan_dates[i] - plan_dates[i - 1]
        assert gap == pd.Timedelta(days=1), f"date gap of {gap} at row {i} (expected 1 day)"
    last_real_date = pd.Timestamp(dates[-1])
    assert plan_dates[9] == last_real_date, "last historical row's date should be the CSV's actual last date"
    assert plan_dates[10] == last_real_date + pd.Timedelta(days=1), (
        "first extrapolated day should continue exactly 1 day after the last real date"
    )
    print("[ok] extrapolated dates are consecutive calendar days with no gap at the real/extrapolated boundary")

    # --- 3. future_assumptions overrides reach the synthetic row -----------
    last_real_row = history_df[
        (history_df["Store ID"] == key_store) & (history_df["Product ID"] == key_product)
    ].sort_values("Date").iloc[-1]
    plan_default = plan  # no override -> discount carried forward from last_real_row
    plan_override = rec_engine.plan_ahead(
        key_store, key_product, start, n_days=n_days, demand_scenario="mean",
        future_assumptions={"discount": 999.0},  # far from any plausible carried-forward value
    )
    default_first_extrap = plan_default.iloc[10]["forecast_mean"]
    override_first_extrap = plan_override.iloc[10]["forecast_mean"]
    expected_delta = 999.0 - float(last_real_row["Discount"])
    actual_delta = override_first_extrap - default_first_extrap
    assert abs(actual_delta - expected_delta) < 1e-6, (
        f"discount override did not propagate to the synthetic row: "
        f"default={default_first_extrap}, override={override_first_extrap}, "
        f"expected delta={expected_delta}, got={actual_delta}"
    )
    print("[ok] future_assumptions['discount'] override reaches the first extrapolated day's demand forecast "
          f"(delta={actual_delta:.1f}, expected={expected_delta:.1f})")

    # --- 4. Units Ordered / Demand Forecast bootstrap evolves, not frozen --
    # With discount D held fixed across all extrapolated days (default
    # carry-forward -- same value every day) and
    # mean(n) = Demand_Forecast_input(n) + INCREMENT + D, a working
    # bootstrap (each day's "Demand Forecast" input = previous day's own
    # forecast_mean output, i.e. Demand_Forecast_input(n+1) = mean(n))
    # means forecast_mean must increase by exactly (INCREMENT + D) every
    # extrapolated day -- D gets folded back in again each time the
    # previous day's own output re-enters as the next day's input. If the
    # bootstrap were bugged into freezing "Demand Forecast" at the last
    # real value instead, forecast_mean would stay completely constant
    # across all extrapolated days (increment of 0), which is the failure
    # mode this test actually needs to rule out.
    expected_increment = ExtrapolationProbeDemandModel.INCREMENT + float(last_real_row["Discount"])
    extrap_forecasts = plan_default.iloc[10:]["forecast_mean"].tolist()
    increments = [b - a for a, b in zip(extrap_forecasts, extrap_forecasts[1:])]
    assert all(abs(inc - expected_increment) < 1e-6 for inc in increments), (
        f"expected forecast_mean to rise by exactly {expected_increment} "
        f"each extrapolated day (proving the 'Demand Forecast' bootstrap from the previous day's "
        f"own output works, rather than freezing -- which would give increments of 0) "
        f"-- got increments {increments}"
    )
    print(f"[ok] extrapolated days bootstrap 'Demand Forecast' from the previous day's own output "
          f"(forecast_mean rises by exactly {expected_increment:.1f} each day, not frozen): "
          f"{[round(f, 1) for f in extrap_forecasts]}")

    # --- 5. no extrapolation needed when the horizon fits in real data ------
    plan_within = rec_engine.plan_ahead(key_store, key_product, dates[0], n_days=10, demand_scenario="mean")
    assert plan_within.attrs["uses_extrapolation"] is False
    assert plan_within.attrs["extrapolated_days"] == 0
    assert (plan_within["data_source"] == "historical").all()
    print("[ok] a horizon that fits within real data uses no extrapolation at all")

    print("\nAll plan_ahead extrapolation tests passed.")


if __name__ == "__main__":
    main()
