"""
test_plan_ahead_responsiveness.py

Diagnostic/regression test for a specific user-reported symptom: "the
Future plan tab always shows the same recommended order quantity every
day -- is that a malfunction?"

Every other plan_ahead test in this suite (test_inference_smoke.py) uses a
FIXED-action fake PPO (it always requests the same order regardless of the
observation), so those tests can prove inventory/rolling-state carry
forward correctly, but they cannot tell us whether the recommended order
quantity itself would vary given a policy that actually reacts to its
inputs -- a fixed-action stub would "pass" even if plan_ahead were broken
in a way that fed every day the same observation.

This test uses a *responsive* fake policy instead: its action is a direct
function of the day's own P90 demand forecast (always the last element of
the observation vector, see replenishment_env._build_observation). If
plan_ahead is correctly threading each day's own forecast/inventory/
rolling state through _build_observation() -> model.predict() -> action,
then both the forecast AND the recommended order should differ from day
to day here.

If a user's real Future plan output looks flat every single day even
after this test passes, the likely explanations are NOT a bug in this
pipeline:
  - the trained policy has converged to a genuine steady-state /
    reorder-to-target strategy (common and expected for a well-trained
    inventory policy once the simulated trajectory reaches equilibrium)
  - current_inventory is saturating at MAX_INVENTORY (the order gets
    fully capacity-capped every day) or bottoming out at 0 repeatedly --
    check the "Starting inventory" column in the UI's day-by-day table to
    tell these apart from a real state-freezing bug
  - something specific to their real trained VecNormalize/PPO artifacts
    that can't be reproduced with synthetic data in this sandbox

Run: python test_plan_ahead_responsiveness.py
"""

import os
import sys
import types

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stubs -- isolated to this process (a separate `python3 test_X.py` run from
# test_inference_smoke.py), so this can safely install a *different* fake
# PPO than the fixed-action one used elsewhere without affecting other tests.
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

    class ResponsiveFakePPO:
        """Unlike the fixed-action FakePPO used in test_inference_smoke.py,
        this policy's action tracks the day's own P90 demand forecast --
        always the last element of the observation vector regardless of
        one-hot width, see replenishment_env._build_observation -- so its
        recommended order genuinely varies as the forecast varies day to
        day. This is what lets this test prove plan_ahead's per-day state
        threading actually works, not just that it doesn't crash."""
        TARGET_MULTIPLIER = 1.1  # aim slightly above the P90 forecast
        SCALE_UNITS = 500.0
        ORDER_STEP = 10

        @classmethod
        def load(cls, path):
            return cls()

        def predict(self, obs, deterministic=True):
            o = np.asarray(obs)[0]
            forecast_q90_scaled = o[-1]
            forecast_q90 = forecast_q90_scaled * self.SCALE_UNITS
            target_order = forecast_q90 * self.TARGET_MULTIPLIER
            action_idx = int(round(target_order / self.ORDER_STEP))
            action_idx = max(0, min(50, action_idx))
            return np.array([action_idx]), None

    sb3_stub.PPO = ResponsiveFakePPO
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
            return obs  # identity -- fine for this test, isolates plan_ahead's own logic

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

from test_env_smoke import make_synthetic_history, FakeDemandModel  # noqa: E402
import inference  # noqa: E402


def ensure_dummy_model_files(scenario_name: str, seed: int = 0):
    model_dir = os.path.join(inference.MODELS_DIR, scenario_name, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    open(os.path.join(model_dir, "ppo_model.zip"), "a").close()
    open(os.path.join(model_dir, "vecnormalize.pkl"), "a").close()


def main():
    # A different seed than test_inference_smoke.py's default, just so this
    # isn't accidentally validated against one specific lucky demand pattern.
    history_df = make_synthetic_history(n_days=120, seed=3)
    demand_model = FakeDemandModel()
    scenario_name = "pure_profit"
    ensure_dummy_model_files(scenario_name)

    rec_engine = inference.Recommender(
        scenario_name, seed=0, history_df=history_df, demand_model=demand_model
    )

    dates = sorted(history_df[
        (history_df["Store ID"] == "S001") & (history_df["Product ID"] == "P0001")
    ]["Date"].unique())

    plan = rec_engine.plan_ahead("S001", "P0001", dates[20], n_days=14, demand_scenario="mean")

    orders = plan["recommended_order_qty"].tolist()
    forecasts_q90 = plan["forecast_q90"].tolist()
    starting_inv = plan["starting_inventory"].tolist()

    # The demand forecast itself must vary day to day (it's driven by
    # rolling sales stats + calendar/price context that change daily) --
    # if this is flat, that WOULD be a real bug in per-day state threading.
    assert len(set(round(f, 3) for f in forecasts_q90)) > 1, (
        f"forecast_q90 is identical every day ({forecasts_q90[0]}) -- the demand "
        "model or rolling state isn't varying with day-of-plan, which would be a bug."
    )
    print("[ok] forecast_q90 varies day to day:", [round(f, 1) for f in forecasts_q90])

    # Since this fake policy's action is a direct function of that forecast,
    # the recommended order must vary too -- proving plan_ahead correctly
    # threads each day's own state through to the decision, not just that
    # it runs without crashing.
    assert len(set(orders)) > 1, (
        f"recommended_order_qty is identical every day ({orders[0]}) even though "
        "forecast_q90 varies -- this WOULD indicate a real bug in plan_ahead's "
        "per-day observation construction."
    )
    print("[ok] recommended_order_qty varies day to day given a responsive policy:", orders)

    # Starting inventory should also evolve (covered for a different dataset
    # in test_inference_smoke.py; re-checked here too since it's the other
    # half of "is this actually a frozen state" diagnosis).
    assert len(set(round(s, 3) for s in starting_inv)) > 1, (
        "starting_inventory is identical every day -- inventory isn't carrying forward."
    )
    print("[ok] starting_inventory evolves day to day:", [round(s, 1) for s in starting_inv])

    print(
        "\nConclusion: plan_ahead's per-day state threading (forecast, inventory, "
        "rolling stats) works correctly against a policy that actually reacts to "
        "its inputs. If a real trained model still shows an identical recommended "
        "order every single day, that's most likely the trained policy's own "
        "converged steady-state behavior, or current_inventory saturating at the "
        "MAX_INVENTORY cap / hitting 0 -- not a bug in this code. Check the "
        "'Starting inventory' column in the UI's day-by-day table to tell those "
        "apart from a real freeze."
    )


if __name__ == "__main__":
    main()
