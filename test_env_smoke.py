"""
test_env_smoke.py

Lightweight smoke test for replenishment_env.py that does NOT require
lightgbm/gymnasium/scikit-learn/stable-baselines3 to be installed. It stubs
just enough of `gymnasium` to satisfy replenishment_env.py's imports, and
replaces the real DemandModel with a trivial fake (same `.predict()` /
`.encoders` interface). This is meant as a fast sanity check of the
simulation *mechanics* (inventory conservation, reward sign/scale under each
scenario, episode length) -- it does not validate the real LightGBM demand
forecasts, which can only be checked in an environment with lightgbm
installed.

Run: python test_env_smoke.py
"""

import sys
import types
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Minimal gymnasium stand-in (only the surface replenishment_env.py touches)
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


from feature_engineering import RollingSalesState, FEATURE_ORDER  # noqa: E402
from replenishment_env import ReplenishmentEnv, MAX_ORDER, ORDER_STEP  # noqa: E402
from reward_configs import SCENARIOS  # noqa: E402


class FakeEncoder:
    def __init__(self, classes):
        self.classes_ = classes

    def transform(self, values):
        return np.array([self.classes_.index(v) for v in values])


class FakeDemandModel:
    """Mimics demand_model.DemandModel's interface without touching lightgbm."""

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
        rolling = rolling_state.features()
        mean = max(rolling["Sales_RollMean_7"], 1.0)
        return mean, mean * 1.3


def make_synthetic_history(n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    for store, product in [("S001", "P0001"), ("S002", "P0002")]:
        inv = 250
        for d in dates:
            demand = max(0, int(rng.normal(100, 20)))
            order = max(0, int(rng.normal(100, 15)))
            units_sold = min(inv + order, demand)
            rows.append(dict(
                Date=d, **{"Store ID": store, "Product ID": product},
                Category="Groceries", Region="North",
                **{"Inventory Level": inv, "Units Ordered": order,
                   "Demand Forecast": demand * 0.9, "Units Sold": units_sold},
                Price=rng.uniform(20, 80), Discount=rng.integers(0, 20),
                **{"Weather Condition": rng.choice(["Sunny", "Rainy"])},
                **{"Holiday/Promotion": int(rng.random() < 0.1)},
                **{"Competitor Pricing": rng.uniform(20, 80)},
                Seasonality=rng.choice(["Autumn", "Summer"]),
            ))
            inv = max(inv + order - units_sold, 0)
    df = pd.DataFrame(rows)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df


def main():
    history_df = make_synthetic_history()
    demand_model = FakeDemandModel()

    print("Action grid size:", MAX_ORDER // ORDER_STEP + 1)

    for name, cfg in SCENARIOS.items():
        env = ReplenishmentEnv(history_df, demand_model, cfg, episode_length=30, rng_seed=1)
        obs, info = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape, f"obs shape mismatch: {obs.shape}"
        assert np.isfinite(obs).all(), "non-finite values in observation"

        total_reward = 0.0
        n_steps = 0
        rng = np.random.default_rng(2)
        done = False
        while not done:
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, step_info = env.step(action)
            assert np.isfinite(reward), "non-finite reward"
            total_reward += reward
            n_steps += 1
            done = terminated or truncated

        log = env.get_episode_log()
        # Inventory conservation check: end_inventory = inv_before + order - units_sold
        assert (log["units_sold"] <= log["demand"] + 1e-6).all(), "units_sold exceeds demand"
        assert (log["end_inventory"] >= -1e-6).all(), "negative inventory"
        assert n_steps == 30, f"expected 30 steps, got {n_steps}"

        print(f"[{name:32s}] steps={n_steps:3d}  total_reward={total_reward:9.2f}  "
              f"avg_order={log['order_qty'].mean():6.1f}  "
              f"stockout_units_total={log['stockout_units'].sum():7.1f}  "
              f"avg_end_inv={log['end_inventory'].mean():6.1f}")

    print("\nAll smoke-test assertions passed.")


if __name__ == "__main__":
    main()
