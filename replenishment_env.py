"""
replenishment_env.py

A gymnasium.Env that simulates daily inventory replenishment for one
(store, product) pair at a time, replayed against real historical demand,
with LightGBM demand forecasts exposed to the agent as observation features.

Episode mechanics
------------------
- One episode = EPISODE_LENGTH consecutive days for a randomly chosen
  (store, product) pair (held-out pairs/date-ranges can be pinned via
  `options` for evaluation -- see reset()).
- Each day:
    1. Agent observes: current (start-of-day) inventory, calendar features,
       price/discount/competitor/weather/promotion context, its own recent
       realized-sales trajectory (lags/rolling stats), and the LightGBM
       mean + q90 demand forecasts for today.
    2. Agent chooses a discrete order quantity (0 .. MAX_ORDER, step
       ORDER_STEP). Lead time is assumed to be zero: the order arrives
       before today's demand is realized (documented assumption -- see
       README.md).
    3. True demand for the day is taken to be the real historical
       "Units Sold" value for that store/product/date (documented
       simplification: the CSV's Units Sold is itself already censored by
       whatever inventory/ordering happened historically, so this is a
       proxy for uncensored demand, not a perfect ground truth).
    4. units_sold = min(available_inventory, demand); stockout units are
       the shortfall. Reward is computed from the shared economic model in
       reward_configs.py, scaled by the scenario's weights.
    5. Rolling/lag features are advanced using the *simulated* realized
       units sold and ending inventory, so future observations reflect the
       agent's own trajectory rather than the original historical trace.

Action space: Discrete(NUM_ACTIONS), action i -> order quantity i * ORDER_STEP.
Observation space: Box(float32) -- see _OBS_SPEC / build_observation().
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from feature_engineering import RollingSalesState
from reward_configs import (
    RewardConfig,
    GROSS_MARGIN,
    HOLDING_RATE_PER_DAY,
    WASTE_COVER_DAYS,
    WASTE_RATE,
    DEVIATION_LAMBDA,
)

MAX_ORDER = 500
ORDER_STEP = 10
NUM_ACTIONS = MAX_ORDER // ORDER_STEP + 1  # 51: {0, 10, ..., 500}

# Warehouse capacity cap: matches the historical max Inventory Level (500).
# Without this, a cheap-enough holding cost lets the agent hoard inventory
# indefinitely over an episode since nothing physically stops it. Orders
# that would push available stock past this cap are truncated at the door
# (rejected/undelivered), not wasted after the fact.
MAX_INVENTORY = 500

EPISODE_LENGTH = 90
MIN_LOOKBACK = 14  # days of real history needed before an episode can start
SCALE_UNITS = 500.0
SCALE_PRICE = 100.0


class ReplenishmentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        history_df,
        demand_model,
        reward_config: RewardConfig,
        episode_length: int = EPISODE_LENGTH,
        rng_seed: int | None = None,
        fixed_pairs: list[tuple[str, str]] | None = None,
    ):
        """
        history_df   : output of feature_engineering.prepare_history(raw_csv_df)
        demand_model : demand_model.DemandModel instance
        reward_config: one entry from reward_configs.SCENARIOS
        fixed_pairs  : optional list of (store_id, product_id) to restrict
                       sampling to (used for held-out eval episodes so train
                       and eval don't draw from the same series).
        """
        super().__init__()
        self.demand_model = demand_model
        self.reward_config = reward_config
        self.episode_length = episode_length
        self.rng = np.random.default_rng(rng_seed)

        self.groups = {
            key: g.reset_index(drop=True)
            for key, g in history_df.groupby(["Store ID", "Product ID"])
        }
        self.pair_keys = fixed_pairs if fixed_pairs else list(self.groups.keys())

        enc = demand_model.encoders
        self._cat_sizes = {
            col: len(enc[col].classes_)
            for col in ["Store ID", "Product ID", "Category", "Region", "Weather Condition", "Seasonality"]
        }
        onehot_dim = sum(self._cat_sizes.values())
        numeric_dim = 16  # see _build_observation() for the exact list (incl. forecast_mean + forecast_q90)
        obs_dim = onehot_dim + numeric_dim

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # episode state
        self._group = None
        self._start_idx = None
        self._t = None
        self._inventory = None
        self._rolling = None
        self._cached_obs = None
        self._episode_log = []

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        if "store_product" in options and "start_idx" in options:
            key = options["store_product"]
            start_idx = options["start_idx"]
        else:
            key = self.pair_keys[self.rng.integers(len(self.pair_keys))]
            group = self.groups[key]
            max_start = len(group) - self.episode_length - 1
            start_idx = int(self.rng.integers(MIN_LOOKBACK, max(MIN_LOOKBACK + 1, max_start)))

        self._group = self.groups[key]
        self._start_idx = start_idx
        self._t = start_idx
        self._episode_log = []

        row0 = self._group.iloc[start_idx]
        self._inventory = float(row0["Inventory Level"])

        seed_sales = self._group["Units Sold"].iloc[max(0, start_idx - 14):start_idx].tolist()
        self._rolling = RollingSalesState(seed_sales, self._inventory)

        self._cached_obs, self._cached_forecast = self._build_observation()
        return self._cached_obs, {"store_product": key, "start_idx": start_idx}

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def step(self, action: int):
        requested_order_qty = float(action) * ORDER_STEP
        row = self._group.iloc[self._t]
        forecast_mean, forecast_q90 = self._cached_forecast

        unit_cost = row["Price"] * (1 - GROSS_MARGIN)
        sale_price = row["Price"] * (1 - row["Discount"] / 100.0)

        # Warehouse capacity cap: the order actually fulfilled cannot push
        # available stock past MAX_INVENTORY. Excess is rejected at the
        # door (not purchased, not paid for), rather than accepted and
        # wasted after the fact.
        max_fulfillable = max(MAX_INVENTORY - self._inventory, 0.0)
        order_qty = min(requested_order_qty, max_fulfillable)

        available = self._inventory + order_qty
        demand = float(row["Units Sold"])  # historical realized demand, used as ground truth
        units_sold = min(available, demand)
        stockout_units = max(demand - available, 0.0)
        end_inventory = available - units_sold

        revenue = units_sold * sale_price
        purchase_cost = order_qty * unit_cost
        holding_cost = end_inventory * unit_cost * HOLDING_RATE_PER_DAY
        lost_margin = max(sale_price - unit_cost, 0.0)
        stockout_cost = stockout_units * lost_margin
        waste_threshold = WASTE_COVER_DAYS * max(forecast_mean, 1.0)
        waste_excess = max(end_inventory - waste_threshold, 0.0)
        waste_cost = waste_excess * unit_cost * WASTE_RATE
        historical_order = float(row["Units Ordered"])
        # Deviation is measured against what the agent *chose* (its raw
        # action), not the capacity-truncated fulfilled quantity -- the
        # penalty is about the agent's decision-making, not the warehouse.
        deviation_units = abs(requested_order_qty - historical_order)
        deviation_cost = deviation_units * unit_cost * DEVIATION_LAMBDA

        cfg = self.reward_config
        reward = (
            revenue
            - purchase_cost
            - cfg.holding_w * holding_cost
            - cfg.stockout_w * stockout_cost
            - cfg.waste_w * waste_cost
            - cfg.deviation_w * deviation_cost
        )
        reward_scaled = reward / 100.0  # keep PPO advantages in a sane numeric range

        self._episode_log.append(
            dict(
                date=row["Date"], order_qty=order_qty, requested_order_qty=requested_order_qty,
                historical_order=historical_order,
                demand=demand, units_sold=units_sold,
                stockout_units=stockout_units, end_inventory=end_inventory, revenue=revenue,
                purchase_cost=purchase_cost, holding_cost=holding_cost,
                stockout_cost=stockout_cost, waste_cost=waste_cost,
                deviation_units=deviation_units, deviation_cost=deviation_cost, reward=reward,
            )
        )

        self._rolling.advance(units_sold, end_inventory)
        self._inventory = end_inventory
        self._t += 1

        terminated = False
        truncated = self._t >= self._start_idx + self.episode_length or self._t >= len(self._group) - 1
        info = {"step_log": self._episode_log[-1]}

        if self._t < len(self._group) - 1:
            # Real row still available -- always return a valid observation
            # (even on the truncating step) so PPO/GAE can bootstrap the
            # value function correctly instead of off a zero vector. SB3's
            # VecEnv wrapper will separately call reset() and swap in the
            # fresh-episode observation for the *next* rollout step.
            self._cached_obs, self._cached_forecast = self._build_observation()
        else:
            self._cached_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            truncated = True

        return self._cached_obs, reward_scaled, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Observation construction
    # ------------------------------------------------------------------ #
    def _build_observation(self):
        row = self._group.iloc[self._t]
        return self._build_observation_from_row(row)

    def _build_observation_from_row(self, row):
        """Same feature/one-hot construction as _build_observation(), but
        takes an explicit `row` (a dict-like/Series with all the same
        fields a real history_df row would have) instead of indexing into
        self._group with self._t. This is what lets inference.plan_ahead()
        keep simulating past the last real historical date for a
        store/product: it builds a synthetic row (carried-forward/assumed
        price, discount, etc., with calendar fields computed from the
        actual future date) and passes it straight in here, without that
        synthetic row needing to actually exist in self._group."""
        forecast_mean, forecast_q90 = self.demand_model.predict(row, self._inventory, self._rolling)
        rolling = self._rolling.features()
        enc = self.demand_model.encoders

        def onehot(col, value):
            n = self._cat_sizes[col]
            v = np.zeros(n, dtype=np.float32)
            from feature_engineering import safe_encode
            v[safe_encode(enc[col], value)] = 1.0
            return v

        parts = [
            onehot("Store ID", row["Store ID"]),
            onehot("Product ID", row["Product ID"]),
            onehot("Category", row["Category"]),
            onehot("Region", row["Region"]),
            onehot("Weather Condition", row["Weather Condition"]),
            onehot("Seasonality", row["Seasonality"]),
        ]
        dow = row["DayOfWeek"]
        month = row["Month"]
        numeric = np.array([
            self._inventory / SCALE_UNITS,
            row["Price"] / SCALE_PRICE,
            row["Discount"] / 100.0,
            row["Competitor Pricing"] / SCALE_PRICE,
            float(row["Holiday/Promotion"]),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
            rolling["Sales_Lag_1"] / SCALE_UNITS,
            rolling["Sales_Lag_7"] / SCALE_UNITS,
            rolling["Sales_RollMean_7"] / SCALE_UNITS,
            rolling["Sales_RollStd_7"] / SCALE_UNITS,
            rolling["Sales_RollMean_14"] / SCALE_UNITS,
            forecast_mean / SCALE_UNITS,
        ], dtype=np.float32)
        # forecast_q90 folded in below to keep numeric_dim accounting simple
        numeric = np.concatenate([numeric, np.array([forecast_q90 / SCALE_UNITS], dtype=np.float32)])

        obs = np.concatenate(parts + [numeric]).astype(np.float32)
        return obs, (forecast_mean, forecast_q90)

    def get_episode_log(self):
        import pandas as pd
        return pd.DataFrame(self._episode_log)
