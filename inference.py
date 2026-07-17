"""
inference.py

The layer a UI calls. Everything upstream (train_rl.py, evaluate_policies.py)
trains and batch-evaluates; nothing so far answers "given today's situation
for one store/product, what should I order right now?" as a single, fast
call. That's what `Recommender` does: load one trained scenario's policy +
VecNormalize once, then reuse it for many recommend()/compare() calls
without reloading anything.

Three entry points:
    Recommender.recommend_from_history(store_id, product_id, date)
        Pull a real historical row and get the model's recommendation for it.
    Recommender.recommend_manual(...)
        Stress-testing: override inventory/demand/price/etc. directly
        (bypassing the LightGBM forecast call) and see how the
        recommendation reacts.
    Recommender.compare_to_historical(store_id, product_id, start_date, n_days)
        Run the trained policy and a historical replay over the same window
        for one (store, product) pair, for a side-by-side comparison plus
        the profit the RL policy would have added over that window.
    Recommender.plan_ahead(store_id, product_id, start_date, n_days, demand_scenario)
        Forward-looking replenishment plan: what to order each day for the
        next n_days, starting from start_date. Unlike compare_to_historical,
        this does NOT use real historical demand (the whole point is that
        future demand isn't known yet) -- it recursively advances inventory
        using the LightGBM forecast itself as the assumed demand each day.

Internally, recommend_from_history / recommend_manual reuse
ReplenishmentEnv._build_observation() by "puppeteering" a live env
instance's internal state rather than re-deriving the feature/one-hot
logic here -- that logic is already written and tested in
replenishment_env.py, and duplicating it would risk the two drifting apart.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from feature_engineering import prepare_history, RollingSalesState, safe_encode
from demand_model import DemandModel
from replenishment_env import (
    ReplenishmentEnv, ORDER_STEP, MAX_INVENTORY, SCALE_UNITS, SCALE_PRICE,
)
from reward_configs import SCENARIOS

MODELS_DIR = "models"
DATA_PATH = "retail_store_inventory.csv"

# Business-friendly labels for the UI -- keeps RL jargon out of the picker.
SCENARIO_LABELS = {
    "historical_baseline": "Conservative (stay close to current practice)",
    "zero_stockout": "Never stock out",
    "high_holding_stockout_tolerant": "Minimize storage cost",
    "pure_profit": "Maximize profit",
}
LABEL_TO_SCENARIO = {v: k for k, v in SCENARIO_LABELS.items()}


class Recommender:
    def __init__(self, scenario_name: str, seed: int = 0,
                 models_dir: str = MODELS_DIR, data_path: str = DATA_PATH,
                 history_df: pd.DataFrame | None = None,
                 demand_model: DemandModel | None = None):
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_name}'. Options: {list(SCENARIOS)}")
        self.scenario_name = scenario_name
        self.cfg = SCENARIOS[scenario_name]

        # history_df/demand_model can be passed in and shared across several
        # Recommender instances (one per scenario) so the UI doesn't reload
        # the CSV + LightGBM models once per scenario.
        self.history_df = history_df if history_df is not None else prepare_history(pd.read_csv(data_path))
        self.demand_model = demand_model if demand_model is not None else DemandModel()

        model_dir = os.path.join(models_dir, scenario_name, f"seed_{seed}")
        model_path = os.path.join(model_dir, "ppo_model.zip")
        vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Train it first: "
                f"python train_rl.py --scenario {scenario_name} --seed {seed}"
            )
        self.model = PPO.load(model_path)

        # A live env instance whose internal state we set directly before
        # calling _build_observation() -- see module docstring.
        self._env = ReplenishmentEnv(self.history_df, self.demand_model, self.cfg, rng_seed=0)
        dummy_vec = DummyVecEnv([lambda: self._env])
        self.vecnorm = VecNormalize.load(vecnorm_path, dummy_vec)
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False

    # ------------------------------------------------------------------ #
    def _predict_action_index(self, obs: np.ndarray) -> int:
        norm_obs = self.vecnorm.normalize_obs(obs[None, :])
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return int(action[0])

    def _finalize(self, action_idx: int, forecast_mean: float, forecast_q90: float,
                   current_inventory: float, context: dict) -> dict:
        requested_qty = action_idx * ORDER_STEP
        fulfilled_qty = min(requested_qty, max(MAX_INVENTORY - current_inventory, 0.0))
        return {
            "scenario": self.scenario_name,
            "scenario_label": SCENARIO_LABELS[self.scenario_name],
            "requested_order_qty": requested_qty,
            "fulfilled_order_qty": fulfilled_qty,
            "capped_by_capacity": fulfilled_qty < requested_qty,
            "forecast_mean": forecast_mean,
            "forecast_q90": forecast_q90,
            "current_inventory": current_inventory,
            **context,
        }

    # ------------------------------------------------------------------ #
    def recommend_for_row(self, row, current_inventory: float,
                           rolling_state: RollingSalesState, context: dict | None = None,
                           return_obs: bool = False):
        """Shared core: given an already-assembled row (real historical OR
        synthetic -- anything with the same fields ReplenishmentEnv expects)
        plus the persisted (current_inventory, rolling_state) for that
        (store, product) series, build the observation and get a decision.

        This is what lets a caller that maintains its OWN per-pair state
        across calls (e.g. streaming_pipeline.py's StreamState, which is
        fed brand-new rows from streaming_data_generator.py rather than
        rows already sitting in self.history_df) reuse the exact same,
        already-tested observation/prediction logic as
        recommend_from_history() and plan_ahead(), instead of duplicating
        it a third time. recommend_from_history() below is now just this
        method plus the bookkeeping needed to look a real row up by date.

        return_obs=True additionally returns the raw observation vector fed
        to the policy -- used by streaming_pipeline.py to feed
        memory_store.CaseMemory the exact same numbers the policy
        itself judged "similar" on, rather than re-deriving a separate
        feature representation. Default False keeps every existing caller
        (recommend_from_history, plan_ahead) returning a plain dict, as
        before.
        """
        self._env._inventory = current_inventory
        self._env._rolling = rolling_state
        obs, (forecast_mean, forecast_q90) = self._env._build_observation_from_row(row)
        action_idx = self._predict_action_index(obs)
        result = self._finalize(action_idx, forecast_mean, forecast_q90, current_inventory, context or {})
        if return_obs:
            return result, obs
        return result

    def recommend_from_history(self, store_id: str, product_id: str, date) -> dict:
        """Recommendation for a real historical day (mainly useful for the
        "decision vs. history" comparison and for demoing with real data)."""
        key = (store_id, product_id)
        if key not in self._env.groups:
            raise ValueError(f"No data for store={store_id}, product={product_id}")
        group = self._env.groups[key]
        matches = group.index[group["Date"] == pd.Timestamp(date)]
        if len(matches) == 0:
            raise ValueError(f"No row for {store_id}/{product_id} on {date}")
        t = int(matches[0])
        row = group.iloc[t]
        current_inventory = float(row["Inventory Level"])
        seed_sales = group["Units Sold"].iloc[max(0, t - 14):t].tolist()
        rolling = RollingSalesState(seed_sales, current_inventory)

        return self.recommend_for_row(
            row, current_inventory, rolling,
            {"store_id": store_id, "product_id": product_id, "date": str(date),
             "historical_order_qty": float(row["Units Ordered"]),
             "price": float(row["Price"]), "discount": float(row["Discount"])},
        )

    # ------------------------------------------------------------------ #
    def recommend_manual(
        self, *,
        store_id: str, product_id: str, category: str, region: str,
        weather: str, seasonality: str,
        price: float, discount: float, competitor_pricing: float, holiday_promo: bool,
        current_inventory: float,
        forecast_mean: float, forecast_q90: float | None = None,
        date=None,
        sales_lag_1: float | None = None, sales_lag_7: float | None = None,
        rollmean_7: float | None = None, rollstd_7: float | None = None, rollmean_14: float | None = None,
    ) -> dict:
        """Stress-testing entry point: every input is a direct override, and
        the LightGBM forecast call is bypassed entirely -- you're supplying
        the demand forecast yourself to see how the *policy* reacts, not
        asking the demand model anything. Rolling/lag features default to
        the supplied forecast_mean (i.e. "recent sales have been running
        about like today's forecast") if not given explicitly."""
        if forecast_q90 is None:
            forecast_q90 = forecast_mean * 1.3
        sales_lag_1 = forecast_mean if sales_lag_1 is None else sales_lag_1
        sales_lag_7 = forecast_mean if sales_lag_7 is None else sales_lag_7
        rollmean_7 = forecast_mean if rollmean_7 is None else rollmean_7
        rollstd_7 = 0.0 if rollstd_7 is None else rollstd_7
        rollmean_14 = forecast_mean if rollmean_14 is None else rollmean_14
        date = pd.Timestamp.today() if date is None else pd.Timestamp(date)

        obs = self._build_manual_observation(
            store_id=store_id, product_id=product_id, category=category, region=region,
            weather=weather, seasonality=seasonality, price=price, discount=discount,
            competitor_pricing=competitor_pricing, holiday_promo=holiday_promo,
            current_inventory=current_inventory, forecast_mean=forecast_mean, forecast_q90=forecast_q90,
            date=date, sales_lag_1=sales_lag_1, sales_lag_7=sales_lag_7,
            rollmean_7=rollmean_7, rollstd_7=rollstd_7, rollmean_14=rollmean_14,
        )
        action_idx = self._predict_action_index(obs)
        return self._finalize(action_idx, forecast_mean, forecast_q90, current_inventory,
                               {"store_id": store_id, "product_id": product_id, "date": str(date.date()),
                                "price": price, "discount": discount})

    def _build_manual_observation(self, *, store_id, product_id, category, region, weather,
                                    seasonality, price, discount, competitor_pricing, holiday_promo,
                                    current_inventory, forecast_mean, forecast_q90, date,
                                    sales_lag_1, sales_lag_7, rollmean_7, rollstd_7, rollmean_14) -> np.ndarray:
        """Mirrors ReplenishmentEnv._build_observation()'s array layout so
        the policy sees an observation in exactly the format it was trained
        on. Kept in sync manually with replenishment_env.py -- if that
        observation layout changes, update this function too."""
        enc = self.demand_model.encoders
        cat_sizes = self._env._cat_sizes

        def onehot(col, value):
            n = cat_sizes[col]
            v = np.zeros(n, dtype=np.float32)
            v[safe_encode(enc[col], value)] = 1.0
            return v

        parts = [
            onehot("Store ID", store_id),
            onehot("Product ID", product_id),
            onehot("Category", category),
            onehot("Region", region),
            onehot("Weather Condition", weather),
            onehot("Seasonality", seasonality),
        ]
        dow = date.dayofweek
        month = date.month
        numeric = np.array([
            current_inventory / SCALE_UNITS,
            price / SCALE_PRICE,
            discount / 100.0,
            competitor_pricing / SCALE_PRICE,
            float(bool(holiday_promo)),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
            sales_lag_1 / SCALE_UNITS,
            sales_lag_7 / SCALE_UNITS,
            rollmean_7 / SCALE_UNITS,
            rollstd_7 / SCALE_UNITS,
            rollmean_14 / SCALE_UNITS,
            forecast_mean / SCALE_UNITS,
        ], dtype=np.float32)
        numeric = np.concatenate([numeric, np.array([forecast_q90 / SCALE_UNITS], dtype=np.float32)])
        return np.concatenate(parts + [numeric]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def compare_to_historical(self, store_id: str, product_id: str, start_date, n_days: int = 90) -> dict:
        """Runs this policy and a historical replay over the same
        (store, product) window, for a side-by-side comparison plus the
        profit uplift the RL policy would have generated."""
        from evaluate_policies import historical_action_for, raw_profit_kpis

        key = (store_id, product_id)
        if key not in self._env.groups:
            raise ValueError(f"No data for store={store_id}, product={product_id}")
        group = self._env.groups[key]
        matches = group.index[group["Date"] == pd.Timestamp(start_date)]
        if len(matches) == 0:
            raise ValueError(f"No row for {store_id}/{product_id} on {start_date}")
        start_idx = int(matches[0])

        env_rl = ReplenishmentEnv(self.history_df, self.demand_model, self.cfg,
                                   episode_length=n_days, rng_seed=0, fixed_pairs=[key])
        obs, _ = env_rl.reset(options={"store_product": key, "start_idx": start_idx})
        done = False
        while not done:
            action_idx = self._predict_action_index(obs)
            obs, reward, terminated, truncated, _ = env_rl.step(action_idx)
            done = terminated or truncated
        rl_log = env_rl.get_episode_log()

        env_hist = ReplenishmentEnv(self.history_df, self.demand_model, self.cfg,
                                     episode_length=n_days, rng_seed=0, fixed_pairs=[key])
        obs, _ = env_hist.reset(options={"store_product": key, "start_idx": start_idx})
        done = False
        while not done:
            row = env_hist._group.iloc[env_hist._t]
            action_idx = historical_action_for(row)
            obs, reward, terminated, truncated, _ = env_hist.step(action_idx)
            done = terminated or truncated
        hist_log = env_hist.get_episode_log()

        rl_kpis = raw_profit_kpis(rl_log)
        hist_kpis = raw_profit_kpis(hist_log)
        return {
            "rl_log": rl_log,
            "historical_log": hist_log,
            "rl_kpis": rl_kpis,
            "historical_kpis": hist_kpis,
            "profit_uplift": rl_kpis["profit"] - hist_kpis["profit"],
        }

    # ------------------------------------------------------------------ #
    def plan_ahead(self, store_id: str, product_id: str, start_date, n_days: int = 14,
                    demand_scenario: str = "mean", future_assumptions: dict | None = None) -> pd.DataFrame:
        """Forward-looking replenishment plan for the next `n_days`, starting
        from `start_date` (treated as "today").

        Real future demand is unknown, so -- unlike compare_to_historical,
        which replays real historical Units Sold to backtest past decisions
        -- this recursively advances the simulated inventory using the
        LightGBM model's own forecast as the assumed demand each day:
        demand_scenario="mean" assumes average demand shows up each day;
        "p90" assumes the 90th-percentile (high-demand) forecast shows up
        each day instead, as a worst-case-for-stockouts companion run. Call
        this twice (once per scenario) to build a best/worst-case band --
        note the policy makes independent decisions along each path, since
        inventory (and therefore the state it observes) evolves differently.

        For each day up to the last real row this store/product has in the
        CSV, exogenous context (price, discount, holiday, weather, season,
        competitor pricing) is pulled straight from that real historical
        record, since in practice these are usually planned in advance even
        though demand is not.

        Beyond that last real date, there is no historical row left to pull
        that context from -- so this now keeps going by *assuming* it
        instead of stopping. `future_assumptions` (a dict, all keys
        optional) lets you specify what "the future" looks like:
            "price", "discount", "competitor_pricing": fixed values applied
                to every extrapolated day. Default: carried forward from
                the last real day for this store/product (i.e. "whatever
                is currently in effect keeps being in effect").
            "holiday_promo": bool, applied to every extrapolated day.
                Default: False -- a promo/holiday is a discrete planned
                event, not something that should be assumed to continue
                just because the last real day happened to have one.
            "weather", "seasonality": fixed values applied to every
                extrapolated day. Default: carried forward from the last
                real day (weather this far out isn't realistically
                knowable either way, so this is a simplification, not a
                forecast).
        Calendar fields (Year/Month/Day/DayOfWeek) are always computed
        correctly from the actual extrapolated date -- there's no need to
        assume those.

        The two remaining inputs the underlying LightGBM demand model was
        trained to take as exogenous context -- "Units Ordered" and "Demand
        Forecast" (see feature_engineering.py's module docstring) -- have
        no real-world analogue for a genuinely future date (no external
        forecast/order exists yet). Rather than freeze these at their last
        real value for the whole extrapolated window, each extrapolated
        day bootstraps them from the *previous simulated day's own output*
        (this policy's own recommended order, and its own mean forecast),
        seeded from the last real row for the first extrapolated day. This
        keeps them evolving plausibly instead of going stale, at the cost
        of the demand model effectively "hearing its own past predictions
        back" -- documented here rather than hidden, since it's a real
        simplification, not a discovered ground truth.

        `plan_df.attrs` reports `real_days` (backed by actual historical
        rows), `extrapolated_days` (backed by the assumptions above), and
        `uses_extrapolation` (whether any days needed it at all) so the UI
        can be transparent about which is which -- also surfaced per row in
        the `data_source` column ("historical" / "extrapolated").
        """
        if demand_scenario not in ("mean", "p90"):
            raise ValueError("demand_scenario must be 'mean' or 'p90'")

        key = (store_id, product_id)
        if key not in self._env.groups:
            raise ValueError(f"No data for store={store_id}, product={product_id}")
        group = self._env.groups[key]
        matches = group.index[group["Date"] == pd.Timestamp(start_date)]
        if len(matches) == 0:
            raise ValueError(f"No row for {store_id}/{product_id} on {start_date}")
        start_idx = int(matches[0])

        available_days = len(group) - start_idx
        real_days = min(n_days, available_days)
        extrapolated_days = max(0, n_days - available_days)
        uses_extrapolation = extrapolated_days > 0

        current_inventory = float(group.iloc[start_idx]["Inventory Level"])
        seed_sales = group["Units Sold"].iloc[max(0, start_idx - 14):start_idx].tolist()
        rolling = RollingSalesState(seed_sales, current_inventory)

        # --- state used only once we run past the real data ------------------
        future_assumptions = future_assumptions or {}
        last_real_row = group.iloc[-1]
        last_real_date = pd.Timestamp(last_real_row["Date"])
        carry = {
            "price": float(future_assumptions.get("price", last_real_row["Price"])),
            "discount": float(future_assumptions.get("discount", last_real_row["Discount"])),
            "competitor_pricing": float(
                future_assumptions.get("competitor_pricing", last_real_row["Competitor Pricing"])
            ),
            "holiday_promo": float(bool(future_assumptions.get("holiday_promo", False))),
            "weather": future_assumptions.get("weather", last_real_row["Weather Condition"]),
            "seasonality": future_assumptions.get("seasonality", last_real_row["Seasonality"]),
            # Bootstrapped fresh each extrapolated day from the previous
            # day's own model output -- see docstring above.
            "units_ordered": float(last_real_row["Units Ordered"]),
            "demand_forecast": float(last_real_row["Demand Forecast"]),
        }

        def make_synthetic_row(date: pd.Timestamp) -> pd.Series:
            return pd.Series({
                "Date": date,
                "Store ID": last_real_row["Store ID"],
                "Product ID": last_real_row["Product ID"],
                "Category": last_real_row["Category"],
                "Region": last_real_row["Region"],
                "Weather Condition": carry["weather"],
                "Seasonality": carry["seasonality"],
                "Price": carry["price"],
                "Discount": carry["discount"],
                "Competitor Pricing": carry["competitor_pricing"],
                "Holiday/Promotion": carry["holiday_promo"],
                "Units Ordered": carry["units_ordered"],
                "Demand Forecast": carry["demand_forecast"],
                "Year": date.year, "Month": date.month, "Day": date.day,
                "DayOfWeek": date.dayofweek,
            })

        rows = []
        for offset in range(n_days):
            t = start_idx + offset
            is_extrapolated = t >= len(group)
            if is_extrapolated:
                date = last_real_date + pd.Timedelta(days=(t - (len(group) - 1)))
                row = make_synthetic_row(date)
            else:
                row = group.iloc[t]
                date = row["Date"]

            self._env._inventory = current_inventory
            self._env._rolling = rolling
            obs, (forecast_mean, forecast_q90) = self._env._build_observation_from_row(row)
            action_idx = self._predict_action_index(obs)
            requested_qty = action_idx * ORDER_STEP
            fulfilled_qty = min(requested_qty, max(MAX_INVENTORY - current_inventory, 0.0))

            available = current_inventory + fulfilled_qty
            assumed_demand = forecast_mean if demand_scenario == "mean" else forecast_q90
            simulated_units_sold = min(available, assumed_demand)
            end_inventory = available - simulated_units_sold
            stockout_risk_units = max(forecast_q90 - available, 0.0)

            rows.append({
                "date": date,
                "day_offset": offset,
                "data_source": "extrapolated" if is_extrapolated else "historical",
                "starting_inventory": current_inventory,
                "recommended_order_qty": fulfilled_qty,
                "requested_order_qty": requested_qty,
                "capped_by_capacity": fulfilled_qty < requested_qty,
                "forecast_mean": forecast_mean,
                "forecast_q90": forecast_q90,
                "assumed_demand": assumed_demand,
                "expected_ending_inventory": end_inventory,
                "stockout_risk_units_if_high_demand": stockout_risk_units,
            })

            rolling.advance(simulated_units_sold, end_inventory)
            current_inventory = end_inventory
            if is_extrapolated:
                # Bootstrap tomorrow's exogenous "prior order/forecast"
                # inputs from today's own output -- see docstring above.
                carry["units_ordered"] = fulfilled_qty
                carry["demand_forecast"] = forecast_mean

        plan_df = pd.DataFrame(rows)
        plan_df.attrs["requested_days"] = n_days
        plan_df.attrs["available_days"] = available_days
        plan_df.attrs["real_days"] = real_days
        plan_df.attrs["extrapolated_days"] = extrapolated_days
        plan_df.attrs["uses_extrapolation"] = uses_extrapolation
        plan_df.attrs["demand_scenario"] = demand_scenario
        plan_df.attrs["future_assumptions"] = future_assumptions
        return plan_df
