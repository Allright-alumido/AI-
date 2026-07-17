"""
streaming_pipeline.py

The Speed Layer of a Lambda Architecture: Ingestion -> Processing ->
Decision -> Serving, running on top of the SAME batch-trained models
(LightGBM + PPO) the rest of this project already produces. It never
retrains anything -- that's the whole point of Lambda Architecture: the
Batch Layer (train_rl.py + the LightGBM training notebook) stays a slow,
offline, periodic job; this module is the fast path that reacts to new
data as it arrives, using whatever the Batch Layer last produced.

    Batch Layer (offline, existing, unchanged)
        retail_store_inventory.csv --train_rl.py/LightGBM training-->
        lgb_sales_model_*.pkl + models/<scenario>/seed_<n>/ppo_model.zip

    Speed Layer (this module, new)
        [Ingestion] streaming_data_generator.BlockBootstrapGenerator
            -> one new simulated settlement row per (store, product) per tick
        [Processing] StreamState
            -> maintains the rolling-sales/inventory/exogenous-carry state
               that has to persist BETWEEN ticks for each series (a real
               streaming system would keep this in something like Flink/
               Redis state; here it's an in-memory dict, same idea at
               prototype scale)
        [Decision] inference.Recommender.recommend_for_row(...)
            -> reuses the exact same, already-tested observation-building
               and PPO inference code the Recommendation/Stress
               Test/Future Plan tabs already call. No duplicated logic.
        [Serving] write_snapshot()
            -> latest state written to stream_state/latest_snapshot.json
               (what a UI would poll) and appended to
               stream_state/stream_log.csv (the running history), decoupled
               from however a UI chooses to display it.

Honesty note: this is a SIMULATION of a streaming architecture for a demo/
pitch, not a production deployment. There is no Kafka/Flink/Spark here --
those would be the natural next step for a real deployment, and are named
explicitly as such rather than implied. What IS real: the separation of
concerns (batch training vs. speed-layer serving), the fact that no
retraining happens on this path, and that the Decision step calls the
project's actual, tested inference code rather than a mocked-up stand-in.

Case Memory note (see memory_store.py): the Decision step's action can now be
passed through a bounded, evidence-gated memory/nudge layer before being
settled. That layer is also NOT retraining anything -- see
memory_store.py's module docstring for exactly what it is.

Usage:
    python streaming_pipeline.py --scenario zero_stockout --ticks 5

Requires: the same environment as train_rl.py/inference.py (gymnasium,
stable-baselines3, lightgbm, joblib, pandas, numpy) -- will not run as-is
in the Cowork sandbox; see README.md's validation note.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import pandas as pd

from feature_engineering import prepare_history, RollingSalesState
from streaming_data_generator import BlockBootstrapGenerator
from reward_configs import SCENARIOS, GROSS_MARGIN, HOLDING_RATE_PER_DAY, WASTE_COVER_DAYS, WASTE_RATE
from replenishment_env import MAX_INVENTORY, ORDER_STEP
from inference import Recommender, SCENARIO_LABELS
from memory_store import CaseMemory

DATA_PATH = "retail_store_inventory.csv"
STREAM_DIR = "stream_state"
SNAPSHOT_PATH = os.path.join(STREAM_DIR, "latest_snapshot.json")
LOG_PATH = os.path.join(STREAM_DIR, "stream_log.csv")


@dataclass
class PairState:
    """Per-(store,product) state that must persist BETWEEN ticks -- this is
    the "Processing" layer's memory. Seeded from that pair's own last real
    historical row so the stream picks up exactly where the batch data left
    off, then evolves from its own simulated trajectory afterward (same
    principle as inference.Recommender.plan_ahead()'s extrapolation)."""
    current_inventory: float
    rolling: RollingSalesState
    carry_units_ordered: float
    carry_demand_forecast: float
    sim_date: pd.Timestamp
    tick: int = 0
    log: list = field(default_factory=list)


class StreamingPipeline:
    def __init__(self, recommenders: dict[str, Recommender], primary_scenario: str,
                 history_df: pd.DataFrame, rng_seed: int = 0, use_case_memory: bool = True):
        """recommenders: {scenario_name: Recommender}, one loaded per
        strategy you want to see react side-by-side each tick (mirrors the
        existing 'Policy Comparison' panel in app.py). primary_scenario
        picks which one's decision actually drives the simulated inventory
        forward -- only one real decision can be acted on per tick, the
        others are shown for comparison only, exactly like the
        Recommendation tab already does with real historical rows.

        use_case_memory: whether the primary scenario's action gets passed
        through memory_store.CaseMemory's bounded, evidence-gated nudge
        before being settled (see memory_store.py's module docstring for
        exactly what this does and does not do). Exposed as a flag so the
        UI can show "PPO alone" vs. "PPO + Case Memory" side by side."""
        if primary_scenario not in recommenders:
            raise ValueError(f"primary_scenario '{primary_scenario}' not in recommenders dict")
        self.recommenders = recommenders
        self.primary_scenario = primary_scenario
        self.history_df = history_df
        self.generator = BlockBootstrapGenerator(history_df, rng_seed=rng_seed)
        self.pairs = sorted(history_df.groupby(["Store ID", "Product ID"]).groups.keys())
        self.states: dict[tuple[str, str], PairState] = {
            key: self._init_state(key) for key in self.pairs
        }
        self.use_case_memory = use_case_memory
        self.memory = CaseMemory(order_step=ORDER_STEP)

    def _init_state(self, key: tuple[str, str]) -> PairState:
        group = self.history_df[
            (self.history_df["Store ID"] == key[0]) & (self.history_df["Product ID"] == key[1])
        ].sort_values("Date")
        last_row = group.iloc[-1]
        seed_sales = group["Units Sold"].iloc[-14:].tolist()
        return PairState(
            current_inventory=float(last_row["Inventory Level"]),
            rolling=RollingSalesState(seed_sales, float(last_row["Inventory Level"])),
            carry_units_ordered=float(last_row["Units Ordered"]),
            carry_demand_forecast=float(last_row["Demand Forecast"]),
            sim_date=pd.Timestamp(last_row["Date"]),
        )

    # ------------------------------------------------------------------ #
    def _settle_day(self, row: dict, state: PairState, fulfilled_qty: float) -> dict:
        """Same economics as replenishment_env.ReplenishmentEnv.step(), run
        against a live-ingested row instead of a pre-loaded dataframe index
        -- kept as one small inlined block (rather than reusing the env's
        step() directly) because the env indexes into a static
        self._group[self._t]; here the row genuinely doesn't exist in any
        dataframe until after we've generated it this tick."""
        unit_cost = row["Price"] * (1 - GROSS_MARGIN)
        sale_price = row["Price"] * (1 - row["Discount"] / 100.0)
        available = state.current_inventory + fulfilled_qty
        demand = float(row["Units Sold"])  # today's bootstrapped ground truth
        units_sold = min(available, demand)
        stockout_units = max(demand - available, 0.0)
        end_inventory = available - units_sold

        revenue = units_sold * sale_price
        purchase_cost = fulfilled_qty * unit_cost
        holding_cost = end_inventory * unit_cost * HOLDING_RATE_PER_DAY
        lost_margin = max(sale_price - unit_cost, 0.0)
        stockout_cost = stockout_units * lost_margin
        waste_threshold = WASTE_COVER_DAYS * max(row.get("_forecast_mean", demand), 1.0)
        waste_cost = max(end_inventory - waste_threshold, 0.0) * unit_cost * WASTE_RATE

        return {
            "units_sold": units_sold, "stockout_units": stockout_units,
            "end_inventory": end_inventory, "revenue": revenue,
            "purchase_cost": purchase_cost, "holding_cost": holding_cost,
            "stockout_cost": stockout_cost, "waste_cost": waste_cost,
            "profit": revenue - purchase_cost - holding_cost - stockout_cost - waste_cost,
        }

    def tick(self) -> dict:
        """One simulated day, across every (store, product) pair:
        Ingestion -> Processing -> Decision -> Serving. Returns the
        snapshot dict that also gets written to disk."""
        results = {}
        for key in self.pairs:
            state = self.states[key]
            state.sim_date = state.sim_date + pd.Timedelta(days=1)
            state.tick += 1

            # --- Ingestion: one new synthetic settlement row -------------
            raw_row = self.generator.next_row(key[0], key[1], state.sim_date)
            row = dict(raw_row)
            row["Units Ordered"] = state.carry_units_ordered
            row["Demand Forecast"] = state.carry_demand_forecast

            # --- Decision: ask every loaded scenario, act on the primary -
            per_scenario = {}
            primary_obs = None
            for name, rec in self.recommenders.items():
                context = {"store_id": key[0], "product_id": key[1], "date": str(state.sim_date.date())}
                if name == self.primary_scenario:
                    rec_out, primary_obs = rec.recommend_for_row(
                        pd.Series(row), state.current_inventory, state.rolling, context, return_obs=True
                    )
                else:
                    rec_out = rec.recommend_for_row(pd.Series(row), state.current_inventory, state.rolling, context)
                per_scenario[name] = rec_out
            primary = dict(per_scenario[self.primary_scenario])
            row["_forecast_mean"] = primary["forecast_mean"]

            # --- Reflection: Case Memory may propose a bounded nudge to the
            # primary scenario's action, based on similar past states for
            # this SAME pair (see memory_store.py). PPO's own action is
            # left untouched unless there's enough consistent evidence.
            ppo_action_idx = round(primary["requested_order_qty"] / ORDER_STEP)
            case_memory_info = {"nudged": False, "explanation": None, "n_neighbors": 0}
            if self.use_case_memory:
                case_memory_info = self.memory.reflect(key, primary_obs, ppo_action_idx)
                if case_memory_info["nudged"]:
                    final_action_idx = case_memory_info["final_action_idx"]
                    new_requested = final_action_idx * ORDER_STEP
                    new_fulfilled = min(new_requested, max(MAX_INVENTORY - state.current_inventory, 0.0))
                    primary["requested_order_qty"] = new_requested
                    primary["fulfilled_order_qty"] = new_fulfilled
                    primary["capped_by_capacity"] = new_fulfilled < new_requested

            # --- Processing: settle the (possibly nudged) primary action --
            outcome = self._settle_day(row, state, primary["fulfilled_order_qty"])

            state.rolling.advance(outcome["units_sold"], outcome["end_inventory"])
            state.current_inventory = outcome["end_inventory"]
            state.carry_units_ordered = primary["fulfilled_order_qty"]
            state.carry_demand_forecast = primary["forecast_mean"]

            # --- Closed loop: log what actually happened back into Case Memory
            # so later ticks (for this same pair) can retrieve this episode.
            final_action_idx = round(primary["requested_order_qty"] / ORDER_STEP)
            self.memory.log_episode(
                key, primary_obs, final_action_idx, outcome["profit"], state.tick, str(state.sim_date.date())
            )

            record = {
                "date": str(state.sim_date.date()),
                "store_id": key[0], "product_id": key[1],
                "template_date": str(pd.Timestamp(raw_row["_template_date"]).date()),
                "primary_scenario": self.primary_scenario,
                "recommended_order_qty": primary["fulfilled_order_qty"],
                "forecast_mean": primary["forecast_mean"],
                "forecast_q90": primary["forecast_q90"],
                **outcome,
                "case_memory": case_memory_info,
                "other_scenarios": {
                    name: {"order_qty": r["fulfilled_order_qty"], "forecast_mean": r["forecast_mean"]}
                    for name, r in per_scenario.items() if name != self.primary_scenario
                },
            }
            state.log.append(record)
            results[f"{key[0]}/{key[1]}"] = record

        snapshot = {
            "tick": next(iter(self.states.values())).tick,
            "sim_date": str(next(iter(self.states.values())).sim_date.date()),
            "primary_scenario": self.primary_scenario,
            "scenario_label": SCENARIO_LABELS[self.primary_scenario],
            "n_pairs": len(self.pairs),
            "results": results,
        }
        return snapshot

    # ------------------------------------------------------------------ #
    def write_snapshot(self, snapshot: dict) -> None:
        """Serving layer: latest full state as JSON (what a UI polls) plus
        an appended flat CSV log (the running history a UI or analyst
        could chart over time)."""
        os.makedirs(STREAM_DIR, exist_ok=True)
        with open(SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        rows = []
        for pair_key, rec in snapshot["results"].items():
            flat = {k: v for k, v in rec.items() if k not in ("other_scenarios", "case_memory")}
            case_memory = rec.get("case_memory", {})
            flat["case_memory_nudged"] = case_memory.get("nudged", False)
            flat["case_memory_explanation"] = case_memory.get("explanation")
            rows.append(flat)
        df = pd.DataFrame(rows)
        write_header = not os.path.exists(LOG_PATH)
        df.to_csv(LOG_PATH, mode="a", header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="pure_profit", choices=list(SCENARIOS.keys()),
                         help="Primary scenario -- its decision actually drives simulated inventory forward.")
    parser.add_argument("--compare-all", action="store_true",
                         help="Also load and show the other 3 strategies' reactions each tick (comparison only).")
    parser.add_argument("--ticks", type=int, default=5, help="Number of simulated days to run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-case-memory", action="store_true",
                         help="Disable the Case Memory nudge layer -- run PPO's raw decisions only.")
    args = parser.parse_args()

    raw = pd.read_csv(DATA_PATH)
    history_df = prepare_history(raw)

    from demand_model import DemandModel
    demand_model = DemandModel()

    scenario_names = list(SCENARIOS.keys()) if args.compare_all else [args.scenario]
    recommenders = {}
    for name in scenario_names:
        try:
            recommenders[name] = Recommender(name, seed=args.seed, history_df=history_df, demand_model=demand_model)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
    if args.scenario not in recommenders:
        raise SystemExit(f"Primary scenario '{args.scenario}' has no trained model -- train it first "
                          f"(python train_rl.py --scenario {args.scenario} --seed {args.seed}).")

    pipeline = StreamingPipeline(recommenders, args.scenario, history_df, rng_seed=args.seed,
                                  use_case_memory=not args.no_case_memory)

    for i in range(args.ticks):
        snapshot = pipeline.tick()
        pipeline.write_snapshot(snapshot)
        print(f"[tick {snapshot['tick']}] sim_date={snapshot['sim_date']}  "
              f"({snapshot['n_pairs']} store/product pairs updated)")
    print(f"\nDone. Latest state: {SNAPSHOT_PATH}\nRunning log: {LOG_PATH}")


if __name__ == "__main__":
    main()
