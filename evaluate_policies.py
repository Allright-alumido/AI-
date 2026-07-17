"""
evaluate_policies.py

Runs the four trained PPO policies plus a "historical replay" baseline
(replays the actual historical Units Ordered from the CSV, clipped/rounded
to the action grid) over the same held-out (store, product) episodes, and
reports a common, unweighted KPI table so the four reward configurations
can be compared on equal footing regardless of what each was trained to
optimize for.

Usage:
    python evaluate_policies.py --episodes-per-pair 3 --seed 0

If you trained multiple seeds per scenario (see train_rl.py), pass a
comma-separated list to average across them and get a feel for training
variance, e.g. --seed 0,1,2. Each seed's policy is evaluated separately and
the table shows the mean +/- std across seeds per scenario.

Requires: gymnasium, stable-baselines3, torch, lightgbm, scikit-learn,
joblib, pandas, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from feature_engineering import prepare_history
from demand_model import DemandModel
from replenishment_env import ReplenishmentEnv, ORDER_STEP, MAX_ORDER
from reward_configs import SCENARIOS
from train_rl import DATA_PATH, MODELS_DIR, make_train_eval_pairs, EVAL_ENV_SEED


def raw_profit_kpis(log: pd.DataFrame) -> dict:
    """Unweighted economic KPIs, comparable across all scenarios/policies."""
    total_demand = log["demand"].sum()
    total_stockout = log["stockout_units"].sum()
    profit = (
        log["revenue"].sum()
        - log["purchase_cost"].sum()
        - log["holding_cost"].sum()
        - log["stockout_cost"].sum()
        - log["waste_cost"].sum()
    )
    return {
        "profit": profit,
        "revenue": log["revenue"].sum(),
        "holding_cost": log["holding_cost"].sum(),
        "stockout_cost_unweighted": log["stockout_cost"].sum(),
        "service_level": 1.0 - (total_stockout / total_demand if total_demand > 0 else 0.0),
        "avg_end_inventory": log["end_inventory"].mean(),
        "avg_order_qty": log["order_qty"].mean(),
        "avg_historical_order_qty": log["historical_order"].mean(),
        "mean_abs_deviation_units": log["deviation_units"].mean(),
    }


def historical_action_for(row) -> int:
    order = float(row["Units Ordered"])
    order = int(round(order / ORDER_STEP)) * ORDER_STEP
    return int(np.clip(order, 0, MAX_ORDER) // ORDER_STEP)


def evaluate_all(episodes_per_pair: int = 3, seeds: list[int] = (0,)):
    """Returns {policy_name: {seed: episode_log_df}}. 'historical_replay' has
    a single pseudo-seed (0) since it isn't a trained model."""
    raw = pd.read_csv(DATA_PATH)
    history_df = prepare_history(raw)
    demand_model = DemandModel()
    _, eval_pairs = make_train_eval_pairs(history_df)

    results: dict[str, dict[int, pd.DataFrame]] = {}

    # --- historical replay baseline (seed-independent) -------------------
    baseline_logs = []
    for pair in eval_pairs:
        env = ReplenishmentEnv(history_df, demand_model, SCENARIOS["pure_profit"],
                                rng_seed=EVAL_ENV_SEED, fixed_pairs=[pair])
        for ep in range(episodes_per_pair):
            obs, info = env.reset(seed=EVAL_ENV_SEED + ep, options={
                "store_product": pair, "start_idx": 14 + ep * 30
            })
            done = False
            while not done:
                row = env._group.iloc[env._t]
                action = historical_action_for(row)
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
            baseline_logs.append(env.get_episode_log())
    results["historical_replay"] = {0: pd.concat(baseline_logs, ignore_index=True)}

    # --- each trained scenario policy, for each requested seed -----------
    for name, cfg in SCENARIOS.items():
        results[name] = {}
        for seed in seeds:
            model_dir = os.path.join(MODELS_DIR, name, f"seed_{seed}")
            model_path = os.path.join(model_dir, "ppo_model.zip")
            vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
            if not os.path.exists(model_path):
                print(f"[skip] no trained model found for '{name}' seed {seed} at {model_path}")
                continue

            model = PPO.load(model_path)
            dummy_env = DummyVecEnv([lambda: ReplenishmentEnv(
                history_df, demand_model, cfg, rng_seed=EVAL_ENV_SEED, fixed_pairs=eval_pairs
            )])
            vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
            vecnorm.training = False
            vecnorm.norm_reward = False

            logs = []
            for pair in eval_pairs:
                env = ReplenishmentEnv(history_df, demand_model, cfg, rng_seed=EVAL_ENV_SEED, fixed_pairs=[pair])
                for ep in range(episodes_per_pair):
                    obs, info = env.reset(seed=EVAL_ENV_SEED + ep, options={
                        "store_product": pair, "start_idx": 14 + ep * 30
                    })
                    done = False
                    while not done:
                        norm_obs = vecnorm.normalize_obs(obs[None, :])
                        action, _ = model.predict(norm_obs, deterministic=True)
                        obs, reward, terminated, truncated, step_info = env.step(int(action[0]))
                        done = terminated or truncated
                    logs.append(env.get_episode_log())
            results[name][seed] = pd.concat(logs, ignore_index=True)

    return results


KPI_COLS = ["profit", "revenue", "holding_cost", "stockout_cost_unweighted",
            "service_level", "avg_end_inventory", "avg_order_qty",
            "avg_historical_order_qty", "mean_abs_deviation_units"]


def summarize(results: dict) -> pd.DataFrame:
    """One row per policy: mean across seeds. If more than one seed was
    evaluated for a policy, a '<kpi>_std' column is added showing the
    across-seed standard deviation, so you can see how much of any
    difference between scenarios is training noise vs. a real effect."""
    rows = []
    for name, per_seed_logs in results.items():
        seed_kpis = [raw_profit_kpis(log) for log in per_seed_logs.values()]
        if not seed_kpis:
            continue
        df_seeds = pd.DataFrame(seed_kpis)
        row = df_seeds.mean().to_dict()
        row["policy"] = name
        row["n_seeds"] = len(seed_kpis)
        if len(seed_kpis) > 1:
            for col in KPI_COLS:
                row[f"{col}_std"] = df_seeds[col].std()
        rows.append(row)
    df = pd.DataFrame(rows).set_index("policy")
    ordered_cols = [c for c in KPI_COLS + ["n_seeds"] + [f"{c}_std" for c in KPI_COLS] if c in df.columns]
    return df[ordered_cols]


def plot_comparison(summary: pd.DataFrame, out_path: str = "policy_comparison.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    has_std = "profit_std" in summary.columns
    yerr = lambda col: summary[f"{col}_std"] if has_std and f"{col}_std" in summary.columns else None
    summary["profit"].plot(kind="bar", ax=axes[0], title="Total profit (unweighted)", yerr=yerr("profit"), capsize=4)
    summary["service_level"].plot(kind="bar", ax=axes[1], title="Service level (1 - stockout rate)", yerr=yerr("service_level"), capsize=4)
    summary["avg_end_inventory"].plot(kind="bar", ax=axes[2], title="Avg ending inventory", yerr=yerr("avg_end_inventory"), capsize=4)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-pair", type=int, default=3)
    parser.add_argument("--seed", type=str, default="0",
                         help="Comma-separated training seed(s) to evaluate, e.g. '0' or '0,1,2'. "
                              "Must match --seed value(s) used in train_rl.py.")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seed.split(",")]

    results = evaluate_all(args.episodes_per_pair, seeds=seeds)
    summary = summarize(results)
    print(summary.to_string())
    summary.to_csv("policy_comparison.csv")
    plot_comparison(summary)


if __name__ == "__main__":
    main()
