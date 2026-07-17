"""
train_rl.py

Trains one PPO agent (same network/hyperparameters) for each of the four
reward scenarios in reward_configs.SCENARIOS, on a train/eval split of the
100 (store, product) series.

Usage:
    python train_rl.py --scenario historical_baseline --timesteps 400000 --seed 0
    python train_rl.py --scenario all --timesteps 400000 --seed 1
    python train_rl.py --scenario all --timesteps 400000 --seed 2

Run the same command with 2-3 different --seed values per scenario before
trusting any single result -- PPO training variance can otherwise be mistaken
for a real difference between scenarios. Each seed's model is saved under
models/<scenario>/seed_<seed>/ so runs don't overwrite each other; point
evaluate_policies.py at whichever seed(s) you want to compare.

Requires: gymnasium, stable-baselines3, torch, lightgbm, scikit-learn,
joblib, pandas, numpy (see requirements.txt). Not runnable as-is in a
sandbox without internet access -- see README.md for how this was validated.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from feature_engineering import prepare_history
from demand_model import DemandModel
from replenishment_env import ReplenishmentEnv
from reward_configs import SCENARIOS

DATA_PATH = "retail_store_inventory.csv"
MODELS_DIR = "models"
EVAL_HOLDOUT_FRACTION = 0.2  # fraction of (store, product) pairs held out for eval

# Fixed regardless of --seed: the train/eval (store, product) split and the
# eval-episode sampling must stay identical across seed reruns, or you'd be
# comparing seeds on different data instead of isolating training variance.
SPLIT_SEED = 42
EVAL_ENV_SEED = 999


def make_train_eval_pairs(history_df, seed=SPLIT_SEED):
    pairs = sorted(history_df.groupby(["Store ID", "Product ID"]).groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * EVAL_HOLDOUT_FRACTION))
    return pairs[n_eval:], pairs[:n_eval]  # train_pairs, eval_pairs


def make_env_fn(history_df, demand_model, scenario_name, pairs, seed):
    cfg = SCENARIOS[scenario_name]

    def _init():
        return ReplenishmentEnv(
            history_df, demand_model, cfg, rng_seed=seed, fixed_pairs=pairs
        )

    return _init


def train_scenario(scenario_name: str, timesteps: int, seed: int):
    print(f"\n=== Training scenario: {scenario_name} (seed={seed}) ===")
    raw = pd.read_csv(DATA_PATH)
    history_df = prepare_history(raw)
    demand_model = DemandModel()

    train_pairs, eval_pairs = make_train_eval_pairs(history_df)

    train_env = DummyVecEnv([make_env_fn(history_df, demand_model, scenario_name, train_pairs, seed)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env_fn(history_df, demand_model, scenario_name, eval_pairs, EVAL_ENV_SEED)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    out_dir = os.path.join(MODELS_DIR, scenario_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=10_000,
        n_eval_episodes=len(eval_pairs),
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        tensorboard_log=None,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)

    model.save(os.path.join(out_dir, "ppo_model"))
    train_env.save(os.path.join(out_dir, "vecnormalize.pkl"))
    print(f"Saved model + VecNormalize stats to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()) + ["all"], default="all"
    )
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0,
                         help="Training seed. Rerun with a few different values (e.g. 0, 1, 2) "
                              "per scenario to check results aren't just training noise.")
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for s in scenarios:
        train_scenario(s, args.timesteps, args.seed)


if __name__ == "__main__":
    main()
