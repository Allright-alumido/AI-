from __future__ import annotations
"""
plot_training_curves.py

Reads PPO's TensorBoard event files directly (via
tensorboard.backend.event_processing.event_accumulator) and plots the
reward curves with matplotlib -- bypassing the `tensorboard` CLI entirely.

Why this exists: `tensorboard --logdir ...` imports its "uploader" feature
on startup, which pulls in google-auth-oauthlib -> google.auth, and on some
conda installs (Windows Anaconda in particular) that import chain is broken
even after reinstalling the packages, due to a `google` namespace-package
conflict between conda- and pip-installed protobuf/google-* packages. The
EventAccumulator API used here reads the same .tfevents files but doesn't
touch the uploader/auth code path at all, so it works regardless of that
conflict. It also means you get a saved .png you can look at without
keeping a local web server running.

Usage:
    python plot_training_curves.py
    python plot_training_curves.py --models-dir models --out training_curves.png
    python plot_training_curves.py --tag eval/mean_reward

Requires: tensorboard (already installed), matplotlib. Does NOT require
google-auth / google-auth-oauthlib to be working.
"""

"""
plot_eval_curves.py

Plots the held-out evaluation reward curve for each trained scenario/seed,
reading it straight from `evaluations.npz` -- the file SB3's `EvalCallback`
writes to `models/<scenario>/seed_<seed>/` via `log_path=out_dir` in
train_rl.py. This file is written with plain numpy, independent of whatever
`tensorboard_log` is set to, so it exists and is readable even if you set
`tensorboard_log=None` in train_rl.py to dodge the TensorBoard/google-auth
issue -- no TensorBoard install or event files needed at all.

This plots `eval/mean_reward`-equivalent data (reward on the held-out
(store, product) pairs, evaluated every `eval_freq` training steps) -- the
more decision-relevant curve for judging convergence, since it's held-out
performance rather than noisy training-rollout reward.

Usage:
    python plot_eval_curves.py
    python plot_eval_curves.py --models-dir models --out eval_curves.png
"""


import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def find_eval_files(models_dir: str) -> dict[str, str]:
    """Returns {run_label: path_to_evaluations.npz}."""
    runs = {}
    pattern = os.path.join(models_dir, "*", "seed_*", "evaluations.npz")
    for path in sorted(glob.glob(pattern)):
        parts = path.replace("\\", "/").split("/")
        scenario, seed_dir = parts[-3], parts[-2]
        runs[f"{scenario}/{seed_dir}"] = path
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out", default="eval_curves.png")
    args = parser.parse_args()

    runs = find_eval_files(args.models_dir)
    if not runs:
        print(f"No evaluations.npz found under {args.models_dir}/*/seed_*/. "
              f"Check that train_rl.py actually ran EvalCallback (it does by default) "
              f"and that training got far enough to log at least one eval_freq checkpoint.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, path in runs.items():
        data = np.load(path)
        timesteps = data["timesteps"]
        results = data["results"]  # shape (n_evals, n_eval_episodes)
        mean_reward = results.mean(axis=1)
        std_reward = results.std(axis=1)
        line, = ax.plot(timesteps, mean_reward, label=label)
        ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward,
                         alpha=0.15, color=line.get_color())

    ax.set_xlabel("training timestep")
    ax.set_ylabel("held-out eval reward (mean +/- std across eval episodes)")
    ax.set_title("PPO held-out evaluation reward by training step")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out} ({len(runs)} runs found)")


if __name__ == "__main__":
    main()