"""
evaluate_lightgbm_accuracy.py

Rigorous, full-dataset accuracy check for the two pretrained LightGBM models
(lgb_sales_model_mean.pkl / lgb_sales_model_q0.9.pkl), reusing the exact same
feature pipeline (feature_engineering.py) and the exact same held-out
(store, product) split (train_rl.make_train_eval_pairs, seed=42, 20% holdout)
that evaluate_policies.py uses to score the PPO policies -- so the LightGBM
accuracy number and the RL policy comparison are reported on the *same*
held-out data, not two different ad-hoc splits.

Unlike a "first 50 rows" chart, this scores every row in the held-out set and
reports:
  - Mean model  : R^2, RMSE, MAE            (standard point-forecast metrics)
  - q0.9 model  : coverage rate + pinball loss (the RIGHT metrics for a
                  quantile model -- R^2/RMSE don't apply, since q0.9 is
                  *supposed* to sit above the actual value most of the time,
                  not track it closely)

Also produces two plots:
  1. lgbm_parity_plot.png      -- predicted vs. actual scatter, ALL held-out
                                   rows (not a cherry-picked subset), with a
                                   y=x reference line and R^2 annotated.
  2. lgbm_full_series_plot.png -- actual vs. mean vs. q0.9, for one
                                   representative held-out (store, product)
                                   pair, over its FULL date range (not just
                                   the first 50 rows).

Usage:
    python evaluate_lightgbm_accuracy.py
    python evaluate_lightgbm_accuracy.py --series-store S003 --series-product P0007

Requires: lightgbm, scikit-learn, joblib, pandas, numpy, matplotlib
(the same environment you used to train the models -- this will NOT run in
the Cowork sandbox, which has no internet access to install these packages).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from feature_engineering import (
    FEATURE_ORDER,
    CATEGORICAL_COLUMNS,
    prepare_history,
    load_label_encoders,
    safe_encode,
)
from train_rl import DATA_PATH, make_train_eval_pairs

LAG_WARMUP_DAYS = 14  # matches RollingSalesState's lookback window


# --------------------------------------------------------------------- #
# 1. Load raw data + reproduce the exact held-out split used elsewhere
# --------------------------------------------------------------------- #
def load_and_split():
    raw = pd.read_csv(DATA_PATH)
    history_df = prepare_history(raw)
    train_pairs, eval_pairs = make_train_eval_pairs(history_df)
    print(f"Held-out (store, product) pairs for evaluation: {len(eval_pairs)} "
          f"of {len(train_pairs) + len(eval_pairs)} total "
          f"(same split evaluate_policies.py uses, SPLIT_SEED=42, 20% holdout)")
    return history_df, eval_pairs


# --------------------------------------------------------------------- #
# 2. Vectorized feature construction (bulk, not the day-by-day
#    RollingSalesState used by the RL environment -- but computed from
#    the *real* historical trace, which is exactly equivalent for a
#    pure "how accurate is this trained model on real history" check).
#    Verified numerically identical to RollingSalesState (after dropping
#    the warm-up rows below) via a synthetic side-by-side check.
# --------------------------------------------------------------------- #
def build_bulk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Store ID", "Product ID", "Date"]).copy()
    g = df.groupby(["Store ID", "Product ID"])["Units Sold"]

    # shift(1) then rolling(...) => window covers days t-1 .. t-N, i.e. the
    # N days strictly BEFORE day t. This matches RollingSalesState.features(),
    # which is always read before advance() records day t's own outcome.
    shifted = g.shift(1)
    df["Sales_Lag_1"] = shifted
    df["Sales_Lag_7"] = g.shift(7)
    df["Sales_RollMean_7"] = shifted.groupby([df["Store ID"], df["Product ID"]]).rolling(7).mean().reset_index(drop=True)
    # ddof=0 (population std) to match RollingSalesState's np.std() default --
    # pandas' .std() defaults to ddof=1 (sample std) and will silently give a
    # slightly different number if you don't override it here.
    df["Sales_RollStd_7"] = shifted.groupby([df["Store ID"], df["Product ID"]]).rolling(7).std(ddof=0).reset_index(drop=True)
    df["Sales_RollMean_14"] = shifted.groupby([df["Store ID"], df["Product ID"]]).rolling(14).mean().reset_index(drop=True)
    df["Inventory_Lag_1"] = df.groupby(["Store ID", "Product ID"])["Inventory Level"].shift(1)

    before = len(df)
    df = df.dropna(subset=[
        "Sales_Lag_1", "Sales_Lag_7", "Sales_RollMean_7",
        "Sales_RollStd_7", "Sales_RollMean_14", "Inventory_Lag_1",
    ])
    print(f"Dropped {before - len(df)} warm-up rows (first ~{LAG_WARMUP_DAYS} days "
          f"per store/product, insufficient lag history) -- {len(df)} rows remain.")
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].apply(lambda v, c=col: safe_encode(encoders[c], v))
    return df


# --------------------------------------------------------------------- #
# 3. Predict + score
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-store", default=None,
                         help="Store ID for the full-length illustrative time series plot "
                              "(default: the held-out pair with the most rows).")
    parser.add_argument("--series-product", default=None)
    args = parser.parse_args()

    import joblib
    mean_model = joblib.load("lgb_sales_model_mean.pkl")
    q90_model = joblib.load("lgb_sales_model_q0.9.pkl")
    encoders = load_label_encoders("label_encoders.pkl")

    history_df, eval_pairs = load_and_split()
    eval_set = set(eval_pairs)
    held_out = history_df[
        history_df.apply(lambda r: (r["Store ID"], r["Product ID"]) in eval_set, axis=1)
    ]

    feat_df = build_bulk_features(held_out)
    encoded = encode_categoricals(feat_df, encoders)
    X = encoded[FEATURE_ORDER].to_numpy(dtype=np.float64)
    y_true = feat_df["Units Sold"].to_numpy(dtype=np.float64)

    pred_mean = mean_model.predict(X)
    pred_q90 = q90_model.predict(X)
    pred_mean = np.clip(pred_mean, 0.0, None)
    pred_q90 = np.clip(pred_q90, pred_mean, None)

    # --- Mean model: standard point-forecast metrics, over ALL held-out rows
    r2 = r2_score(y_true, pred_mean)
    rmse = np.sqrt(mean_squared_error(y_true, pred_mean))
    mae = mean_absolute_error(y_true, pred_mean)

    # --- q0.9 model: coverage + pinball loss, NOT R^2/RMSE (see docstring)
    coverage = float((y_true <= pred_q90).mean())

    def pinball_loss(y, pred, q=0.9):
        diff = y - pred
        return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

    pinball = pinball_loss(y_true, pred_q90)

    print("\n" + "=" * 60)
    print(f"Evaluated on {len(y_true)} held-out rows "
          f"({held_out[['Store ID', 'Product ID']].drop_duplicates().shape[0]} store/product series)")
    print("-" * 60)
    print(f"Mean model   -- R^2: {r2:.4f}   RMSE: {rmse:.2f}   MAE: {mae:.2f}")
    print(f"q0.9 model   -- Coverage: {coverage:.1%}  (target ~90%)   "
          f"Pinball loss: {pinball:.2f}")
    print("=" * 60 + "\n")

    pd.DataFrame({
        "metric": ["r2", "rmse", "mae", "q90_coverage", "q90_pinball_loss", "n_rows"],
        "value": [r2, rmse, mae, coverage, pinball, len(y_true)],
    }).to_csv("lightgbm_accuracy_metrics.csv", index=False)
    print("Saved lightgbm_accuracy_metrics.csv")

    # --- Plot 1: parity plot, every held-out row ------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, pred_mean, s=4, alpha=0.25, color="tab:orange", label="Predicted (mean)")
    lims = [0, max(y_true.max(), pred_mean.max()) * 1.02]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="y = x (perfect prediction)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Units Sold")
    ax.set_ylabel("Predicted (mean)")
    ax.set_title(f"Mean Model: Predicted vs. Actual, all {len(y_true)} held-out rows\n"
                 f"R² = {r2:.4f}, RMSE = {rmse:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("lgbm_parity_plot.png", dpi=150)
    print("Saved lgbm_parity_plot.png")

    # --- Plot 2: one full-length series, illustrative --------------------
    if args.series_store and args.series_product:
        series_key = (args.series_store, args.series_product)
    else:
        series_key = (
            feat_df.groupby(["Store ID", "Product ID"]).size().idxmax()
        )
    mask = (feat_df["Store ID"] == series_key[0]) & (feat_df["Product ID"] == series_key[1])
    s_dates = feat_df.loc[mask, "Date"]
    s_actual = y_true[mask.to_numpy()]
    s_mean = pred_mean[mask.to_numpy()]
    s_q90 = pred_q90[mask.to_numpy()]
    order = np.argsort(s_dates.to_numpy())

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(s_dates.to_numpy()[order], s_actual[order], "o-", markersize=3, label="Actual", color="tab:blue")
    ax2.plot(s_dates.to_numpy()[order], s_mean[order], "x--", markersize=4, label="Predicted (mean)", color="tab:orange")
    ax2.plot(s_dates.to_numpy()[order], s_q90[order], "^--", markersize=3, label="Predicted (q0.9)", color="tab:green")
    ax2.set_title(f"Mean vs Quantile Forecast -- {series_key[0]} / {series_key[1]}, "
                  f"full held-out series ({mask.sum()} days)")
    ax2.legend()
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig("lgbm_full_series_plot.png", dpi=150)
    print("Saved lgbm_full_series_plot.png "
          f"(store={series_key[0]}, product={series_key[1]}, {mask.sum()} days -- "
          "pass --series-store/--series-product to pick a different one)")


if __name__ == "__main__":
    main()
