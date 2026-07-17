"""
feature_engineering.py

Reproduces, at inference/simulation time, the same feature set the two LightGBM
models (lgb_sales_model_mean.pkl / lgb_sales_model_q0.9.pkl) were trained on.

The trained feature order (from feature_order.pkl) is:

    Store ID, Product ID, Category, Region, Inventory Level, Units Ordered,
    Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion,
    Competitor Pricing, Seasonality, Year, Month, Day, DayOfWeek,
    Sales_Lag_1, Sales_Lag_7, Sales_RollMean_7, Sales_RollStd_7,
    Sales_RollMean_14, Inventory_Lag_1

Two important design decisions carried through the RL environment (see
README.md "Assumptions" for the full rationale):

1. "Demand Forecast" and "Units Ordered" are treated as *exogenous context*
   taken from the historical record for that Store/Product/Date -- they are
   background signals the original forecasting pipeline had available, not
   quantities the RL agent controls. The RL agent's own order decision is a
   separate action and never overwrites the historical "Units Ordered" used
   here as a model input.
2. "Inventory Level" and the Sales_Lag_* / Inventory_Lag_1 rolling features
   ARE recomputed live from the simulated trajectory (the agent's own past
   decisions), since those are genuinely observable state at decision time
   and should reflect what actually happened in the simulated episode, not
   the original historical trace.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Exact training-time feature order (also loadable from feature_order.pkl,
# kept here as an explicit constant so the RL code does not silently break if
# that file is regenerated in a different order).
FEATURE_ORDER = [
    "Store ID", "Product ID", "Category", "Region", "Inventory Level",
    "Units Ordered", "Demand Forecast", "Price", "Discount",
    "Weather Condition", "Holiday/Promotion", "Competitor Pricing",
    "Seasonality", "Year", "Month", "Day", "DayOfWeek",
    "Sales_Lag_1", "Sales_Lag_7", "Sales_RollMean_7", "Sales_RollStd_7",
    "Sales_RollMean_14", "Inventory_Lag_1",
]

CATEGORICAL_COLUMNS = [
    "Store ID", "Product ID", "Category", "Region",
    "Weather Condition", "Seasonality",
]

UNSEEN_TOKEN = "__unseen__"


def load_feature_order(path: str = "feature_order.pkl") -> list:
    import joblib
    order = joblib.load(path)
    assert order == FEATURE_ORDER, (
        "feature_order.pkl does not match the FEATURE_ORDER constant in "
        "feature_engineering.py -- update the constant or investigate why "
        "the training artifact changed."
    )
    return order


def load_label_encoders(path: str = "label_encoders.pkl") -> dict:
    """label_encoders.pkl was written with joblib (sklearn LabelEncoder
    objects containing NumPy object arrays), so it must be loaded with
    joblib.load rather than plain pickle.load."""
    import joblib
    return joblib.load(path)


def safe_encode(encoder, value) -> int:
    """Encode a single categorical value, falling back to the '__unseen__'
    bucket that was appended to every encoder's classes_ at training time."""
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    if UNSEEN_TOKEN in classes:
        return classes.index(UNSEEN_TOKEN)
    raise ValueError(
        f"Value {value!r} not seen at training time and no '{UNSEEN_TOKEN}' "
        f"fallback exists in this encoder."
    )


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """One-time cleanup of the raw CSV: parse dates, sort, add calendar
    fields. Does not touch lag/rolling columns (those are simulation state,
    built incrementally by the environment)."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    return out


class RollingSalesState:
    """Maintains the Sales_Lag_1/7, Sales_RollMean_7/14, Sales_RollStd_7 and
    Inventory_Lag_1 features incrementally for a single (store, product)
    episode, seeded from real history immediately preceding the episode so
    the first few steps are not cold-started at zero."""

    def __init__(self, seed_sales: list[float], seed_inventory: float):
        # seed_sales: realized Units Sold for the up-to-14 days strictly
        # before the episode start, oldest -> newest.
        self.sales_history: list[float] = list(seed_sales[-14:])
        self.prev_inventory = float(seed_inventory)

    def features(self) -> dict:
        h = self.sales_history
        lag1 = h[-1] if len(h) >= 1 else 0.0
        lag7 = h[-7] if len(h) >= 7 else (h[0] if h else 0.0)
        last7 = h[-7:] if h else [0.0]
        last14 = h[-14:] if h else [0.0]
        return {
            "Sales_Lag_1": lag1,
            "Sales_Lag_7": lag7,
            "Sales_RollMean_7": float(np.mean(last7)),
            "Sales_RollStd_7": float(np.std(last7)) if len(last7) > 1 else 0.0,
            "Sales_RollMean_14": float(np.mean(last14)),
            "Inventory_Lag_1": self.prev_inventory,
        }

    def advance(self, realized_units_sold: float, end_inventory: float) -> None:
        self.sales_history.append(realized_units_sold)
        if len(self.sales_history) > 14:
            self.sales_history = self.sales_history[-14:]
        self.prev_inventory = float(end_inventory)


def build_feature_vector(
    exogenous_row: pd.Series,
    current_inventory: float,
    rolling_state: RollingSalesState,
    encoders: dict,
    feature_order: list = FEATURE_ORDER,
) -> np.ndarray:
    """Assemble a single feature row, in exact training order, as a plain
    numpy array (avoids any DataFrame column-name/underscore mismatch
    between the raw CSV names and LightGBM's internal sanitized names)."""
    rolling = rolling_state.features()
    values = {
        "Store ID": safe_encode(encoders["Store ID"], exogenous_row["Store ID"]),
        "Product ID": safe_encode(encoders["Product ID"], exogenous_row["Product ID"]),
        "Category": safe_encode(encoders["Category"], exogenous_row["Category"]),
        "Region": safe_encode(encoders["Region"], exogenous_row["Region"]),
        "Inventory Level": current_inventory,
        "Units Ordered": exogenous_row["Units Ordered"],  # exogenous context, see module docstring
        "Demand Forecast": exogenous_row["Demand Forecast"],
        "Price": exogenous_row["Price"],
        "Discount": exogenous_row["Discount"],
        "Weather Condition": safe_encode(encoders["Weather Condition"], exogenous_row["Weather Condition"]),
        "Holiday/Promotion": exogenous_row["Holiday/Promotion"],
        "Competitor Pricing": exogenous_row["Competitor Pricing"],
        "Seasonality": safe_encode(encoders["Seasonality"], exogenous_row["Seasonality"]),
        "Year": exogenous_row["Year"],
        "Month": exogenous_row["Month"],
        "Day": exogenous_row["Day"],
        "DayOfWeek": exogenous_row["DayOfWeek"],
        **rolling,
    }
    return np.array([values[f] for f in feature_order], dtype=np.float64).reshape(1, -1)
