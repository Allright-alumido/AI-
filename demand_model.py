"""
demand_model.py

Thin wrapper around the two pre-trained LightGBM models:
  - lgb_sales_model_mean.pkl : point forecast (mean) of Units Sold
  - lgb_sales_model_q0.9.pkl : 90th-percentile forecast of Units Sold

These are used purely as *observation features* for the RL agent -- at
decision time the agent sees "what a demand model thinks will happen today",
not the true realized demand (which the environment reveals only after the
order action is taken). This is what actually makes the LightGBM models
relevant to the RL problem instead of redundant with it.

Requires: lightgbm, scikit-learn, joblib (the environment these models were
trained in). Not runnable in a sandbox without internet access to install
those packages -- see README.md for validation notes.
"""

from __future__ import annotations

import numpy as np
import joblib

from feature_engineering import (
    FEATURE_ORDER,
    RollingSalesState,
    build_feature_vector,
    load_label_encoders,
)


class DemandModel:
    def __init__(
        self,
        mean_model_path: str = "lgb_sales_model_mean.pkl",
        q90_model_path: str = "lgb_sales_model_q0.9.pkl",
        label_encoders_path: str = "label_encoders.pkl",
    ):
        self.mean_model = joblib.load(mean_model_path)
        self.q90_model = joblib.load(q90_model_path)
        self.encoders = load_label_encoders(label_encoders_path)

    def predict(
        self,
        exogenous_row,
        current_inventory: float,
        rolling_state: RollingSalesState,
    ) -> tuple[float, float]:
        """Returns (mean_forecast, q90_forecast) for one store/product/day."""
        x = build_feature_vector(
            exogenous_row, current_inventory, rolling_state, self.encoders, FEATURE_ORDER
        )
        mean_forecast = float(self.mean_model.predict(x)[0])
        q90_forecast = float(self.q90_model.predict(x)[0])
        # Demand forecasts cannot be negative and q90 should not fall below
        # the mean; both are pathological-but-possible with GBM extrapolation.
        mean_forecast = max(mean_forecast, 0.0)
        q90_forecast = max(q90_forecast, mean_forecast)
        return mean_forecast, q90_forecast
