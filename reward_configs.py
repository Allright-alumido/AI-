from __future__ import annotations

"""
reward_configs.py

One agent architecture (same observation/action space, same PPO network),
trained four times under four different reward weightings. All four share
the same underlying economic reward:

    reward = revenue
             - purchase_cost
             - holding_w    * holding_cost
             - stockout_w   * stockout_cost
             - waste_w      * waste_cost
             - deviation_w  * deviation_cost      (scenario 1 only)

where (per step, for one store/product/day):
    unit_cost            = price * (1 - GROSS_MARGIN)
    sale_price           = price * (1 - discount / 100)
    revenue              = units_sold * sale_price
    purchase_cost        = order_qty * unit_cost
    holding_cost         = end_inventory * unit_cost * HOLDING_RATE_PER_DAY
    lost_margin_per_unit = sale_price - unit_cost
    stockout_cost        = stockout_units * lost_margin_per_unit
    waste_cost           = max(end_inventory - WASTE_COVER_DAYS * demand_forecast_mean, 0)
                            * unit_cost * WASTE_RATE
    deviation_cost       = abs(order_qty - historical_units_ordered) * unit_cost * DEVIATION_LAMBDA
"""

from dataclasses import dataclass


# Shared economic constants (identical across all four scenarios -- only the
# *weights* below change per scenario).
GROSS_MARGIN = 0.45            # unit_cost = price * (1 - GROSS_MARGIN)
HOLDING_RATE_PER_DAY = 0.03    # 3% of unit cost, per unit, per day of ending inventory (raised from 1%: at 1% the
                                # holding cost was too cheap relative to the stockout penalty, so profit-seeking
                                # policies were hoarding inventory well beyond the historical operating range)
WASTE_COVER_DAYS = 3.0         # inventory beyond 3x forecast demand is "at risk" stock
WASTE_RATE = 0.02              # 2% of unit cost per excess unit per day
DEVIATION_LAMBDA = 0.5         # scenario 1 only: penalty scale on |order - historical order|


@dataclass(frozen=True)
class RewardConfig:
    name: str
    holding_w: float
    stockout_w: float
    waste_w: float
    deviation_w: float
    description: str


SCENARIOS: dict[str, RewardConfig] = {
    "historical_baseline": RewardConfig(
        name="historical_baseline",
        holding_w=1.0,
        stockout_w=1.0,
        waste_w=1.0,
        deviation_w=1.0,
        description=(
            "Conservative mode: full economic reward plus a penalty for "
            "deviating from the historical Units Ordered decision."
        ),
    ),
    "zero_stockout": RewardConfig(
        name="zero_stockout",
        holding_w=1.0,
        stockout_w=8.0,
        waste_w=1.0,
        deviation_w=0.0,
        description="Stockout penalty weight increased 8x; no deviation penalty.",
    ),
    "high_holding_stockout_tolerant": RewardConfig(
        name="high_holding_stockout_tolerant",
        holding_w=4.0,
        stockout_w=0.25,
        waste_w=2.0,
        deviation_w=0.0,
        description=(
            "Holding/waste cost weight increased, stockout penalty reduced "
            "to a quarter of its base weight."
        ),
    ),
    "pure_profit": RewardConfig(
        name="pure_profit",
        holding_w=1.0,
        stockout_w=1.0,
        waste_w=1.0,
        deviation_w=0.0,
        description=(
            "Raw economic reward -- actual revenue and costs, no scenario-"
            "specific emphasis on any single objective."
        ),
    ),
}
