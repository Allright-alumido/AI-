# Replenishment RL: one agent, four reward scenarios

This trains a PPO agent to choose daily order quantities for the retail
replenishment problem, using your two LightGBM models
(`lgb_sales_model_mean.pkl`, `lgb_sales_model_q0.9.pkl`) as demand-forecast
features the agent observes each day. The same agent architecture (network,
hyperparameters, observation/action space) is trained four separate times,
once per reward configuration, so you get four policies to compare.

## Files

| File | Purpose |
|---|---|
| `feature_engineering.py` | Rebuilds the exact feature set the LightGBM models were trained on (lags, rolling stats, calendar fields, label encoding with an `__unseen__` fallback). |
| `demand_model.py` | Loads the two LightGBM models and exposes `predict(...) -> (mean, q90)`. |
| `reward_configs.py` | The shared economic reward formula plus the four scenarios' weights. |
| `replenishment_env.py` | The `gymnasium.Env` simulation itself. |
| `train_rl.py` | Trains PPO for one or all four scenarios; saves models + VecNormalize stats under `models/<scenario>/seed_<seed>/`. |
| `evaluate_policies.py` | Runs all four trained policies plus a historical-replay baseline on held-out data; prints/saves a comparison table and plot. |
| `plot_eval_curves.py` | Plots held-out eval-reward convergence curves from `evaluations.npz` (works even if `tensorboard_log=None`). |
| `plot_training_curves.py` | Same idea but reads real TensorBoard event files directly, bypassing the `tensorboard` CLI (useful if it's broken by the `google-auth` conda conflict). |
| `inference.py` | `Recommender` class: loads one trained scenario once, then answers "what should I order right now" for a real historical situation, a manually-specified stress-test situation, a side-by-side comparison against historical decisions, or a forward-looking multi-day plan (`plan_ahead`). This is the layer the UI calls. |
| `explain.py` | Turns a recommendation into a plain-language explanation -- tries an LLM (Anthropic/Gemini/OpenAI, whichever has an API key set) and falls back to a template if none is configured. |
| `priorities.py` | Scans every (store, product) pair for a strategy and triages each one Red/Yellow/Green ("Today's Priority") -- see "Running the UI" below. |
| `app.py` | The Streamlit UI (see "Running the UI" below). |
| `requirements.txt` | Pinned dependency floor. |
| `test_env_smoke.py` / `test_inference_smoke.py` / `test_app_smoke.py` / `test_priorities_smoke.py` / `test_plan_ahead_responsiveness.py` / `test_plan_ahead_extrapolation.py` | Mock-based tests validating logic without needing lightgbm/gymnasium/stable-baselines3/streamlit installed -- see "Validation note" below. |

## Setup

```bash
pip install -r requirements.txt
# place retail_store_inventory.csv, lgb_sales_model_mean.pkl,
# lgb_sales_model_q0.9.pkl, feature_order.pkl, label_encoders.pkl
# in the same directory as these scripts.
```

## Run

```bash
# train all four scenarios, one seed (defaults to 400k timesteps each; adjust as needed)
python train_rl.py --scenario all --timesteps 400000 --seed 0

# recommended: rerun with 2-3 seeds per scenario to separate real effects from
# PPO training noise before drawing conclusions from the comparison table
python train_rl.py --scenario all --timesteps 400000 --seed 1
python train_rl.py --scenario all --timesteps 400000 --seed 2

# or one scenario at a time
python train_rl.py --scenario zero_stockout --timesteps 400000 --seed 0

# compare all trained policies against the historical replay baseline
# (single seed)
python evaluate_policies.py --episodes-per-pair 3 --seed 0
# averaged across seeds, with std shown per KPI
python evaluate_policies.py --episodes-per-pair 3 --seed 0,1,2
```

Note: `GROSS_MARGIN`, `HOLDING_RATE_PER_DAY`, `WASTE_COVER_DAYS`, `WASTE_RATE`,
and `MAX_INVENTORY` are shared across all four scenarios -- if you change any
of them, retrain **all four**, not just the ones that looked off in a
previous run, or the comparison table stops being apples-to-apples.

Training logs go to `models/<scenario>/seed_<seed>/tb/` (viewable with
`tensorboard --logdir models`). Evaluation produces `policy_comparison.csv`
and `policy_comparison.png`.

## The four scenarios

All four share the identical economic reward:

```
reward = revenue
         - purchase_cost
         - holding_w    * holding_cost
         - stockout_w   * stockout_cost
         - waste_w      * waste_cost
         - deviation_w  * deviation_cost   (only scenario 1 uses this term)
```

| Scenario | holding_w | stockout_w | waste_w | deviation_w | Intent |
|---|---|---|---|---|---|
| `historical_baseline` | 1.0 | 1.0 | 1.0 | 1.0 | Conservative: penalizes deviating from the historical `Units Ordered` decision. |
| `zero_stockout` | 1.0 | 8.0 | 1.0 | 0 | Stockout penalty weight increased 8x. |
| `high_holding_stockout_tolerant` | 4.0 | 0.25 | 2.0 | 0 | Holding/waste cost weight increased, stockout penalty cut to a quarter. |
| `pure_profit` | 1.0 | 1.0 | 1.0 | 0 | Raw economic reward, no scenario-specific tilt. |

Edit the constants at the top of `reward_configs.py` (`GROSS_MARGIN`,
`HOLDING_RATE_PER_DAY`, `WASTE_COVER_DAYS`, `WASTE_RATE`,
`DEVIATION_LAMBDA`) to change the shared economics, or the per-scenario
weights in `SCENARIOS` to retune emphasis.

## Running the UI

Once at least one scenario has a trained model under `models/<scenario>/seed_0/`
(seed 0 is the default the UI loads -- pass a different seed by editing
`DEFAULT_SEED` in `app.py` if you want a different one), start the app from a
terminal:

```bash
streamlit run app.py
```

This opens in your browser (Streamlit is not run inside a Jupyter cell).
It has four tabs:

- **Recommendation** -- opens with **Today's Priority**: every (store,
  product) pair for the selected strategy, scanned once and triaged:
  - 🔴 **High** -- either, assuming today plays out as expected, this
    policy's own forward plan (`plan_ahead`, "mean" demand scenario) still
    shows a stockout risk tomorrow if demand spikes to the high (P90)
    case, or a promotion/holiday starts tomorrow.
  - 🟡 **Medium** -- today's safety buffer is *thin* relative to P90
    demand (under `SAFETY_BUFFER_RATIO_THRESHOLD`, default 20% -- not
    just an outright deficit). A live, present-day cushion concern,
    distinct from the forward-looking High check above -- see
    `priorities.py`'s module docstring for why these two checks
    deliberately use different demand assumptions, and for why a ratio
    (not a strict "buffer < 0") is used: the strict version turned out to
    be nearly unreachable in practice, so almost everything landed in
    High or Low with nothing in between.
  - 🟢 **Low** -- none of the above; shown collapsed in an expander so the
    table isn't dominated by "nothing to see here" rows.

  Each row has a "View" button; clicking it auto-fills the Store/Product/
  Date fields below with that pair and immediately shows its
  recommendation, so you don't have to hunt for it manually. This section
  is powered by `priorities.compute_todays_priorities()`, cached per
  (strategy, calendar day) so the full network scan only re-runs once a
  day, not on every click. See `priorities.py`'s module docstring for why
  "today" is the second-to-last date in the CSV rather than the literal
  last one (the file has no real "tomorrow" row to check promo/holiday
  context against otherwise, since it's a static historical snapshot, not
  a live feed).

  Then: pick a strategy from the sidebar (shown in plain language: "Never
  stock out", "Minimize storage cost", etc., mapped internally to the four
  scenarios), pick a store/product/date, and get the recommended order
  quantity plus a plain-language explanation of why.
- **Stress test** -- override the situation directly (current inventory,
  expected demand, price, discount, holiday flag) with sliders and watch the
  recommendation react live. This bypasses the LightGBM forecast call on
  purpose -- you're testing how the *strategy* responds to a scenario you
  specify, not asking the demand model to predict anything.
- **Decision vs. history** -- pick a store/product and a date window, and
  see a side-by-side of what the selected strategy would have ordered versus
  what was actually ordered historically, plus the estimated additional
  profit the strategy would have generated over that window.
- **Future plan** -- pick a store/product, a start date ("today"), and how
  many days ahead to simulate (3-60), and get a forward-looking, day-by-day
  order plan. Unlike the other tabs, this does not use real historical
  demand at all -- since actual future demand isn't known yet, it
  recursively advances a simulated inventory trajectory using the LightGBM
  forecast itself as the assumed demand each day (`Recommender.plan_ahead`
  in `inference.py`). It runs the simulation twice -- once assuming average
  ("mean") demand shows up each day, once assuming the 90th-percentile
  ("p90", high-demand) forecast shows up instead -- and plots both as an
  ending-inventory band, so you can see how quickly uncertainty compounds
  the further out you look. It also flags any day where the high-demand
  case would risk a stockout. Price/discount/holiday/weather/season for
  each simulated day are pulled from the real historical record for that
  date where one exists, since in practice these are usually planned
  ahead of time even though demand isn't.

  **Extrapolating past the end of the data.** If the requested horizon
  runs past the last real date this store/product has data for, the plan
  no longer stops there -- it keeps going by *assuming* price/discount/
  promo/weather for those days instead of reading them from a real row
  (there's no such row to read). An "Assumptions for dates beyond the
  data" expander lets you set Discount/Price/Competitor pricing/Holiday-
  promotion for those extrapolated days (defaults to whatever was last in
  effect for that store/product); everything up to the real data's end
  still uses actual historical context, untouched by these inputs. The
  day-by-day table's **Data source** column marks each row "Historical" or
  "Extrapolated (assumed)" so it's always clear which is which, and the
  plan's warning banner reports the real/extrapolated day counts. Two
  inputs the LightGBM demand model itself needs as exogenous context --
  the historical "Units Ordered" and "Demand Forecast" columns -- have no
  real-world analogue for a genuinely future date; rather than freeze
  these at their last real value, each extrapolated day bootstraps them
  from the *previous simulated day's own output* (this policy's own
  order, and its own mean forecast), so they keep evolving instead of
  going stale. All of this is implemented in
  `Recommender.plan_ahead`/`future_assumptions` in `inference.py` -- see
  its docstring for the full rationale, and `test_plan_ahead_extrapolation.py`
  for the tests proving the horizon is never truncated, the extrapolated
  dates are gap-free, assumption overrides actually reach the model, and
  the bootstrap actually evolves day to day rather than freezing.

  Treat extrapolated days with more caution than real ones: they're a
  projection built on your stated assumptions, not a forecast grounded in
  what's actually planned for that date.

  **If the recommended order looks identical for several days in a row,
  that is very likely not a bug.** A well-trained inventory policy
  naturally settles into a steady reorder-to-target pattern once the
  simulated trajectory stabilizes -- `test_plan_ahead_responsiveness.py`
  demonstrates this concretely: even a synthetic policy that reacts
  directly to the day's own demand forecast converges to a near-constant
  order after the first few days. Use the **Starting inventory** column
  in the day-by-day table to tell a real freeze apart from expected
  convergence: if inventory is also flat and pinned at 0 or at
  `MAX_INVENTORY`, the order is being floor/capacity-limited every day
  (worth investigating via the Stress test tab); if inventory is moving
  but the order has converged, that's the policy working as intended.

  **If you can't seem to pick a different Store/Product/Start date and it
  keeps reverting -- it should now just work.** The Store/Product/Date
  selectboxes (in every tab, not just this one) used to use both `key=`
  and `index=` together, and once Streamlit had a stored value for a
  widget's `key`, it kept using that value on every rerun and ignored
  `index` -- so if changing the Store made the previously-selected
  Product invalid, the widget had nothing valid to fall back to and
  appeared to "snap back." This is fixed by `cascading_selectbox()` in
  `app.py`, which validates/repairs the stored selection *before*
  creating the widget instead of leaning on Streamlit's own key/index
  precedence; see its docstring and the regression test in
  `test_app_smoke.py`. (Separately: since the Start date field defaults
  to the *last* available date for a pair, being stuck there used to also
  mean every plan request hit the end of the data and truncated hard --
  that specific symptom no longer applies at all now that the plan
  extrapolates past the end of the data instead of stopping there, see
  above.)

The "based on backtesting" line under the strategy picker pulls from
`policy_comparison.csv` if it exists (run `evaluate_policies.py` first to
generate it) -- without it, the picker still works, just without the
aggregate stats.

**On AI-generated explanations:** `explain.py` checks for `ANTHROPIC_API_KEY`,
then `GOOGLE_API_KEY`/`GEMINI_API_KEY`, then `OPENAI_API_KEY` in your
environment, and uses whichever is set (install the matching package from
the commented-out lines in `requirements.txt`). If none are set, or the
package isn't installed, or the call fails for any reason, it silently falls
back to a template explanation built from the actual numbers -- so this
works whether or not you end up with LLM API access.

The Recommendation tab also includes: a numeric decision breakdown (Demand
P90 / Current Stock / Safety Buffer / Expected Ending Inventory -- a
business-friendly approximation of the reasoning, not a literal trace of
the neural network); a Recommendation Confidence badge + progress bar
based on how wide the mean-to-P90 demand gap is; a Policy Comparison
section showing what all four strategies would recommend for the same
store/product/date (loads whichever of the four are trained, shows "Not
trained yet" for the rest) with bar charts of order quantity and, if
`policy_comparison.csv` exists, backtested service level/inventory/profit;
and a "What changed?" note that appears when the order quantity jumps more
than 20% versus the previous day, listing which of demand/inventory/
promotion/holiday likely drove it.

## Assumptions (please review before trusting the results)

The CSV (`retail_store_inventory.csv`, 5 stores x 20 products x 731 days)
does not include an explicit unit cost, holding-cost rate, or stockout
penalty, so these had to be assumed:

- **Gross margin 45%**: `unit_cost = Price * (1 - 0.45)`.
- **Holding cost**: 3% of unit cost per unit of ending inventory per day
  (raised from an initial 1% -- at 1% it was too cheap relative to the
  stockout penalty, and profit-seeking policies hoarded inventory well
  beyond the historical range; see the capacity cap below too).
- **Warehouse capacity cap (`MAX_INVENTORY = 500`)**: orders that would push
  available stock above 500 units (matching the historical max Inventory
  Level) are truncated at the door -- rejected, not purchased -- rather than
  accepted and left to become waste. Without this, nothing physically
  stopped an agent from accumulating stock indefinitely across an episode.
- **Waste/obsolescence cost**: ending inventory beyond 3x the forecast
  demand is charged an extra 2%-of-unit-cost/day penalty (there's no expiry
  data in the CSV, so this is a proxy for overstock risk, not real spoilage).
- **Lead time = 0**: an order placed "today" is assumed to arrive before
  today's demand is realized. If your real process has a lead time, add a
  pending-orders queue to `ReplenishmentEnv.step()`.
- **Demand ground truth**: the environment uses the historical `Units Sold`
  value as the "true" demand draw for each day. This is a common
  simplification, but note `Units Sold` in the source data is itself already
  censored by whatever inventory/ordering happened historically -- it's a
  proxy for uncensored demand, not a perfect ground truth.
- **`Demand Forecast` and `Units Ordered` as LightGBM inputs**: the models
  were trained with these as input features. At the moment the agent needs
  today's forecast (before it has chosen today's order), the RL agent's own
  action can't be used as a model input without a circularity problem, so
  the environment feeds the model the *historical* `Units Ordered`/`Demand
  Forecast` values for that store/product/date as exogenous context. The
  agent's actual chosen order quantity drives the simulated inventory
  dynamics and reward, independently of what's fed to the forecast model.
- **Action space**: discrete order quantities from 0 to 500 in steps of 10
  (51 actions), matching the CSV's observed Inventory Level range.
- **Episodes**: 90 consecutive days for one randomly sampled (store,
  product) pair; 20% of the 100 (store, product) pairs are held out
  entirely for evaluation and never seen during training.

## Validation note

This sandbox has no internet access, so `lightgbm`, `gymnasium`,
`stable-baselines3`, and `torch` could not be installed or exercised here.
The code above is written to run in a normal Python environment that has
these packages (the same one you used to train the LightGBM models). Before
full-scale training, sanity-check the pipeline with a short run, e.g.:

```bash
python train_rl.py --scenario pure_profit --timesteps 5000 --seed 0
```

and confirm `models/pure_profit/seed_0/ppo_model.zip` is produced and
`evaluate_policies.py --seed 0` runs without error, before committing to the
full 400k-timestep runs across all four scenarios.

## Sanity-checking `historical_replay` against the raw CSV

`policy_comparison.csv`'s `historical_replay` row is produced by replaying
the CSV's own `Units Ordered`/`Units Sold` through the environment's reward
formula. As an independent check that this number is right, compute profit
directly from the CSV without touching any of this code:

```python
import pandas as pd
from reward_configs import GROSS_MARGIN, HOLDING_RATE_PER_DAY, WASTE_COVER_DAYS, WASTE_RATE

df = pd.read_csv("retail_store_inventory.csv")
unit_cost = df["Price"] * (1 - GROSS_MARGIN)
sale_price = df["Price"] * (1 - df["Discount"] / 100)
revenue = df["Units Sold"] * sale_price
purchase_cost = df["Units Ordered"] * unit_cost
holding_cost = df["Inventory Level"] * unit_cost * HOLDING_RATE_PER_DAY
lost_margin = (sale_price - unit_cost).clip(lower=0)
# CSV-implied stockout: demand exceeded available stock that day
stockout_units = (df["Demand Forecast"] - df["Inventory Level"]).clip(lower=0)
stockout_cost = stockout_units * lost_margin
print((revenue - purchase_cost - holding_cost - stockout_cost).sum())
```

This won't match `historical_replay` exactly -- the env only evaluates a
90-day window on the 20 held-out (store, product) pairs, while this snippet
sums over the full 2-year, 100-pair dataset, and it uses `Inventory Level`
directly rather than the env's simulated `end_inventory` -- but the
per-unit economics and the overall order of magnitude should line up. If
they're wildly different (say, off by 10x or the sign flips), that points
to a bug rather than a scope difference.

## Live Stream Monitor (Tab 5) -- simulated real-time architecture

A Lambda Architecture demo, layered on top of everything above without
retraining anything: a **Batch Layer** (unchanged -- `train_rl.py` +
LightGBM training produce the models everything else loads) and a **Speed
Layer** (new) that simulates a retailer's daily settlement arriving on a
compressed timeline and reacts to it with the already-trained models.

- `streaming_data_generator.py` -- **Ingestion**. `BlockBootstrapGenerator`
  generates one new simulated settlement day per (store, product) pair per
  tick, by resampling a random contiguous 7-day block from that pair's own
  real history (preserving weekly seasonality) plus 5% multiplicative
  noise. TimeGAN and LLM-based generation were deliberately not used --
  see the module docstring for why. Identity fields (Store/Product/
  Category/Region) are never resampled; calendar fields always come from
  the true simulated date.
- `streaming_pipeline.py` -- **Processing + Decision + Serving**.
  `PairState` persists each pair's rolling sales/inventory/exogenous-carry
  state between ticks. `StreamingPipeline.tick()` ingests one new day per
  pair, calls `inference.Recommender.recommend_for_row()` (the same
  tested inference code the other tabs use -- see below) to get a
  decision, settles the day's economics with the same formulas as
  `ReplenishmentEnv.step()`, and writes the result to
  `stream_state/latest_snapshot.json` + `stream_state/stream_log.csv`.
  Run standalone: `python streaming_pipeline.py --scenario pure_profit
  --ticks 5` (add `--compare-all` to also load the other 3 strategies for
  side-by-side comparison, same spirit as the Recommendation tab's Policy
  Comparison panel).
- `inference.py` gained one new public method, `Recommender
  .recommend_for_row(row, current_inventory, rolling_state, context)` --
  the shared core `recommend_from_history()` already used internally,
  now exposed so the streaming pipeline (which has its own persisted
  per-pair state, not a row sitting in `history_df`) can reuse it too
  instead of a third copy of the same logic.
- `proposal_assets/make_lambda_architecture.py` generates
  `lambda_architecture.png`, diagramming Batch vs. Speed layer and the
  Ingestion -> Processing -> Decision -> Serving pipeline, with an
  explicit note that this is a demo simulation (no Kafka/Flink/Spark),
  not a production streaming deployment.
- **Tab 5 (`app.py`)** wires this into the UI: auto-play (via the
  optional `streamlit-autorefresh` package, graceful fallback to
  manual-step-only if it isn't installed) is the primary interaction
  mode, "Advance 1 hour" is always available as a manual complement, and
  a **smart pause** automatically stops auto-play once at least
  `STOCKOUT_ALERT_THRESHOLD` (5%) of tracked pairs go into stockout in the
  same tick (so a live demo never scrolls past a genuinely significant
  moment unattended) -- see `tab_live_stream()` and the
  `_check_stream_alerts()` helper it calls. Gated on a percentage, not a
  raw count: with dozens/hundreds of pairs running simultaneously, at
  least one stocking out on any given tick is common and not, by itself,
  noteworthy -- pausing on that alone would make auto-play pause almost
  every tick and be unusable for a demo. Tunable via the
  `STOCKOUT_ALERT_THRESHOLD` module constant.

Tested by `test_streaming_smoke.py` (generator schema/anchoring, tick-by-
tick inventory/state bookkeeping, JSON/CSV serving output, compare-all
mode) and the Live Stream-specific assertions added to
`test_app_smoke.py` (`_init_stream_pipeline` graceful degradation when a
scenario isn't trained yet, `_check_stream_alerts` pause logic,
`_stream_tick_and_check_alerts` updating session state).

## Case Memory layer -- bounded, evidence-gated case-based decision support

Layered on top of the Live Stream Monitor's frozen PPO policy: a lightweight
episodic memory ("Case Memory") that can nudge the primary strategy's order
quantity based on similar past states for the SAME (store, product) pair,
and that keeps learning online through simple retrieval -- **without ever
retraining the PPO network**. This distinction matters and is stated
explicitly in `memory_store.py`'s module docstring: this is case-based
reasoning with online feedback, not online reinforcement learning.

- `memory_store.py` -- `CaseMemory`:
  - `log_episode(pair_key, features, action_idx, reward, tick, date)` --
    structured state logging. `features` is the exact observation vector
    PPO's network was shown (obtained via
    `Recommender.recommend_for_row(..., return_obs=True)`), so "similar"
    means what the policy itself would consider similar.
  - `_retrieve_similar(pair_key, features, k=5)` -- dynamic state
    reconstruction: k-NN by Euclidean distance, scoped to the SAME pair's
    own history only (a Groceries item in one store isn't a meaningful
    "similar case" for Electronics in another).
  - `reflect(pair_key, features, ppo_action_idx)` -- pure lookup +
    arithmetic (never mutates memory). Requires at least `MIN_NEIGHBORS`
    (3) similar past cases; compares the best-performing differing-action
    neighbor's reward against the retrieved neighbors' average; only
    proposes a nudge if that margin clears `MIN_MARGIN` (5%). The nudge is
    capped at `NUDGE_CAP_STEPS` (1 action step = 10 units) regardless of
    how far away the better-performing historical action actually was, and
    never pushes the action below 0. Returns a plain-language `explanation`
    string for the UI.
  - Closed loop: whatever action actually executes each tick (nudged or
    not) plus its resulting profit gets logged back via `log_episode()`,
    so later ticks for that same pair can retrieve it -- the memory keeps
    growing and refining its own suggestions run after run.
  - Documented limitation: reward is realized against one specific
    historical/simulated demand draw, so comparing rewards across
    different episodes at "similar" states compares across different
    demand realizations too -- this is a heuristic, approximate signal
    appropriate for a bounded advisory nudge, not a rigorous causal
    estimate of the truly optimal action.

- `streaming_pipeline.py` integration: `StreamingPipeline(..., use_case_memory=True)`
  builds one shared `CaseMemory` instance. Each `tick()`, after the
  primary scenario's `Recommender.recommend_for_row(..., return_obs=True)`
  call, `memory.reflect()` may adjust the requested order quantity (with
  the warehouse capacity cap re-applied afterward), then
  `memory.log_episode()` records what actually executed. The per-pair
  result dict gains a `"case_memory"` sub-dict (`nudged`, `final_action_idx`,
  `nudge_steps`, `explanation`, `n_neighbors`). `--no-case-memory` on the CLI
  (or `use_case_memory=False` programmatically) disables nudging entirely --
  episodes still get logged (harmless bookkeeping) but `reflect()` is
  never consulted, so the primary scenario's action is always PPO's own.

- **Tab 5 (`app.py`)** UI: a "🧠 Enable Case Memory" checkbox next to
  the existing "compare all strategies" checkbox (default on, toggling it
  rebuilds the pipeline via the same `needs_init` mechanism used for
  scenario/compare-all changes). Below the results table, a "🧠 Case Memory
  memory" expander shows this tick's nudge explanation when one applied,
  or a status note (memory size, why no nudge happened) when Case Memory is
  on but didn't act, or a plain note when Case Memory is off.

Tested by `test_memory_smoke.py` (pure `CaseMemory` unit tests, no
Streamlit/gymnasium/stable-baselines3 stubs needed: insufficient-neighbors
gate, all-neighbors-agree no-op, weak-margin no-op, a clear-evidence nudge
correctly capped even when the better historical action was 5 steps away,
nudge never goes negative, retrieval is correctly scoped per pair with no
cross-pair leakage, and the closed-loop property that a newly-logged
episode becomes retrievable evidence for a later tick) plus integration
assertions added to `test_streaming_smoke.py`
(`test_pipeline_case_memory_wiring`: every tick's result carries a
well-formed `"case_memory"` sub-dict, memory grows by exactly one episode per
pair per tick, and `use_case_memory=False` always reports the untouched
default) and `test_app_smoke.py` (`_init_stream_pipeline` correctly passes
`use_case_memory` through to the constructed pipeline).
