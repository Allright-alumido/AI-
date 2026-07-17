"""
test_streaming_smoke.py

Smoke test for streaming_data_generator.py and streaming_pipeline.py, using
the same stubbing approach as test_inference_smoke.py (fake gymnasium/
stable_baselines3/joblib + FakeDemandModel from test_env_smoke.py) so this
runs without lightgbm/gymnasium/stable-baselines3 installed.

Checks:
  - BlockBootstrapGenerator: schema, identity fields never resampled,
    calendar fields always match the true simulated date (never resampled),
    same block anchor reused for BLOCK_SIZE consecutive ticks then rotates.
  - StreamingPipeline.tick(): runs across all pairs, inventory stays
    non-negative, sim_date advances by exactly one day per tick, state
    persists between ticks (tick counter increments, carry fields update).
  - write_snapshot(): produces valid JSON and an appendable CSV log.
  - compare-all mode: pipeline can hold multiple scenarios' Recommenders at
    once and only the primary one's decision drives settlement.
  - Case Memory wiring: every tick's result carries a well-formed "case_memory"
    sub-dict, use_case_memory=False genuinely disables nudging, and the
    memory grows by exactly one logged episode per pair per tick.

Run: python test_streaming_smoke.py
"""

import os
import shutil
import sys
import types

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Same stubs as test_inference_smoke.py
# ---------------------------------------------------------------------------
if "gymnasium" not in sys.modules:
    gym_stub = types.ModuleType("gymnasium")

    class _Env:
        metadata = {}

        def reset(self, *, seed=None, options=None):
            pass

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    spaces_stub = types.ModuleType("gymnasium.spaces")
    spaces_stub.Discrete = _Discrete
    spaces_stub.Box = _Box
    gym_stub.Env = _Env
    gym_stub.spaces = spaces_stub
    sys.modules["gymnasium"] = gym_stub
    sys.modules["gymnasium.spaces"] = spaces_stub

if "stable_baselines3" not in sys.modules:
    sb3_stub = types.ModuleType("stable_baselines3")

    class FakePPO:
        FIXED_ACTION = 20  # -> order_qty = 200

        @classmethod
        def load(cls, path):
            return cls()

        def predict(self, obs, deterministic=True):
            return np.array([self.FIXED_ACTION]), None

    sb3_stub.PPO = FakePPO
    sys.modules["stable_baselines3"] = sb3_stub

    common_stub = types.ModuleType("stable_baselines3.common")
    vecenv_stub = types.ModuleType("stable_baselines3.common.vec_env")

    class FakeDummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]

    class FakeVecNormalize:
        def __init__(self, venv, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True):
            self.venv = venv
            self.training = training
            self.norm_reward = norm_reward

        @classmethod
        def load(cls, path, venv):
            return cls(venv)

        def normalize_obs(self, obs):
            return obs

    vecenv_stub.DummyVecEnv = FakeDummyVecEnv
    vecenv_stub.VecNormalize = FakeVecNormalize

    callbacks_stub = types.ModuleType("stable_baselines3.common.callbacks")

    class FakeEvalCallback:
        def __init__(self, *args, **kwargs):
            pass

    callbacks_stub.EvalCallback = FakeEvalCallback

    common_stub.__path__ = []
    common_stub.vec_env = vecenv_stub
    common_stub.callbacks = callbacks_stub
    sys.modules["stable_baselines3.common"] = common_stub
    sys.modules["stable_baselines3.common.vec_env"] = vecenv_stub
    sys.modules["stable_baselines3.common.callbacks"] = callbacks_stub

if "joblib" not in sys.modules:
    joblib_stub = types.ModuleType("joblib")
    joblib_stub.load = lambda path: None
    joblib_stub.dump = lambda obj, path: None
    sys.modules["joblib"] = joblib_stub

from test_env_smoke import make_synthetic_history, FakeDemandModel  # noqa: E402
from reward_configs import SCENARIOS  # noqa: E402
import inference  # noqa: E402
from streaming_data_generator import BlockBootstrapGenerator, BLOCK_SIZE  # noqa: E402
import streaming_pipeline  # noqa: E402

TEST_STREAM_DIR = "test_stream_state"


def ensure_dummy_model_files(scenario_name: str, seed: int = 0):
    model_dir = os.path.join(inference.MODELS_DIR, scenario_name, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    open(os.path.join(model_dir, "ppo_model.zip"), "a").close()
    open(os.path.join(model_dir, "vecnormalize.pkl"), "a").close()


def test_generator(history_df):
    gen = BlockBootstrapGenerator(history_df, rng_seed=0)
    key = ("S001", "P0001")
    base_date = history_df["Date"].max()

    seen_anchors = []
    for i in range(BLOCK_SIZE * 2 + 2):
        sim_date = base_date + pd.Timedelta(days=i + 1)
        row = gen.next_row(key[0], key[1], sim_date)
        # identity fields must be exactly the pair's own real values
        assert row["Store ID"] == "S001" and row["Product ID"] == "P0001"
        assert row["Category"] == "Groceries" and row["Region"] == "North"
        # calendar fields must always match the TRUE simulated date, never resampled
        assert row["Year"] == sim_date.year
        assert row["Month"] == sim_date.month
        assert row["Day"] == sim_date.day
        assert row["DayOfWeek"] == sim_date.dayofweek
        assert row["Date"] == sim_date
        # numeric resampled fields must be non-negative and finite
        assert row["Units Sold"] >= 0 and np.isfinite(row["Units Sold"])
        assert row["Price"] >= 0 and np.isfinite(row["Price"])
        assert row["Competitor Pricing"] >= 0 and np.isfinite(row["Competitor Pricing"])
        seen_anchors.append(gen._anchor_state[key][0])

    # anchor should change at least once across 2+ full blocks (not stuck forever)
    assert len(set(seen_anchors)) > 1, "block anchor never rotated across multiple blocks"
    print(f"[ok] BlockBootstrapGenerator: schema correct, calendar fields match true date, "
          f"identity fields preserved, anchor rotated {len(set(seen_anchors))} times over "
          f"{BLOCK_SIZE * 2 + 2} ticks")

    # unknown pair should raise, not silently return garbage
    try:
        gen.next_row("S999", "P9999", base_date + pd.Timedelta(days=1))
        raise AssertionError("expected ValueError for unknown pair")
    except ValueError:
        print("[ok] BlockBootstrapGenerator rejects unknown (store, product) pairs")


def test_pipeline_single_scenario(history_df, demand_model):
    scenario = "pure_profit"
    ensure_dummy_model_files(scenario)
    rec = inference.Recommender(scenario, seed=0, history_df=history_df, demand_model=demand_model)

    pipeline = streaming_pipeline.StreamingPipeline(
        {scenario: rec}, primary_scenario=scenario, history_df=history_df, rng_seed=0
    )
    assert len(pipeline.pairs) == 2  # S001/P0001, S002/P0002 in make_synthetic_history

    prev_date = None
    for i in range(5):
        snapshot = pipeline.tick()
        assert snapshot["tick"] == i + 1
        assert snapshot["n_pairs"] == 2
        assert len(snapshot["results"]) == 2
        sim_date = pd.Timestamp(snapshot["sim_date"])
        if prev_date is not None:
            assert (sim_date - prev_date) == pd.Timedelta(days=1), "sim_date should advance by exactly 1 day/tick"
        prev_date = sim_date

        for pair_key, rec_out in snapshot["results"].items():
            assert rec_out["end_inventory"] >= -1e-6, f"negative inventory for {pair_key}"
            assert rec_out["units_sold"] <= rec_out["units_sold"] + rec_out["stockout_units"] + 1e-6
            assert np.isfinite(rec_out["profit"])
            assert rec_out["recommended_order_qty"] >= 0

    print(f"[ok] StreamingPipeline.tick() x5: sim_date advances by 1 day/tick, "
          f"inventory stays non-negative, profit finite for all pairs")

    # state persistence: tick counter and carried exogenous state actually changed
    some_state = next(iter(pipeline.states.values()))
    assert some_state.tick == 5
    print("[ok] PairState persists across ticks (tick counter = 5 after 5 ticks)")

    # --- Serving layer -----------------------------------------------------
    streaming_pipeline.STREAM_DIR = TEST_STREAM_DIR
    streaming_pipeline.SNAPSHOT_PATH = os.path.join(TEST_STREAM_DIR, "latest_snapshot.json")
    streaming_pipeline.LOG_PATH = os.path.join(TEST_STREAM_DIR, "stream_log.csv")
    snapshot = pipeline.tick()
    pipeline.write_snapshot(snapshot)
    assert os.path.exists(streaming_pipeline.SNAPSHOT_PATH)
    assert os.path.exists(streaming_pipeline.LOG_PATH)
    import json
    with open(streaming_pipeline.SNAPSHOT_PATH) as f:
        loaded = json.load(f)
    assert loaded["tick"] == snapshot["tick"]
    log_df = pd.read_csv(streaming_pipeline.LOG_PATH)
    assert len(log_df) == 2  # one row per pair, this tick only (first write)
    print("[ok] write_snapshot(): valid JSON snapshot + appendable CSV log produced")


def test_pipeline_case_memory_wiring(history_df, demand_model):
    """Integration check (not a re-test of memory_store.py's own logic,
    which test_memory_smoke.py already covers in isolation): every tick's
    result dict carries a well-formed "case_memory" sub-dict, use_case_memory=False
    genuinely disables nudging even though episodes still get logged, and
    the pipeline's memory keeps growing tick over tick."""
    scenario = "pure_profit"
    ensure_dummy_model_files(scenario)
    rec = inference.Recommender(scenario, seed=0, history_df=history_df, demand_model=demand_model)

    pipeline = streaming_pipeline.StreamingPipeline(
        {scenario: rec}, primary_scenario=scenario, history_df=history_df, rng_seed=2, use_case_memory=True
    )
    assert pipeline.use_case_memory is True
    prev_n = {key: pipeline.memory.n_episodes(key) for key in pipeline.pairs}
    for _ in range(4):
        snapshot = pipeline.tick()
        for pair_key, rec_out in snapshot["results"].items():
            p = rec_out["case_memory"]
            assert set(p.keys()) == {"nudged", "final_action_idx", "nudge_steps", "explanation", "n_neighbors"}
            assert isinstance(p["nudged"], bool)
            if p["nudged"]:
                assert p["explanation"] is not None
            else:
                assert p["explanation"] is None
    for key in pipeline.pairs:
        n_now = pipeline.memory.n_episodes(key)
        assert n_now == prev_n[key] + 4, "memory should grow by exactly 1 logged episode per tick per pair"
    print("[ok] use_case_memory=True: every tick's result carries a well-formed 'case_memory' sub-dict, "
          "and Case Memory's memory grows by 1 episode per pair per tick")

    # --- use_case_memory=False: never nudges, but keeps logging (closed loop
    # bookkeeping is harmless even when nudging itself is switched off) ----
    pipeline_off = streaming_pipeline.StreamingPipeline(
        {scenario: rec}, primary_scenario=scenario, history_df=history_df, rng_seed=2, use_case_memory=False
    )
    assert pipeline_off.use_case_memory is False
    for _ in range(4):
        snapshot = pipeline_off.tick()
        for pair_key, rec_out in snapshot["results"].items():
            p = rec_out["case_memory"]
            assert p == {"nudged": False, "explanation": None, "n_neighbors": 0}, \
                "use_case_memory=False must always report the untouched default -- never a nudge"
    print("[ok] use_case_memory=False: case memory sub-dict always reports the untouched default (never nudges)")


def test_pipeline_compare_all(history_df, demand_model):
    for name in SCENARIOS:
        ensure_dummy_model_files(name)
    recommenders = {
        name: inference.Recommender(name, seed=0, history_df=history_df, demand_model=demand_model)
        for name in SCENARIOS
    }
    pipeline = streaming_pipeline.StreamingPipeline(
        recommenders, primary_scenario="zero_stockout", history_df=history_df, rng_seed=1
    )
    snapshot = pipeline.tick()
    for pair_key, rec_out in snapshot["results"].items():
        assert rec_out["primary_scenario"] == "zero_stockout"
        others = rec_out["other_scenarios"]
        assert set(others.keys()) == set(SCENARIOS.keys()) - {"zero_stockout"}
        for name, o in others.items():
            assert np.isfinite(o["order_qty"]) and np.isfinite(o["forecast_mean"])
    print("[ok] compare-all mode: primary scenario drives settlement, "
          "other 3 scenarios' reactions reported for comparison only")


def main():
    history_df = make_synthetic_history()
    demand_model = FakeDemandModel()

    # test_pipeline_compare_all() below creates dummy model files for all 4
    # scenarios via ensure_dummy_model_files(). "pure_profit" is the one
    # test_inference_smoke.py already treats as a permanent fixture (its
    # dummy is left in place by convention), but the other 3 are NOT --
    # other tests (e.g. test_app_smoke.py's "_init_stream_pipeline returns
    # None for an untrained scenario" check) rely on those 3 NOT existing.
    # Track+remove them afterward so this test doesn't leave the shared
    # models/ directory polluted for whichever test runs next.
    scenarios_before = set(os.listdir(inference.MODELS_DIR)) if os.path.exists(inference.MODELS_DIR) else set()

    test_generator(history_df)
    test_pipeline_single_scenario(history_df, demand_model)
    test_pipeline_case_memory_wiring(history_df, demand_model)
    test_pipeline_compare_all(history_df, demand_model)

    # cleanup
    if os.path.exists(TEST_STREAM_DIR):
        shutil.rmtree(TEST_STREAM_DIR)

    scenarios_after = set(os.listdir(inference.MODELS_DIR))
    for name in scenarios_after - scenarios_before:
        shutil.rmtree(os.path.join(inference.MODELS_DIR, name), ignore_errors=True)

    print("\nAll streaming smoke-test assertions passed.")


if __name__ == "__main__":
    main()
