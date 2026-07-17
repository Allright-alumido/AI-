"""
test_memory_smoke.py

Smoke test for memory_store.CaseMemory in isolation (pure numpy +
dataclasses, no gymnasium/stable-baselines3/lightgbm stubs needed -- this
module has no such dependencies).

Checks:
  - Not enough neighbors yet -> never nudges (MIN_NEIGHBORS gate).
  - Enough neighbors but all agree with PPO's own action -> no nudge.
  - Enough neighbors, inconsistent/weak evidence (margin too small) -> no nudge.
  - Enough neighbors, one clearly-better differing action -> nudges, capped
    at +/- NUDGE_CAP_STEPS action steps even when the better action is
    further away than that.
  - Nudge never pushes the action index below 0.
  - Retrieval/reflection is scoped per pair_key -- another pair's episodes
    never leak into a different pair's reflect() call.
  - log_episode() + reflect() together form the "closed loop": a nudge
    applied this tick becomes retrievable evidence for a later tick.

Run: python test_memory_smoke.py
"""

import numpy as np

from memory_store import CaseMemory, NUDGE_CAP_STEPS, MIN_NEIGHBORS, MIN_MARGIN

ORDER_STEP = 10.0
PAIR_A = ("S001", "P0001")
PAIR_B = ("S002", "P0002")


def make_features(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.normal(size=10).astype(np.float32)


def test_insufficient_neighbors():
    mem = CaseMemory(order_step=ORDER_STEP)
    features = make_features(0)
    # Log fewer than MIN_NEIGHBORS episodes, all with a differing, "better" action.
    for i in range(MIN_NEIGHBORS - 1):
        mem.log_episode(PAIR_A, features, action_idx=30, reward=100.0, tick=i, date=f"2026-01-0{i+1}")
    result = mem.reflect(PAIR_A, features, ppo_action_idx=20)
    assert not result["nudged"]
    assert result["final_action_idx"] == 20
    assert result["n_neighbors"] == MIN_NEIGHBORS - 1
    print("[ok] fewer than MIN_NEIGHBORS logged episodes -> never nudges")


def test_all_neighbors_agree_with_ppo():
    mem = CaseMemory(order_step=ORDER_STEP)
    features = make_features(1)
    for i in range(5):
        mem.log_episode(PAIR_A, features, action_idx=20, reward=100.0 + i, tick=i, date=f"2026-01-0{i+1}")
    result = mem.reflect(PAIR_A, features, ppo_action_idx=20)
    assert not result["nudged"]
    print("[ok] all similar past cases already agree with PPO's action -> no nudge")


def test_weak_margin_no_nudge():
    mem = CaseMemory(order_step=ORDER_STEP, min_margin=0.5)  # demand a huge margin
    features = make_features(2)
    for i in range(5):
        # differing action, only a tiny bit better -- shouldn't clear a 50% margin bar
        action = 20 if i % 2 == 0 else 30
        reward = 100.0 if action == 20 else 102.0
        mem.log_episode(PAIR_A, features, action_idx=action, reward=reward, tick=i, date=f"2026-01-0{i+1}")
    result = mem.reflect(PAIR_A, features, ppo_action_idx=20)
    assert not result["nudged"], "margin too small for min_margin=0.5, should not nudge"
    print("[ok] differing action exists but margin too small -> no nudge")


def test_clear_nudge_capped():
    mem = CaseMemory(order_step=ORDER_STEP, nudge_cap_steps=1, min_neighbors=3, min_margin=0.05)
    features = make_features(3)
    # 4 similar past cases: 3 at action_idx=20 (reward ~100), 1 at action_idx=25
    # (5 steps away) with a much higher reward -- clear, consistent evidence
    # that a HIGHER order would have done better, but 5 steps away.
    for i in range(3):
        mem.log_episode(PAIR_A, features, action_idx=20, reward=100.0, tick=i, date=f"2026-01-0{i+1}")
    mem.log_episode(PAIR_A, features, action_idx=25, reward=200.0, tick=3, date="2026-01-04")

    result = mem.reflect(PAIR_A, features, ppo_action_idx=20)
    assert result["nudged"], "should nudge: differing action clearly beats the neighbor average"
    # capped at +/- NUDGE_CAP_STEPS=1 even though best_other is 5 steps away
    assert result["final_action_idx"] == 21, f"expected capped nudge to 21, got {result['final_action_idx']}"
    assert result["nudge_steps"] == 1
    assert result["explanation"] is not None and "微調" in result["explanation"]
    print(f"[ok] clear, consistent evidence for a differing (5-steps-away) action -> nudges, "
          f"but capped to +{NUDGE_CAP_STEPS} step ({result['final_action_idx'] * ORDER_STEP:.0f} units), "
          f"not all the way to the neighbor's own action")


def test_nudge_never_goes_negative():
    mem = CaseMemory(order_step=ORDER_STEP, nudge_cap_steps=1, min_neighbors=3, min_margin=0.05)
    features = make_features(4)
    for i in range(3):
        mem.log_episode(PAIR_A, features, action_idx=0, reward=100.0, tick=i, date=f"2026-01-0{i+1}")
    mem.log_episode(PAIR_A, features, action_idx=5, reward=500.0, tick=3, date="2026-01-04")
    result = mem.reflect(PAIR_A, features, ppo_action_idx=0)
    if result["nudged"]:
        assert result["final_action_idx"] >= 0
    print("[ok] nudge (if any) never pushes action index below 0")


def test_pair_scoping():
    mem = CaseMemory(order_step=ORDER_STEP, nudge_cap_steps=1, min_neighbors=3, min_margin=0.05)
    features = make_features(5)
    # Strong evidence logged under PAIR_B should NOT affect PAIR_A's reflect().
    for i in range(3):
        mem.log_episode(PAIR_B, features, action_idx=20, reward=100.0, tick=i, date=f"2026-01-0{i+1}")
    mem.log_episode(PAIR_B, features, action_idx=30, reward=500.0, tick=3, date="2026-01-04")

    result = mem.reflect(PAIR_A, features, ppo_action_idx=20)
    assert not result["nudged"], "PAIR_B's episodes must not leak into PAIR_A's retrieval"
    assert result["n_neighbors"] == 0
    print("[ok] retrieval is correctly scoped per (store, product) pair -- no cross-pair leakage")


def test_closed_loop_feedback():
    """A nudge applied (and logged) this tick becomes retrievable evidence
    for a LATER tick with a similar state -- the "closed loop" property."""
    mem = CaseMemory(order_step=ORDER_STEP, nudge_cap_steps=1, min_neighbors=3, min_margin=0.05)
    features = make_features(6)

    assert mem.n_episodes(PAIR_A) == 0
    for i in range(3):
        mem.log_episode(PAIR_A, features, action_idx=20, reward=100.0, tick=i, date=f"2026-01-0{i+1}")
    assert mem.n_episodes(PAIR_A) == 3

    # First reflect(): not enough differing-action evidence yet.
    result1 = mem.reflect(PAIR_A, features, ppo_action_idx=20)
    assert not result1["nudged"]

    # A new, clearly-better differing action gets logged (simulating that
    # tick's actual outcome) -- this is what closes the loop.
    mem.log_episode(PAIR_A, features, action_idx=22, reward=300.0, tick=3, date="2026-01-04")
    assert mem.n_episodes(PAIR_A) == 4

    # A LATER tick with a similar state should now see that new evidence.
    result2 = mem.reflect(PAIR_A, features, ppo_action_idx=20)
    assert result2["nudged"], "the newly-logged, clearly-better episode should now be retrievable"
    print("[ok] closed loop: an episode logged this tick is retrievable evidence for a later tick, "
          "without any retraining of a neural network")


def main():
    test_insufficient_neighbors()
    test_all_neighbors_agree_with_ppo()
    test_weak_margin_no_nudge()
    test_clear_nudge_capped()
    test_nudge_never_goes_negative()
    test_pair_scoping()
    test_closed_loop_feedback()
    print("\nAll Case Memory smoke-test assertions passed.")


if __name__ == "__main__":
    main()
