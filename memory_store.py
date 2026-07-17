"""
memory_store.py

"Case Memory": a lightweight, non-parametric episodic memory layered on top of
the already-trained, FROZEN PPO policy that streaming_pipeline.py calls.

IMPORTANT -- what this is, and what it deliberately is NOT:
This is NOT online reinforcement learning. The PPO network's weights are
never touched, never gradient-updated, never retrained. What Case Memory does
is closer to case-based reasoning / retrieval-augmented decision support,
in three steps that map directly onto how it was originally pitched:

  1. Structured state logging (log_episode):
     Every tick, for the (store, product) pair being decided on, remember
     the exact feature vector the policy was shown (the same numbers PPO's
     network itself looks at -- see inference.Recommender.recommend_for_row's
     `return_obs=True`), the action index that was actually executed that
     tick (PPO's own, or PPO's nudged by Case Memory), and the profit that
     resulted.

  2. Dynamic state reconstruction (retrieve -- "_retrieve_similar"):
     Given a new state, find the k most similar past states for the SAME
     pair by Euclidean distance over that same feature vector. Retrieval is
     scoped per-pair by design: a Groceries item in Store S001 isn't a
     meaningful "similar case" for Electronics in Store S004, and keeping
     comparisons within one pair's own history also means the one-hot
     store/product/category/region dimensions (which are constant within a
     pair) contribute zero noise to the distance calculation.

  3. Reflection + bounded nudge (reflect):
     Compare the best-performing similar past case whose action DIFFERED
     from PPO's current suggestion against the average outcome across all
     retrieved neighbors. If there is enough consistent evidence (at least
     MIN_NEIGHBORS retrieved cases, and the differing action's reward beats
     the neighbor average by at least MIN_MARGIN), propose a nudge -- capped
     at +/- NUDGE_CAP_STEPS action steps (NUDGE_CAP_STEPS * ORDER_STEP
     units), and never below zero. If the evidence is thin or inconsistent,
     reflect() returns "nudged": False and PPO's own action is left alone.

  4. Closed loop:
     Whatever action actually gets executed this tick (nudged or not) and
     the reward it produces gets written back via log_episode() -- so the
     memory keeps growing and refining its own suggestions run after run,
     entirely through this simple retrieval mechanism, WITHOUT retraining
     any neural network.

Known limitation, documented rather than hidden: reward here is the profit
realized for ONE specific historical/simulated demand draw, so comparing
rewards across different episodes at "similar" states is comparing across
different demand realizations too, not a controlled experiment. This is a
heuristic, approximate signal appropriate for a bounded advisory nudge --
NOT a rigorous causal estimate of "what action is truly best." That
distinction should be stated plainly in any write-up of this feature: this
is case-based decision support with online feedback, not causal policy
optimization and not online RL.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

NUDGE_CAP_STEPS = 1     # max +/- action steps (each = order_step units) Case Memory may nudge by
MIN_NEIGHBORS = 3       # minimum # of similar past cases required before nudging at all
MIN_MARGIN = 0.05       # differing-action neighbor's reward must beat the neighbor-average
                        # reward by at least this fractional margin before nudging


@dataclass
class Episode:
    pair_key: tuple[str, str]
    features: np.ndarray   # the exact obs vector the policy was given
    action_idx: int        # action index actually executed (post-nudge if any)
    reward: float          # profit realized that tick
    tick: int
    date: str


class CaseMemory:
    """One memory store shared across all (store, product) pairs a
    StreamingPipeline tracks. See module docstring for the full design and
    its explicit limitations."""

    def __init__(self, order_step: float, nudge_cap_steps: int = NUDGE_CAP_STEPS,
                 min_neighbors: int = MIN_NEIGHBORS, min_margin: float = MIN_MARGIN):
        self.order_step = order_step
        self.nudge_cap_steps = nudge_cap_steps
        self.min_neighbors = min_neighbors
        self.min_margin = min_margin
        self._episodes: dict[tuple, list[Episode]] = {}

    def n_episodes(self, pair_key) -> int:
        return len(self._episodes.get(pair_key, []))

    def log_episode(self, pair_key, features: np.ndarray, action_idx: int,
                     reward: float, tick: int, date: str) -> None:
        self._episodes.setdefault(pair_key, []).append(
            Episode(pair_key, np.asarray(features, dtype=np.float32), int(action_idx), float(reward),
                     int(tick), str(date))
        )

    def _retrieve_similar(self, pair_key, features: np.ndarray, k: int = 5) -> list[Episode]:
        history = self._episodes.get(pair_key, [])
        if not history:
            return []
        features = np.asarray(features, dtype=np.float32)
        dists = np.array([np.linalg.norm(ep.features - features) for ep in history])
        order = np.argsort(dists)[:k]
        return [history[i] for i in order]

    def reflect(self, pair_key, features: np.ndarray, ppo_action_idx: int) -> dict:
        """Pure lookup + arithmetic -- never mutates memory. Returns:
            {"nudged": bool, "final_action_idx": int, "nudge_steps": int,
             "explanation": str | None, "n_neighbors": int}
        Caller (streaming_pipeline.py) is responsible for re-applying the
        warehouse capacity cap to whatever final_action_idx implies, since a
        nudge can push the requested quantity back above MAX_INVENTORY."""
        neighbors = self._retrieve_similar(pair_key, features, k=5)
        result = {
            "nudged": False, "final_action_idx": int(ppo_action_idx), "nudge_steps": 0,
            "explanation": None, "n_neighbors": len(neighbors),
        }
        if len(neighbors) < self.min_neighbors:
            return result

        baseline_reward = float(np.mean([n.reward for n in neighbors]))
        candidates = [n for n in neighbors if n.action_idx != ppo_action_idx]
        if not candidates:
            return result  # every similar past case already agrees with PPO

        best = max(candidates, key=lambda n: n.reward)
        if baseline_reward == 0 or best.reward <= baseline_reward:
            return result
        margin = (best.reward - baseline_reward) / abs(baseline_reward)
        if margin < self.min_margin:
            return result

        direction = 1 if best.action_idx > ppo_action_idx else -1
        raw_diff = abs(best.action_idx - ppo_action_idx)
        nudge_steps = min(self.nudge_cap_steps, raw_diff) * direction
        final_action_idx = max(0, int(ppo_action_idx + nudge_steps))

        result.update({
            "nudged": final_action_idx != ppo_action_idx,
            "final_action_idx": final_action_idx,
            "nudge_steps": final_action_idx - ppo_action_idx,
            "explanation": (
                f"偵測到 {len(neighbors)} 筆相似歷史情境（含 {best.date}）：訂購量為 "
                f"{best.action_idx * self.order_step:.0f} 單位時，獲利較這些相似情境的平均高出約 "
                f"{margin * 100:.0f}%，已將本次建議微調 "
                f"{'+' if final_action_idx - ppo_action_idx > 0 else ''}"
                f"{(final_action_idx - ppo_action_idx) * self.order_step:.0f} 單位"
                f"（原建議 {ppo_action_idx * self.order_step:.0f} -> "
                f"{final_action_idx * self.order_step:.0f}）。"
            ),
        })
        return result
