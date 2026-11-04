"""Just-in-Time Reward Calibration for Emo-Reward (training-free, non-parametric).

This is the Emo-Reward analogue of JitRL (Li et al., 2026, arXiv:2601.18510):
instead of modulating an LLM *policy* logit with a retrieval-estimated *action*
advantage, we modulate the *reward-fusion weights* with a retrieval-estimated
*tool-reliability* advantage. No neural network is trained; the reward system
"learns" purely by writing and reading a non-parametric RewardMemory.

JitRL --> Emo-Reward mapping
---------------------------
  state  s            -> reward "context signature" z = (valence, arousal,
                          face_ratio, motion_regime, ...)  of (prompt+EmoScript+video)
  action a            -> a reward primitive / tool k (macro/micro/.../r1omni/motion)
  reward / return G   -> downstream outcome y (human preference, offline VideoAlign,
                          GRPO advantage sign, KL stability, reward-hacking tag)
  memory M            -> historical reward traces
  advantage A(s,a)    -> tool-reliability advantage A_k(z): is tool k more
                          predictive of a good outcome than the average tool, here?
  logit update        -> closed-form KL-constrained reward-weight update
        z' = z + beta*A   (JitRL)   <==>   w_k(z) = w0_k * exp(eta * A_k(z) / tau)

Reliability advantage (principled, train-free)
---------------------------------------------
For a query z, retrieve top-k similar traces N(z). For each tool k, its
*reliability* in this regime is how well its dense score s_k co-varies with the
outcome y over the neighborhood (Pearson correlation):

    rho_k(z) = Cov_{i in N(z)}(s_k^i, y_i) / (std(s_k) * std(y) + eps)

We center across tools to get an advantage (how much more reliable than the
average active tool), exactly mirroring A = Q - V:

    A_k(z) = rho_k(z) - mean_j rho_j(z)            (= Q_k - V)

and apply the KL-constrained closed-form weight update (analogue of Eq. 9/10):

    w_k(z) ∝ w0_k * exp(eta * A_k(z) / tau),   renormalized over active tools.

Constant / low-variance tools (the documented failure mode: micro/body/temporal
== 0.5, saturated clip_autt, decaying macro) get rho_k ~ 0 -> A_k < 0 -> they are
automatically down-weighted; tools whose score tracks the good outcome are
up-weighted. Everything stays close to the base calibration w0 (KL term).
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Order-stable list of reward primitives (tools) we calibrate.
TOOL_KEYS: Tuple[str, ...] = (
    "macro", "micro", "body", "temporal",
    "chromatic", "kinetic", "clip_autt", "r1omni", "motion",
)

# Context signature fields used for retrieval similarity.
SIG_KEYS: Tuple[str, ...] = (
    "valence", "arousal", "face_ratio", "motion_regime", "subtle",
)


@dataclass
class RewardTrace:
    """One scored video -> one reward trace m_i = {z_i, s^k, g^k, R_i, y_i}."""
    signature: Dict[str, float]          # z_i  (SIG_KEYS -> float in [0,1])
    tool_scores: Dict[str, float]        # s^k  (TOOL_KEYS -> raw tool score [0,1])
    gates: Dict[str, float] = field(default_factory=dict)   # g^k
    final_reward: float = 0.0            # R_i  (fused reward at scoring time)
    outcome: float = 0.0                 # y_i  (downstream outcome, higher=better)
    meta: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "RewardTrace":
        d = json.loads(line)
        return RewardTrace(
            signature=dict(d.get("signature", {})),
            tool_scores=dict(d.get("tool_scores", {})),
            gates=dict(d.get("gates", {})),
            final_reward=float(d.get("final_reward", 0.0)),
            outcome=float(d.get("outcome", 0.0)),
            meta=dict(d.get("meta", {})),
        )


def _feature_vec(sig: Dict[str, float]) -> List[float]:
    return [float(sig.get(key, 0.5)) for key in SIG_KEYS]


def _dist(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _pearson(xs: List[float], ys: List[float], eps: float = 1e-6) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    denom = math.sqrt(vx) * math.sqrt(vy) + eps
    return cov / denom


class RewardMemory:
    """Non-parametric reward memory + just-in-time weight calibration.

    Thread-safe append; pure-python retrieval (no external vector DB needed).
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        eta: float = 1.0,
        tau: float = 1.0,
        top_k: int = 12,
        sim_radius: float = 0.6,
        min_neighbors: int = 4,
        max_log_ratio: float = 1.0,   # clamp |log(w/w0)| for stability (KL trust region)
        persist_path: Optional[str] = None,
    ):
        self.base_weights = dict(base_weights or {k: 1.0 for k in TOOL_KEYS})
        self.eta = float(eta)
        self.tau = max(float(tau), 1e-6)
        self.top_k = int(top_k)
        self.sim_radius = float(sim_radius)
        self.min_neighbors = int(min_neighbors)
        self.max_log_ratio = float(max_log_ratio)
        self.persist_path = persist_path
        self._traces: List[RewardTrace] = []
        self._feats: List[List[float]] = []
        self._lock = threading.Lock()
        if persist_path and os.path.isfile(persist_path):
            self.load(persist_path)

    # ----- memory write stream -----
    def add(self, trace: RewardTrace) -> None:
        with self._lock:
            self._traces.append(trace)
            self._feats.append(_feature_vec(trace.signature))
            if self.persist_path:
                with open(self.persist_path, "a", encoding="utf-8") as f:
                    f.write(trace.to_json() + "\n")

    def __len__(self) -> int:
        return len(self._traces)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = RewardTrace.from_json(line)
                except Exception:
                    continue
                self._traces.append(t)
                self._feats.append(_feature_vec(t.signature))

    # ----- retrieval (Top-k neighbors within radius) -----
    def retrieve(self, sig: Dict[str, float], k: Optional[int] = None) -> List[RewardTrace]:
        k = k or self.top_k
        if not self._traces:
            return []
        q = _feature_vec(sig)
        scored = sorted(
            ((_dist(q, f), i) for i, f in enumerate(self._feats)), key=lambda x: x[0]
        )
        out = []
        for d, i in scored[: max(k, self.min_neighbors)]:
            if d <= self.sim_radius or len(out) < self.min_neighbors:
                out.append(self._traces[i])
            if len(out) >= k:
                break
        return out

    # ----- tool-reliability advantage  A_k(z) = rho_k - mean_j rho_j -----
    def tool_advantages(self, sig: Dict[str, float], k: Optional[int] = None) -> Dict[str, float]:
        neighbors = self.retrieve(sig, k)
        if len(neighbors) < self.min_neighbors:
            return {key: 0.0 for key in TOOL_KEYS}
        ys = [n.outcome for n in neighbors]
        rho: Dict[str, float] = {}
        for key in TOOL_KEYS:
            sk = [n.tool_scores.get(key, 0.5) for n in neighbors]
            # a tool that is constant in this regime carries no information -> rho 0
            rho[key] = _pearson(sk, ys)
        mean_rho = sum(rho.values()) / max(len(rho), 1)
        return {key: rho[key] - mean_rho for key in TOOL_KEYS}

    # ----- closed-form KL-constrained weight update -----
    def calibrated_weights(
        self,
        sig: Dict[str, float],
        active_tools: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return JIT-calibrated, renormalized fusion weights for active tools.

        w_k(z) ∝ w0_k * exp(eta * A_k(z) / tau), clamped to a KL trust region,
        then renormalized over active tools so sum == sum of base active weights.
        """
        adv = self.tool_advantages(sig, k)
        tools = list(active_tools) if active_tools else list(TOOL_KEYS)
        raw: Dict[str, float] = {}
        for key in tools:
            w0 = max(self.base_weights.get(key, 1.0), 0.0)
            log_ratio = self.eta * adv.get(key, 0.0) / self.tau
            log_ratio = max(-self.max_log_ratio, min(self.max_log_ratio, log_ratio))
            raw[key] = w0 * math.exp(log_ratio)
        base_sum = sum(max(self.base_weights.get(key, 1.0), 0.0) for key in tools)
        raw_sum = sum(raw.values())
        if raw_sum <= 1e-12 or base_sum <= 1e-12:
            return {key: self.base_weights.get(key, 1.0) for key in tools}
        scale = base_sum / raw_sum
        return {key: raw[key] * scale for key in tools}

    def explain(self, sig: Dict[str, float], k: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        adv = self.tool_advantages(sig, k)
        w = self.calibrated_weights(sig, k=k)
        return {
            "n_neighbors": float(len(self.retrieve(sig, k))),
            "advantage": adv,
            "base_weight": dict(self.base_weights),
            "calibrated_weight": w,
        }


def _self_test() -> None:
    """Synthetic regime: 'no-face, restrained sadness'. r1omni/VAD are predictive of
    the good outcome; clip_autt (face AU) is pure noise. JIT should up-weight r1omni
    and down-weight clip_autt relative to base."""
    import random
    random.seed(0)
    base = {k: (0.30 if k == "macro" else 0.10) for k in TOOL_KEYS}
    base.update({"micro": 0.10, "temporal": 0.10, "motion": 0.24, "r1omni": 0.12})
    mem = RewardMemory(base_weights=base, eta=1.0, tau=0.5, top_k=20)
    for _ in range(200):
        latent_good = random.random()  # ground-truth quality of this sample
        sig = {"valence": 0.2 + 0.1 * random.random(), "arousal": 0.25,
               "face_ratio": 0.0, "motion_regime": 0.3, "subtle": 1.0}
        scores = {
            "r1omni": min(1.0, max(0.0, latent_good + random.gauss(0, 0.08))),  # predictive
            "motion": min(1.0, max(0.0, latent_good + random.gauss(0, 0.15))),  # weak predictive
            "clip_autt": random.random(),                                        # pure noise
            "micro": 0.5, "body": 0.5, "temporal": 0.5,                          # constants
            "macro": random.random() * 0.1,                                      # decaying noise
            "chromatic": 0.8, "kinetic": 0.75,
        }
        outcome = latent_good + random.gauss(0, 0.05)  # human/offline outcome
        mem.add(RewardTrace(signature=sig, tool_scores=scores, final_reward=0.0, outcome=outcome))

    q = {"valence": 0.22, "arousal": 0.25, "face_ratio": 0.0, "motion_regime": 0.3, "subtle": 1.0}
    info = mem.explain(q)
    adv = info["advantage"]
    w = info["calibrated_weight"]
    print("neighbors:", info["n_neighbors"])
    print("advantage r1omni=%.3f clip_autt=%.3f micro=%.3f" % (adv["r1omni"], adv["clip_autt"], adv["micro"]))
    print("base   r1omni=%.3f clip_autt=%.3f" % (base["r1omni"], base["clip_autt"]))
    print("calib  r1omni=%.3f clip_autt=%.3f" % (w["r1omni"], w["clip_autt"]))
    assert adv["r1omni"] > adv["clip_autt"], "predictive tool should outrank noise"
    assert w["r1omni"] > base["r1omni"], "predictive tool should be up-weighted"
    assert w["clip_autt"] <= base["clip_autt"] + 1e-9, "noise tool should not be up-weighted"
    print("SELF_TEST_OK")


if __name__ == "__main__":
    _self_test()
