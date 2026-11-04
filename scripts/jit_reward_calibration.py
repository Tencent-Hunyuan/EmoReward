#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline driver for Just-in-Time Reward Calibration (Emo-Reward x JitRL).

Produces the headline "reward system continually learns" proof WITHOUT any GPU
or training: it replays a stream of scored videos, growing a non-parametric
RewardMemory, and compares how well STATIC reward-fusion vs JIT-calibrated
reward-fusion agree with the downstream outcome (human preference / offline
VideoAlign) as the memory grows — the analogue of JitRL's Avg-vs-Final curve.

Two input modes:
  --demo               run on a synthetic, reproducible stream (works out of the box)
  --traces <jsonl>     each line is a RewardTrace JSON (signature, tool_scores, gates,
                       final_reward, outcome). See reward_memory.RewardTrace.

Outputs (under --out_dir):
  jit_learning_curve.png / .pdf   sliding-window pairwise-agreement curve (static vs JIT)
  jit_summary.json                final metrics + example weight calibration

To build real traces from SAGE_GRPO_results/<exp>/rl_samples + offline labels,
see the `signature/outcome` field contract in the guideline (Guideline 0623, sec.
"From rollouts to reward traces").
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyvideo.models.reward_models.emo_reward.reward_memory import (  # noqa: E402
    RewardMemory, RewardTrace, TOOL_KEYS,
)


def fuse(tool_scores, weights, gates=None):
    """Active-tool dynamically-renormalized fusion: R = sum g w s / sum g w."""
    num = den = 0.0
    for k in TOOL_KEYS:
        s = tool_scores.get(k)
        if s is None:
            continue
        g = (gates or {}).get(k, 1.0)
        w = weights.get(k, 0.0)
        num += g * w * s
        den += g * w
    return num / den if den > 1e-9 else 0.0


def pairwise_agreement(rewards, outcomes):
    """Fraction of comparable pairs whose reward order matches the outcome order."""
    n = len(rewards)
    ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            do = outcomes[i] - outcomes[j]
            dr = rewards[i] - rewards[j]
            if abs(do) < 1e-9:
                continue
            tot += 1
            if (do > 0 and dr > 0) or (do < 0 and dr < 0):
                ok += 1
    return ok / tot if tot else 0.5


def synth_stream(n=600, seed=0):
    """Synthetic multi-regime stream where the *reliable* tool depends on the regime
    (e.g. no-face => trust r1omni/motion not AU). Static weights cannot know this."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        face = rng.choice([0.0, 0.0, 0.0, 0.7])     # ~75% faceless (matches real data)
        subtle = 1.0 if rng.random() < 0.5 else 0.0
        latent = rng.random()                        # ground-truth quality
        if face < 0.35:  # no-face regime: r1omni + motion + chromatic are informative; AU noise
            scores = {
                "r1omni": clip(latent + rng.gauss(0, 0.08)),
                "motion": clip(latent + rng.gauss(0, 0.15)),
                "chromatic": clip(latent + rng.gauss(0, 0.2)),
                "clip_autt": rng.random(), "micro": 0.5, "body": 0.5, "temporal": 0.5,
                "kinetic": clip(0.7 + rng.gauss(0, 0.1)), "macro": rng.random() * 0.1,
            }
        else:            # face regime: AU (micro/clip_autt) informative
            scores = {
                "micro": clip(latent + rng.gauss(0, 0.1)),
                "clip_autt": clip(latent + rng.gauss(0, 0.12)),
                "r1omni": clip(latent + rng.gauss(0, 0.2)),
                "motion": clip(0.6 + rng.gauss(0, 0.15)), "body": 0.5, "temporal": 0.5,
                "chromatic": 0.8, "kinetic": 0.75, "macro": rng.random() * 0.1,
            }
        sig = {"valence": clip(0.3 + 0.4 * latent), "arousal": clip(0.3 + 0.4 * rng.random()),
               "face_ratio": face, "motion_regime": clip(0.3 + 0.4 * rng.random()), "subtle": subtle}
        outcome = latent + rng.gauss(0, 0.05)
        out.append((sig, scores, outcome))
    return out


def clip(x):
    return max(0.0, min(1.0, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--traces", default=None, help="JSONL of RewardTrace lines")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..",
                    "SAGE-GRPO-logs", "paper_draft"))
    ap.add_argument("--window", type=int, default=80)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    base = {k: 0.10 for k in TOOL_KEYS}
    base.update({"macro": 0.30, "micro": 0.35, "temporal": 0.25, "motion": 0.24, "r1omni": 0.12})

    if args.traces:
        stream = []
        with open(args.traces) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = RewardTrace.from_json(line)
                stream.append((t.signature, t.tool_scores, t.outcome))
    else:
        stream = synth_stream()

    mem = RewardMemory(base_weights=base, eta=args.eta, tau=args.tau, top_k=args.top_k)
    static_r, jit_r, ys = [], [], []
    curve_x, curve_static, curve_jit = [], [], []

    for idx, (sig, scores, outcome) in enumerate(stream):
        active = [k for k in TOOL_KEYS if k in scores]
        rs = fuse(scores, base)
        wj = mem.calibrated_weights(sig, active_tools=active)
        rj = fuse(scores, wj)
        static_r.append(rs); jit_r.append(rj); ys.append(outcome)
        mem.add(RewardTrace(signature=sig, tool_scores=scores, final_reward=rj, outcome=outcome))
        if idx >= args.window and idx % 10 == 0:
            w0 = idx - args.window
            curve_x.append(idx)
            curve_static.append(pairwise_agreement(static_r[w0:idx], ys[w0:idx]))
            curve_jit.append(pairwise_agreement(jit_r[w0:idx], ys[w0:idx]))

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "n_samples": len(stream),
        "final_agreement_static": curve_static[-1] if curve_static else None,
        "final_agreement_jit": curve_jit[-1] if curve_jit else None,
        "mean_agreement_static": (sum(curve_static) / len(curve_static)) if curve_static else None,
        "mean_agreement_jit": (sum(curve_jit) / len(curve_jit)) if curve_jit else None,
        "example_calibration_noface": mem.explain(
            {"valence": 0.25, "arousal": 0.3, "face_ratio": 0.0, "motion_regime": 0.4, "subtle": 1.0}),
    }
    with open(os.path.join(args.out_dir, "jit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(curve_x, curve_static, "--", color="#888", lw=2, label="Static Emo-Reward (fixed weights)")
        ax.plot(curve_x, curve_jit, "-", color="#8b2f5e", lw=2.2, label="JIT Emo-Reward (memory-calibrated)")
        ax.set_xlabel("# reward traces seen (memory size)")
        ax.set_ylabel("reward–outcome pairwise agreement")
        ax.set_title("Reward system continually learns (train-free): JIT vs Static")
        ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0.45, 1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "jit_learning_curve.png"), dpi=200)
        fig.savefig(os.path.join(args.out_dir, "jit_learning_curve.pdf"))
        print("WROTE jit_learning_curve.{png,pdf}")
    except Exception as e:
        print("plot skipped:", e)

    print(json.dumps({k: v for k, v in summary.items() if k != "example_calibration_noface"}, indent=2))


if __name__ == "__main__":
    main()
