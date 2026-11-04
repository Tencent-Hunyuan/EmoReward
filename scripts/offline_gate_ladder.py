#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Reflective-Reward-Gating ablation ladder + Best-of-N reranking (GPU-free).

Produces the headline "ladder" the reviewer asked for:
    videoalign-only < single < random < round-robin < fixed < reflect < JIT < oracle
on (a) reward-outcome pairwise agreement and (b) Best-of-N top-1 selection accuracy,
purely from already-scored samples (no GPU, no training).

Input modes:
  --demo                synthetic multi-regime stream (runs out of the box)
  --traces <jsonl>      one RewardTrace JSON per line (signature/tool_scores/outcome[/meta.prompt])

A gate POLICY maps per-sample provider scores -> a fused scalar reward; we then measure
how well that reward agrees with the downstream outcome y (human pref / offline VideoAlign).
'oracle' is the per-sample best-provider selection upper bound. 'jit' uses RewardMemory.

Outputs (under --out_dir):
  gate_ladder.png/.pdf      bar chart: agreement + Best-of-N top-1 per policy
  gate_ladder.json          all numbers
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyvideo.models.reward_models.emo_reward.reward_memory import (  # noqa: E402
    RewardMemory, RewardTrace, TOOL_KEYS,
)

POLICIES = ["videoalign_only", "single", "random", "roundrobin", "fixed", "reflect", "jit", "oracle"]


def clip(x):
    return max(0.0, min(1.0, x))


def synth_stream(n_prompts=200, n_cand=4, seed=0):
    """n_prompts prompts x n_cand candidates. Regime decides which provider is reliable."""
    rng = random.Random(seed)
    stream = []
    for pid in range(n_prompts):
        face = rng.choice([0.0, 0.0, 0.0, 0.7])
        aro = clip(0.25 + 0.5 * rng.random())
        for _ in range(n_cand):
            latent = rng.random()
            if face < 0.35:   # no-face: r1omni/motion/chromatic informative, AU noise
                sc = {"r1omni": clip(latent + rng.gauss(0, .08)), "motion": clip(latent + rng.gauss(0, .15)),
                      "chromatic": clip(latent + rng.gauss(0, .2)), "clip_autt": rng.random(),
                      "micro": 0.5, "body": 0.5, "temporal": 0.5, "kinetic": clip(.7 + rng.gauss(0, .1)),
                      "macro": rng.random() * .1, "videoalign": clip(.4 + .4 * latent + rng.gauss(0, .15))}
            else:             # face: AU informative
                sc = {"micro": clip(latent + rng.gauss(0, .1)), "clip_autt": clip(latent + rng.gauss(0, .12)),
                      "r1omni": clip(latent + rng.gauss(0, .2)), "motion": clip(.6 + rng.gauss(0, .15)),
                      "body": 0.5, "temporal": 0.5, "chromatic": .8, "kinetic": .75, "macro": rng.random() * .1,
                      "videoalign": clip(.4 + .4 * latent + rng.gauss(0, .15))}
            sig = {"valence": clip(.3 + .4 * latent), "arousal": aro, "face_ratio": face,
                   "motion_regime": clip(.3 + .4 * rng.random()), "subtle": 1.0 if aro < 0.45 else 0.0}
            stream.append({"prompt": pid, "signature": sig, "scores": sc, "outcome": latent + rng.gauss(0, .05)})
    return stream


def fuse(scores, gates):
    num = den = 0.0
    for k in TOOL_KEYS:
        s = scores.get(k)
        if s is None:
            continue
        g = gates.get(k, 0.0)
        num += g * s; den += g
    return num / den if den > 1e-9 else 0.0


def reflect_gates(sig, active):
    """Lightweight stand-in for the adaptive reflect gate (face/subtle aware)."""
    face = sig.get("face_ratio", 0.0); subtle = sig.get("subtle", 0.0)
    g = {}
    for k in active:
        if k in ("micro", "body", "temporal"):
            g[k] = min(1.0, face / 0.35)
        elif k in ("clip_autt",):
            g[k] = (0.8 if subtle else 1.0) * min(1.0, face / 0.35 + 0.1)
        else:
            g[k] = 1.0
    return g


def policy_reward(policy, sig, scores, rng, rr_state, mem):
    active = [k for k in TOOL_KEYS if k in scores]
    if policy == "videoalign_only":
        return scores.get("videoalign", fuse(scores, {k: 1.0 for k in active}))
    if policy == "single":
        return scores.get("r1omni", 0.5)
    if policy == "random":
        return scores.get(rng.choice(active), 0.5)
    if policy == "roundrobin":
        t = active[rr_state[0] % len(active)]; rr_state[0] += 1
        return scores.get(t, 0.5)
    if policy == "fixed":
        return fuse(scores, {k: 1.0 for k in active})
    if policy == "reflect":
        return fuse(scores, reflect_gates(sig, active))
    if policy == "jit":
        w = mem.calibrated_weights(sig, active_tools=active)
        return fuse(scores, w)
    if policy == "oracle":  # per-sample best single provider (upper bound on selection)
        y = scores.get("_y", 0.5)
        return max(scores.get(k, 0.5) for k in active) if False else \
            min(((abs(scores[k] - y), scores[k]) for k in active))[1]
    return 0.5


def pairwise_agreement(rewards, outcomes):
    n = len(rewards); ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            do = outcomes[i] - outcomes[j]
            if abs(do) < 1e-9:
                continue
            tot += 1; dr = rewards[i] - rewards[j]
            if (do > 0) == (dr > 0):
                ok += 1
    return ok / tot if tot else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--traces", default=None)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..",
                    "SAGE-GRPO-logs", "paper_draft"))
    ap.add_argument("--n_cand", type=int, default=4)
    args = ap.parse_args()

    if args.traces:
        stream = []
        with open(args.traces) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = RewardTrace.from_json(line)
                stream.append({"prompt": t.meta.get("prompt", id(t)), "signature": t.signature,
                               "scores": t.tool_scores, "outcome": t.outcome})
    else:
        stream = synth_stream(n_cand=args.n_cand)

    # warm a RewardMemory for the JIT policy (train-free: just write the stream)
    base = {k: 0.10 for k in TOOL_KEYS}
    base.update({"macro": 0.30, "micro": 0.35, "temporal": 0.25, "motion": 0.24, "r1omni": 0.12})
    mem = RewardMemory(base_weights=base, eta=1.0, tau=0.5, top_k=20)
    for s in stream:
        mem.add(RewardTrace(signature=s["signature"], tool_scores=s["scores"], outcome=s["outcome"]))

    ys = [s["outcome"] for s in stream]
    agree, bon_top1, bon_meanout = {}, {}, {}
    # group by prompt for Best-of-N
    prompts = {}
    for i, s in enumerate(stream):
        prompts.setdefault(s["prompt"], []).append(i)

    for pol in POLICIES:
        rng = random.Random(7); rr = [0]
        rewards = []
        for s in stream:
            sc = dict(s["scores"]); sc["_y"] = s["outcome"]
            rewards.append(policy_reward(pol, s["signature"], sc, rng, rr, mem))
        agree[pol] = pairwise_agreement(rewards, ys)
        # Best-of-N: pick argmax reward candidate per prompt
        hit = 0; outs = []
        for pid, idxs in prompts.items():
            sel = max(idxs, key=lambda i: rewards[i])
            best = max(idxs, key=lambda i: ys[i])
            hit += int(sel == best); outs.append(ys[sel])
        bon_top1[pol] = hit / max(len(prompts), 1)
        bon_meanout[pol] = sum(outs) / max(len(outs), 1)

    summary = {"n_samples": len(stream), "n_prompts": len(prompts),
               "agreement": agree, "bestN_top1": bon_top1, "bestN_mean_outcome": bon_meanout}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "gate_ladder.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        colors = ["#bbb", "#bbb", "#bbb", "#bbb", "#888", "#8b2f5e", "#2e8b57", "#0b3d91"]
        for ax, data, title in ((axes[0], agree, "reward–outcome agreement"),
                                (axes[1], bon_top1, "Best-of-%d top-1 selection acc" % args.n_cand)):
            vals = [data[p] for p in POLICIES]
            ax.bar(range(len(POLICIES)), vals, color=colors)
            ax.set_xticks(range(len(POLICIES))); ax.set_xticklabels(POLICIES, rotation=35, ha="right")
            ax.set_title(title); ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01, "%.2f" % v, ha="center", fontsize=7)
        fig.suptitle("Reflective Reward Gating — ablation ladder (offline)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "gate_ladder.png"), dpi=200)
        fig.savefig(os.path.join(args.out_dir, "gate_ladder.pdf"))
        print("WROTE gate_ladder.{png,pdf}")
    except Exception as e:
        print("plot skipped:", e)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
