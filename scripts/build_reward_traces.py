#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build RewardMemory traces (traces.jsonl) from existing rl_samples (GPU-free).

Prerequisite step for the g7 JIT advantage gate: JIT needs a non-parametric
RewardMemory, which is built from historical <state, per-provider scores, outcome>
traces. Every prior run (g1..g6, or the 06.11/06.12 runs) logs, per rollout sample,
all `emo_reward_emo_*` provider scores + `videoalign_overall` in
`<run>/rl_samples/*/group_*/*_rewards.json`. This script harvests them into the
RewardTrace JSONL consumed by:
  - scripts/export_jit_gate_table.py  (-> jit_gate_table.json for g7)
  - scripts/jit_reward_calibration.py (C1 learning curve)
  - scripts/offline_gate_ladder.py    (B ladder + oracle)

State signature z = (valence, arousal, face_ratio, motion_regime, subtle):
  target VAD is NOT logged per-sample, so valence/arousal are proxied from the
  prompt via the SAME keyword lists the taxonomy planner uses (subtle / high-energy
  / positive / negative). This reproduces the binary regime split used at inference
  (regime = face{0/1}_aro{0/1}). Use --use_taxonomy_planner for the exact planner
  VAD (heavier import).

outcome y (higher = better) defaults to `videoalign_overall` (the general video
quality label); override with --outcome emo_total, or provide an external
label CSV via --label_csv (columns: video_path,outcome).

Usage:
  python3 scripts/build_reward_traces.py \
      --results_dir SAGE-GRPO-logs/SAGE_GRPO_results \
      --runs 2026.06.11-17.40.28,2026.06.12-15.02.45 \
      --out SAGE-GRPO-logs/paper_draft/traces.jsonl
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyvideo.models.reward_models.emo_reward.reward_memory import (  # noqa: E402
    RewardTrace, TOOL_KEYS,
)

# provider-score key mapping: rl_samples key -> RewardMemory tool key
_TOOL_MAP = {
    "macro": "emo_reward_emo_macro",
    "micro": "emo_reward_emo_micro",
    "body": "emo_reward_emo_body_compliance",
    "temporal": "emo_reward_emo_temporal",
    "chromatic": "emo_reward_emo_chromatic",
    "kinetic": "emo_reward_emo_kinetic",
    "clip_autt": "emo_reward_emo_clip_autt",
    "r1omni": "emo_reward_emo_r1omni",
    "motion": "emo_reward_emo_motion_naturalness",
}
_FACE_KEYS = ("emo_reward_emo_micro", "emo_reward_emo_body_compliance",
              "emo_reward_emo_temporal", "emo_reward_emo_au_compliance")

_SUBTLE = ("calm", "gentle", "quiet", "soft", "still", "slight", "subtle", "restrained",
           "tender", "peaceful", "serene", "微笑", "轻", "缓慢", "平静", "安静", "克制", "温柔")
_HIGH = ("run", "running", "dance", "dancing", "jump", "shout", "scream", "fight", "explode",
         "冲", "奔跑", "跳", "激烈", "狂喜", "愤怒", "尖叫")
_POS = ("joy", "happy", "smile", "excited", "love", "warm", "delight", "cheer",
        "喜", "快乐", "微笑", "温暖", "幸福", "开心")
_NEG = ("sad", "cry", "tears", "heartbreak", "anger", "fear", "grief", "lonely",
        "悲", "哭", "泪", "心碎", "愤怒", "恐惧", "孤独")


def _has(text, words):
    t = str(text or "").lower()
    return any(w in t for w in words)


def _proxy_signature(sample, use_planner=False):
    prompt = sample.get("prompt", "")
    subtle = 1.0 if _has(prompt, _SUBTLE) else 0.0
    if use_planner:
        try:
            from hyvideo.models.reward_models.emo_reward.react_reward import EmotionPlanner
            plan = EmotionPlanner(vlm_planner=None).plan(prompt, 5.0)
            valence, arousal = float(plan.target_vad[0]), float(plan.target_vad[1])
        except Exception:
            use_planner = False
    if not use_planner:
        arousal = 0.8 if _has(prompt, _HIGH) else (0.2 if subtle else 0.5)
        valence = 0.8 if _has(prompt, _POS) else (0.2 if _has(prompt, _NEG) else 0.5)
    face_ratio = 1.0 if any(abs(float(sample.get(k, 0.5)) - 0.5) > 1e-4 for k in _FACE_KEYS) else 0.0
    motion_regime = float(sample.get("emo_reward_emo_motion_naturalness",
                          sample.get("emo_reward_emo_kinetic", 0.5)))
    return {"valence": valence, "arousal": arousal, "face_ratio": face_ratio,
            "motion_regime": max(0.0, min(1.0, motion_regime)), "subtle": subtle}


def _iter_samples(run_dir):
    for fp in glob.glob(os.path.join(run_dir, "rl_samples", "*", "group_*", "*_rewards.json")):
        try:
            arr = json.load(open(fp))
        except Exception:
            continue
        if isinstance(arr, list):
            for s in arr:
                if isinstance(s, dict):
                    yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="SAGE-GRPO-logs/SAGE_GRPO_results")
    ap.add_argument("--runs", default="2026.06.11-17.40.28,2026.06.12-15.02.45",
                    help="comma list of run dir names under results_dir (glob ok)")
    ap.add_argument("--out", default="SAGE-GRPO-logs/paper_draft/traces.jsonl")
    ap.add_argument("--outcome", default="videoalign_overall",
                    choices=["videoalign_overall", "overall", "emo_total"])
    ap.add_argument("--label_csv", default=None, help="optional CSV video_path,outcome (overrides --outcome)")
    ap.add_argument("--use_taxonomy_planner", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    labels = {}
    if args.label_csv and os.path.isfile(args.label_csv):
        import csv
        for row in csv.DictReader(open(args.label_csv)):
            labels[row["video_path"]] = float(row["outcome"])

    run_dirs = []
    for r in args.runs.split(","):
        r = r.strip()
        if not r:
            continue
        run_dirs += sorted(glob.glob(os.path.join(args.results_dir, r)) +
                           glob.glob(os.path.join(args.results_dir, r + "*")))
    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        print(f"[build_traces] no run dirs matched under {args.results_dir}; runs={args.runs}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    out_key = "emo_reward_" + args.outcome if args.outcome != "emo_total" else "emo_reward_emo_total"
    with open(args.out, "w", encoding="utf-8") as f:
        for rd in run_dirs:
            cnt = 0
            for s in _iter_samples(rd):
                sig = _proxy_signature(s, args.use_taxonomy_planner)
                tool_scores = {k: float(s.get(src, 0.5)) for k, src in _TOOL_MAP.items()}
                gates = {"reflect_gate": float(s.get("emo_reward_emo_reflect_gate", 1.0))}
                vp = str(s.get("video_path", ""))
                y = labels.get(vp)
                if y is None:
                    y = float(s.get(out_key, s.get("emo_reward_videoalign_overall", 0.0)))
                if y != y:  # NaN guard
                    continue
                final_reward = float(s.get("emo_reward_emo_total", s.get("avg", 0.0)))
                tr = RewardTrace(signature=sig, tool_scores=tool_scores, gates=gates,
                                 final_reward=final_reward, outcome=y,
                                 meta={"run": os.path.basename(rd), "prompt": str(s.get("prompt", ""))[:120]})
                f.write(tr.to_json() + "\n")
                n += 1
                cnt += 1
                if args.max_samples and n >= args.max_samples:
                    break
            print(f"[build_traces] {os.path.basename(rd)}: {cnt} traces")
            if args.max_samples and n >= args.max_samples:
                break
    print(f"[build_traces] wrote {n} traces -> {args.out}  (outcome={args.outcome}, tools={list(_TOOL_MAP)})")


if __name__ == "__main__":
    main()
