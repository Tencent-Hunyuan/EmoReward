#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the JIT gate reliability-multiplier table for g7 (GPU-free).

Consumes traces.jsonl (from build_reward_traces.py) and emits jit_gate_table.json
in the EXACT format the g7 gate consumes in react_reward.ReflectionRewarder:

    regime key : "face{0/1}_aro{0/1}"  (int(face_ratio>=0.35), int(arousal>=0.5))
    table      : { regime: { tool: multiplier }, ..., "default": {...} }
    gate usage : effective_gate_k = reflect_gate_k * multiplier_k     (per tool)

Mathematics (per regime R, over the traces whose signature falls in R):
    rho_k(R)  = Pearson( s_k^i , y_i )         # provider score vs. outcome
    A_k(R)    = rho_k(R) - (1/|T|) * sum_j rho_j(R)   # reliability advantage (A=Q-V)
    m_k(R)    = clip( exp(eta * A_k(R) / tau), exp(-cap), exp(cap) )   # KL-form tilt
This is the closed-form KL-regularized weight tilt of JitRL applied to reward
providers (see hyvideo/.../reward_memory.py). Reliable providers (rho high ->
A_k>0) get m_k>1; constant/low-variance providers (rho~0 -> A_k<0) get m_k<1.
The fusion's active-tool re-normalization absorbs the overall scale, so we emit
the raw tilt (no base-weight renormalization needed).

Only the 7 fusion providers are emitted (macro/micro/body/temporal/clip_autt/
r1omni/motion); chromatic/kinetic are logged-only and ignored by the gate.

Usage:
  python3 scripts/export_jit_gate_table.py \
      --traces SAGE-GRPO-logs/paper_draft/traces.jsonl \
      --out SAGE-GRPO-logs/paper_draft/jit_gate_table.json \
      --eta 1.0 --tau 0.5 --cap 1.0 --min_regime 20
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyvideo.models.reward_models.emo_reward.reward_memory import (  # noqa: E402
    RewardTrace, _pearson,
)

# providers consumed by the g7 gate (react_reward._GATE_TOOLS)
GATE_TOOLS = ("macro", "micro", "body", "temporal", "clip_autt", "r1omni", "motion")


def _regime(sig):
    return "face%d_aro%d" % (int(float(sig.get("face_ratio", 0.0)) >= 0.35),
                             int(float(sig.get("arousal", 0.5)) >= 0.5))


def _multipliers(traces, eta, tau, cap):
    """Return {tool: multiplier} for a set of traces (a regime, or all=default)."""
    ys = [t.outcome for t in traces]
    rho = {}
    for k in GATE_TOOLS:
        sk = [float(t.tool_scores.get(k, 0.5)) for t in traces]
        rho[k] = _pearson(sk, ys)
    mean_rho = sum(rho.values()) / max(len(rho), 1)
    mult = {}
    for k in GATE_TOOLS:
        a_k = rho[k] - mean_rho
        log_m = max(-cap, min(cap, eta * a_k / max(tau, 1e-6)))
        mult[k] = round(math.exp(log_m), 4)
    return mult, {k: round(rho[k], 3) for k in GATE_TOOLS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="SAGE-GRPO-logs/paper_draft/traces.jsonl")
    ap.add_argument("--out", default="SAGE-GRPO-logs/paper_draft/jit_gate_table.json")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=1.0, help="|log multiplier| trust region (1.0 -> [0.37,2.72])")
    ap.add_argument("--min_regime", type=int, default=20, help="min traces per regime; else fall back to default")
    args = ap.parse_args()

    if not os.path.isfile(args.traces):
        print(f"[export_jit] traces not found: {args.traces}  (run build_reward_traces.py first)")
        sys.exit(1)
    traces = [RewardTrace.from_json(l) for l in open(args.traces) if l.strip()]
    if len(traces) < args.min_regime:
        print(f"[export_jit] too few traces ({len(traces)}); need >= {args.min_regime}")
        sys.exit(1)

    by_regime = {}
    for t in traces:
        by_regime.setdefault(_regime(t.signature), []).append(t)

    default_mult, default_rho = _multipliers(traces, args.eta, args.tau, args.cap)
    table = {"default": default_mult}
    report = {"n_traces": len(traces), "default": {"n": len(traces), "rho": default_rho, "mult": default_mult},
              "regimes": {}}
    for regime in ("face0_aro0", "face0_aro1", "face1_aro0", "face1_aro1"):
        grp = by_regime.get(regime, [])
        if len(grp) >= args.min_regime:
            mult, rho = _multipliers(grp, args.eta, args.tau, args.cap)
            table[regime] = mult
            report["regimes"][regime] = {"n": len(grp), "rho": rho, "mult": mult}
        else:
            # not enough data -> inherit default (gate will also fall back to default)
            report["regimes"][regime] = {"n": len(grp), "note": "insufficient -> use default"}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(table, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(report, open(args.out.replace(".json", "_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"[export_jit] wrote {args.out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # sanity: warn if every multiplier ~1 (no signal) or extreme
    allm = [m for r in table.values() for m in r.values()]
    if max(allm) - min(allm) < 0.05:
        print("[export_jit] WARNING: multipliers nearly uniform (~1.0) -> traces carry little "
              "provider-reliability signal; g7 will behave ~ g6 (reflect). Check outcome labels/coverage.")


if __name__ == "__main__":
    main()
