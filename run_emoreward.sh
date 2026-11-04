#!/bin/bash
# =============================================================================
# EmoReward — single-command training pipeline (Plan · Act · Reflect · Evolve)
#
# EmoReward self-evolves a reward across two training runs. This script chains
# both runs end to end:
#
#   Stage 1  emoreward without RewardMemory
#            Reflective Reward Gating only (instance-level phase/conflict/face/
#            subtlety gates + active-provider renormalization; unit memory
#            multipliers m_k=1). Runs a short reference run that logs reward
#            traces into rl_samples.
#
#   (offline, GPU-free) distill RewardMemory from the Stage-1 traces:
#            build traces.jsonl  ->  export the regime-keyed reliability-
#            multiplier table (jit_gate_table.json) used by Stage 2.
#
#   Stage 2  emoreward with RewardMemory
#            Reflect gates composed with the frozen memory multipliers m_k(r)
#            exported above. This is the full method — a complete training run.
#
# Everything except the gate policy / fusion weights is held fixed between the
# two stages, matching the paper's controlled setup. Other reward policies in
# the ablation ladder (VideoAlign-only, single-provider, random/round-robin
# routing, fixed fusion) are reachable without editing this script by exporting
# REWARD_GATE_POLICY — see the README.
#
# Usage (from the repo root; 32-GPU recipe via run_post_train_multinode.sh):
#   bash run_emoreward.sh                 # Stage 1 (100 steps) -> table -> Stage 2
#   STAGE1_STEPS=100 bash run_emoreward.sh
#   SKIP_STAGE1=1 STAGE1_DIR=SAGE_GRPO_results/<ref_run> bash run_emoreward.sh
#                                         # reuse an existing reference run's traces
# =============================================================================
set -eu
cd "$(dirname "$0")"

# ---- shared config (identical across both stages) ---------------------------
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export CLIP_AUTT_WEIGHT="${CLIP_AUTT_WEIGHT:-0.0}"
export R1OMNI_WEIGHT="${R1OMNI_WEIGHT:-0.12}"
export R1OMNI_VAD_MODE="${R1OMNI_VAD_MODE:-alignment}"
export ENABLE_CHROMATIC="${ENABLE_CHROMATIC:-True}"
export ENABLE_KINETIC="${ENABLE_KINETIC:-True}"
export ENABLE_MOTION_NATURALNESS="${ENABLE_MOTION_NATURALNESS:-True}"
export MOTION_NATURALNESS_WEIGHT="${MOTION_NATURALNESS_WEIGHT:-0.24}"
export MOTION_NATURALNESS_TARGET_AROUSAL="${MOTION_NATURALNESS_TARGET_AROUSAL:-True}"
export ADAPTIVE_REFLECT_CONTROL="${ADAPTIVE_REFLECT_CONTROL:-True}"
export REFLECT_WARMUP_START_STEP="${REFLECT_WARMUP_START_STEP:-50}"
export REFLECT_FULL_ON_STEP="${REFLECT_FULL_ON_STEP:-140}"
export REFLECT_CONFLICT_SUPPRESSION="${REFLECT_CONFLICT_SUPPRESSION:-True}"
export REFLECT_SUBTLE_PROMPT_GATING="${REFLECT_SUBTLE_PROMPT_GATING:-True}"
export EMO_REWARD_PLAN_ENABLED="${EMO_REWARD_PLAN_ENABLED:-False}"
export EMO_REWARD_COMBINE_VIDEOALIGN="${EMO_REWARD_COMBINE_VIDEOALIGN:-true}"
# R_total = (alpha*R_vid + beta*R_emo)/(alpha+beta); the emotional reward is on.
export VIDEOALIGN_FUSION_ALPHA="${VIDEOALIGN_FUSION_ALPHA:-0.5}"
export VIDEOALIGN_FUSION_BETA="${VIDEOALIGN_FUSION_BETA:-0.5}"

RESULTS_DIR="${RESULTS_DIR:-SAGE_GRPO_results}"
OUT_DIR="${OUT_DIR:-SAGE-GRPO-logs/paper_draft}"
TRACES="${TRACES:-$OUT_DIR/traces.jsonl}"
TABLE="${TABLE:-$OUT_DIR/jit_gate_table.json}"
OUTCOME="${OUTCOME:-videoalign_overall}"
# RewardMemory tilt hyper-parameters (temperature tau, log-multiplier cap c,
# minimum per-regime support n_min) — see scripts/export_jit_gate_table.py.
ETA="${ETA:-1.0}"; TAU="${TAU:-0.5}"; CAP="${CAP:-1.0}"; MIN_REGIME="${MIN_REGIME:-20}"

# =============================================================================
# Stage 1 — emoreward without RewardMemory
#   Reflective Reward Gating, memory multipliers m_k=1. A short reference run
#   whose rl_samples become the experience RewardMemory is distilled from.
# =============================================================================
STAGE1_STEPS="${STAGE1_STEPS:-100}"
STAGE1_TAG="${STAGE1_TAG:-emoreward_reflect_ref}"

if [ "${SKIP_STAGE1:-0}" != "1" ]; then
  echo "==== [EmoReward] Stage 1/2: emoreward without RewardMemory (${STAGE1_STEPS} steps) ===="
  STAGE1_START="${STAGE1_START:-$(date "+%Y.%m.%d-%H.%M.%S")}"
  STAGE1_DIR="${RESULTS_DIR}/${STAGE1_START}_${STAGE1_TAG}"
  START_EXPR_TIME="$STAGE1_START" \
  EXPERIMENT_TAG="$STAGE1_TAG" \
  MAX_STEPS="$STAGE1_STEPS" \
  SAVE_INTERVAL="$STAGE1_STEPS" \
  REWARD_GATE_POLICY="adaptive" \
  REWARD_GATE_JIT_TABLE="" \
    bash run_post_train_multinode.sh
else
  echo "==== [EmoReward] Stage 1 skipped; reusing reference run ${STAGE1_DIR:?set STAGE1_DIR when SKIP_STAGE1=1} ===="
fi

# =============================================================================
# Distill RewardMemory from the Stage-1 traces (offline, no GPU).
#   rl_samples -> traces.jsonl -> regime-keyed reliability-multiplier table.
# =============================================================================
echo "==== [EmoReward] Evolve: build traces from ${STAGE1_DIR} -> ${TRACES} ===="
python3 scripts/build_reward_traces.py \
    --results_dir "$RESULTS_DIR" --runs "$(basename "$STAGE1_DIR")" \
    --out "$TRACES" --outcome "$OUTCOME"

echo "==== [EmoReward] Evolve: export RewardMemory multiplier table -> ${TABLE} ===="
python3 scripts/export_jit_gate_table.py --traces "$TRACES" --out "$TABLE" \
    --eta "$ETA" --tau "$TAU" --cap "$CAP" --min_regime "$MIN_REGIME"

# =============================================================================
# Stage 2 — emoreward with RewardMemory
#   Reflect gates composed with the frozen memory multipliers m_k(r) from the
#   table above. The full method; a complete training run. If the table is
#   missing the gate falls back to adaptive (= Stage 1 behaviour).
# =============================================================================
STAGE2_STEPS="${STAGE2_STEPS:-10000}"
STAGE2_TAG="${STAGE2_TAG:-emoreward_full}"
echo "==== [EmoReward] Stage 2/2: emoreward with RewardMemory (${STAGE2_STEPS} steps) ===="
EXPERIMENT_TAG="$STAGE2_TAG" \
MAX_STEPS="$STAGE2_STEPS" \
SAVE_INTERVAL="${SAVE_INTERVAL:-100}" \
REWARD_GATE_POLICY="jit" \
REWARD_GATE_JIT_TABLE="$TABLE" \
  bash run_post_train_multinode.sh
