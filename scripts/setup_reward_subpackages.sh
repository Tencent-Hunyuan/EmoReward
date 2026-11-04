#!/usr/bin/env bash
# =============================================================================
# EmoReward — reward sub-package setup
#
# The EmoReward reward model (hyvideo/models/reward_models/<emo_reward>) drives four
# heterogeneous emotional-reward providers that live as *external* code packages
# inside the reward-module directory. This repo tracks the glue code only; the
# provider packages (and their weights) are fetched separately so the repo stays clean.
#
#   CLIP-AUTT                    facial AU / valence primitive   (CLIP ViT-B/32)
#   R1-Omni                      semantic emotion + VAD          (HumanOmni-0.5B)
#   OpenFace-3.0                 AU / gaze / landmark extraction
#   facial-landmarks-recognition dlib 68-pt landmarks (used by the OpenFace pipeline)
#
# Two modes:
#   (1) CLONE (default): git clone each upstream at its pinned commit.
#   (2) COPY  : one-key copy from an existing local SAGE-GRPO / EmoReward checkout,
#               code only (weights skipped; download per REWARD_SUBPACKAGES.md):
#                 SAGE_GRPO_SRC=/path/to/SAGE-GRPO bash scripts/setup_reward_subpackages.sh
#
# Weights are NEVER handled here — see REWARD_SUBPACKAGES.md for weight downloads.
# Providers degrade gracefully: a provider whose weights are absent is dropped and
# its fusion weight is redistributed, so the pipeline still runs for a smoke test.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REWARD_ROOT="$REPO_ROOT/hyvideo/models/reward_models"

# Resolve the reward-module directory name (open-source layout uses emo_reward).
detect_module_dir() {
  local root="$1"
  if [ -d "$root/emo_reward" ]; then echo "emo_reward"; return 0; fi
  echo "emo_reward"   # default for a fresh layout
}
MODULE_NAME="$(detect_module_dir "$REWARD_ROOT")"
MODULE_DIR="$REWARD_ROOT/$MODULE_NAME"

# name|upstream|pinned-commit   (the four providers actually called by the code)
SUBPKGS=(
  "CLIP-AUTT|https://github.com/osamazeeshan/CLIP-AUTT.git|595a3d16f3a2d1e1215ecdb0b39732123dd7bdcb"
  "R1-Omni|https://github.com/HumanMLLM/R1-Omni.git|17cafcae0b0a1c454896f4eb7a1c006bf13e1f4d"
  "OpenFace-3.0|https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git|662a555a8566ae3aec8139cb8c72acf6d06e0eb3"
  "facial-landmarks-recognition|https://github.com/italojs/facial-landmarks-recognition.git|66c562fc6c51634bc18a02d123a7589d4fbde2ef"
)

echo "== EmoReward reward sub-package setup =="
echo "   module dir : $MODULE_DIR"
mkdir -p "$MODULE_DIR"

if [ -n "${SAGE_GRPO_SRC:-}" ]; then
  SRC_MODULE_NAME="$(detect_module_dir "$SAGE_GRPO_SRC/hyvideo/models/reward_models")"
  SRC_MODULE_DIR="$SAGE_GRPO_SRC/hyvideo/models/reward_models/$SRC_MODULE_NAME"
  echo "   copy source: $SRC_MODULE_DIR (code only)"
fi

for spec in "${SUBPKGS[@]}"; do
  name="${spec%%|*}"; rest="${spec#*|}"; url="${rest%%|*}"; sha="${rest##*|}"
  dst="$MODULE_DIR/$name"

  if [ -d "$dst/.git" ] || [ -n "$(ls -A "$dst" 2>/dev/null || true)" ]; then
    echo "-- $name : already present, skipping"
    continue
  fi

  if [ -n "${SAGE_GRPO_SRC:-}" ]; then
    # COPY mode: pull code from a local checkout, excluding weights and .git.
    src="$SRC_MODULE_DIR/$name"
    if [ ! -d "$src" ]; then
      echo "-- $name : NOT found under $src — skipping" >&2
      continue
    fi
    echo "-- $name : copying code from $src (weights excluded)"
    mkdir -p "$dst"
    rsync -a \
      --exclude='.git' \
      --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.safetensors' \
      --exclude='*.bin' --exclude='*.dat' --exclude='*.tar' --exclude='*.pkl' \
      --exclude='WeightFiles' --exclude='weights' \
      --exclude='R1-Omni-0.5B' --exclude='whisper-large-v3' \
      --exclude='siglip-base-patch16-224' --exclude='AI-ModelScope' \
      "$src/" "$dst/"
  else
    # CLONE mode: fetch upstream at the pinned commit.
    echo "-- $name : cloning $url @ ${sha:0:8}"
    git clone --quiet "$url" "$dst"
    git -C "$dst" checkout --quiet "$sha" || \
      echo "   (warning: pinned commit $sha not found; using default branch HEAD)" >&2
  fi
done

echo ""
echo "== Done. Sub-package CODE is in place. =="
echo "   Next: download the provider WEIGHTS following REWARD_SUBPACKAGES.md,"
echo "   and the base checkpoints following checkpoints-download.md."
