#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# General video reward. Base-model downloads follow checkpoints-download.md.
hf download KlingTeam/VideoReward --local-dir ./ckpts/VideoReward

# Optional HPSv3/COHP image-generation model; not required for video training.
if [[ "${DOWNLOAD_FLUX_SCHNELL:-0}" == "1" ]]; then
  hf download black-forest-labs/FLUX.1-schnell --local-dir ./ckpts/FLUX.1-schnell
fi

echo "See checkpoints-download.md for base-model and optional checkpoint setup."
