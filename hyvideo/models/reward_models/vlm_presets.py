"""
Shared paths / aliases for vision-language backbones used by EmoReward (plan)
and optionally VideoAlign (reward backbone via env / model_config).

By default the presets resolve to public Hugging Face repo ids, so a fresh
checkout works without any shared filesystem. Point ``PUBLIC_VLM_ROOT`` at a
local directory of pre-downloaded weights to use that instead (each preset is
then resolved as ``$PUBLIC_VLM_ROOT/<model-dir>``).
"""

import os
from typing import Dict

# Empty by default -> presets fall back to Hugging Face repo ids below.
PUBLIC_VLM_ROOT = os.getenv("PUBLIC_VLM_ROOT", "").strip()

# Public Hugging Face repo ids (used when PUBLIC_VLM_ROOT is unset).
_HF_IDS: Dict[str, str] = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3-vl-30b-a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3-vl-30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen2-vl-2b-nj8": "Qwen/Qwen2-VL-2B-Instruct",
}


def _resolve_preset(key: str) -> str:
    """HF id by default, or $PUBLIC_VLM_ROOT/<model-dir> when the root is set."""
    hf_id = _HF_IDS[key]
    if PUBLIC_VLM_ROOT:
        return os.path.join(PUBLIC_VLM_ROOT, hf_id.split("/", 1)[-1])
    return hf_id


# Aliases (use in emo_reward_config / CLI). Values are repo ids or absolute dirs.
VLM_PATH_PRESETS: Dict[str, str] = {key: _resolve_preset(key) for key in _HF_IDS}


def resolve_vlm_path(path_or_preset: str) -> str:
    """
    Resolve preset name to a path / HF id, or pass through existing paths / HF ids.

    HF ids (e.g. 'Qwen/Qwen2-VL-2B-Instruct') are returned unchanged.
    """
    if path_or_preset is None:
        return VLM_PATH_PRESETS["qwen2-vl-2b"]
    key = str(path_or_preset).strip()
    if key in VLM_PATH_PRESETS:
        return VLM_PATH_PRESETS[key]
    return key

