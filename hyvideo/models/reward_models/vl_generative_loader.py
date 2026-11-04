"""
Load Qwen2-VL / Qwen2.5-VL / Qwen3-VL(MoE) for **text generation** (no images).

Used by EmoReward Plan phase and by offline prompt scripts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def load_vl_generative_model(
    model_path: str,
    device: Union[str, torch.device],
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a Qwen*VL generative model + processor by architecture in config.json.

    Args:
        model_path: Absolute directory or HF model id
        device: Used when device_map is None (single-device .to(device))
        torch_dtype: bf16 recommended
        device_map: e.g. "auto" for large MoE on multiple GPUs; None -> .to(device)
        trust_remote_code: must be True for some Qwen3 checkpoints
    """
    import transformers
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    arch_list = getattr(config, "architectures", None) or []
    if not arch_list:
        raise ValueError(f"No architectures in config for {model_path}")
    cls_name = arch_list[0]
    try:
        model_cls = getattr(transformers, cls_name)
    except AttributeError as e:
        raise ImportError(
            f"Transformers missing class {cls_name} for {model_path}. "
            f"Upgrade transformers for Qwen2.5-VL / Qwen3-VL support."
        ) from e

    kwargs: Dict[str, Any] = dict(
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    if device_map:
        kwargs["device_map"] = device_map

    model = model_cls.from_pretrained(model_path, **kwargs)
    model.eval()

    if not device_map:
        dev = torch.device(device) if isinstance(device, str) else device
        model.to(dev)

    return model, processor


def _model_input_device(model, fallback: Union[str, torch.device]) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback) if isinstance(fallback, str) else fallback


def vl_generate_text(
    model,
    processor,
    messages: List[Dict[str, Any]],
    device: Union[str, torch.device],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Run chat-templated text generation (text-only user content)."""
    dev = _model_input_device(model, device)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    in_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[0, in_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()
