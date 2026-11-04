"""
Unified reward interface for video generation models.
Provides a standard interface for all reward models including video and image rewards.
All reward functions are self-contained without external flow_grpo dependencies.
"""
import os
import json
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Union, Callable

# Model base path (override via MODEL_BASE env var for shared filesystem)
_MODEL_BASE = os.getenv("MODEL_BASE", "./ckpts")

# VideoAlign model path: default to local path, override via VIDEOALIGN_MODEL_PATH.
# Fusion scorer uses this; relative "./ckpts/VideoReward" only resolves when cwd is
# the SAGE-GRPO repo root, so cross-repo callers (e.g. flowgrpo emo adapter) must set
# VIDEOALIGN_MODEL_PATH to an absolute path.
_DEFAULT_VIDEOALIGN_PATH = os.getenv("VIDEOALIGN_MODEL_PATH", "./ckpts/VideoReward")
REWARD_MODEL_PATH = {
    "videoalign": _DEFAULT_VIDEOALIGN_PATH,
    "hpsv3": os.path.join(_MODEL_BASE, "HPSv3", "HPSv3.safetensors"),
    # Must match HPSv3 backbone (7B); 2B path causes shape mismatch on lm_head / rm_head.
    # Override with QWEN2_VL_7B_PATH to use a local copy.
    "qwen2_vl_7b": os.getenv("QWEN2_VL_7B_PATH", "Qwen/Qwen2-VL-7B-Instruct"),
}


# ============================================================================
# Video Reward Models
# ============================================================================

def videoalign_local_score(device, reward_checkpoint_mode="none"):
    """Local VideoAlign reward model."""
    from hyvideo.models.reward_models.videoalign.inference import VideoVLMRewardInference
    
    dtype = torch.bfloat16
    scorer = VideoVLMRewardInference(
        REWARD_MODEL_PATH["videoalign"],
        device=device,
        dtype=dtype,
        reward_checkpoint_mode=reward_checkpoint_mode,
    )
    
    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        """
        Args:
            video_paths: List of video file paths
            prompts: List of text prompts
            metadata: Optional metadata for each sample
            
        Returns:
            scores_dict: Dict of {metric_name: List[float]}
            meta_dict: Additional metadata (empty for now)
        """
        if metadata is None:
            metadata = [{}] * len(video_paths)
        
        scores_dict = {}
        all_entries = []
        
        # Batch inference
        for video_path, prompt in zip(video_paths, prompts):
            try:
                reward_entry = scorer.reward([video_path], [prompt])[0]
                all_entries.append(reward_entry)
            except Exception as e:
                # Return NaN on error
                all_entries.append({})
        
        # Collect all metrics
        if all_entries:
            # Get all possible keys from entries
            all_keys = set()
            for entry in all_entries:
                all_keys.update(entry.keys())
            
            # Extract each metric
            for key in all_keys:
                scores = []
                for entry in all_entries:
                    val = entry.get(key)
                    try:
                        scores.append(float(val) if val is not None else float('nan'))
                    except (TypeError, ValueError):
                        scores.append(float('nan'))
                scores_dict[key] = scores
        
        return scores_dict, {}
    
    return _fn


def video_align_remote_score(server_url: str, model: str = "video_align"):
    """
    Implementing your own remote video reward service here if required.
    Default using local video reward service.
    """
    
    pass


def hpsv3_local_score(device, num_frames=4):
    """HPSv3 reward model for video evaluation.
    Extracts key frames from videos and scores each frame with HPSv3,
    then averages per-frame scores as the video-level score.

    Args:
        device: Device to run the model on
        num_frames: Number of frames to extract per video for scoring
    """
    from hyvideo.models.reward_models.hpsv3 import HPSv3RewardInferencer
    import tempfile

    inferencer = HPSv3RewardInferencer(
        checkpoint_path=REWARD_MODEL_PATH["hpsv3"],
        device=device,
        model_name_or_path=REWARD_MODEL_PATH["qwen2_vl_7b"],
    )

    def _extract_video_frames(video_path, n_frames):
        """Extract evenly-spaced frames from a video file."""
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            if total <= 0:
                return []
            indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
            frames_np = vr.get_batch(indices.tolist()).asnumpy()
            return [Image.fromarray(f) for f in frames_np]
        except ImportError:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
            return frames

    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        if metadata is None:
            metadata = [{}] * len(video_paths)

        scores = []
        temp_files = []

        try:
            for video_path, prompt in zip(video_paths, prompts):
                try:
                    if video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        frame_paths = [video_path]
                    else:
                        frames = _extract_video_frames(video_path, num_frames)
                        if not frames:
                            scores.append(float('nan'))
                            continue
                        frame_paths = []
                        for frame in frames:
                            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                            temp_files.append(tf.name)
                            frame.save(tf.name, format='PNG')
                            frame_paths.append(tf.name)

                    frame_prompts = [prompt] * len(frame_paths)
                    rewards = inferencer.reward(frame_prompts, image_paths=frame_paths)
                    frame_scores = [r[0].item() for r in rewards]
                    scores.append(sum(frame_scores) / len(frame_scores))
                except Exception as e:
                    print(f"Warning: Error scoring video {video_path} with HPSv3: {e}")
                    scores.append(float('nan'))
        finally:
            for tf in temp_files:
                try:
                    if os.path.exists(tf):
                        os.remove(tf)
                except Exception:
                    pass

        return scores, {}

    return _fn


# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_key(key: str) -> str:
    """Normalize key to lowercase for case-insensitive lookup."""
    return str(key).strip().lower()


_VIDEO_MODELS = frozenset(["videoalign_local", "video_align", "video_score2", "hpsv3", "emo_reward"])


def _init_reward_metadata(video_paths_or_images, metadata, is_image_input: bool) -> List[Dict]:
    """Initialize metadata list for reward computation."""
    if metadata is not None:
        return metadata
    batch_size = video_paths_or_images.shape[0] if is_image_input else len(video_paths_or_images)
    return [{}] * batch_size


def _should_skip_model(model_name: str, is_image_input: bool) -> bool:
    """Return True if model should be skipped for given input type."""
    is_video_model = model_name in _VIDEO_MODELS
    if is_video_model and is_image_input:
        print(f"Warning: Skipping video model '{model_name}' for image input")
        return True
    if not is_video_model and not is_image_input:
        print(f"Warning: Skipping image model '{model_name}' for video path input")
        return True
    return False


def _store_model_scores(model_name: str, scores, all_scores: Dict, model_raw_scores: Dict) -> None:
    """Store scores from a model into all_scores and model_raw_scores."""
    if isinstance(scores, dict):
        model_raw_scores[model_name] = scores
        for key, values in scores.items():
            prefixed_key = f"{model_name}_{_normalize_key(key)}"
            all_scores[prefixed_key] = values
    elif isinstance(scores, (list, np.ndarray)):
        scores_list = list(scores) if isinstance(scores, np.ndarray) else scores
        model_raw_scores[model_name] = scores_list
        all_scores[model_name] = scores_list
    else:
        raise ValueError(f"Unexpected score format from {model_name}: {type(scores)}")


def _compute_video_model_weighted_scores(raw_scores: Dict, per_model_metric_weights, model_weight: float) -> List[float]:
    """Compute weighted scores for a video model with multiple metrics."""
    if per_model_metric_weights:
        model_metric_scores = []
        for metric_name, weight in per_model_metric_weights.items():
            metric_key = _normalize_key(metric_name)
            metric_values = raw_scores.get(metric_key, raw_scores.get(metric_name))
            if metric_values is not None:
                model_metric_scores.append([weight * s for s in metric_values])
        if model_metric_scores:
            model_sum = [sum(col) for col in zip(*model_metric_scores)]
            return [model_weight * s for s in model_sum]
    else:
        all_metric_values = list(raw_scores.values())
        if all_metric_values:
            model_avg = [sum(col) / len(all_metric_values) for col in zip(*all_metric_values)]
            return [model_weight * s for s in model_avg]
    return None


def _compute_weighted_avg(
    weighted_scores_list: List[List[float]],
    output_scores: Dict,
    video_paths_or_images,
    is_image_input: bool,
) -> None:
    """Compute final weighted average and set output_scores['avg']."""
    if weighted_scores_list:
        output_scores['avg'] = [sum(col) for col in zip(*weighted_scores_list)]
        return
    all_values = [v for k, v in output_scores.items() if k != 'avg']
    if all_values:
        output_scores['avg'] = [sum(col) / len(all_values) for col in zip(*all_values)]
        return
    batch_size = video_paths_or_images.shape[0] if is_image_input else len(video_paths_or_images)
    output_scores['avg'] = [0.0] * batch_size


# ============================================================================
# Multi-Reward Interface
# ============================================================================

def multi_video_score(device, reward_config: Dict[str, Any]):
    """
    Unified multi-reward interface with flexible weighting.
    Only the **standard format** (Example 3) is supported; all other old formats are no longer supported.
    
    Args:
        device: Device to run models on
        reward_config: Dict with configuration
        
            Example:
            {
                "models": {
                    "videoalign_local": {
                        "weight": 1.0,
                        "sub_reward": {"VQ": 0.5, "MQ": 0.5, "TA": 1.0},
                    },
                    "video_align": {"weight": 1.0, "server_url": "..."},
                },
            }
            
            Constraints:
            - Must contain a "models" field.
            - Each model's config must be a dict and must contain at least the "weight" key.
            - The optional "sub_reward" field should be a dictionary mapping metric_name to weight.
            - If there is only one reward model, its weight must be exactly 1.0.
    
    Returns:
        reward_fn: Function (video_paths, prompts, metadata) -> (scores_dict, meta_dict)
    """
    # Must contain "models"
    if "models" not in reward_config:
        raise ValueError(
            "reward_config must contain a 'models' field with the standard format: "
            '{"models": {"model_name": {"weight": float, "sub_reward": {...}}}}'
        )
    
    model_weights = reward_config["models"]
    
    # Parse each model configuration (only standard format is supported)
    # model_configs: {model_name: {"model_weight": float, "metric_weights": dict or None}}
    model_configs: Dict[str, Dict[str, Any]] = {}
    for model_name, weight_config in model_weights.items():
        if not isinstance(weight_config, dict):
            raise ValueError(
                f"Model config for '{model_name}' must be a dict with at least 'weight' key, "
                f"got {type(weight_config)}"
            )
        
        if "weight" not in weight_config:
            raise ValueError(
                f"Model config for '{model_name}' must contain 'weight', "
                f"optionally with 'sub_reward', got keys: {list(weight_config.keys())}"
            )
        
        model_weight = float(weight_config["weight"])
        metric_weights = weight_config.get("sub_reward", None)
        
        if metric_weights is not None and not isinstance(metric_weights, dict):
            raise ValueError(
                f"'sub_reward' for model '{model_name}' must be a dict of metric_name -> weight, "
                f"got {type(metric_weights)}"
            )
        
        model_configs[model_name] = {
            "model_weight": model_weight,
            "metric_weights": metric_weights,
        }
    
    # When only one reward model is used, its weight must be 1.0
    if len(model_configs) == 1:
        only_model_name = next(iter(model_configs.keys()))
        only_model_weight = model_configs[only_model_name]["model_weight"]
        if abs(only_model_weight - 1.0) > 1e-6:
            raise ValueError(
                f"When only one reward model is used ('{only_model_name}'), "
                f"its weight must be 1.0, got {only_model_weight}"
            )
    
    # Initialize all reward functions
    reward_fns = {}
    for model_name in model_configs.keys():
        if model_name == "videoalign_local":
            checkpoint_mode = reward_config.get("reward_checkpoint_mode", "none")
            reward_fns[model_name] = videoalign_local_score(device, checkpoint_mode)
        elif model_name in ["video_align", "video_score2"]:
            server_url = reward_config.get("server_url", reward_config.get("remote_reward_url"))
            if server_url is None:
                raise ValueError(f"server_url required for {model_name}")
            reward_fns[model_name] = video_align_remote_score(server_url, model=model_name)
        elif model_name == "hpsv3":
            reward_fns[model_name] = hpsv3_local_score(device)
        elif model_name == "emo_reward":
            from hyvideo.models.reward_models.emo_reward.react_reward import emo_reward_score
            emo_reward_config = dict(weight_config.get("emo_reward_config", {}))
            if "reward_checkpoint_mode" not in emo_reward_config:
                emo_reward_config["reward_checkpoint_mode"] = reward_config.get("reward_checkpoint_mode", "v3")
            emo_reward_config["_dist_sp_size"] = int(reward_config.get("_dist_sp_size", 1) or 1)
            emo_reward_config["_dist_rank"] = int(reward_config.get("_dist_rank", 0))
            reward_fns[model_name] = emo_reward_score(device, config=emo_reward_config)
        else:
            raise ValueError(
                f"Unknown reward model: {model_name}. "
                "Supported: videoalign_local, video_align, video_score2, hpsv3, emo_reward."
            )
    
    def _fn(video_paths_or_images: Union[List[str], torch.Tensor], prompts: List[str], metadata: List[Dict] = None):
        """
        Compute rewards for videos/images.
        
        Args:
            video_paths_or_images: Either a list of video file paths (for video models)
                                   or a torch.Tensor of images (for image models)
            prompts: List of text prompts
            metadata: Optional metadata for each sample
        
        Weighting logic:
        1. Each video model's metrics are first weighted by their metric weights
        2. Then the weighted sum of each model's metrics is multiplied by model weight
        3. Image models are directly multiplied by their model weight
        4. Final avg = sum of all weighted scores
        
        Returns scores_dict with keys like:
        - "videoalign_local_vq", "videoalign_local_mq", "videoalign_local_ta" (video model metrics)
        - "aesthetic" (image model)
        - "tencent_remote_vq", "tencent_remote_mq" (another video model)
        - "avg" (weighted average)
        """
        is_image_input = isinstance(video_paths_or_images, torch.Tensor)
        metadata = _init_reward_metadata(video_paths_or_images, metadata, is_image_input)

        all_scores = {}
        all_meta = {}
        model_raw_scores = {}

        for model_name in model_configs.keys():
            if _should_skip_model(model_name, is_image_input):
                continue
            reward_fn = reward_fns[model_name]
            scores, meta = reward_fn(video_paths_or_images, prompts, metadata)
            _store_model_scores(model_name, scores, all_scores, model_raw_scores)
            all_meta.update(meta)

        output_scores = all_scores.copy()
        weighted_scores_list = []

        for model_name, config in model_configs.items():
            if model_name not in model_raw_scores:
                continue
            model_weight = config["model_weight"]
            per_model_metric_weights = config["metric_weights"]
            raw_scores = model_raw_scores[model_name]

            if isinstance(raw_scores, dict):
                weighted = _compute_video_model_weighted_scores(
                    raw_scores, per_model_metric_weights, model_weight
                )
                if weighted is not None:
                    weighted_scores_list.append(weighted)
            else:
                weighted_scores_list.append([model_weight * s for s in raw_scores])

        _compute_weighted_avg(
            weighted_scores_list, output_scores, video_paths_or_images, is_image_input
        )
        assert 'avg' in output_scores.keys(), f"avg not in output_scores: {output_scores.keys()}"
        return output_scores, all_meta

    return _fn


def get_reward_fn(args, device, logger):
    """
    Factory function to create reward function from args.
    
    Args:
        args: Training arguments
        device: Device
        logger: Logger
        
    Returns:
        reward_fn: Unified reward function
    """
    if hasattr(args, "public_vlm_root") and getattr(args, "public_vlm_root", None):
        os.environ["PUBLIC_VLM_ROOT"] = str(args.public_vlm_root)
    if hasattr(args, "videoalign_vlm_preset") and getattr(args, "videoalign_vlm_preset", None):
        os.environ["VIDEOALIGN_VLM_PRESET"] = str(args.videoalign_vlm_preset)

    def _apply_args_to_emo_reward_config(rc: Dict[str, Any]) -> None:
        models = rc.get("models")
        if not isinstance(models, dict):
            return
        entry = models.get("emo_reward")
        if not isinstance(entry, dict):
            return
        emo_cfg = dict(entry.get("emo_reward_config") or {})
        if getattr(args, "emo_reward_vlm_preset", None):
            emo_cfg["emo_reward_plan_model_path"] = str(args.emo_reward_vlm_preset)
        if hasattr(args, "emo_reward_plan_enabled"):
            emo_cfg["emo_reward_plan_enabled"] = bool(getattr(args, "emo_reward_plan_enabled"))
        if hasattr(args, "emo_reward_plan_backend"):
            emo_cfg["emo_reward_plan_backend"] = str(getattr(args, "emo_reward_plan_backend"))
        if hasattr(args, "emo_reward_plan_gemini_model"):
            emo_cfg["emo_reward_plan_gemini_model"] = str(getattr(args, "emo_reward_plan_gemini_model"))
        if hasattr(args, "reward_checkpoint_mode") and getattr(args, "reward_checkpoint_mode", None):
            emo_cfg.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)
        if hasattr(args, "emo_reward_combine_videoalign"):
            emo_cfg["combine_videoalign"] = bool(args.emo_reward_combine_videoalign)
        if hasattr(args, "videoalign_fusion_alpha"):
            emo_cfg["videoalign_fusion_alpha"] = float(args.videoalign_fusion_alpha)
        if hasattr(args, "videoalign_fusion_beta"):
            emo_cfg["videoalign_fusion_beta"] = float(args.videoalign_fusion_beta)
        # Emotion scorer and channel controls
        for _attr in (
            "enable_emotion_scorers",
            "clip_autt_weight",
            "r1omni_weight",
            "r1omni_vad_mode",
            "enable_chromatic",
            "enable_kinetic",
            "enable_motion_naturalness",
            "motion_naturalness_weight",
            "motion_naturalness_target_arousal",
            "adaptive_reflect_control",
            "reflect_warmup_start_step",
            "reflect_full_on_step",
            "reflect_conflict_suppression",
            "reflect_subtle_prompt_gating",
            "emo_reward_plan_enabled",
            "emo_reward_plan_backend",
            "emo_reward_plan_gemini_model",
        ):
            if hasattr(args, _attr):
                _val = getattr(args, _attr)
                if isinstance(_val, bool):
                    emo_cfg[_attr] = bool(_val)
                elif isinstance(_val, float):
                    emo_cfg[_attr] = float(_val)
                else:
                    emo_cfg[_attr] = _val
        entry["emo_reward_config"] = emo_cfg

    # External config has highest priority
    if hasattr(args, "reward_config") and isinstance(args.reward_config, dict) and args.reward_config:
        reward_config = args.reward_config.copy()
        # Add reward_checkpoint_mode if not in config
        if hasattr(args, "reward_checkpoint_mode"):
            reward_config.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)
        if hasattr(args, "remote_reward_url"):
            reward_config.setdefault("server_url", args.remote_reward_url)
        logger.info(f"Using external reward_config: {reward_config}")
    else:
        # Fallback: Build reward_config from reward_model and optional reward_weights
        logger.warning("reward_config not provided, falling back to default config with reward_model only")
        
        reward_model = getattr(args, "reward_model", "videoalign_local")
        if reward_model == "auto":
            reward_model = "videoalign_local"
        
        # Models that output multiple sub-metrics (VQ, MQ, TA)
        _multi_metric_models = {"videoalign_local", "video_align", "video_score2"}
        _va_style_weights = frozenset({"VQ", "MQ", "TA", "Overall", "overall"})

        # Use reward_weights from args if provided (e.g. from post_train --reward_weights), else default
        rw = getattr(args, "reward_weights", None)
        if reward_model in _multi_metric_models:
            if isinstance(rw, dict) and rw:
                sub_reward = rw
            elif isinstance(rw, str) and rw.strip():
                try:
                    parsed = json.loads(rw.strip())
                    sub_reward = parsed if isinstance(parsed, dict) and parsed else {"VQ": 0.5, "MQ": 0.5, "TA": 0.5}
                except (json.JSONDecodeError, TypeError):
                    sub_reward = {"VQ": 0.5, "MQ": 0.5, "TA": 0.5}
            else:
                sub_reward = {"VQ": 0.5, "MQ": 0.5, "TA": 0.5}
        elif reward_model == "emo_reward":
            if isinstance(rw, dict) and rw:
                sub_reward = (
                    {"emo_total": 1.0}
                    if set(rw.keys()) <= _va_style_weights
                    else rw
                )
            elif isinstance(rw, str) and rw.strip():
                try:
                    parsed = json.loads(rw.strip())
                    if isinstance(parsed, dict) and parsed:
                        sub_reward = (
                            {"emo_total": 1.0}
                            if set(parsed.keys()) <= _va_style_weights
                            else parsed
                        )
                    else:
                        sub_reward = {"emo_total": 1.0}
                except (json.JSONDecodeError, TypeError):
                    sub_reward = {"emo_total": 1.0}
            else:
                sub_reward = {"emo_total": 1.0}
        else:
            sub_reward = None
        
        # Build standard format reward_config
        model_entry = {"weight": 1.0}
        if sub_reward is not None:
            model_entry["sub_reward"] = sub_reward
        reward_config = {
            "models": {
                reward_model: model_entry
            }
        }
        
        # Model-specific config
        if reward_model in ["video_align", "video_score2"]:
            reward_config["server_url"] = getattr(args, "remote_reward_url", None)
        elif reward_model == "videoalign_local":
            reward_config["reward_checkpoint_mode"] = getattr(args, "reward_checkpoint_mode", "v2")
            # reward_config["reward_checkpoint_mode"] = "v3"
        elif reward_model == "emo_reward":
            emo_reward_cfg: Dict[str, Any] = {
                "reward_checkpoint_mode": getattr(args, "reward_checkpoint_mode", "v3"),
                "combine_videoalign": bool(getattr(args, "emo_reward_combine_videoalign", True)),
                "videoalign_fusion_alpha": float(getattr(args, "videoalign_fusion_alpha", 0.5)),
                "videoalign_fusion_beta": float(getattr(args, "videoalign_fusion_beta", 0.5)),
                "enable_motion_naturalness": bool(getattr(args, "enable_motion_naturalness", True)),
                "motion_naturalness_weight": float(getattr(args, "motion_naturalness_weight", 0.20)),
                "motion_naturalness_target_arousal": bool(getattr(args, "motion_naturalness_target_arousal", True)),
                "adaptive_reflect_control": bool(getattr(args, "adaptive_reflect_control", True)),
                "reflect_warmup_start_step": int(getattr(args, "reflect_warmup_start_step", 40)),
                "reflect_full_on_step": int(getattr(args, "reflect_full_on_step", 100)),
                "reflect_conflict_suppression": bool(getattr(args, "reflect_conflict_suppression", True)),
                "reflect_subtle_prompt_gating": bool(getattr(args, "reflect_subtle_prompt_gating", True)),
                "emo_reward_plan_enabled": bool(getattr(args, "emo_reward_plan_enabled", False)),
                "emo_reward_plan_backend": str(getattr(args, "emo_reward_plan_backend", "gemini")),
                "emo_reward_plan_gemini_model": str(getattr(args, "emo_reward_plan_gemini_model", "gemini-2.5-flash")),
            }
            if getattr(args, "emo_reward_vlm_preset", None):
                emo_reward_cfg["emo_reward_plan_model_path"] = str(args.emo_reward_vlm_preset)
            model_entry["emo_reward_config"] = emo_reward_cfg
        
        if hasattr(args, "remote_reward_url"):
            reward_config.setdefault("server_url", args.remote_reward_url)
        if hasattr(args, "reward_checkpoint_mode"):
            reward_config.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)
    
    _apply_args_to_emo_reward_config(reward_config)

    # EmoReward: skip heavy Qwen2-VL / VideoReward init on SP non-leader ranks
    # (rewards are only computed when rank % sp_size == 0; see grpo_utils.prepare_samples_online).
    reward_config = dict(reward_config)
    reward_config["_dist_sp_size"] = int(getattr(args, "sp_size", 1) or 1)
    if dist.is_initialized():
        reward_config["_dist_rank"] = int(dist.get_rank())
    else:
        reward_config["_dist_rank"] = 0

    logger.info(f"Creating reward function with config: {reward_config}")
    
    return multi_video_score(device, reward_config)


def create_reward_fn_from_config(config: Dict[str, Any], device, logger=None):
    """
    Create reward function from config dict.
    
    Args:
        config: Reward configuration
        device: Device
        logger: Optional logger
        
    Returns:
        reward_fn: Unified reward function
    """
    if logger:
        logger.info(f"Creating reward function from config: {config}")

    config = dict(config)
    config.setdefault("_dist_sp_size", 1)
    config.setdefault("_dist_rank", int(dist.get_rank()) if dist.is_initialized() else 0)

    return multi_video_score(device, config)
