"""
emotion_scorers.py — Lazy-loading emotion understanding model wrappers for SAGE-GRPO Reflect phase.

Models integrated:
  1. CLIP-AUTT  : CLIP ViT-B/32 + text_adapter/temporal trained on BioVid (pain/neutral)
                  → emotion AU similarity score, class probability
  2. R1-Omni   : HumanOmni-0.5B multimodal, VLM emotion description + valence/arousal estimation
                  → parsed emotion label, valence score [0,1], arousal score [0,1]

Both scorers return a unified EmotionScoreResult dataclass that feeds into ReflectionRewarder.

Usage:
    from emotion_scorers import ClipAuttScorer, R1OmniScorer, EmotionScoreResult
    scorer = ClipAuttScorer(checkpoint_path="...", clip_autt_root="...")
    result = scorer.score_video(video_path)
    print(result.class_probs, result.au_similarity_mean)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmotionScoreResult:
    """Unified output from any emotion scorer."""
    scorer_name: str
    video_path: str
    # Classification
    predicted_class: str = "unknown"
    class_probs: Dict[str, float] = field(default_factory=dict)
    # Continuous VAD
    valence: float = 0.5          # [0=negative, 1=positive]
    arousal: float = 0.5          # [0=calm, 1=excited]
    # AU similarity (CLIP-AUTT specific)
    au_similarity_mean: float = 0.5
    au_similarity: Dict[str, float] = field(default_factory=dict)
    # Text output (R1-Omni specific)
    description: str = ""
    # Combined score for reward weighting [0,1]
    emotion_reward: float = 0.5
    # Whether this scorer produced valid output (False = fallback 0.5)
    active: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# CLIP-AUTT Scorer
# ---------------------------------------------------------------------------

_AU_PROMPTS_47 = [
    "inner_eyebrows_raised",
    "outer_eyebrows_raised",
    "eyebrows_lowered_and_pulled_together",
    "upper_eyelids_lifted",
    "cheeks_raised_and_eye_corners_tightened",
    "eyelids_tightened",
    "nose_wrinkled",
    "upper_lip_lifted",
    "nasolabial_folds_deepened",
    "lip_corners_pulled_up_in_a_smile",
    "cheeks_puffed_upward",
    "cheek_dimples_formed_lip_corners_tightened_laterally",
    "lip_corners_pulled_down",
    "lower_lip_pulled_down",
    "chin_pushed_up_lower_lip_bulged",
    "lips_puckered_forward",
    "lips_stretched_horizontally",
    "lips_funneled_rounded_and_protruded",
    "lips_pressed_together_and_tightened",
    "lips_strongly_pressed",
    "lips_slightly_parted",
    "jaw_lowered_mouth_opened",
    "mouth_widely_stretched_open",
    "lips_sucked_into_mouth",
    "upper_eyelids_drooping",
    "eyes_slitted_narrow_eye_opening",
    "eyes_fully_closed",
    "squinting_lower_and_upper_lids_compressed",
    "blink_rapid_eyelid_closure",
    "one_eye_closed_wink",
    "head_turn_left",
    "head_turn_right",
    "head_tilt_up",
    "head_tilt_down",
    "eyes_turn_left",
    "eyes_turn_right",
    "eyes_look_up",
    "eyes_look_down",
    "jaw_thrust_forward",
    "jaw_clenched",
    "lip_bite",
    "lip_wipe_with_tongue",
    "lick_lips",
    "cheek_blow",
    "nostrils_dilated",
    "nostrils_compressed",
    "tongue_show",
]


def _add_to_sys_path(root: str) -> None:
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_clip_custom_clip(clip_autt_root: str):
    """
    Import ``CLIP-AUTT/clip/custom_clip.py`` while avoiding collision with
    the system-level ``clip`` package (OpenAI CLIP) **and** any unrelated
    ``data`` module already cached in ``sys.modules``.

    Two pollution sources we must neutralise:

    1. ``sys.modules["clip"]`` — OpenAI CLIP (``import clip``) is often
       imported first by other dependencies, so a plain
       ``from clip.custom_clip import get_coop`` resolves against the
       *system* ``clip`` which has no ``custom_clip`` submodule.

    2. ``sys.modules["data"]`` — inside CLIP-AUTT's ``clip/custom_clip.py``
       there are top-level imports ``from data.biovid_prompts import ...``
       and ``from data.bah_prompts import ...`` that expect the sibling
       ``CLIP-AUTT/data/`` package. But many training stacks register an
       unrelated ``data`` module first (e.g. ``OpenFace-3.0/Pytorch_Retinaface/*``
       uses ``from data.config import cfg_mnet`` and caches a non-package
       ``data`` under that key). Python's import machinery then returns
       that cached module instead of traversing ``sys.path`` — producing
       ``ModuleNotFoundError: No module named 'data.biovid_prompts';
       'data' is not a package``.

       This exact failure was observed in the 2026.05.02-23.58 training
       run after the stdlib→loguru logging bridge exposed the previously
       silent traceback.

    Fix strategy (idempotent, restores state on exit):
      * snapshot ``sys.modules["clip"]`` and ``sys.modules["data"]`` plus
        all their children
      * evict the polluted entries, insert ``CLIP-AUTT/`` on ``sys.path``
      * force-load the local ``clip`` package via spec_from_file_location
      * let Python resolve ``data.*`` via sys.path (CLIP-AUTT/data/*.py)
      * in ``finally``: stash the local clip+data modules under private
        prefixes (so the loaded model object still has live references),
        then restore every entry we saved at the start.
    """
    import importlib

    clip_dir = os.path.join(clip_autt_root, "clip")
    clip_init = os.path.join(clip_dir, "__init__.py")
    if not os.path.isfile(clip_init):
        raise ImportError(f"CLIP-AUTT clip package not found at {clip_init}")

    data_dir = os.path.join(clip_autt_root, "data")
    data_init = os.path.join(data_dir, "__init__.py")
    # data/ should have an __init__.py in the CLIP-AUTT repo; if it doesn't,
    # this was missed during the repo copy and Python still treats it as a
    # namespace package — but many callers (custom_clip.py's ``from data.xxx
    # import ...``) rely on regular-package semantics, so warn loudly.
    if not os.path.isfile(data_init):
        logger.debug(
            "[ClipAutt] %s has no __init__.py; relying on namespace-package "
            "semantics (may break 'from data.xxx import ...' if another "
            "non-package 'data' module is cached in sys.modules).",
            data_dir,
        )

    # --- snapshot what we need to restore ---
    saved_clip = sys.modules.get("clip")
    saved_clip_subs = {k: sys.modules[k] for k in list(sys.modules) if k.startswith("clip.")}
    saved_data = sys.modules.get("data")
    saved_data_subs = {k: sys.modules[k] for k in list(sys.modules) if k.startswith("data.")}
    path_inserted = clip_autt_root not in sys.path
    # Initialize BEFORE the try block so the finally clause cannot raise
    # NameError. Previously this was only bound inside the try, so any
    # ImportError before line "_local_clip_modules = ..." would trigger a
    # NameError in finally that masked the real exception.
    _local_clip_modules: Dict[str, Any] = {}
    _local_data_modules: Dict[str, Any] = {}

    try:
        # 1. Put CLIP-AUTT root on sys.path (for ``from data.xxx import ...``)
        if path_inserted:
            sys.path.insert(0, clip_autt_root)

        # 2. Evict cached system ``clip`` and ``data`` (and all children),
        #    forcing the upcoming import to traverse sys.path and find our
        #    local CLIP-AUTT/clip and CLIP-AUTT/data packages.
        sys.modules.pop("clip", None)
        for k in list(sys.modules):
            if k.startswith("clip."):
                sys.modules.pop(k, None)
        sys.modules.pop("data", None)
        for k in list(sys.modules):
            if k.startswith("data."):
                sys.modules.pop(k, None)

        # 3. Force-import the local clip package from CLIP-AUTT/clip/.
        #    We use spec_from_file_location to guarantee this binds to the
        #    file on disk rather than re-resolving "clip" against sys.path
        #    (which would still work here, but being explicit is safer).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "clip", clip_init,
            submodule_search_locations=[clip_dir],
        )
        local_clip = importlib.util.module_from_spec(spec)
        sys.modules["clip"] = local_clip
        spec.loader.exec_module(local_clip)
        # Note: ``spec.loader.exec_module`` runs custom_clip.py's top-level
        # ``from data.biovid_prompts import ...``, which relies on having
        # ``clip_autt_root`` on sys.path AND no polluted ``data`` module
        # — both conditions we established in steps 1-2.

        # 4. Now import custom_clip — it will find the local clip + data packages
        from clip.custom_clip import get_coop as _get_coop

        # Stash everything the loaded model may reference at runtime.
        _local_clip_modules = {k: sys.modules[k] for k in list(sys.modules)
                               if k == "clip" or k.startswith("clip.")}
        _local_data_modules = {k: sys.modules[k] for k in list(sys.modules)
                               if k == "data" or k.startswith("data.")}

    finally:
        # 5. Restore sys.modules — but we must keep the local clip modules
        #    alive because the loaded model object references them at runtime.
        #    We store them under a private prefix so they don't shadow system
        #    clip / data.
        for k, mod in _local_clip_modules.items():
            private_key = k.replace("clip", "_clip_autt_local", 1)
            sys.modules[private_key] = mod
        for k, mod in _local_data_modules.items():
            private_key = k.replace("data", "_clip_autt_local_data", 1)
            sys.modules[private_key] = mod

        # Restore pre-existing ``clip`` state
        if saved_clip is not None:
            sys.modules["clip"] = saved_clip
        else:
            sys.modules.pop("clip", None)
        for k in list(sys.modules):
            if k.startswith("clip.") and k not in saved_clip_subs:
                sys.modules.pop(k, None)
        sys.modules.update(saved_clip_subs)

        # Restore pre-existing ``data`` state (this is the critical 2026-05-03 fix)
        if saved_data is not None:
            sys.modules["data"] = saved_data
        else:
            sys.modules.pop("data", None)
        for k in list(sys.modules):
            if k.startswith("data.") and k not in saved_data_subs:
                sys.modules.pop(k, None)
        sys.modules.update(saved_data_subs)

        # Remove CLIP-AUTT root from sys.path if we added it
        if path_inserted and clip_autt_root in sys.path:
            sys.path.remove(clip_autt_root)

    return _get_coop


def _extract_frames(video_path: str, seq_len: int = 16, frame_stride: int = 4) -> List:
    """Extract PIL frames from video using decord or cv2."""
    from PIL import Image

    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = list(range(0, min(total, seq_len * frame_stride), frame_stride))[:seq_len]
        if len(indices) < seq_len:
            indices += [indices[-1]] * (seq_len - len(indices))
        frames_np = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames_np]
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, min(total, seq_len * frame_stride), frame_stride))[:seq_len]
        if len(indices) < seq_len:
            indices += [indices[-1]] * (seq_len - len(indices))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        while len(frames) < seq_len:
            frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
        return frames[:seq_len]


def _build_clip_video_tensor(frames: List, resolution: int = 224):
    """Build [T, 3, H, W] float32 CLIP-normalized tensor."""
    import torch
    import torchvision.transforms as T
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        from PIL import Image
        BICUBIC = Image.BICUBIC

    tfm = T.Compose([
        T.Resize(resolution, interpolation=BICUBIC),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return torch.stack([tfm(f) for f in frames])


class ClipAuttScorer:
    """
    CLIP-AUTT emotion scorer.

    Loads CLIP ViT-B/32 + BioVid-trained text adapter + temporal classifier.
    Scores a video clip for pain/neutral expression and per-AU similarity.

    The checkpoint trains for pain-vs-neutral domain, but AU similarity vectors
    are expression-agnostic and serve as a proxy for emotional expressiveness.

    Score mapping for Reflect reward:
      - emotion_reward = mean_au_similarity_across_pain_related_aus
        (higher = more expressive, regardless of specific label)
      - class_probs["pain"] treated as proxy for negative-valence intensity
    """

    # Class-level sentinel: emit exactly one INFO log line on the first
    # successful score_video() call per process. Without this we either
    # see nothing (previous behaviour, since the warning path is silent)
    # or we'd spam one line per sample. A single confirmation at training
    # start is enough to verify the scorer is live.
    _FIRST_SUCCESS_LOGGED = False

    def __init__(
        self,
        checkpoint_path: str,
        clip_autt_root: str,
        arch: str = "ViT-B/32",
        seq_len: int = 16,
        frame_stride: int = 4,
        resolution: int = 224,
        classnames: Tuple[str, ...] = ("neutral", "pain"),
        num_aus: int = 46,
        gpu: int = 0,
    ):
        self.checkpoint_path = checkpoint_path
        self.clip_autt_root = clip_autt_root
        self.arch = arch
        self.seq_len = seq_len
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.classnames = list(classnames)
        self.num_aus = num_aus
        self.gpu = gpu
        self._model = None
        self._device = None
        self._au_prompts = _AU_PROMPTS_47[:num_aus]

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        get_coop = _import_clip_custom_clip(self.clip_autt_root)

        self._device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        model = get_coop(
            self.arch,
            "biosub0",
            self.gpu,
            n_ctx=16,
            ctx_init=None,
            num_aus=len(self._au_prompts),
            num_classes=len(self.classnames),
            au_prompts=self._au_prompts,
            is_video_clip=True,
            frame_stride=self.frame_stride,
            key_frames=self.seq_len,
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self._device)
        if "text_adapter" in ckpt:
            model.text_adapter.load_state_dict(ckpt["text_adapter"])
        if "temporal_classifier" in ckpt:
            model.temporal_classifier.load_state_dict(ckpt["temporal_classifier"])
        if "temporal" in ckpt:
            model.temporal.load_state_dict(ckpt["temporal"])
        if "temporal_proj" in ckpt:
            model.temporal_proj.load_state_dict(ckpt["temporal_proj"])
        model = model.to(self._device)
        model.eval()
        self._model = model
        logger.info("[ClipAutt] Model loaded from %s", self.checkpoint_path)

    def score_video(self, video_path: str) -> EmotionScoreResult:
        import torch
        import torch.nn.functional as F

        result = EmotionScoreResult(scorer_name="clip_autt", video_path=video_path)
        try:
            # Flush any cached allocations held by prior reward sub-models
            # (VideoAlign / Qwen2-VL / R1-Omni subprocess) before we load CLIP
            # and allocate new tensors. This is a cheap no-op on the happy
            # path; on VRAM-constrained SP ranks it can prevent an OOM.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            self._load()
            frames = _extract_frames(video_path, self.seq_len, self.frame_stride)
            tensor = _build_clip_video_tensor(frames, self.resolution)
            tensor = tensor.unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits, au_sim, _ = self._model.temporal_clip_au_forward(
                    tensor, self._au_prompts, class_prompts=None, adapt_tar=False,
                )

            probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_cls = self.classnames[int(torch.argmax(logits, dim=-1).item())]
            au_sim_list = au_sim[0].cpu().tolist()

            result.predicted_class = pred_cls
            result.class_probs = {cn: round(p, 5) for cn, p in zip(self.classnames, probs)}
            result.au_similarity = {
                self._au_prompts[i]: round(float(au_sim_list[i]), 4)
                for i in range(len(self._au_prompts))
            }
            result.au_similarity_mean = float(np.mean(au_sim_list))

            # Derive valence: pain→negative (0.15), neutral→positive (0.75)
            pain_prob = result.class_probs.get("pain", 0.0)
            result.valence = 1.0 - pain_prob          # pain is negative-valence
            result.arousal = min(1.0, pain_prob * 1.5 + 0.2)  # pain correlates with high arousal

            # emotion_reward: map cosine similarity [-1, 1] → [0, 1] linearly.
            # np.clip(x, 0, 1) would destroy signal when au_sim < 0 (e.g. no-face videos).
            # Linear rescaling preserves relative ordering even when all values are negative.
            result.emotion_reward = float((result.au_similarity_mean + 1.0) / 2.0)
            result.active = True

            # One-shot INFO to confirm CLIP-AUTT is actually producing signal.
            # Bridged to loguru via the stdlib→loguru InterceptHandler installed
            # at the top of post_train.py, so this lands in train.log.
            if not ClipAuttScorer._FIRST_SUCCESS_LOGGED:
                logger.info(
                    "[ClipAutt] First successful score: video=%s pred=%s "
                    "au_sim=%.4f emotion_reward=%.4f "
                    "(subsequent scores suppressed to avoid log spam)",
                    os.path.basename(video_path), pred_cls,
                    result.au_similarity_mean, result.emotion_reward,
                )
                ClipAuttScorer._FIRST_SUCCESS_LOGGED = True
        except Exception as exc:
            import traceback
            result.error = str(exc)
            # Emit the full traceback, not just str(exc). Previously we only
            # logged ``str(exc)``, and even that was being dropped because
            # stdlib logging wasn't bridged to loguru — which is exactly why
            # CLIP-AUTT failed silently in 11 training runs.
            logger.warning(
                "[ClipAutt] score_video FAILED for %s: %s\n%s",
                video_path, exc, traceback.format_exc(),
            )
        return result

    def score_videos(self, video_paths: List[str]) -> List[EmotionScoreResult]:
        return [self.score_video(vp) for vp in video_paths]

    def unload(self) -> None:
        self._model = None
        import torch
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# R1-Omni Scorer
# ---------------------------------------------------------------------------

# Emotion→VAD mapping for R1-Omni parsed labels
_LABEL_TO_VAD: Dict[str, Tuple[float, float]] = {
    "happy": (0.85, 0.70),
    "happiness": (0.85, 0.70),
    "joy": (0.90, 0.80),
    "excited": (0.80, 0.90),
    "surprise": (0.60, 0.85),
    "surprised": (0.60, 0.85),
    "neutral": (0.50, 0.30),
    "calm": (0.55, 0.15),
    "sad": (0.15, 0.30),
    "sadness": (0.15, 0.30),
    "fear": (0.10, 0.80),
    "fearful": (0.10, 0.80),
    "anger": (0.10, 0.85),
    "angry": (0.10, 0.85),
    "disgust": (0.10, 0.60),
    "contempt": (0.20, 0.40),
    "pain": (0.15, 0.75),
    "distress": (0.10, 0.80),
}


def _parse_r1omni_output(text: str) -> Tuple[str, float, float]:
    """
    Parse R1-Omni free-text output into (label, valence, arousal).
    Uses simple keyword search on emotion vocabulary.
    Returns (label, valence, arousal) with fallback to (neutral, 0.5, 0.5).
    """
    text_lower = text.lower()
    best_label = "neutral"
    best_count = 0
    for label in _LABEL_TO_VAD:
        count = text_lower.count(label)
        if count > best_count:
            best_count = count
            best_label = label

    # Parse explicit "valence: X.X" / "arousal: X.X" mentions
    import re
    val_match = re.search(r"valence[:\s]+([0-9.]+)", text_lower)
    aro_match = re.search(r"arousal[:\s]+([0-9.]+)", text_lower)

    vad = _LABEL_TO_VAD.get(best_label, (0.5, 0.5))
    valence = float(val_match.group(1)) if val_match else vad[0]
    arousal = float(aro_match.group(1)) if aro_match else vad[1]

    # Clamp to [0,1]
    valence = max(0.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))

    return best_label, valence, arousal


class R1OmniScorer:
    """
    R1-Omni (HumanOmni-0.5B) emotion scorer.

    Runs inference via subprocess to avoid VRAM conflicts with the main model.
    Parses the free-text output to extract emotion label, valence, arousal.

    emotion_reward is computed as alignment with a target VAD (passed from the EmotionPlan).
    When target_vad is None, uses raw valence as the reward signal.
    """

    DEFAULT_INSTRUCT = (
        "Describe the emotion expressed in this video. "
        "State the predicted emotion label, valence (0=negative, 1=positive), "
        "and arousal (0=calm, 1=excited) as numbers."
    )

    def __init__(
        self,
        r1omni_root: str,
        model_path: str,
        gpu: int = 0,
        timeout_sec: int = 180,
        modal: str = "video_audio",
        instruct: Optional[str] = None,
    ):
        self.r1omni_root = r1omni_root
        self.model_path = model_path
        self.gpu = gpu
        self.timeout_sec = timeout_sec
        self.modal = modal
        self.instruct = instruct or self.DEFAULT_INSTRUCT
        # R1-Omni bert path co-located with r1omni_root
        self._bert_path = str(Path(r1omni_root) / "AI-ModelScope" / "bert-base-uncased")
        self._infer_script = str(Path(r1omni_root) / "inference.py")

    def score_video(
        self,
        video_path: str,
        target_vad: Optional[Tuple[float, float]] = None,
    ) -> EmotionScoreResult:
        result = EmotionScoreResult(scorer_name="r1omni", video_path=video_path)
        try:
            env = {
                **os.environ,
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": str(self.gpu),
                "PYTHONPATH": self.r1omni_root + ":" + os.environ.get("PYTHONPATH", ""),
            }
            proc = subprocess.run(
                [
                    sys.executable,
                    self._infer_script,
                    "--model_path", self.model_path,
                    "--video_path", video_path,
                    "--instruct", self.instruct,
                    "--modal", self.modal,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=self.r1omni_root,
                env=env,
            )
            output = proc.stdout.strip()
            if not output:
                output = proc.stderr.strip()[:500]
            if not output:
                raise RuntimeError("Empty R1-Omni output")

            label, valence, arousal = _parse_r1omni_output(output)
            result.description = output
            result.predicted_class = label
            result.valence = valence
            result.arousal = arousal
            result.class_probs = {label: 1.0}

            # Compute emotion_reward as alignment with target VAD (if provided)
            if target_vad is not None:
                tv, ta = target_vad
                val_align = 1.0 - abs(valence - tv)
                aro_align = 1.0 - abs(arousal - ta)
                result.emotion_reward = float((val_align + aro_align) / 2.0)
            else:
                # Just use valence as reward (positive emotion → higher reward)
                result.emotion_reward = valence

            result.active = True
        except subprocess.TimeoutExpired:
            result.error = "R1-Omni inference timed out"
            logger.warning("[R1Omni] Timeout for %s", video_path)
        except Exception as exc:
            import traceback
            result.error = str(exc)
            logger.warning(
                "[R1Omni] score_video FAILED for %s: %s\n%s",
                video_path, exc, traceback.format_exc(),
            )
        return result

    def score_videos(
        self,
        video_paths: List[str],
        target_vad: Optional[Tuple[float, float]] = None,
    ) -> List[EmotionScoreResult]:
        return [self.score_video(vp, target_vad) for vp in video_paths]


# ---------------------------------------------------------------------------
# Combined EmotionReflector — used by ReflectionRewarder
# ---------------------------------------------------------------------------

class EmotionReflector:
    """
    Combines CLIP-AUTT and R1-Omni into a single emotion scoring interface
    for the Reflect phase of the ReAct reward loop.

    Score fusion:
        emotion_reflect_score = clip_weight * clip_autt_reward + r1omni_weight * r1omni_reward

    When a scorer is unavailable (no checkpoint, import error), its weight is
    redistributed to the remaining active scorer.
    """

    def __init__(
        self,
        clip_autt_scorer: Optional[ClipAuttScorer] = None,
        r1omni_scorer: Optional[R1OmniScorer] = None,
        clip_weight: float = 0.5,
        r1omni_weight: float = 0.5,
    ):
        self.clip_autt_scorer = clip_autt_scorer
        self.r1omni_scorer = r1omni_scorer
        self.clip_weight = clip_weight
        self.r1omni_weight = r1omni_weight

    def reflect(
        self,
        video_path: str,
        target_vad: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Score video with all available emotion scorers.

        Returns a dict with:
          - emotion_reflect_score: fused [0,1] reward
          - clip_autt: EmotionScoreResult dict or None
          - r1omni: EmotionScoreResult dict or None
          - active: bool
        """
        clip_result: Optional[EmotionScoreResult] = None
        r1omni_result: Optional[EmotionScoreResult] = None

        if self.clip_autt_scorer is not None:
            clip_result = self.clip_autt_scorer.score_video(video_path)

        if self.r1omni_scorer is not None:
            r1omni_result = self.r1omni_scorer.score_video(video_path, target_vad=target_vad)

        # Fuse scores (redistribute weights if scorer failed)
        scores: List[Tuple[float, float]] = []
        if clip_result is not None and clip_result.active:
            scores.append((clip_result.emotion_reward, self.clip_weight))
        if r1omni_result is not None and r1omni_result.active:
            scores.append((r1omni_result.emotion_reward, self.r1omni_weight))

        if scores:
            total_w = sum(w for _, w in scores)
            fused = sum(s * w for s, w in scores) / max(total_w, 1e-8)
            active = True
        else:
            fused = 0.5
            active = False

        return {
            "emotion_reflect_score": float(fused),
            "active": active,
            "clip_autt": clip_result.__dict__ if clip_result is not None else None,
            "r1omni": r1omni_result.__dict__ if r1omni_result is not None else None,
        }

    def reflect_batch(
        self,
        video_paths: List[str],
        target_vad: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        return [self.reflect(vp, target_vad) for vp in video_paths]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_clip_autt_scorer(
    emo_reward_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    gpu: int = 0,
    num_aus: int = 46,
) -> Optional[ClipAuttScorer]:
    """
    Build a ClipAuttScorer from the emo_reward directory.
    Returns None if checkpoint not found.
    """
    if emo_reward_root is None:
        emo_reward_root = str(Path(__file__).parent)

    clip_autt_root = str(Path(emo_reward_root) / "CLIP-AUTT")
    if checkpoint_path is None:
        checkpoint_path = str(
            Path(emo_reward_root) / "CLIP-AUTT" / "WeightFiles" / "Biovid"
            / "lab_srcs77_biovid_ep10_bs8_sql16_str2_vid.pth"
        )

    if not Path(checkpoint_path).exists():
        logger.warning("[ClipAutt] Checkpoint not found: %s", checkpoint_path)
        return None
    if not Path(clip_autt_root).exists():
        logger.warning("[ClipAutt] CLIP-AUTT root not found: %s", clip_autt_root)
        return None

    return ClipAuttScorer(
        checkpoint_path=checkpoint_path,
        clip_autt_root=clip_autt_root,
        gpu=gpu,
        num_aus=num_aus,
    )


def build_r1omni_scorer(
    emo_reward_root: Optional[str] = None,
    gpu: int = 0,
    modal: str = "video_audio",
) -> Optional[R1OmniScorer]:
    """
    Build an R1OmniScorer from the emo_reward directory.
    Returns None if R1-Omni model weights not found.
    """
    if emo_reward_root is None:
        emo_reward_root = str(Path(__file__).parent)

    r1omni_root = str(Path(emo_reward_root) / "R1-Omni")
    model_path = str(Path(r1omni_root) / "R1-Omni-0.5B")

    if not Path(model_path).exists():
        logger.warning("[R1Omni] Model dir not found: %s", model_path)
        return None
    if not (Path(r1omni_root) / "inference.py").exists():
        logger.warning("[R1Omni] inference.py not found in %s", r1omni_root)
        return None

    return R1OmniScorer(
        r1omni_root=r1omni_root,
        model_path=model_path,
        gpu=gpu,
        modal=modal,
    )


def build_emotion_reflector(
    emo_reward_root: Optional[str] = None,
    gpu: int = 0,
    clip_weight: float = 0.5,
    r1omni_weight: float = 0.5,
) -> EmotionReflector:
    """Build a combined EmotionReflector from the emo_reward directory."""
    clip_scorer = build_clip_autt_scorer(emo_reward_root=emo_reward_root, gpu=gpu)
    r1omni_scorer = build_r1omni_scorer(emo_reward_root=emo_reward_root, gpu=gpu)

    if clip_scorer is None and r1omni_scorer is None:
        logger.warning("[EmotionReflector] Neither CLIP-AUTT nor R1-Omni available.")

    return EmotionReflector(
        clip_autt_scorer=clip_scorer,
        r1omni_scorer=r1omni_scorer,
        clip_weight=clip_weight,
        r1omni_weight=r1omni_weight,
    )
