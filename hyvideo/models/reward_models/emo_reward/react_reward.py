"""
EmoReward: Agentic Hierarchical Reward Loop (ReAct Paradigm)

Implements the three-phase reward computation described in the Proposal:
  1. Plan-and-Solve: VLM decomposes target emotion into physical acceptance criteria (FACS AU plan)
  2. Act: Lightweight AU/emotion extractors produce frame-level physical observation vectors
  3. Reflection & Hierarchical Reward: VLM synthesizes plan + observation into macro/micro rewards

This module is a **standalone reward branch** that runs alongside existing SAGE-GRPO rewards.
It does NOT modify any existing reward logic; instead it contributes an independent score dict
that gets merged into the multi-model weighted average through the standard `multi_video_score` interface.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │                    ReAct Reward Loop                  │
    │  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
    │  │  Plan &   │──>│  Act     │──>│  Reflect &       │ │
    │  │  Solve    │   │ (AU Obs) │   │  Hierarchical    │ │
    │  │  (VLM)    │   │ (Small)  │   │  Reward (VLM)    │ │
    │  └──────────┘   └──────────┘   └──────────────────┘ │
    │       │                               │              │
    │       │        Macro Reward ──────────┤              │
    │       │        Micro Reward ──────────┤              │
    └──────────────────────────────────────────────────────┘
"""

import os
import json
import re
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable

from .emotion_taxonomy import (
    EMOTION_TAXONOMY,
    EMOTION_BY_NAME_EN,
    FACS_ACTION_UNITS,
    EmotionCategory,
    get_emotion,
    get_required_aus_for_emotion,
    get_muscle_regions_for_emotion,
)
from .motion_naturalness import MotionNaturalnessResult, MotionNaturalnessScorer

_VALENCE_TO_SCALAR = {
    "positive": 0.85,
    "negative": 0.15,
    "mixed": 0.50,
    "neutral": 0.50,
}

_AROUSAL_TO_SCALAR = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.75,
    "extreme": 0.95,
}

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EmotionPlan:
    """Output of Plan-and-Solve phase: physical acceptance criteria for target emotion."""
    target_emotion: EmotionCategory
    expected_au_activations: Dict[int, float]     # AU_id -> expected intensity [0, 1]
    expected_body_markers: List[str]
    temporal_window: Tuple[float, float]           # (start_sec, end_sec) where emotion peak expected
    target_vad: Tuple[float, float] = (0.5, 0.5)   # (valence, arousal)
    confidence: float = 1.0


@dataclass
class FrameObservation:
    """Output of Act phase: per-frame physical measurements from lightweight models."""
    frame_idx: int
    timestamp_sec: float
    au_activations: Dict[int, float]               # AU_id -> detected intensity [0, 1]
    body_marker_detected: Dict[str, bool]
    emotion_label: str = ""
    emotion_logits: Optional[List[float]] = None
    gaze: Optional[Tuple[float, float]] = None
    face_detected: bool = True
    face_bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2) normalized
    channel_signals: Dict[str, float] = field(default_factory=dict)  # chromatic/kinetic/compositional


@dataclass
class HierarchicalReward:
    """Physics-only EmoReward score (AU / body / temporal), fused with VideoAlign in EmoRewardScorer."""
    total_score: float
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Phase 1: Plan-and-Solve (Emotion → Physical Acceptance Criteria)
# ============================================================================

class EmotionPlanner:
    """
    Converts a text prompt's target emotion into objective physical acceptance criteria.

    In production, this calls a VLM to reason about what AUs and body markers
    constitute the target emotion. For efficiency, we provide a fast rule-based
    fallback using the emotion taxonomy, with optional VLM override.
    """

    def __init__(self, vlm_planner: Optional[Callable] = None):
        """
        Args:
            vlm_planner: Optional callable (prompt: str) -> dict with keys:
                         'emotion_name', 'au_expectations', 'body_expectations', 'temporal_window'
                         When None, uses taxonomy-based deterministic planning.
        """
        self.vlm_planner = vlm_planner

    def plan(self, prompt: str, video_duration_sec: float = 5.0) -> EmotionPlan:
        """
        Plan-and-Solve: analyze prompt to determine target emotion and physical criteria.

        If a VLM planner is provided, it performs:
          Thought: VLM reasons about the emotion's anatomical definition
          Plan: VLM produces expected AU activations and body markers

        Otherwise, uses taxonomy lookup as deterministic fallback.
        """
        if self.vlm_planner is not None:
            return self._vlm_plan(prompt, video_duration_sec)
        return self._taxonomy_plan(prompt, video_duration_sec)

    def _vlm_plan(self, prompt: str, video_duration_sec: float) -> EmotionPlan:
        """VLM-based planning with ReAct reasoning chain."""
        try:
            vlm_output = self.vlm_planner(prompt)
            emotion_name = vlm_output.get("emotion_name", "")
            emotion = get_emotion(emotion_name)

            if emotion is None:
                return self._taxonomy_plan(prompt, video_duration_sec)

            au_expectations = vlm_output.get("au_expectations", {})
            if not au_expectations:
                au_expectations = {au: 0.8 for au in emotion.required_aus}
                au_expectations.update({au: 0.4 for au in emotion.optional_aus})

            body_expectations = vlm_output.get("body_expectations", emotion.body_markers)
            temporal_window = vlm_output.get(
                "temporal_window",
                (0.2 * video_duration_sec, 0.8 * video_duration_sec),
            )
            # The planner emits temporal_window as a FRACTION of duration (0..1);
            # observations are timestamped in SECONDS. Convert fraction->seconds.
            try:
                ts, te = float(temporal_window[0]), float(temporal_window[1])
                if 0.0 <= ts <= 1.0 and 0.0 <= te <= 1.0 and te <= 1.0:
                    ts, te = ts * video_duration_sec, te * video_duration_sec
                if te < ts:
                    ts, te = te, ts
                temporal_window = (ts, te)
            except Exception:
                temporal_window = (0.2 * video_duration_sec, 0.8 * video_duration_sec)

            target_vad = self._emotion_to_vad(emotion)
            if "target_vad" in vlm_output and isinstance(vlm_output["target_vad"], (list, tuple)) and len(vlm_output["target_vad"]) == 2:
                try:
                    target_vad = (
                        float(vlm_output["target_vad"][0]),
                        float(vlm_output["target_vad"][1]),
                    )
                except Exception:
                    pass

            return EmotionPlan(
                target_emotion=emotion,
                expected_au_activations=au_expectations,
                expected_body_markers=body_expectations,
                temporal_window=temporal_window,
                target_vad=target_vad,
                confidence=vlm_output.get("confidence", 0.9),
            )
        except Exception:
            return self._taxonomy_plan(prompt, video_duration_sec)

    def _taxonomy_plan(self, prompt: str, video_duration_sec: float) -> EmotionPlan:
        """
        Deterministic planning using the emotion taxonomy.
        Matches prompt keywords against emotion names to find the target emotion.
        """
        prompt_lower = prompt.lower()
        best_match = None
        best_score = 0

        for emo in EMOTION_TAXONOMY:
            score = 0
            en_name = emo.name_en.lower()
            cn_name = emo.name_cn

            if en_name in prompt_lower:
                score = len(en_name)
            elif cn_name in prompt:
                score = len(cn_name)
            else:
                for word in en_name.split("/"):
                    word = word.strip().lower()
                    if len(word) > 3 and word in prompt_lower:
                        score = max(score, len(word))
                for word in cn_name.split("/"):
                    word = word.strip()
                    if len(word) > 1 and word in prompt:
                        score = max(score, len(word) * 2)

            if score > best_score:
                best_score = score
                best_match = emo

        if best_match is None:
            best_match = get_emotion("Calmness")

        au_expectations = {au: 0.8 for au in best_match.required_aus}
        au_expectations.update({au: 0.4 for au in best_match.optional_aus})

        return EmotionPlan(
            target_emotion=best_match,
            expected_au_activations=au_expectations,
            expected_body_markers=best_match.body_markers,
            temporal_window=(0.2 * video_duration_sec, 0.8 * video_duration_sec),
            target_vad=self._emotion_to_vad(best_match),
            confidence=0.7 if best_score > 0 else 0.3,
        )

    @staticmethod
    def _emotion_to_vad(emotion: EmotionCategory) -> Tuple[float, float]:
        """Map taxonomy valence/arousal enums to continuous VAD-like coordinates."""
        valence_key = getattr(emotion.valence, "value", "neutral")
        arousal_key = getattr(emotion.arousal, "value", "medium")
        v = _VALENCE_TO_SCALAR.get(str(valence_key).lower(), 0.5)
        a = _AROUSAL_TO_SCALAR.get(str(arousal_key).lower(), 0.5)
        return (v, a)


# ============================================================================
# Phase 2: Act (Lightweight Model Observation)
# ============================================================================

class PhysicalObserver:
    """
    Calls lightweight AU/emotion extraction models to obtain frame-level physical
    observation vectors from generated videos.

    This is the bridge between the VLM's plan and the small model's measurements.
    The actual AU extraction model is injected via `au_extractor_fn`.
    """

    def __init__(
        self,
        au_extractor_fn: Optional[Callable] = None,
        sample_fps: float = 8.0,
    ):
        """
        Args:
            au_extractor_fn: Callable (video_path: str, frame_indices: List[int]) -> List[dict]
                Each dict: {'au_activations': {au_id: float}, 'face_detected': bool,
                           'face_bbox': (x1,y1,x2,y2), 'body_markers': {marker: bool}}
                When None, returns empty observations (no small model available).
            sample_fps: Target FPS for frame sampling.
        """
        self.au_extractor_fn = au_extractor_fn
        self.sample_fps = sample_fps

    def observe(
        self,
        video_path: str,
        video_fps: float = 24.0,
        video_duration_sec: float = 5.0,
    ) -> List[FrameObservation]:
        """
        Act phase: extract physical observations from video frames.

        Returns a list of FrameObservation, one per sampled frame.
        """
        total_frames = int(video_fps * video_duration_sec)
        sample_interval = max(1, int(video_fps / self.sample_fps))
        frame_indices = list(range(0, total_frames, sample_interval))

        if self.au_extractor_fn is not None:
            try:
                raw_observations = self.au_extractor_fn(video_path, frame_indices)
                return self._parse_raw_observations(raw_observations, frame_indices, video_fps)
            except Exception:
                pass

        # Even without an AU extractor, compute lightweight channel signals via OpenCV
        return self._observations_with_channel_signals(video_path, frame_indices, video_fps)

    def _parse_raw_observations(
        self,
        raw_obs: List[Dict],
        frame_indices: List[int],
        video_fps: float,
    ) -> List[FrameObservation]:
        observations = []
        for i, (fidx, raw) in enumerate(zip(frame_indices, raw_obs)):
            gaze = raw.get("gaze")
            gaze_tuple = None
            if isinstance(gaze, (list, tuple)) and len(gaze) >= 2:
                try:
                    gaze_tuple = (float(gaze[0]), float(gaze[1]))
                except Exception:
                    gaze_tuple = None

            observations.append(FrameObservation(
                frame_idx=fidx,
                timestamp_sec=fidx / video_fps,
                au_activations=raw.get("au_activations", {}),
                body_marker_detected=raw.get("body_markers", {}),
                emotion_label=str(raw.get("emotion_label", "")).lower(),
                emotion_logits=(
                    list(raw.get("emotion_logits"))
                    if isinstance(raw.get("emotion_logits"), (list, tuple))
                    else None
                ),
                gaze=gaze_tuple,
                face_detected=raw.get("face_detected", True),
                face_bbox=raw.get("face_bbox"),
                channel_signals=dict(raw.get("channel_signals") or {}),
            ))
        return observations

    def _empty_observations(
        self,
        frame_indices: List[int],
        video_fps: float,
    ) -> List[FrameObservation]:
        return [
            FrameObservation(
                frame_idx=fidx,
                timestamp_sec=fidx / video_fps,
                au_activations={},
                body_marker_detected={},
                emotion_label="",
                emotion_logits=None,
                gaze=None,
                face_detected=False,
            )
            for fidx in frame_indices
        ]

    def _observations_with_channel_signals(
        self,
        video_path: str,
        frame_indices: List[int],
        video_fps: float,
    ) -> List[FrameObservation]:
        """Build observations using only lightweight OpenCV channel signals (no AU extractor needed)."""
        try:
            from .openface_adapter import (
                _load_frames_for_channel_signals,
                _compute_channel_signals,
            )
            frame_map = _load_frames_for_channel_signals(video_path, frame_indices)
        except Exception:
            frame_map = {}

        observations = []
        prev_frame = None
        for fidx in frame_indices:
            frame_bgr = frame_map.get(fidx)
            try:
                cs = _compute_channel_signals(frame_bgr, prev_frame, {})
            except Exception:
                cs = {}
            observations.append(FrameObservation(
                frame_idx=fidx,
                timestamp_sec=fidx / video_fps,
                au_activations={},
                body_marker_detected={},
                emotion_label="",
                emotion_logits=None,
                gaze=None,
                face_detected=False,
                channel_signals=cs,
            ))
            if frame_bgr is not None:
                prev_frame = frame_bgr
        return observations


# ============================================================================
# Phase 3: Reflection & Hierarchical Reward Computation
# ============================================================================

class ReflectionRewarder:
    """
    Full Reflect phase: AU micro reward, body markers, temporal peak alignment,
    **plus** a VLM-based macro score that ingests the AU observation summary,
    **and** deep emotion understanding via CLIP-AUTT + R1-Omni.

    Reflect pipeline (in order of computation):
      1. Channel alignment  (chromatic/kinetic/compositional — pure CV, always active)
      2. AU micro reward    (OpenFace AU compliance — requires face detection)
      3. Body compliance    (body marker detection — requires face detection)
      4. Temporal alignment (peak-AU timing — requires face detection)
      5. Macro reward       (Qwen3-VL-30B VLM or heuristic — optional)
      6. [NEW] CLIP-AUTT emotion reward  (CLIP AU text adapter — works on any face/non-face)
      7. [NEW] R1-Omni emotion reward    (HumanOmni-0.5B VLM — parsed VAD score)

    When VLM components are None, the corresponding branch uses its own fallback.
    """

    def __init__(
        self,
        vlm_reflector: Optional[Callable] = None,
        macro_weight: float = 0.30,
        micro_weight: float = 0.35,
        body_weight: float = 0.10,
        temporal_weight: float = 0.25,
        # Emotion understanding model weights (new Reflect branches)
        clip_autt_weight: float = 0.15,
        r1omni_weight: float = 0.10,
        emotion_reflector: Optional[Any] = None,  # EmotionReflector instance
        # VAD mode and channel controls
        r1omni_vad_mode: str = "valence",
        enable_chromatic: bool = True,
        enable_kinetic: bool = True,
        motion_naturalness_weight: float = 0.20,
        adaptive_reflect_control: bool = True,
        reflect_warmup_start_step: int = 40,
        reflect_full_on_step: int = 100,
        reflect_conflict_suppression: bool = True,
        reflect_subtle_prompt_gating: bool = True,
        # ---- Reflective Reward Gating: gate-policy ablation ladder ----
        # adaptive (=our method) | fixed | single | random | roundrobin | jit
        gate_policy: str = "adaptive",
        gate_single_tool: str = "r1omni",
        gate_jit_table: Optional[str] = None,
        gate_seed: int = 0,
    ):
        self.vlm_reflector = vlm_reflector
        self.emotion_reflector = emotion_reflector  # EmotionReflector | None
        # gate-policy state
        self._gate_policy = str(gate_policy or "adaptive").strip().lower()
        self._gate_single_tool = str(gate_single_tool or "r1omni").strip()
        self._gate_rr_counter = 0
        self._gate_rng = __import__("random").Random(int(gate_seed))
        self._gate_jit_table = None
        if gate_jit_table and os.path.isfile(str(gate_jit_table)):
            try:
                with open(str(gate_jit_table), "r", encoding="utf-8") as _f:
                    self._gate_jit_table = json.load(_f)
            except Exception:
                self._gate_jit_table = None
        raw = {
            "macro": float(macro_weight),
            "micro": float(micro_weight),
            "body": float(body_weight),
            "temporal": float(temporal_weight),
            "clip_autt": float(clip_autt_weight) if emotion_reflector is not None else 0.0,
            "r1omni": float(r1omni_weight) if emotion_reflector is not None else 0.0,
            "motion": float(motion_naturalness_weight),
        }
        denom = sum(max(v, 0.0) for v in raw.values())
        if denom <= 1e-8:
            denom = 1.0
        self.macro_weight = raw["macro"] / denom
        self.micro_weight = raw["micro"] / denom
        self.body_weight = raw["body"] / denom
        self.temporal_weight = raw["temporal"] / denom
        self.clip_autt_weight = raw["clip_autt"] / denom
        self.r1omni_weight = raw["r1omni"] / denom
        self.motion_naturalness_weight = raw["motion"] / denom
        self._r1omni_vad_mode = r1omni_vad_mode
        self._enable_chromatic = enable_chromatic
        self._enable_kinetic = enable_kinetic
        self._adaptive_reflect_control = bool(adaptive_reflect_control)
        self._reflect_warmup_start_step = int(reflect_warmup_start_step)
        self._reflect_full_on_step = max(
            int(reflect_full_on_step),
            self._reflect_warmup_start_step + 1,
        )
        self._reflect_conflict_suppression = bool(reflect_conflict_suppression)
        self._reflect_subtle_prompt_gating = bool(reflect_subtle_prompt_gating)

    def reflect(
        self,
        plan: EmotionPlan,
        observations: List[FrameObservation],
        prompt: str,
        video_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        motion_result: Optional[MotionNaturalnessResult] = None,
    ) -> HierarchicalReward:
        micro_score, au_compliance, micro_active = self._compute_micro_reward(plan, observations)
        body_score, body_active = self._compute_body_compliance(plan, observations)
        temporal_score, temporal_active = self._compute_temporal_alignment(plan, observations)
        channel_scores, channel_active = self._compute_channel_alignment(plan, observations)
        motion_score = float(getattr(motion_result, "motion_naturalness_reward", 0.5))
        motion_confidence = float(getattr(motion_result, "confidence", 0.0))
        motion_active = bool(getattr(motion_result, "active", False))
        motion_error = str(getattr(motion_result, "error", ""))

        if self.vlm_reflector is not None:
            macro_score, macro_active, reasoning = self._vlm_macro_reward(
                plan, observations, prompt, video_path,
            )
        else:
            macro_score, macro_active, reasoning = self._heuristic_macro_reward(plan, observations)

        clip_autt_score = 0.5
        clip_autt_active = False
        r1omni_score = 0.5
        r1omni_active = False
        emotion_reflect_details: Dict[str, Any] = {}

        if self.emotion_reflector is not None and video_path and os.path.isfile(video_path):
            try:
                if self._r1omni_vad_mode == "alignment":
                    vad_for_r1 = plan.target_vad if hasattr(plan, "target_vad") else None
                else:
                    vad_for_r1 = None
                er = self.emotion_reflector.reflect(video_path, target_vad=vad_for_r1)
                emotion_reflect_details = er
                if er.get("clip_autt") and er["clip_autt"].get("active"):
                    clip_autt_score = float(er["clip_autt"].get("emotion_reward", 0.5))
                    clip_autt_active = True
                if er.get("r1omni") and er["r1omni"].get("active"):
                    r1omni_score = float(er["r1omni"].get("emotion_reward", 0.5))
                    if self._r1omni_vad_mode == "mean":
                        _r1_val = float(er["r1omni"].get("valence", 0.5))
                        _r1_aro = float(er["r1omni"].get("arousal", 0.5))
                        r1omni_score = (_r1_val + _r1_aro) / 2.0
                    r1omni_active = True
            except Exception as _er_e:
                import logging as _logging
                import traceback as _tb
                _logging.getLogger(__name__).warning(
                    "[Reflect] EmotionReflector FAILED for %s: %s\n%s",
                    video_path, _er_e, _tb.format_exc(),
                )

        control = self._build_tool_control(
            plan=plan,
            observations=observations,
            prompt=prompt,
            metadata=metadata,
            clip_autt_score=clip_autt_score,
            clip_autt_active=clip_autt_active,
            r1omni_score=r1omni_score,
            r1omni_active=r1omni_active,
            motion_active=motion_active,
            motion_confidence=motion_confidence,
        )

        # Reflective Reward Gating: pick effective gates per the configured gate policy.
        _active_keys = [k for k, on in (
            ("macro", macro_active), ("micro", micro_active), ("body", body_active),
            ("temporal", temporal_active), ("clip_autt", clip_autt_active),
            ("r1omni", r1omni_active), ("motion", motion_active),
        ) if on]
        eg = self._effective_gates(
            control["tool_gates"], _active_keys,
            face_ratio=control["face_gate"] * 0.35,  # face_gate = min(1, face_ratio/0.35)
            arousal=float(getattr(plan, "target_vad", (0.5, 0.5))[1]),
        )

        parts: List[Tuple[float, float]] = []
        if macro_active:
            parts.append((macro_score, self.macro_weight * eg["macro"]))
        if micro_active:
            parts.append((micro_score, self.micro_weight * eg["micro"]))
        if body_active:
            parts.append((body_score, self.body_weight * eg["body"]))
        if temporal_active:
            parts.append((temporal_score, self.temporal_weight * eg["temporal"]))
        if clip_autt_active:
            parts.append((clip_autt_score, self.clip_autt_weight * eg["clip_autt"]))
        if r1omni_active:
            parts.append((r1omni_score, self.r1omni_weight * eg["r1omni"]))
        if motion_active:
            parts.append((motion_score, self.motion_naturalness_weight * eg["motion"]))

        parts = [(score, weight) for score, weight in parts if weight > 1e-8]
        if parts:
            wsum = sum(weight for _, weight in parts)
            total = sum(score * weight for score, weight in parts) / max(wsum, 1e-8)
        elif motion_active:
            total = motion_score
        elif macro_active or micro_active or body_active or temporal_active or clip_autt_active or r1omni_active:
            fallback_parts: List[Tuple[float, float]] = []
            if macro_active:
                fallback_parts.append((macro_score, self.macro_weight))
            if micro_active:
                fallback_parts.append((micro_score, self.micro_weight))
            if body_active:
                fallback_parts.append((body_score, self.body_weight))
            if temporal_active:
                fallback_parts.append((temporal_score, self.temporal_weight))
            if clip_autt_active:
                fallback_parts.append((clip_autt_score, self.clip_autt_weight))
            if r1omni_active:
                fallback_parts.append((r1omni_score, self.r1omni_weight))
            wsum = sum(weight for _, weight in fallback_parts)
            total = sum(score * weight for score, weight in fallback_parts) / max(wsum, 1e-8)
        elif channel_active:
            total = float(sum(channel_scores.values()) / max(len(channel_scores), 1))
        else:
            total = 0.0

        return HierarchicalReward(
            total_score=total,
            details={
                "target_emotion": plan.target_emotion.name_en,
                "plan_confidence": plan.confidence,
                "target_vad": plan.target_vad,
                "num_observations": len(observations),
                "faces_detected": sum(1 for o in observations if o.face_detected),
                "macro_active": macro_active,
                "macro_score": macro_score,
                "macro_reasoning": reasoning,
                "micro_active": micro_active,
                "body_active": body_active,
                "temporal_active": temporal_active,
                "channel_active": channel_active,
                "micro_score": micro_score,
                "body_score": body_score,
                "temporal_score": temporal_score,
                "au_compliance": au_compliance,
                "chromatic_score": channel_scores.get("chromatic_score", 0.5),
                "kinetic_score": channel_scores.get("kinetic_score", 0.5),
                "compositional_score": channel_scores.get("compositional_score", 0.5),
                "clip_autt_active": clip_autt_active,
                "clip_autt_score": clip_autt_score,
                "r1omni_active": r1omni_active,
                "r1omni_score": r1omni_score,
                "motion_naturalness_active": motion_active,
                "motion_naturalness_score": motion_score,
                "motion_naturalness_confidence": motion_confidence,
                "motion_naturalness_error": motion_error,
                "reflect_gate": control["reflect_gate"],
                "reflect_phase_gate": control["phase_gate"],
                "reflect_face_gate": control["face_gate"],
                "reflect_conflict_gate": control["conflict_gate"],
                "reflect_subtle_prompt": control["subtle_prompt"],
                "reflect_tool_gates": control["tool_gates"],
                "reflect_step": control["current_step"],
                "reflect_max_steps": control["max_steps"],
                "emotion_reflect": emotion_reflect_details,
            },
        )

    def _build_tool_control(
        self,
        plan: EmotionPlan,
        observations: List[FrameObservation],
        prompt: str,
        metadata: Optional[Dict[str, Any]],
        clip_autt_score: float,
        clip_autt_active: bool,
        r1omni_score: float,
        r1omni_active: bool,
        motion_active: bool,
        motion_confidence: float,
    ) -> Dict[str, Any]:
        prompt_lower = str(prompt or "").lower()
        current_step, max_steps = self._extract_progress(metadata)
        phase_gate = self._compute_phase_gate(current_step) if self._adaptive_reflect_control else 1.0

        num_obs = max(len(observations), 1)
        face_ratio = sum(1 for o in observations if o.face_detected) / float(num_obs)
        face_gate = max(0.0, min(1.0, face_ratio / 0.35))

        subtle_prompt = self._is_subtle_prompt(prompt_lower, plan)
        subtle_face_gate = 0.85 if subtle_prompt and self._reflect_subtle_prompt_gating else 1.0
        subtle_temporal_gate = 0.75 if subtle_prompt and self._reflect_subtle_prompt_gating else 1.0
        subtle_emotion_gate = 0.80 if subtle_prompt and self._reflect_subtle_prompt_gating else 1.0

        conflict_gate = 1.0
        if clip_autt_active and r1omni_active:
            disagreement = abs(float(clip_autt_score) - float(r1omni_score))
            if self._reflect_conflict_suppression and disagreement > 0.20:
                conflict_gate = max(0.35, 1.0 - (disagreement - 0.20) / 0.60)

        reflect_gate = max(0.10, phase_gate) * conflict_gate
        motion_gate = 1.0 if not motion_active else (0.55 + 0.45 * max(0.0, min(1.0, motion_confidence)))
        tool_gates = {
            "macro": reflect_gate,
            "micro": reflect_gate * face_gate * subtle_face_gate,
            "body": reflect_gate * face_gate * subtle_face_gate,
            "temporal": reflect_gate * face_gate * subtle_temporal_gate,
            "clip_autt": reflect_gate * subtle_emotion_gate,
            "r1omni": reflect_gate * subtle_emotion_gate,
            "motion": motion_gate,
        }
        for key, value in tool_gates.items():
            tool_gates[key] = float(max(0.0, min(1.0, value)))

        return {
            "current_step": current_step,
            "max_steps": max_steps,
            "phase_gate": float(max(0.0, min(1.0, phase_gate))),
            "face_gate": float(max(0.0, min(1.0, face_gate))),
            "conflict_gate": float(max(0.0, min(1.0, conflict_gate))),
            "reflect_gate": float(max(0.0, min(1.0, reflect_gate))),
            "subtle_prompt": bool(subtle_prompt),
            "tool_gates": tool_gates,
        }

    @staticmethod
    def _extract_progress(metadata: Optional[Dict[str, Any]]) -> Tuple[int, int]:
        if not isinstance(metadata, dict):
            return 0, 0
        current_step = 0
        max_steps = 0
        for key in ("update_step", "update_steps", "global_step", "step"):
            value = metadata.get(key)
            if value is not None:
                try:
                    current_step = int(value)
                    break
                except Exception:
                    pass
        for key in ("max_steps", "total_steps", "max_train_steps"):
            value = metadata.get(key)
            if value is not None:
                try:
                    max_steps = int(value)
                    break
                except Exception:
                    pass
        return max(0, current_step), max(0, max_steps)

    def _compute_phase_gate(self, current_step: int) -> float:
        if current_step <= self._reflect_warmup_start_step:
            return 0.0
        if current_step >= self._reflect_full_on_step:
            return 1.0
        denom = max(float(self._reflect_full_on_step - self._reflect_warmup_start_step), 1.0)
        return (float(current_step) - float(self._reflect_warmup_start_step)) / denom

    # ---- Reflective Reward Gating: gate-policy ablation ladder ----
    _GATE_TOOLS = ("macro", "micro", "body", "temporal", "clip_autt", "r1omni", "motion")

    def _effective_gates(
        self,
        adaptive_gates: Dict[str, float],
        active_keys: List[str],
        face_ratio: float,
        arousal: float,
    ) -> Dict[str, float]:
        """Return the per-tool gate actually used in fusion, per the gate policy.

        adaptive   : our reflective gating (phase/conflict/face/subtle) -- unchanged.
        fixed      : disable adaptive gating -> fixed weighted multi-provider fusion (all gates=1).
        single     : only `gate_single_tool` is on (random/round-robin/oracle ablation control).
        random     : pick ONE active provider uniformly at random per call.
        roundrobin : cycle providers across calls.
        jit        : multiply base gate by a precomputed reliability weight table (regime-keyed);
                     falls back to `adaptive` when no table is loaded.
        """
        policy = self._gate_policy
        if policy in ("adaptive", "", "reflect"):
            return adaptive_gates
        active = [k for k in self._GATE_TOOLS if k in active_keys] or list(self._GATE_TOOLS)
        gates = {k: 0.0 for k in self._GATE_TOOLS}
        if policy == "fixed":
            return {k: 1.0 for k in self._GATE_TOOLS}
        if policy == "single":
            t = self._gate_single_tool if self._gate_single_tool in self._GATE_TOOLS else "r1omni"
            gates[t] = 1.0
            return gates
        if policy == "random":
            gates[self._gate_rng.choice(active)] = 1.0
            return gates
        if policy == "roundrobin":
            t = active[self._gate_rr_counter % len(active)]
            self._gate_rr_counter += 1
            gates[t] = 1.0
            return gates
        if policy == "jit":
            if not self._gate_jit_table:
                return adaptive_gates  # no offline-calibrated table -> behave as adaptive
            regime = "face%d_aro%d" % (int(face_ratio >= 0.35), int(arousal >= 0.5))
            mult = self._gate_jit_table.get(regime) or self._gate_jit_table.get("default") or {}
            return {k: float(adaptive_gates.get(k, 0.0)) * float(mult.get(k, 1.0))
                    for k in self._GATE_TOOLS}
        return adaptive_gates

    @staticmethod
    def _is_subtle_prompt(prompt_lower: str, plan: EmotionPlan) -> bool:
        subtle_keywords = (
            "calm", "gentle", "quiet", "soft", "still", "slight", "subtle", "restrained",
            "tender", "peaceful", "serene", "light smile", "微笑", "轻", "缓慢", "平静", "安静", "克制", "温柔",
        )
        high_energy_keywords = (
            "run", "running", "dance", "dancing", "jump", "shout", "scream", "fight", "explode",
            "冲", "奔跑", "跳", "激烈", "狂喜", "愤怒", "尖叫",
        )
        if any(token in prompt_lower for token in subtle_keywords):
            return True
        if any(token in prompt_lower for token in high_energy_keywords):
            return False
        try:
            return float(plan.target_vad[1]) <= 0.40
        except Exception:
            return False

    # --- Macro Reward (VLM or heuristic) ---

    def _vlm_macro_reward(
        self,
        plan: EmotionPlan,
        observations: List[FrameObservation],
        prompt: str,
        video_path: str,
    ) -> Tuple[float, bool, str]:
        """Send AU obs summary to the VLM reflector; returns (score, active, reasoning)."""
        try:
            obs_summary = self._summarize_observations(plan, observations)
            obs_summary["_video_path"] = video_path
            result = self.vlm_reflector(plan, obs_summary, prompt)
            return float(result.get("macro_score", 0.5)), True, result.get("reasoning", "")
        except Exception:
            s, active, r = self._heuristic_macro_reward(plan, observations)
            return s, active, r

    def _heuristic_macro_reward(
        self,
        plan: EmotionPlan,
        observations: List[FrameObservation],
    ) -> Tuple[float, bool, str]:
        if not observations:
            return 0.3, False, "no_obs"
        face_ratio = sum(1 for o in observations if o.face_detected) / max(len(observations), 1)
        if face_ratio < 0.1:
            return 0.3, False, f"face_ratio={face_ratio:.2f}"
        _, compliance, micro_active = self._compute_micro_reward(plan, observations)
        if not micro_active:
            return 0.3, False, f"micro_inactive,face_ratio={face_ratio:.2f}"
        score = 0.3 + 0.7 * compliance * plan.confidence
        return score, True, f"heuristic:compliance={compliance:.3f},face={face_ratio:.2f}"

    def _compute_micro_reward(
        self,
        plan: EmotionPlan,
        observations: List[FrameObservation],
    ) -> Tuple[float, float, bool]:
        if not observations or not plan.expected_au_activations:
            return 0.5, 0.5, False

        has_data = any(o.face_detected and o.au_activations for o in observations)
        if not has_data:
            return 0.5, 0.5, False

        au_scores = []
        for au_id, target_intensity in plan.expected_au_activations.items():
            frame_scores = []
            for obs in observations:
                if not obs.face_detected:
                    continue
                observed = obs.au_activations.get(au_id, 0.0)
                iou = self._single_au_iou(target_intensity, observed)
                frame_scores.append(iou)

            if frame_scores:
                au_scores.append(sum(frame_scores) / len(frame_scores))

        if not au_scores:
            return 0.5, 0.5, False

        micro = sum(au_scores) / len(au_scores)
        required_aus_detected = sum(
            1 for au_id in plan.target_emotion.required_aus
            if any(
                obs.au_activations.get(au_id, 0.0) > 0.3
                for obs in observations
                if obs.face_detected
            )
        )
        total_required = max(len(plan.target_emotion.required_aus), 1)
        compliance = required_aus_detected / total_required

        return micro, compliance, True

    @staticmethod
    def _single_au_iou(target: float, observed: float) -> float:
        """Compute IoU-like overlap for a single AU activation pair."""
        intersection = min(target, observed)
        union = max(target, observed)
        if union < 1e-6:
            return 1.0
        return intersection / union

    def _compute_body_compliance(
        self,
        plan: EmotionPlan,
        observations: List[FrameObservation],
    ) -> Tuple[float, bool]:
        if not plan.expected_body_markers:
            return 0.5, False

        has_body_data = any(o.body_marker_detected for o in observations)
        if not has_body_data:
            return 0.5, False

        marker_scores = []
        for marker in plan.expected_body_markers:
            detected_in_any_frame = any(
                obs.body_marker_detected.get(marker, False)
                for obs in observations
            )
            marker_scores.append(1.0 if detected_in_any_frame else 0.0)

        return sum(marker_scores) / max(len(marker_scores), 1), True

    def _compute_temporal_alignment(
        self,
        plan: EmotionPlan,
        observations: List[FrameObservation],
    ) -> Tuple[float, bool]:
        if not observations or not plan.expected_au_activations:
            return 0.5, False

        has_data = any(o.face_detected and o.au_activations for o in observations)
        if not has_data:
            return 0.5, False

        frame_intensities = []
        for obs in observations:
            if not obs.face_detected:
                frame_intensities.append((obs.timestamp_sec, 0.0))
                continue
            total_activation = sum(
                obs.au_activations.get(au, 0.0)
                for au in plan.expected_au_activations
            )
            frame_intensities.append((obs.timestamp_sec, total_activation))

        if not frame_intensities:
            return 0.5, False

        peak_time, _peak_val = max(frame_intensities, key=lambda x: x[1])
        t_start, t_end = plan.temporal_window

        if t_start <= peak_time <= t_end:
            return 1.0, True

        distance = min(abs(peak_time - t_start), abs(peak_time - t_end))
        window_size = max(t_end - t_start, 0.1)
        penalty = min(distance / window_size, 1.0)
        return max(0.0, 1.0 - penalty), True

    def _compute_channel_alignment(
        self,
        plan: EmotionPlan,
        observations: List[FrameObservation],
    ) -> Tuple[Dict[str, float], bool]:
        """
        Aggregate per-frame channel_signals across all observations into per-channel scores.

        Returns a dict of channel scores [0, 1]:
            {
                'chromatic_score': float,
                'kinetic_score': float,
                'compositional_score': float,
            }
        and an `active` bool indicating whether any real (non-0.5 default) data was found.

        The channel scores are further adjusted by the target emotion's VAD profile:
          - chromatic: biased by target valence (positive → higher is better; negative → lower preferred)
          - kinetic: biased by target arousal (high arousal → higher motion is better)
          - compositional: used raw (framing quality is prompt-independent)
        """
        _CHANNEL_KEYS = ("chromatic_score", "kinetic_score", "compositional_score")
        raw_values: Dict[str, List[float]] = {k: [] for k in _CHANNEL_KEYS}

        for obs in observations:
            cs = obs.channel_signals
            if not cs:
                continue
            for key in _CHANNEL_KEYS:
                val = cs.get(key)
                if val is not None:
                    raw_values[key].append(float(val))

        # Check whether we have real data (some values differ from the 0.5 fallback)
        has_data = any(
            vals and any(abs(v - 0.5) > 1e-4 for v in vals)
            for vals in raw_values.values()
        )

        means: Dict[str, float] = {}
        for key in _CHANNEL_KEYS:
            vals = raw_values[key]
            means[key] = float(sum(vals) / len(vals)) if vals else 0.5

        # VAD-aware alignment adjustment (soft rescaling, not hard clipping)
        target_valence, target_arousal = plan.target_vad  # both in [0, 1]

        # chromatic: for high-valence (positive) emotions warm/saturated frames are ideal (score→1);
        #            for low-valence (negative) emotions, subdued frames can also match (score→0)
        #            We compute alignment as: 1 - |mean_chromatic - target_valence|
        if self._enable_chromatic:
            chromatic_align = 1.0 - abs(means["chromatic_score"] - target_valence)
        else:
            chromatic_align = 0.5  # neutral fallback when disabled

        # kinetic: high-arousal emotions expect more motion; low-arousal expect stillness
        #          alignment = 1 - |mean_kinetic - target_arousal|
        if self._enable_kinetic:
            kinetic_align = 1.0 - abs(means["kinetic_score"] - target_arousal)
        else:
            kinetic_align = 0.5  # neutral fallback when disabled

        # compositional: raw mean (no VAD dependency; good framing always helps)
        compositional_align = means["compositional_score"]

        return {
            "chromatic_score": float(max(0.0, min(1.0, chromatic_align))),
            "kinetic_score": float(max(0.0, min(1.0, kinetic_align))),
            "compositional_score": float(max(0.0, min(1.0, compositional_align))),
        }, has_data

    @staticmethod
    def _summarize_observations(
        plan: EmotionPlan,
        observations: List[FrameObservation],
    ) -> Dict[str, Any]:
        """Summarize frame observations into a compact dict for VLM consumption."""
        if not observations:
            return {"num_frames": 0}

        au_timeseries = {}
        for au_id in plan.expected_au_activations:
            au_timeseries[au_id] = [
                (obs.timestamp_sec, obs.au_activations.get(au_id, 0.0))
                for obs in observations
                if obs.face_detected
            ]

        return {
            "num_frames": len(observations),
            "face_detection_rate": sum(1 for o in observations if o.face_detected) / len(observations),
            "au_timeseries": au_timeseries,
            "time_range": (observations[0].timestamp_sec, observations[-1].timestamp_sec),
        }


# ============================================================================
# Unified ReAct Reward Interface (compatible with SAGE-GRPO multi_video_score)
# ============================================================================

class EmoRewardScorer:
    """
    End-to-end ReAct reward pipeline.

    Wraps Plan-and-Solve, Act, and Reflection into a single callable
    compatible with SAGE-GRPO's reward_inferencer interface.
    """

    def __init__(
        self,
        vlm_planner: Optional[Callable] = None,
        vlm_reflector: Optional[Callable] = None,
        au_extractor_fn: Optional[Callable] = None,
        micro_weight: float = 0.5,
        body_weight: float = 0.25,
        temporal_weight: float = 0.25,
        video_fps: float = 24.0,
        video_duration_sec: float = 5.0,
        sample_fps: float = 8.0,
        lazy_init_pack: Optional[Tuple[Any, Dict[str, Any]]] = None,
        # Emotion understanding scorers (CLIP-AUTT + R1-Omni)
        emotion_reflector: Optional[Any] = None,
        clip_autt_weight: float = 0.15,
        r1omni_weight: float = 0.10,
        enable_motion_naturalness: bool = True,
        motion_naturalness_weight: float = 0.20,
        motion_naturalness_target_arousal: bool = True,
        adaptive_reflect_control: bool = True,
        reflect_warmup_start_step: int = 40,
        reflect_full_on_step: int = 100,
    ):
        self.video_fps = video_fps
        self.video_duration_sec = video_duration_sec
        self._lazy_device: Optional[Any] = None
        self._lazy_cfg: Optional[Dict[str, Any]] = None
        self._videoalign_fn: Optional[Callable] = None
        self._combine_videoalign = True
        self._videoalign_fusion_alpha: float = 0.5
        self._videoalign_fusion_beta: float = 0.5
        self._emotion_reflector = emotion_reflector
        self._clip_autt_weight = clip_autt_weight
        self._r1omni_weight = r1omni_weight
        self._enable_motion_naturalness = bool(enable_motion_naturalness)
        self._motion_naturalness_target_arousal = bool(motion_naturalness_target_arousal)
        self._motion_naturalness_weight = float(motion_naturalness_weight)
        self._motion_scorer = MotionNaturalnessScorer() if self._enable_motion_naturalness else None

        if lazy_init_pack is not None:
            self._lazy_device, self._lazy_cfg = lazy_init_pack
            self.planner = EmotionPlanner(vlm_planner=None)
            self.observer = PhysicalObserver(au_extractor_fn=None, sample_fps=sample_fps)
            self.rewarder = ReflectionRewarder(
                vlm_reflector=None,
                micro_weight=micro_weight,
                body_weight=body_weight,
                temporal_weight=temporal_weight,
                emotion_reflector=emotion_reflector,
                clip_autt_weight=clip_autt_weight,
                r1omni_weight=r1omni_weight,
                motion_naturalness_weight=motion_naturalness_weight,
                adaptive_reflect_control=adaptive_reflect_control,
                reflect_warmup_start_step=reflect_warmup_start_step,
                reflect_full_on_step=reflect_full_on_step,
            )
            return

        self.planner = EmotionPlanner(vlm_planner=vlm_planner)
        self.observer = PhysicalObserver(
            au_extractor_fn=au_extractor_fn,
            sample_fps=sample_fps,
        )
        self.rewarder = ReflectionRewarder(
            vlm_reflector=vlm_reflector,
            micro_weight=micro_weight,
            body_weight=body_weight,
            temporal_weight=temporal_weight,
            emotion_reflector=emotion_reflector,
            clip_autt_weight=clip_autt_weight,
            r1omni_weight=r1omni_weight,
            motion_naturalness_weight=motion_naturalness_weight,
            adaptive_reflect_control=adaptive_reflect_control,
            reflect_warmup_start_step=reflect_warmup_start_step,
            reflect_full_on_step=reflect_full_on_step,
        )

    def _ensure_heavy_models(self) -> None:
        """Load Qwen2-VL / VideoReward / OpenFace on first reward call (lazy init)."""
        if self._lazy_cfg is None:
            return
        import logging
        _log = logging.getLogger(__name__)
        cfg = self._lazy_cfg
        device = self._lazy_device
        self._lazy_cfg = None
        self._lazy_device = None

        _log.info("[EmoReward] Lazy-loading heavy reward models (after generation) on device=%s", device)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        vlm_planner = None
        vlm_reflector = None
        vlm_path = cfg.get("vlm_model_path", "auto")
        if vlm_path and str(vlm_path).lower() != "none":
            vlm_planner, vlm_reflector = _build_vlm_components(str(vlm_path), device, cfg)
        au_fn = None
        au_path = cfg.get("au_extractor_model_path", "openface")
        if au_path and str(au_path).lower() != "none":
            au_fn = _build_au_extractor(au_path, device)

        self.planner = EmotionPlanner(vlm_planner=vlm_planner)
        self.observer = PhysicalObserver(
            au_extractor_fn=au_fn,
            sample_fps=cfg.get("sample_fps", 8.0),
        )

        # --- [NEW] Emotion understanding scorers (lazy init) ---
        emotion_reflector = self._emotion_reflector  # may be pre-set or None
        if emotion_reflector is None and cfg.get("enable_emotion_scorers", True):
            try:
                from .emotion_scorers import build_emotion_reflector
                emo_reward_root = str(os.path.dirname(os.path.abspath(__file__)))
                gpu_idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
                emotion_reflector = build_emotion_reflector(
                    emo_reward_root=emo_reward_root,
                    gpu=gpu_idx,
                    clip_weight=float(cfg.get("clip_autt_weight", 0.5)),
                    r1omni_weight=float(cfg.get("r1omni_weight", 0.5)),
                )
                _log.info("[EmoReward] EmotionReflector built (clip_autt=%s, r1omni=%s)",
                          emotion_reflector.clip_autt_scorer is not None,
                          emotion_reflector.r1omni_scorer is not None)
            except Exception as _er_e:
                import traceback as _tb
                _log.warning(
                    "[EmoReward] EmotionReflector build FAILED (%s); skipping.\n%s",
                    _er_e, _tb.format_exc(),
                )

        self.rewarder = ReflectionRewarder(
            vlm_reflector=vlm_reflector,
            macro_weight=cfg.get("macro_weight", 0.30),
            micro_weight=cfg.get("micro_weight", 0.35),
            body_weight=cfg.get("body_weight", 0.10),
            temporal_weight=cfg.get("temporal_weight", 0.25),
            clip_autt_weight=float(cfg.get("clip_autt_weight", 0.15)),
            r1omni_weight=float(cfg.get("r1omni_weight", 0.10)),
            emotion_reflector=emotion_reflector,
            r1omni_vad_mode=str(cfg.get("r1omni_vad_mode", "valence")),
            enable_chromatic=bool(cfg.get("enable_chromatic", True)),
            enable_kinetic=bool(cfg.get("enable_kinetic", True)),
            motion_naturalness_weight=float(cfg.get("motion_naturalness_weight", self._motion_naturalness_weight)),
            adaptive_reflect_control=bool(cfg.get("adaptive_reflect_control", True)),
            reflect_warmup_start_step=int(cfg.get("reflect_warmup_start_step", 40)),
            reflect_full_on_step=int(cfg.get("reflect_full_on_step", 100)),
            reflect_conflict_suppression=bool(cfg.get("reflect_conflict_suppression", True)),
            reflect_subtle_prompt_gating=bool(cfg.get("reflect_subtle_prompt_gating", True)),
            gate_policy=str(cfg.get("reward_gate_policy", "adaptive")),
            gate_single_tool=str(cfg.get("reward_gate_single_tool", "r1omni")),
            gate_jit_table=cfg.get("reward_gate_jit_table") or None,
        )
        self._combine_videoalign = bool(cfg.get("combine_videoalign", True))
        self._videoalign_fusion_alpha = float(cfg.get("videoalign_fusion_alpha", 0.5))
        self._videoalign_fusion_beta = float(cfg.get("videoalign_fusion_beta", 0.5))
        self._videoalign_fn = None
        if self._combine_videoalign:
            try:
                from hyvideo.models.reward_models.rewards import videoalign_local_score

                ck = cfg.get("reward_checkpoint_mode", "v3")
                self._videoalign_fn = videoalign_local_score(device, ck)
                _log.info("[EmoReward] VideoAlign local scorer attached (fusion a=%s, b=%s).",
                          self._videoalign_fusion_alpha, self._videoalign_fusion_beta)
            except Exception as _va_e:
                _log.warning("[EmoReward] VideoAlign attach failed (%s); EmoReward-only scores.", _va_e)

        _log.info("[EmoReward] Lazy load done (planner=%s, reflector=%s, au=%s).",
                  bool(vlm_planner), bool(vlm_reflector), bool(au_fn))

    def __call__(
        self,
        video_paths: List[str],
        prompts: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """
        SAGE-GRPO compatible interface.

        Exposed metrics:
          - emo_physics_total, emo_total (fused with VideoAlign Overall when enabled)
          - EQ (alias of emo_total after fusion)
          - emo_chromatic, emo_kinetic, emo_compositional  (channel alignment scores)
          - emo_micro, emo_body_compliance, emo_temporal, emo_au_compliance, emo_macro
          - videoalign_overall, overall, vq, mq, ta (when combine_videoalign)
        """
        if metadata is None:
            metadata = [{}] * len(video_paths)

        self._ensure_heavy_models()

        scores: Dict[str, List[float]] = {
            "emo_total": [],
            "emo_physics_total": [],
            "EQ": [],
            "emo_chromatic": [],
            "emo_kinetic": [],
            "emo_compositional": [],
            "emo_micro": [],
            "emo_body_compliance": [],
            "emo_temporal": [],
            "emo_au_compliance": [],
            "emo_macro": [],
            # [NEW] Emotion understanding scores
            "emo_clip_autt": [],
            "emo_r1omni": [],
            "emo_motion_naturalness": [],
            "emo_motion_confidence": [],
            "emo_reflect_gate": [],
        }

        for video_path, prompt, meta in zip(video_paths, prompts, metadata):
            try:
                reward = self._score_single(video_path, prompt, meta)
                phys = float(reward.total_score)
                details = reward.details
                scores["emo_physics_total"].append(phys)
                scores["emo_total"].append(phys)
                scores["EQ"].append(phys)
                scores["emo_chromatic"].append(float(details.get("chromatic_score", 0.5)))
                scores["emo_kinetic"].append(float(details.get("kinetic_score", 0.5)))
                scores["emo_compositional"].append(float(details.get("compositional_score", 0.5)))
                scores["emo_micro"].append(float(details.get("micro_score", 0.5)))
                scores["emo_body_compliance"].append(float(details.get("body_score", 0.5)))
                scores["emo_temporal"].append(float(details.get("temporal_score", 0.5)))
                scores["emo_au_compliance"].append(float(details.get("au_compliance", 0.5)))
                scores["emo_macro"].append(float(details.get("macro_score", 0.5)))
                # [NEW] Emotion model scores
                scores["emo_clip_autt"].append(float(details.get("clip_autt_score", 0.5)))
                scores["emo_r1omni"].append(float(details.get("r1omni_score", 0.5)))
                scores["emo_motion_naturalness"].append(float(details.get("motion_naturalness_score", 0.5)))
                scores["emo_motion_confidence"].append(float(details.get("motion_naturalness_confidence", 0.0)))
                scores["emo_reflect_gate"].append(float(details.get("reflect_gate", 1.0)))
            except Exception:
                for key in scores:
                    scores[key].append(float("nan"))

        if self._videoalign_fn is not None and len(video_paths) > 0:
            try:
                va_inputs_vp: List[Optional[str]] = []
                va_inputs_pr: List[str] = []
                va_map: List[int] = []
                for i, (vp, pr) in enumerate(zip(video_paths, prompts)):
                    if vp and isinstance(vp, str) and os.path.isfile(vp):
                        va_inputs_vp.append(vp)
                        va_inputs_pr.append(pr)
                        va_map.append(i)
                n = len(video_paths)
                overall_vec = [float("nan")] * n
                vq_vec = [float("nan")] * n
                mq_vec = [float("nan")] * n
                ta_vec = [float("nan")] * n
                if va_inputs_vp:
                    va_inputs_meta = [metadata[i] for i in va_map]
                    va_scores, _ = self._videoalign_fn(va_inputs_vp, va_inputs_pr, va_inputs_meta)
                    for j, orig_i in enumerate(va_map):
                        for key, vec_key in (
                            ("Overall", overall_vec),
                            ("VQ", vq_vec),
                            ("MQ", mq_vec),
                            ("TA", ta_vec),
                        ):
                            col = va_scores.get(key)
                            if col is not None and j < len(col):
                                vec_key[orig_i] = float(col[j])
                scores["videoalign_overall"] = overall_vec
                scores["overall"] = list(overall_vec)
                scores["vq"] = vq_vec
                scores["mq"] = mq_vec
                scores["ta"] = ta_vec
                a = self._videoalign_fusion_alpha
                b = self._videoalign_fusion_beta
                denom = max(abs(a) + abs(b), 1e-8)
                fused: List[float] = []
                for i in range(n):
                    phys = scores["emo_physics_total"][i]
                    vo = overall_vec[i]
                    if vo == vo:  # not NaN
                        fused.append((a * vo + b * phys) / denom)
                    else:
                        fused.append(phys)
                scores["emo_total"] = fused
                scores["EQ"] = list(fused)
            except Exception:
                pass

        return scores, {}

    def _score_single(
        self,
        video_path: str,
        prompt: str,
        meta: Dict,
    ) -> HierarchicalReward:
        plan = self.planner.plan(prompt, self.video_duration_sec)

        observations = self.observer.observe(
            video_path,
            video_fps=self.video_fps,
            video_duration_sec=self.video_duration_sec,
        )

        motion_result = None
        if self._motion_scorer is not None and video_path and os.path.isfile(video_path):
            target_arousal = None
            if self._motion_naturalness_target_arousal:
                try:
                    target_arousal = float(plan.target_vad[1])
                except Exception:
                    target_arousal = None
            motion_result = self._motion_scorer.score_video(
                video_path,
                target_arousal=target_arousal,
            )

        reward = self.rewarder.reflect(
            plan,
            observations,
            prompt,
            video_path=video_path,
            metadata=meta,
            motion_result=motion_result,
        )
        return reward

    def clear_cache(self) -> None:
        """Reserved for future caches; no-op after error-mask removal."""
        return


# ============================================================================
# Factory Function (for rewards.py integration)
# ============================================================================

def emo_reward_score(
    device,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Factory function to create EmoReward reward scorer.

    Compatible with SAGE-GRPO's multi_video_score model registration.

    Args:
        device: Torch device
        config: Optional configuration dict with keys:
            - vlm_model_path: Plan VLM ("auto" → Qwen2-VL preset). Set "none" for taxonomy-only planning.
            - au_extractor_model_path: "openface" (default) or "none".
            - reward_checkpoint_mode: VideoAlign checkpoint mapping (default "v3")
            - micro_weight, body_weight, temporal_weight: physics blend (only active signals contribute)
            - video_fps, video_duration_sec, sample_fps
            - _dist_sp_size, _dist_rank: SP-aware lazy init (from get_reward_fn)
            - load_emo_reward_heavy_only_on_sp_rank0: default True for distributed training
            - combine_videoalign: fuse VideoAlign Overall with physics (default True)
            - videoalign_fusion_alpha, videoalign_fusion_beta
    """
    import logging
    _log = logging.getLogger(__name__)

    if config is None:
        config = {}

    cfg = dict(config)
    sp_size = max(1, int(cfg.pop("_dist_sp_size", 1) or 1))
    rank = int(cfg.pop("_dist_rank", 0))
    only_sp0 = cfg.get("load_emo_reward_heavy_only_on_sp_rank0", True)
    skip_heavy = (
        only_sp0
        and dist.is_initialized()
        and sp_size > 1
        and (rank % sp_size) != 0
    )

    if skip_heavy:
        _log.info(
            "[EmoReward] Rank %s: SP non-leader (sp_size=%s) — no Qwen2-VL / VideoReward / OpenFace "
            "(prepare_samples_online computes rewards only on rank %% sp_size == 0).",
            rank, sp_size,
        )
        scorer = EmoRewardScorer(
            vlm_planner=None,
            vlm_reflector=None,
            au_extractor_fn=None,
            micro_weight=cfg.get("micro_weight", 0.5),
            body_weight=cfg.get("body_weight", 0.25),
            temporal_weight=cfg.get("temporal_weight", 0.25),
            video_fps=cfg.get("video_fps", 24.0),
            video_duration_sec=cfg.get("video_duration_sec", 5.0),
            sample_fps=cfg.get("sample_fps", 8.0),
            lazy_init_pack=None,
            enable_motion_naturalness=cfg.get("enable_motion_naturalness", True),
            motion_naturalness_weight=cfg.get("motion_naturalness_weight", 0.20),
            motion_naturalness_target_arousal=cfg.get("motion_naturalness_target_arousal", True),
            adaptive_reflect_control=cfg.get("adaptive_reflect_control", True),
            reflect_warmup_start_step=cfg.get("reflect_warmup_start_step", 40),
            reflect_full_on_step=cfg.get("reflect_full_on_step", 100),
        )
    else:
        _log.info(
            "[EmoReward] Rank %s: deferring heavy model load until first reward() "
            "(avoids VRAM spike before sequence-parallel rollout).",
            rank,
        )
        scorer = EmoRewardScorer(
            vlm_planner=None,
            vlm_reflector=None,
            au_extractor_fn=None,
            micro_weight=cfg.get("micro_weight", 0.5),
            body_weight=cfg.get("body_weight", 0.25),
            temporal_weight=cfg.get("temporal_weight", 0.25),
            video_fps=cfg.get("video_fps", 24.0),
            video_duration_sec=cfg.get("video_duration_sec", 5.0),
            sample_fps=cfg.get("sample_fps", 8.0),
            lazy_init_pack=(device, cfg),
            enable_motion_naturalness=cfg.get("enable_motion_naturalness", True),
            motion_naturalness_weight=cfg.get("motion_naturalness_weight", 0.20),
            motion_naturalness_target_arousal=cfg.get("motion_naturalness_target_arousal", True),
            adaptive_reflect_control=cfg.get("adaptive_reflect_control", True),
            reflect_warmup_start_step=cfg.get("reflect_warmup_start_step", 40),
            reflect_full_on_step=cfg.get("reflect_full_on_step", 100),
        )

    def _fn(
        video_paths: List[str],
        prompts: List[str],
        metadata: Optional[List[Dict]] = None,
    ):
        return scorer(video_paths, prompts, metadata)

    _fn._scorer = scorer
    return _fn


def _build_vlm_components(
    vlm_model_path: str,
    device,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    """
    Build Plan VLM (Qwen2-VL family) for emotion → AU expectations.
    Semantic video scores come from VideoAlign attached in EmoRewardScorer, not a second VideoReward pass.

    Args:
        vlm_model_path: Model path or one of "auto", "videoalign" (resolved to VideoReward ckpt dir for path only).
        device: torch device
        config: Optional config dict (emo_reward_plan_model_path, reward_checkpoint_mode, videoreward_path, ...).
    """
    import logging
    _log = logging.getLogger(__name__)

    if config is None:
        config = {}

    reward_checkpoint_mode = config.get("reward_checkpoint_mode", "v3")
    videoreward_path = config.get("videoreward_path", "./ckpts/VideoReward")
    plan_enabled = bool(config.get("emo_reward_plan_enabled", False))
    plan_backend = str(config.get("emo_reward_plan_backend", "gemini")).strip().lower()
    plan_gemini_model = str(config.get("emo_reward_plan_gemini_model", "gemini-2.5-flash")).strip()

    if vlm_model_path in ("auto", "videoalign"):
        vlm_model_path = videoreward_path

    device_obj = torch.device(device) if isinstance(device, str) else device

    # ------------------------------------------------------------------
    # 1. Build Planner backend
    # ------------------------------------------------------------------
    planner_fn = None
    if not plan_enabled or plan_backend in ("taxonomy", "none", "off", "disabled"):
        _log.info(
            "[EmoReward-VLM] Planner disabled or taxonomy-only (enabled=%s, backend=%s).",
            plan_enabled,
            plan_backend,
        )
    elif plan_backend == "gemini":
        try:
            from .gemini_planner import call_gemini_text
            from .prompt_templates import PLAN_SYSTEM_PROMPT, build_plan_messages, parse_plan_response

            _log.info("[EmoReward-VLM] Using Gemini planner backend model=%s", plan_gemini_model)

            def _planner_fn(prompt: str) -> Optional[Dict[str, Any]]:
                messages = [
                    message
                    for message in build_plan_messages(prompt)
                    if str(message.get("role", "")).strip().lower() != "system"
                ]
                answer, _cot, raw_response, request_id = call_gemini_text(
                    messages=messages,
                    system_prompt=PLAN_SYSTEM_PROMPT,
                    model=plan_gemini_model,
                )
                if not answer:
                    _log.warning(
                        "[EmoReward-VLM] Gemini planner returned empty answer (request_id=%s): %s",
                        request_id,
                        raw_response[:512],
                    )
                    return None
                parsed = parse_plan_response(answer)
                if parsed is None:
                    _log.warning(
                        "[EmoReward-VLM] Gemini planner parse failed (request_id=%s): %s",
                        request_id,
                        answer[:512],
                    )
                return parsed

            planner_fn = _planner_fn
            _log.info("[EmoReward-VLM] Gemini planner backend ready.")
        except Exception as exc:
            _log.warning(
                "[EmoReward-VLM] Failed to init Gemini planner backend: %s. Using taxonomy fallback.",
                exc,
            )
    elif plan_backend == "local_vlm":
        try:
            from hyvideo.models.reward_models.vlm_presets import resolve_vlm_path
            from hyvideo.models.reward_models.vl_generative_loader import (
                load_vl_generative_model,
                vl_generate_text,
            )
            from .prompt_templates import build_plan_messages, parse_plan_response

            plan_path_key = config.get("emo_reward_plan_model_path") or config.get(
                "qwen2_vl_base_path", "qwen2-vl-2b",
            )
            base_model_path = resolve_vlm_path(plan_path_key)
            plan_device_map = config.get("emo_reward_plan_device_map")
            _log.info(
                "[EmoReward-VLM] Loading local Plan VLM from %s (device_map=%s)",
                base_model_path,
                plan_device_map,
            )
            planner_model, planner_processor = load_vl_generative_model(
                base_model_path,
                device_obj,
                torch_dtype=torch.bfloat16,
                device_map=plan_device_map,
                trust_remote_code=True,
            )

            def _planner_fn(prompt: str) -> Optional[Dict[str, Any]]:
                messages = build_plan_messages(prompt)
                response_text = vl_generate_text(
                    planner_model,
                    planner_processor,
                    messages,
                    device_obj,
                    max_new_tokens=512,
                    temperature=0.35,
                    do_sample=True,
                )
                return parse_plan_response(response_text)

            planner_fn = _planner_fn
            _log.info("[EmoReward-VLM] Local Plan VLM loaded successfully.")
        except Exception as exc:
            _log.warning(
                "[EmoReward-VLM] Failed to load local Plan VLM: %s. Using taxonomy fallback.",
                exc,
            )
    else:
        _log.warning(
            "[EmoReward-VLM] Unknown planner backend '%s'. Using taxonomy fallback.",
            plan_backend,
        )

    # ------------------------------------------------------------------
    # 2. Build VLM Reflector (scoring: VideoReward model → TA as macro seed)
    # ------------------------------------------------------------------
    reflector_fn = None
    try:
        from hyvideo.models.reward_models.videoalign.inference import VideoVLMRewardInference

        _log.info("[EmoReward-VLM] Loading VideoReward for Reflect from %s (mode=%s)",
                  vlm_model_path, reward_checkpoint_mode)
        reflector_inference = VideoVLMRewardInference(
            vlm_model_path,
            device=str(device_obj),
            dtype=torch.bfloat16,
            reward_checkpoint_mode=reward_checkpoint_mode,
        )

        def _reflector_fn(
            plan: "EmotionPlan",
            obs_summary: Dict[str, Any],
            prompt: str,
        ) -> Dict[str, Any]:
            try:
                reward_entry = reflector_inference.reward(
                    [obs_summary.get("_video_path", "")], [prompt],
                )[0]
                ta_score = max(0.0, min(1.0, float(reward_entry.get("TA", 0.5))))
            except Exception:
                ta_score = 0.5

            face_rate = obs_summary.get("face_detection_rate", 0.0)
            au_ts = obs_summary.get("au_timeseries", {})
            required_aus = set(
                getattr(getattr(plan, "target_emotion", None), "required_aus", [])
            )
            au_compliance = 0.0
            if required_aus and au_ts:
                detected = sum(
                    1 for au_id in required_aus
                    if au_ts.get(au_id) and any(v > 0.3 for _, v in au_ts[au_id])
                )
                au_compliance = detected / max(len(required_aus), 1)

            physical_weight = min(face_rate, 1.0)
            macro_score = 0.6 * ta_score + 0.4 * (au_compliance * physical_weight)
            reasoning = f"TA={ta_score:.2f},AU_comp={au_compliance:.2f},face={face_rate:.2f}"
            return {"macro_score": macro_score, "reasoning": reasoning}

        reflector_fn = _reflector_fn
        _log.info("[EmoReward-VLM] Reflect VLM (VideoReward) loaded.")
    except Exception as exc:
        _log.warning("[EmoReward-VLM] Failed to load Reflect VLM: %s. Using heuristic fallback.", exc)

    return planner_fn, reflector_fn


def _build_au_extractor(au_model_path: str, device) -> Optional[Callable]:
    """
    Build AU extraction function from model path using OpenFace 3.0.

    The returned callable has the signature:
        (video_path: str, frame_indices: List[int]) -> List[dict]
    Each dict contains:
        {'au_activations': {au_id: float}, 'face_detected': bool,
         'face_bbox': (x1,y1,x2,y2), 'body_markers': {str: bool}}

    Args:
        au_model_path: Path to OpenFace weights directory, or one of:
            - A directory containing .pth / .pkl weight files
            - The string "openface" to auto-resolve from the bundled OpenFace-3.0
        device: torch device (e.g. torch.device('cuda'))
    """
    try:
        from .openface_adapter import build_openface_au_extractor, OPENFACE_DIR

        if au_model_path.lower() in ("openface", "openface-3.0", "auto"):
            weights_dir = os.path.join(OPENFACE_DIR, "weights")
        elif os.path.isdir(au_model_path):
            weights_dir = au_model_path
        else:
            weights_dir = os.path.dirname(au_model_path)

        device_str = str(device) if not isinstance(device, str) else device
        au_fn = build_openface_au_extractor(
            weights_dir=weights_dir,
            device=device_str,
            face_conf_threshold=0.5,
        )
        return au_fn

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to build OpenFace AU extractor: %s. "
            "Falling back to empty observations.", e)
        return None
