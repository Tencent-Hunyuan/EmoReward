"""
Motion naturalness scorer for generated videos.

This module intentionally stays lightweight: it uses classical CV signals only,
so it can run alongside EmoReward/VideoAlign without introducing another heavy model.

Core idea:
  - reward smooth, coherent motion trajectories
  - penalize abrupt global camera jerk
  - reward frame-to-frame warp consistency
  - penalize temporal flicker after coarse motion compensation

The scorer returns a [0, 1] reward plus interpretable sub-signals. A small
arousal-conditioned style term can be injected when the caller provides a
target arousal, but the main objective remains motion plausibility rather than
motion amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging
import math
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionNaturalnessResult:
    scorer_name: str
    video_path: str
    motion_naturalness_reward: float = 0.5
    flow_smoothness: float = 0.5
    camera_smoothness: float = 0.5
    warp_consistency: float = 0.5
    flicker_consistency: float = 0.5
    activity_score: float = 0.5
    arousal_match: float = 0.5
    frame_count: int = 0
    confidence: float = 0.0
    active: bool = False
    error: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)


class MotionNaturalnessScorer:
    """Classical CV motion-naturalness scorer."""

    def __init__(
        self,
        resize_short_side: int = 160,
        max_frames: int = 18,
        frame_stride: int = 4,
        farneback_levels: int = 3,
        farneback_winsize: int = 15,
        farneback_iterations: int = 3,
        farneback_poly_n: int = 5,
        farneback_poly_sigma: float = 1.2,
    ):
        self.resize_short_side = int(resize_short_side)
        self.max_frames = int(max_frames)
        self.frame_stride = int(frame_stride)
        self.farneback_levels = int(farneback_levels)
        self.farneback_winsize = int(farneback_winsize)
        self.farneback_iterations = int(farneback_iterations)
        self.farneback_poly_n = int(farneback_poly_n)
        self.farneback_poly_sigma = float(farneback_poly_sigma)

    def score_video(
        self,
        video_path: str,
        target_arousal: Optional[float] = None,
    ) -> MotionNaturalnessResult:
        result = MotionNaturalnessResult(
            scorer_name="motion_naturalness",
            video_path=video_path,
        )
        try:
            frames = self._load_frames(video_path)
            result.frame_count = len(frames)
            if len(frames) < 3:
                result.error = "too_few_frames"
                return result

            flows = []
            mean_mags = []
            global_motion = []
            warp_maes = []
            flicker_maes = []
            brightness = []

            prev = frames[0]
            brightness.append(float(prev.mean()))
            for curr in frames[1:]:
                brightness.append(float(curr.mean()))
                flow = cv2.calcOpticalFlowFarneback(
                    prev,
                    curr,
                    None,
                    pyr_scale=0.5,
                    levels=self.farneback_levels,
                    winsize=self.farneback_winsize,
                    iterations=self.farneback_iterations,
                    poly_n=self.farneback_poly_n,
                    poly_sigma=self.farneback_poly_sigma,
                    flags=0,
                )
                flows.append(flow)
                mag = np.linalg.norm(flow, axis=2)
                mean_mags.append(float(mag.mean()))
                global_vec = flow.reshape(-1, 2).mean(axis=0)
                global_motion.append(global_vec)
                warp_maes.append(self._warp_mae(prev, curr, flow))
                flicker_maes.append(self._compensated_flicker(prev, curr, global_vec))
                prev = curr

            global_motion_arr = np.asarray(global_motion, dtype=np.float32)
            mean_mags_arr = np.asarray(mean_mags, dtype=np.float32)
            brightness_arr = np.asarray(brightness, dtype=np.float32)
            warp_maes_arr = np.asarray(warp_maes, dtype=np.float32)
            flicker_maes_arr = np.asarray(flicker_maes, dtype=np.float32)

            # Smooth motion magnitude changes are preferred.
            mag_delta = np.diff(mean_mags_arr) if len(mean_mags_arr) >= 2 else np.zeros(1, dtype=np.float32)
            flow_smoothness = self._exp_score(float(np.mean(np.abs(mag_delta))), scale=0.35)

            # Penalize jerk of the global camera / dominant motion trajectory.
            cam_acc = np.diff(global_motion_arr, axis=0) if len(global_motion_arr) >= 2 else np.zeros((1, 2), dtype=np.float32)
            cam_jerk = np.diff(cam_acc, axis=0) if len(cam_acc) >= 2 else cam_acc
            camera_smoothness = self._exp_score(
                float(np.mean(np.linalg.norm(cam_jerk, axis=1))) if len(cam_jerk) else 0.0,
                scale=0.65,
            )

            # If optical flow can warp frame_t to frame_t+1 with low error, motion is more coherent.
            warp_consistency = self._exp_score(float(np.mean(warp_maes_arr)), scale=0.085)

            # After coarse global compensation, large residual flicker suggests temporal instability.
            flicker_consistency = self._exp_score(float(np.mean(flicker_maes_arr)), scale=0.11)

            activity_score = 1.0 - math.exp(-float(np.mean(mean_mags_arr)) / 1.5)
            activity_score = float(max(0.0, min(1.0, activity_score)))
            if target_arousal is not None:
                arousal = float(max(0.0, min(1.0, target_arousal)))
                arousal_match = 1.0 - abs(activity_score - arousal)
            else:
                arousal_match = 0.5

            base_reward = (
                0.34 * flow_smoothness
                + 0.24 * camera_smoothness
                + 0.24 * warp_consistency
                + 0.18 * flicker_consistency
            )
            if target_arousal is None:
                reward = base_reward
            else:
                reward = 0.82 * base_reward + 0.18 * arousal_match

            frame_conf = min(1.0, max(0.0, (len(frames) - 2) / 8.0))
            motion_conf = min(1.0, max(0.0, float(np.std(mean_mags_arr)) / 0.035))
            confidence = 0.6 * frame_conf + 0.4 * motion_conf

            result.motion_naturalness_reward = float(max(0.0, min(1.0, reward)))
            result.flow_smoothness = float(flow_smoothness)
            result.camera_smoothness = float(camera_smoothness)
            result.warp_consistency = float(warp_consistency)
            result.flicker_consistency = float(flicker_consistency)
            result.activity_score = float(activity_score)
            result.arousal_match = float(max(0.0, min(1.0, arousal_match)))
            result.confidence = float(max(0.0, min(1.0, confidence)))
            result.active = True
            result.debug = {
                "mean_motion_magnitude": float(np.mean(mean_mags_arr)),
                "motion_magnitude_std": float(np.std(mean_mags_arr)),
                "brightness_std": float(np.std(brightness_arr)),
                "warp_mae_mean": float(np.mean(warp_maes_arr)),
                "flicker_mae_mean": float(np.mean(flicker_maes_arr)),
            }
        except Exception as exc:
            import traceback

            result.error = str(exc)
            logger.warning(
                "[MotionNaturalness] score_video FAILED for %s: %s\n%s",
                video_path,
                exc,
                traceback.format_exc(),
            )
        return result

    def _load_frames(self, video_path: str) -> List[np.ndarray]:
        if not video_path or not os.path.isfile(video_path):
            return []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        indices = list(range(0, total, max(self.frame_stride, 1)))[: self.max_frames]
        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            short = min(h, w)
            if short > self.resize_short_side:
                scale = self.resize_short_side / float(short)
                gray = cv2.resize(
                    gray,
                    (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(gray.astype(np.float32) / 255.0)
        cap.release()
        return frames

    @staticmethod
    def _exp_score(value: float, scale: float) -> float:
        scale = max(float(scale), 1e-6)
        return float(max(0.0, min(1.0, math.exp(-max(0.0, value) / scale))))

    @staticmethod
    def _warp_mae(prev: np.ndarray, curr: np.ndarray, flow: np.ndarray) -> float:
        h, w = prev.shape
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        remap_x = grid_x + flow[..., 0].astype(np.float32)
        remap_y = grid_y + flow[..., 1].astype(np.float32)
        warped = cv2.remap(
            prev,
            remap_x,
            remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return float(np.mean(np.abs(warped - curr)))

    @staticmethod
    def _compensated_flicker(prev: np.ndarray, curr: np.ndarray, global_vec: np.ndarray) -> float:
        h, w = prev.shape
        tx, ty = float(global_vec[0]), float(global_vec[1])
        affine = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        shifted = cv2.warpAffine(
            prev,
            affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return float(np.mean(np.abs(shifted - curr)))

