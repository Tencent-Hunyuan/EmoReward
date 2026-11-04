"""
OpenFace 3.0 Adapter — bridges OpenFace-3.0 AU/Emotion/Gaze extraction with
the EmoReward ReAct reward pipeline.

This module provides:
  1. OpenFaceExtractor: loads all three OpenFace sub-models (RetinaFace, STAR,
     MLT) and offers a unified API for video-level and frame-level analysis.
  2. build_openface_au_extractor(): factory returning an `au_extractor_fn`
     compatible with react_reward.PhysicalObserver.
  3. OpenFaceVideoAnalyzer: higher-level class that caches per-video results and
     exposes structured output (Emotion, Gaze, AU, Landmarks).

Architecture fit (ReAct Reward Loop):
    ┌──────────────────────────────────────────────────────┐
    │          Act (AU Obs) — OpenFace 3.0 Adapter         │
    │  RetinaFace ──> STAR Landmarks ──> MLT (multi-task)  │
    │  ─── per-frame: emotion logits, gaze, 8 AU scores ──│
    └──────────────────────────────────────────────────────┘

The 8 AU channels produced by MLT correspond to:
    Index 0 → AU1  (Inner Brow Raise)
    Index 1 → AU2  (Outer Brow Raise)
    Index 2 → AU4  (Brow Lowerer)
    Index 3 → AU6  (Cheek Raiser)
    Index 4 → AU9  (Nose Wrinkler)
    Index 5 → AU12 (Lip Corner Puller)
    Index 6 → AU25 (Lips Part)
    Index 7 → AU26 (Jaw Drop)
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENFACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenFace-3.0")

AU_INDEX_TO_ID = {0: 1, 1: 2, 2: 4, 3: 6, 4: 9, 5: 12, 6: 25, 7: 26}
AU_ID_TO_INDEX = {v: k for k, v in AU_INDEX_TO_ID.items()}

EXPRESSION_LABELS = [
    "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"
]


# ---------------------------------------------------------------------------
# OpenFaceExtractor — core model wrapper
# ---------------------------------------------------------------------------

class OpenFaceExtractor:
    """
    Wraps the three OpenFace-3.0 sub-models into a single inference engine.

    Models loaded:
        - RetinaFace (MobileNet-0.25): face detection + 5-point landmarks
        - STAR alignment: 98-point facial landmark refinement
        - MLT (EfficientNet-B0-ns multi-task): emotion (8-class), gaze (yaw/pitch),
          AU intensity (8 AUs via GNN head)

    Options:
        weights_dir  — directory containing .pth (MLT, RetinaFace) and .pkl (STAR)
        device       — 'cpu', 'cuda', 'cuda:0', etc.
        face_conf_threshold — minimum RetinaFace confidence for a valid face
        bbox_scale_factor   — how much to enlarge the detection box before crop
        enable_landmarks    — whether to also run STAR landmark detection
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        device: str = "cpu",
        face_conf_threshold: float = 0.5,
        bbox_scale_factor: float = 0.15,
        enable_landmarks: bool = False,
    ):
        self.device_str = device
        self.face_conf_threshold = face_conf_threshold
        self.bbox_scale_factor = bbox_scale_factor
        self.enable_landmarks = enable_landmarks
        self._models_loaded = False

        self.weights_dir = weights_dir or os.path.join(OPENFACE_DIR, "weights")

        self._mlt_model = None
        self._retinaface_model = None
        self._alignment = None
        self._transform = None
        self._device = None

    def load_models(self):
        """Lazily load all sub-models into memory."""
        if self._models_loaded:
            return

        _ensure_openface_on_path()
        import torch
        from torchvision import transforms

        self._device = torch.device(self.device_str)
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        from model.MLT import MLT
        from Pytorch_Retinaface.models.retinaface import RetinaFace
        from Pytorch_Retinaface.data import cfg_mnet
        from Pytorch_Retinaface.detect import load_model as rf_load_model

        mlt_path, retina_path, landmark_path = _resolve_weight_paths(self.weights_dir)

        self._mlt_model = MLT()
        self._mlt_model.load_state_dict(
            torch.load(mlt_path, map_location=self._device))
        self._mlt_model.eval().to(self._device)

        self._retinaface_model = RetinaFace(cfg=cfg_mnet, phase="test")
        self._retinaface_model = rf_load_model(
            self._retinaface_model, retina_path,
            self._device.type == "cpu")
        self._retinaface_model.eval().to(self._device)

        if self.enable_landmarks and landmark_path:
            from STAR.demo import Alignment
            config = {
                "config_name": "alignment",
                "device_id": (self._device.index
                              if self._device.type == "cuda" else -1),
            }
            args_ns = argparse.Namespace(**config)
            dev_ids = ([0] if self._device.type == "cuda" else [-1])
            self._alignment = Alignment(
                args_ns, landmark_path,
                dl_framework="pytorch", device_ids=dev_ids)

        self._models_loaded = True
        logger.info("OpenFace 3.0 models loaded on %s", self.device_str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frame(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze a single BGR frame.

        Returns a list of per-face dicts:
          {
            'face_detected': True,
            'confidence': float,
            'bbox': (x1, y1, x2, y2),
            'emotion_logits': np.ndarray (8,),
            'emotion_label': str,
            'emotion_index': int,
            'gaze': np.ndarray (2,),        # [yaw, pitch]
            'au_raw': np.ndarray (8,),       # raw MLT output per AU index
            'au_activations': {int: float},  # {AU_ID: intensity}
            'landmarks': np.ndarray or None, # (98, 2) if enabled
          }
        """
        self.load_models()
        import torch
        faces = self._detect_and_crop(frame_bgr)
        results = []
        for face_info in faces:
            cropped_bgr = face_info["crop"]
            emo_logits, gaze, au_raw = self._predict_multitask(cropped_bgr)

            au_activations = {}
            for idx, au_id in AU_INDEX_TO_ID.items():
                val = float(au_raw[idx])
                au_activations[au_id] = max(0.0, min(1.0, val))

            emo_idx = int(np.argmax(emo_logits))

            landmarks = None
            if self.enable_landmarks and self._alignment is not None:
                landmarks = self._detect_landmarks_single(
                    frame_bgr, face_info["det"])

            results.append({
                "face_detected": True,
                "confidence": float(face_info["confidence"]),
                "bbox": face_info["bbox"],
                "emotion_logits": emo_logits,
                "emotion_label": EXPRESSION_LABELS[emo_idx],
                "emotion_index": emo_idx,
                "gaze": gaze,
                "au_raw": au_raw,
                "au_activations": au_activations,
                "landmarks": landmarks,
            })
        return results

    def analyze_video(
        self,
        video_path: str,
        frame_indices: Optional[List[int]] = None,
        sample_fps: float = 8.0,
        max_frames: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Analyze selected frames of a video.

        Args:
            video_path: path to video file
            frame_indices: explicit frame indices to process. If None, samples
                           at `sample_fps` up to `max_frames`.

        Returns:
            List of per-frame dicts (same structure as analyze_frame output,
            plus 'frame_idx' and 'timestamp_sec' fields).
            If no face is found in a frame, returns a stub dict with
            face_detected=False.
        """
        self.load_models()
        import cv2 as _cv2

        cap = _cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(_cv2.CAP_PROP_FPS) or 24.0
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))

        if frame_indices is None:
            interval = max(1, int(fps / sample_fps))
            frame_indices = list(range(0, min(total, max_frames * interval), interval))

        frame_set = set(frame_indices)
        sorted_indices = sorted(frame_set)
        frame_map: Dict[int, np.ndarray] = {}

        idx = 0
        ptr = 0
        while cap.isOpened() and ptr < len(sorted_indices):
            ret, frame = cap.read()
            if not ret:
                break
            if idx == sorted_indices[ptr]:
                frame_map[idx] = frame
                ptr += 1
                while ptr < len(sorted_indices) and sorted_indices[ptr] == idx:
                    ptr += 1
            idx += 1
        cap.release()

        all_results = []
        for fi in frame_indices:
            frame = frame_map.get(fi)
            if frame is None:
                all_results.append({
                    "frame_idx": fi,
                    "timestamp_sec": fi / fps,
                    "face_detected": False,
                    "au_activations": {},
                    "body_markers": {},
                })
                continue

            faces = self.analyze_frame(frame)
            if not faces:
                all_results.append({
                    "frame_idx": fi,
                    "timestamp_sec": fi / fps,
                    "face_detected": False,
                    "au_activations": {},
                    "body_markers": {},
                })
            else:
                best = max(faces, key=lambda f: f["confidence"])
                best["frame_idx"] = fi
                best["timestamp_sec"] = fi / fps
                best["body_markers"] = {}
                all_results.append(best)

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_and_crop(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Run RetinaFace and crop the detected faces."""
        import torch
        from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
        from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
        from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
        from Pytorch_Retinaface.data import cfg_mnet

        img = np.float32(frame_bgr.copy())
        im_h, im_w, _ = img.shape
        scale = torch.Tensor([im_w, im_h, im_w, im_h]).to(self._device)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img_t = torch.from_numpy(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            loc, conf, landms = self._retinaface_model(img_t)

        priorbox = PriorBox(cfg_mnet, image_size=(im_h, im_w))
        priors = priorbox.forward().to(self._device)
        boxes = decode(loc.data.squeeze(0), priors.data, cfg_mnet["variance"])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        scale1 = torch.Tensor([img_t.shape[3], img_t.shape[2]] * 5).to(self._device)
        landms = decode_landm(landms.data.squeeze(0), priors.data, cfg_mnet["variance"])
        landms = (landms * scale1).cpu().numpy()

        inds = np.where(scores > 0.02)[0]
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]
        order = scores.argsort()[::-1]
        boxes, landms, scores = boxes[order], landms[order], scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep]
        landms = landms[keep]
        dets = np.concatenate((dets, landms), axis=1)

        faces = []
        for det in dets:
            c = det[4]
            if c < self.face_conf_threshold:
                continue
            x1, y1, x2, y2 = det[:4].astype(int)
            sf = self.bbox_scale_factor
            bw, bh = x2 - x1, y2 - y1
            dx, dy = int(bw * sf), int(bh * sf)
            x1s = max(x1 - dx, 0)
            y1s = max(y1 - dy, 0)
            x2s = min(x2 + dx, im_w)
            y2s = min(y2 + dy, im_h)
            crop = frame_bgr[y1s:y2s, x1s:x2s]
            if crop.size == 0:
                continue
            faces.append({
                "crop": crop,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(c),
                "det": det,
            })
        return faces

    def _predict_multitask(self, face_bgr: np.ndarray):
        """Run MLT on a cropped face. Returns (emotion, gaze, au) as numpy."""
        import torch
        from PIL import Image
        import cv2 as _cv2

        face_rgb = _cv2.cvtColor(face_bgr, _cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(face_rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emo, gaze, au = self._mlt_model(tensor)
        return (emo[0].cpu().numpy(),
                gaze[0].cpu().numpy(),
                au[0].cpu().numpy())

    def _detect_landmarks_single(self, image_bgr, det):
        """Run STAR alignment on a single detection."""
        x1, y1, x2, y2 = det[:4].astype(int)
        center_w = (x2 + x1) / 2.0
        center_h = (y2 + y1) / 2.0
        s = min(x2 - x1, y2 - y1) / 200.0 * 1.05
        try:
            return self._alignment.analyze(image_bgr, float(s),
                                           float(center_w), float(center_h))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# OpenFaceVideoAnalyzer — higher-level wrapper with caching & batching
# ---------------------------------------------------------------------------

class OpenFaceVideoAnalyzer:
    """
    Video-level analyzer that caches results and provides aggregate statistics.

    Designed for integration with the ReAct reward pipeline. Holds a single
    OpenFaceExtractor instance and provides structured outputs.

    Usage:
        analyzer = OpenFaceVideoAnalyzer(weights_dir="...", device="cuda")
        result = analyzer.analyze(video_path, sample_fps=8.0)
        # result keys: 'per_frame', 'aggregate', 'au_timeseries', 'emotion_distribution'
    """

    def __init__(self, weights_dir: Optional[str] = None, device: str = "cpu",
                 face_conf_threshold: float = 0.5, enable_landmarks: bool = False):
        self.extractor = OpenFaceExtractor(
            weights_dir=weights_dir,
            device=device,
            face_conf_threshold=face_conf_threshold,
            enable_landmarks=enable_landmarks,
        )
        self._cache: Dict[str, Dict] = {}

    def analyze(
        self,
        video_path: str,
        frame_indices: Optional[List[int]] = None,
        sample_fps: float = 8.0,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Full analysis pipeline with aggregate statistics."""
        cache_key = f"{video_path}:{frame_indices}:{sample_fps}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        per_frame = self.extractor.analyze_video(
            video_path, frame_indices=frame_indices, sample_fps=sample_fps)

        detected = [f for f in per_frame if f.get("face_detected", False)]
        n_total = len(per_frame)
        n_detected = len(detected)

        # AU timeseries: {AU_ID: [(timestamp, value), ...]}
        au_ts = {}
        for au_id in AU_INDEX_TO_ID.values():
            au_ts[au_id] = [
                (f["timestamp_sec"], f["au_activations"].get(au_id, 0.0))
                for f in detected
            ]

        # Emotion distribution across detected frames
        emo_counts = {label: 0 for label in EXPRESSION_LABELS}
        for f in detected:
            emo_counts[f.get("emotion_label", "Neutral")] += 1

        # Mean AU intensities
        mean_au = {}
        for au_id in AU_INDEX_TO_ID.values():
            vals = [f["au_activations"].get(au_id, 0.0) for f in detected]
            mean_au[au_id] = float(np.mean(vals)) if vals else 0.0

        result = {
            "per_frame": per_frame,
            "aggregate": {
                "total_frames": n_total,
                "faces_detected": n_detected,
                "face_detection_rate": n_detected / max(n_total, 1),
                "mean_au_intensities": mean_au,
                "dominant_emotion": max(emo_counts, key=emo_counts.get) if detected else "Unknown",
            },
            "au_timeseries": au_ts,
            "emotion_distribution": emo_counts,
        }

        if use_cache:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# Factory: build_openface_au_extractor
# ---------------------------------------------------------------------------

def build_openface_au_extractor(
    weights_dir: Optional[str] = None,
    device: str = "cpu",
    face_conf_threshold: float = 0.5,
) -> Callable:
    """
    Factory that returns an `au_extractor_fn` compatible with
    react_reward.PhysicalObserver.

    The returned callable has the signature:
        (video_path: str, frame_indices: List[int]) -> List[dict]

    Each dict in the returned list contains:
        {
            'au_activations': {au_id_int: float},  # e.g. {1: 0.82, 4: 0.13, ...}
            'face_detected': bool,
            'face_bbox': (x1, y1, x2, y2) or None,
            'body_markers': {},    # placeholder — OpenFace doesn't detect body
            'emotion_label': str,  # bonus: emotion label for logging
            'emotion_logits': list,
            'gaze': list,
            'channel_signals': {   # per-frame channel scores [0, 1]
                'chromatic_score': float,       # color saturation/warmth proxy
                'kinetic_score': float,         # frame-to-frame motion magnitude proxy
                'compositional_score': float,   # face centrality and relative size proxy
            },
        }

    Usage in react_reward.py:
        from .openface_adapter import build_openface_au_extractor
        au_fn = build_openface_au_extractor(weights_dir="...", device="cuda")
        observer = PhysicalObserver(au_extractor_fn=au_fn)
    """
    extractor = OpenFaceExtractor(
        weights_dir=weights_dir,
        device=device,
        face_conf_threshold=face_conf_threshold,
        enable_landmarks=False,
    )

    def _au_extractor_fn(
        video_path: str,
        frame_indices: List[int],
    ) -> List[Dict[str, Any]]:
        results = extractor.analyze_video(
            video_path, frame_indices=frame_indices)

        # Fetch raw frames for channel signal computation
        raw_frames = _load_frames_for_channel_signals(video_path, frame_indices)

        formatted = []
        prev_frame = None
        for i, r in enumerate(results):
            frame_bgr = raw_frames.get(i)
            channel_signals = _compute_channel_signals(frame_bgr, prev_frame, r)
            if frame_bgr is not None:
                prev_frame = frame_bgr

            formatted.append({
                "au_activations": r.get("au_activations", {}),
                "face_detected": r.get("face_detected", False),
                "face_bbox": r.get("bbox"),
                "body_markers": r.get("body_markers", {}),
                "emotion_label": r.get("emotion_label", "Unknown"),
                "emotion_logits": (r.get("emotion_logits").tolist()
                                   if r.get("emotion_logits") is not None
                                   else []),
                "gaze": (r.get("gaze").tolist()
                         if r.get("gaze") is not None else []),
                "channel_signals": channel_signals,
            })
        return formatted

    _au_extractor_fn._extractor = extractor
    return _au_extractor_fn


def _load_frames_for_channel_signals(
    video_path: str,
    frame_indices: List[int],
) -> Dict[int, Any]:
    """
    Load raw BGR frames at the given frame_indices positions for channel-signal computation.
    Returns a dict mapping list-position-index -> frame_bgr (numpy array).
    Falls back gracefully if video cannot be opened.
    """
    frame_map: Dict[int, Any] = {}
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frame_map
        sorted_unique = sorted(set(frame_indices))
        # Map frame_index (original) → list position(s)
        idx_to_pos: Dict[int, List[int]] = {}
        for pos, fi in enumerate(frame_indices):
            idx_to_pos.setdefault(fi, []).append(pos)
        ptr = 0
        frame_counter = 0
        while cap.isOpened() and ptr < len(sorted_unique):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_counter == sorted_unique[ptr]:
                for pos in idx_to_pos[frame_counter]:
                    frame_map[pos] = frame.copy()
                ptr += 1
                while ptr < len(sorted_unique) and sorted_unique[ptr] == frame_counter:
                    ptr += 1
            frame_counter += 1
        cap.release()
    except Exception:
        pass
    return frame_map


def _compute_channel_signals(
    frame_bgr: Optional[Any],
    prev_frame_bgr: Optional[Any],
    au_result: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute per-frame chromatic, kinetic, and compositional channel signals.

    chromatic_score  [0,1] : Mean HSV saturation × warmth heuristic.
                             Higher → more saturated / warm-toned frame.
    kinetic_score    [0,1] : Normalized mean absolute frame difference.
                             Higher → more motion between frames.
    compositional_score [0,1]: How well the detected face is centred and
                             sized relative to the frame. 1.0 = ideal rule-of-thirds centred.

    Falls back to 0.5 (neutral) if the required data is unavailable.
    """
    signals: Dict[str, float] = {
        "chromatic_score": 0.5,
        "kinetic_score": 0.5,
        "compositional_score": 0.5,
    }

    if frame_bgr is None:
        return signals

    try:
        import cv2 as _cv2

        # --- chromatic_score ---
        hsv = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0          # [0,1]
        hue = hsv[:, :, 0]                          # [0,179] in OpenCV

        mean_sat = float(np.mean(saturation))
        # Warm hue range (red-yellow): 0-30 and 150-179 in OpenCV HSV
        warm_mask = ((hue <= 30) | (hue >= 150)).astype(np.float32)
        mean_warm = float(np.mean(warm_mask))
        # Blend saturation and warmth equally
        chromatic = 0.5 * mean_sat + 0.5 * mean_warm
        signals["chromatic_score"] = float(np.clip(chromatic, 0.0, 1.0))

        # --- kinetic_score ---
        if prev_frame_bgr is not None:
            try:
                # Resize to same shape if needed
                h1, w1 = frame_bgr.shape[:2]
                h2, w2 = prev_frame_bgr.shape[:2]
                if h1 != h2 or w1 != w2:
                    prev_resized = _cv2.resize(prev_frame_bgr, (w1, h1))
                else:
                    prev_resized = prev_frame_bgr
                diff = np.abs(frame_bgr.astype(np.float32) - prev_resized.astype(np.float32))
                mean_diff = float(np.mean(diff)) / 255.0   # [0,1]
                # Typical non-trivial motion: ~0.02–0.10; saturate at 0.20
                kinetic = float(np.clip(mean_diff / 0.20, 0.0, 1.0))
                signals["kinetic_score"] = kinetic
            except Exception:
                pass

        # --- compositional_score ---
        bbox = au_result.get("bbox")
        face_detected = au_result.get("face_detected", False)
        if face_detected and bbox is not None:
            try:
                h, w = frame_bgr.shape[:2]
                x1, y1, x2, y2 = bbox
                face_cx = (x1 + x2) / 2.0
                face_cy = (y1 + y2) / 2.0
                face_w = float(x2 - x1)
                face_h = float(y2 - y1)

                # Centrality: distance from frame centre, normalised
                cx_norm = float(face_cx) / max(w, 1)
                cy_norm = float(face_cy) / max(h, 1)
                dist_from_centre = np.sqrt((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2)
                # max possible dist ≈ 0.707; penalise linearly
                centrality = float(np.clip(1.0 - dist_from_centre / 0.5, 0.0, 1.0))

                # Size ratio: ideal face occupies 10–40% of frame area
                face_area_ratio = (face_w * face_h) / max(w * h, 1)
                # Sigmoid-like: peak at ~0.20 area ratio
                if face_area_ratio < 0.05:
                    size_score = face_area_ratio / 0.05
                elif face_area_ratio < 0.40:
                    size_score = 1.0
                else:
                    size_score = max(0.0, 1.0 - (face_area_ratio - 0.40) / 0.40)

                compositional = 0.6 * centrality + 0.4 * size_score
                signals["compositional_score"] = float(np.clip(compositional, 0.0, 1.0))
            except Exception:
                pass

    except Exception:
        pass

    return signals


# ---------------------------------------------------------------------------
# Utility: path management
# ---------------------------------------------------------------------------

def _ensure_openface_on_path():
    """Add OpenFace-3.0 and its submodules to sys.path for imports."""
    dirs = [
        OPENFACE_DIR,
        os.path.join(OPENFACE_DIR, "STAR"),
        os.path.join(OPENFACE_DIR, "Pytorch_Retinaface"),
    ]
    for d in dirs:
        if d not in sys.path:
            sys.path.insert(0, d)


def _resolve_weight_paths(
    weights_dir: str,
) -> Tuple[str, str, Optional[str]]:
    """
    Auto-discover model weight files in the given directory.

    Returns: (mlt_path, retinaface_path, landmark_path_or_None)
    """
    import glob

    pth_files = glob.glob(os.path.join(weights_dir, "*.pth"))
    pkl_files = glob.glob(os.path.join(weights_dir, "*.pkl"))

    mlt_path = retina_path = landmark_path = None

    for p in pth_files:
        bn = os.path.basename(p).lower()
        if "mobilenet" in bn or "retinaface" in bn:
            retina_path = p
        elif "stage" in bn or "mlt" in bn or "backbone" in bn:
            mlt_path = p

    for p in pkl_files:
        bn = os.path.basename(p).lower()
        if any(kw in bn for kw in ("landmark", "star", "wflw", "300w")):
            landmark_path = p

    if mlt_path is None:
        for p in pth_files:
            if p != retina_path:
                mlt_path = p
                break

    if mlt_path is None:
        raise FileNotFoundError(
            f"MLT weight (.pth) not found in {weights_dir}. "
            "Download from https://huggingface.co/nutPace/openface_weights")
    if retina_path is None:
        raise FileNotFoundError(
            f"RetinaFace weight (.pth) not found in {weights_dir}. "
            "Expected mobilenet0.25_Final.pth")

    return mlt_path, retina_path, landmark_path
