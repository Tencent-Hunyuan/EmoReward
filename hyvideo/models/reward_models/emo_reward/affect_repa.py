"""
Agent-Guided AffectREPA (AG-AffectREPA)

Dynamic representation alignment module that uses VLM-generated error masks
to selectively enforce intermediate latent alignment with a frozen FACS-encoder
only in emotionally faulted regions.

Core Formula:
    L_Total = L_SAGE-GRPO + λ · Mask_Agent ⊙ L_AffectREPA

Design Principles (from Proposal Section 3):
1. Dynamic Fault Localization: ReAct Reflection produces spatio-temporal error masks
2. Straightforward RL Fusion: AffectREPA auxiliary loss added during SAGE-GRPO backward pass
3. FACS-Teacher Alignment: Aligns T2V model hidden states with frozen FACS encoder features
4. "Fix Where It Fails": Only regions identified as emotionally distorted are aligned

Inspired by:
- REPA (Representation Alignment for Generation, arxiv:2410.06940)
- AG-REPA (Attribution-Guided REPA, arxiv:2603.01006)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any, Callable

from .emotion_taxonomy import (
    BODY_REGION_GROUPS,
    AU_TO_MUSCLE_REGION,
)


# ============================================================================
# FACS Feature Extraction Interface (placeholder for SOTA emotion model)
# ============================================================================

def get_facs_features(
    video_frames: torch.Tensor,
    model: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract FACS-aligned feature representations from video frames using a
    specialized emotion understanding model (frozen FACS Teacher Encoder).

    This is the placeholder function that awaits implementation with an actual
    SOTA emotion understanding model (e.g., OpenFace, LibreFace, AU-focused
    pretrained encoder, or any model producing per-region emotion features).

    Args:
        video_frames: Input video frames tensor.
            Shape: [B, C, T, H, W] (batch, channels, temporal, height, width)
            or [B, T, C, H, W] depending on the model's expected input format.
            Values should be in [0, 1] or [-1, 1] range depending on model.

        model: The frozen FACS teacher encoder model instance.
            When None, returns zero features (graceful degradation).
            When implemented, this should be a model that outputs:
            - Per-frame feature maps with spatial structure preserved
            - Feature dimension should encode AU-relevant information

            Recommended model architectures to implement:
            1. OpenFace-based: Extract AU intensities + intermediate CNN features
            2. LibreFace: AU detection with transformer features
            3. Custom: Fine-tuned DINOv2/MAE on AU detection task
            4. Hume AI-style: Multi-dimensional emotion feature extractor

        device: Target device for the output features.

    Returns:
        features: FACS-aligned feature tensor.
            Shape: [B, T, D_facs] where D_facs is the feature dimension.
            or [B, T, H', W', D_facs] if spatial structure is preserved.
            These features should encode per-region emotional information
            that can be aligned with the T2V model's intermediate hidden states.

    Implementation Guide:
        When implementing with an actual model, ensure:
        1. The model is FROZEN (no gradients) - it serves as a teacher
        2. Features should be L2-normalized for stable alignment
        3. Spatial structure should be preserved when possible (for masked alignment)
        4. The feature dimension D_facs should be documented in the config

    Example implementation sketch:
        ```python
        def get_facs_features(video_frames, model, device):
            model.eval()
            with torch.no_grad():
                # Reshape if needed: [B, C, T, H, W] -> [B*T, C, H, W]
                B, C, T, H, W = video_frames.shape
                frames_2d = video_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                # Extract features
                features = model.encode(frames_2d.to(device))  # [B*T, D_facs]
                # Reshape back: [B*T, D_facs] -> [B, T, D_facs]
                features = features.reshape(B, T, -1)
                # L2 normalize
                features = F.normalize(features, dim=-1)
                return features
        ```
    """
    if model is None:
        if video_frames is None:
            return torch.zeros(1, 1, 512)
        B = video_frames.shape[0]
        T = video_frames.shape[2] if video_frames.dim() == 5 else 1
        target_device = device if device is not None else video_frames.device
        return torch.zeros(B, T, 512, device=target_device)

    raise NotImplementedError(
        "FACS feature extraction model not yet implemented. "
        "Please provide a frozen FACS teacher encoder that outputs "
        "per-frame emotion-aligned features. See docstring for implementation guide."
    )


# ============================================================================
# Spatial Error Mask Construction
# ============================================================================

class SpatioTemporalErrorMask:
    """
    Converts region-level error masks (from ReAct Reflection) into spatial masks
    applicable to the T2V model's latent space for selective REPA alignment.

    The mapping works as follows:
    1. ReAct produces error_mask: {region_name: severity} (from react_reward.py)
    2. This class maps region names to approximate spatial locations in the latent
    3. The resulting tensor mask is applied element-wise to the REPA loss
    """

    REGION_TO_SPATIAL_PRIOR: Dict[str, Tuple[float, float, float, float]] = {
        "face_upper":    (0.05, 0.25, 0.20, 0.75),
        "face_mid":      (0.20, 0.30, 0.35, 0.70),
        "face_lower":    (0.35, 0.30, 0.50, 0.70),
        "head_neck":     (0.00, 0.30, 0.55, 0.70),
        "torso":         (0.40, 0.20, 0.75, 0.80),
        "upper_limb":    (0.30, 0.05, 0.70, 0.95),
        "lower_limb":    (0.70, 0.15, 1.00, 0.85),
    }

    def __init__(self, min_mask_value: float = 0.1, gaussian_sigma: float = 0.1):
        self.min_mask_value = min_mask_value
        self.gaussian_sigma = gaussian_sigma

    def build_spatial_mask(
        self,
        error_mask: Dict[str, float],
        spatial_size: Tuple[int, int],
        temporal_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Build a spatial attention mask from region-level error severity.

        Args:
            error_mask: {region_name: error_severity [0, 1]}
            spatial_size: (H_latent, W_latent) of the latent space
            temporal_size: Number of temporal frames in latent
            device: Target device

        Returns:
            mask: [1, temporal_size, H_latent, W_latent] with values in [min_mask_value, 1.0]
        """
        H, W = spatial_size
        mask = torch.full((1, temporal_size, H, W), self.min_mask_value, device=device)

        if not error_mask:
            return mask

        for region_name, severity in error_mask.items():
            if severity < 0.05:
                continue

            group_name = self._resolve_group(region_name)
            if group_name is None:
                continue

            bbox = self.REGION_TO_SPATIAL_PRIOR.get(group_name)
            if bbox is None:
                continue

            y1, x1, y2, x2 = bbox
            iy1, iy2 = int(y1 * H), int(y2 * H)
            ix1, ix2 = int(x1 * W), int(x2 * W)
            iy1, iy2 = max(0, iy1), min(H, iy2)
            ix1, ix2 = max(0, ix1), min(W, ix2)

            region_mask = torch.zeros(H, W, device=device)
            if iy2 > iy1 and ix2 > ix1:
                region_mask[iy1:iy2, ix1:ix2] = severity

            if self.gaussian_sigma > 0 and region_mask.sum() > 0:
                region_mask = self._gaussian_blur_2d(region_mask, self.gaussian_sigma)

            mask[0, :, :, :] = torch.maximum(mask[0, :, :, :], region_mask.unsqueeze(0))

        return mask.clamp(self.min_mask_value, 1.0)

    def _resolve_group(self, region_name: str) -> Optional[str]:
        for group, members in BODY_REGION_GROUPS.items():
            if region_name == group or region_name in members:
                return group
        return None

    @staticmethod
    def _gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply approximate Gaussian blur to a 2D tensor."""
        if sigma <= 0:
            return tensor
        kernel_size = max(3, int(6 * sigma) | 1)
        padding = kernel_size // 2

        x = torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype) - padding
        kernel_1d = torch.exp(-0.5 * (x / max(sigma * kernel_size, 1e-6)) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        t = tensor.unsqueeze(0).unsqueeze(0)
        t = F.pad(t, (padding, padding, padding, padding), mode="reflect")
        t = F.conv2d(t, kernel_2d)
        return t.squeeze(0).squeeze(0)


# ============================================================================
# AG-AffectREPA Loss Module
# ============================================================================

class AGAffectREPALoss(nn.Module):
    """
    Agent-Guided AffectREPA loss for selective representation alignment.

    L_AffectREPA = mean( Mask_Agent ⊙ ||proj(h_t2v) - sg(f_facs)||^2 )

    Where:
    - h_t2v: Intermediate hidden states from the T2V model (HunyuanVideo)
    - f_facs: Features from frozen FACS teacher encoder
    - proj: Learnable linear projection to align dimensions
    - sg: Stop-gradient (FACS features are frozen)
    - Mask_Agent: Spatio-temporal mask from ReAct Reflection
    """

    def __init__(
        self,
        t2v_hidden_dim: int = 3072,
        facs_feature_dim: int = 512,
        projection_dim: int = 512,
        loss_type: str = "cosine",
        temperature: float = 0.07,
    ):
        """
        Args:
            t2v_hidden_dim: Dimension of T2V model intermediate hidden states
            facs_feature_dim: Dimension of FACS teacher features
            projection_dim: Dimension of the alignment projection space
            loss_type: "mse" for L2 loss, "cosine" for cosine similarity loss
            temperature: Temperature for cosine similarity (only used when loss_type="cosine")
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

        self.t2v_projector = nn.Sequential(
            nn.Linear(t2v_hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.facs_projector = nn.Sequential(
            nn.Linear(facs_feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(
        self,
        t2v_hidden_states: torch.Tensor,
        facs_features: torch.Tensor,
        error_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked AffectREPA loss.

        Args:
            t2v_hidden_states: Hidden states from T2V model. Shape: [B, N, D_t2v]
                where N is the flattened spatial-temporal token count.
            facs_features: Features from FACS teacher. Shape: [B, N, D_facs]
                Must be stop-gradiented (detached) before passing.
            error_mask: Optional spatial mask. Shape: [B, N] or broadcastable.
                Values in [0, 1] indicating error severity per token.

        Returns:
            loss: Scalar alignment loss.
        """
        h_proj = self.t2v_projector(t2v_hidden_states)
        f_proj = self.facs_projector(facs_features.detach())

        h_proj = F.normalize(h_proj, dim=-1)
        f_proj = F.normalize(f_proj, dim=-1)

        if self.loss_type == "cosine":
            sim = (h_proj * f_proj).sum(dim=-1) / self.temperature
            per_token_loss = 1.0 - sim
        else:
            per_token_loss = F.mse_loss(h_proj, f_proj, reduction="none").mean(dim=-1)

        if error_mask is not None:
            if error_mask.dim() < per_token_loss.dim():
                error_mask = error_mask.unsqueeze(-1) if error_mask.dim() == 1 else error_mask
            per_token_loss = per_token_loss * error_mask

        return per_token_loss.mean()


# ============================================================================
# Integration Helper: Compute AffectREPA for SAGE-GRPO Training Step
# ============================================================================

class AffectREPAIntegration:
    """
    High-level integration class for using AG-AffectREPA within SAGE-GRPO's
    training loop. Handles feature extraction, mask construction, and loss computation.

    Usage in train_one_step (grpo_utils.py):
        ```python
        if affect_repa is not None:
            affect_repa_loss = affect_repa.compute_loss(
                t2v_hidden_states=model_hidden_states,  # from transformer forward
                video_frames=decoded_video_frames,       # from VAE decode
                error_mask=error_mask_from_react,         # from ReAct reward
            )
            loss = policy_loss + kl_loss + lambda_repa * affect_repa_loss
        ```
    """

    def __init__(
        self,
        facs_model: Optional[Any] = None,
        t2v_hidden_dim: int = 3072,
        facs_feature_dim: int = 512,
        projection_dim: int = 512,
        loss_type: str = "cosine",
        lambda_repa: float = 0.1,
        device: torch.device = torch.device("cuda"),
    ):
        self.facs_model = facs_model
        self.lambda_repa = lambda_repa
        self.device = device

        self.loss_module = AGAffectREPALoss(
            t2v_hidden_dim=t2v_hidden_dim,
            facs_feature_dim=facs_feature_dim,
            projection_dim=projection_dim,
            loss_type=loss_type,
        ).to(device)

        self.mask_builder = SpatioTemporalErrorMask()

    @property
    def is_active(self) -> bool:
        """Whether AffectREPA is active (FACS model available)."""
        return self.facs_model is not None

    def compute_loss(
        self,
        t2v_hidden_states: torch.Tensor,
        video_frames: Optional[torch.Tensor] = None,
        error_mask: Optional[Dict[str, float]] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute the AG-AffectREPA auxiliary loss.

        Args:
            t2v_hidden_states: Intermediate features from T2V model [B, N, D]
            video_frames: Decoded video frames [B, C, T, H, W] for FACS feature extraction
            error_mask: Region-level error mask from ReAct reward
            spatial_size: (H_latent, W_latent) for spatial mask construction

        Returns:
            Weighted AffectREPA loss (lambda_repa * raw_loss), or 0 if inactive.
        """
        if not self.is_active:
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        facs_features = get_facs_features(
            video_frames,
            model=self.facs_model,
            device=self.device,
        )

        mask_tensor = None
        if error_mask is not None and spatial_size is not None:
            B = t2v_hidden_states.shape[0]
            T = facs_features.shape[1] if facs_features.dim() >= 3 else 1
            spatial_mask = self.mask_builder.build_spatial_mask(
                error_mask, spatial_size, temporal_size=T, device=self.device,
            )
            mask_tensor = spatial_mask.reshape(1, -1).expand(B, -1)

        if facs_features.dim() == 3:
            facs_features = facs_features.reshape(facs_features.shape[0], -1, facs_features.shape[-1])

        min_tokens = min(t2v_hidden_states.shape[1], facs_features.shape[1])
        t2v_aligned = t2v_hidden_states[:, :min_tokens]
        facs_aligned = facs_features[:, :min_tokens]
        if mask_tensor is not None:
            mask_tensor = mask_tensor[:, :min_tokens]

        raw_loss = self.loss_module(t2v_aligned, facs_aligned, mask_tensor)
        return self.lambda_repa * raw_loss

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return learnable parameters (projectors only; FACS model is frozen)."""
        return list(self.loss_module.parameters())


# ============================================================================
# Factory
# ============================================================================

def build_affect_repa(
    config: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device("cuda"),
) -> AffectREPAIntegration:
    """
    Build AffectREPA integration from config.

    Args:
        config: Dict with keys:
            - facs_model_path: str, path to frozen FACS teacher model
            - t2v_hidden_dim: int (default 3072 for HunyuanVideo)
            - facs_feature_dim: int (default 512)
            - projection_dim: int (default 512)
            - loss_type: str, "cosine" or "mse"
            - lambda_repa: float (default 0.1)
        device: Target device

    Returns:
        AffectREPAIntegration instance
    """
    if config is None:
        config = {}

    facs_model = None
    facs_model_path = config.get("facs_model_path")
    if facs_model_path is not None:
        facs_model = _load_facs_model(facs_model_path, device)

    return AffectREPAIntegration(
        facs_model=facs_model,
        t2v_hidden_dim=config.get("t2v_hidden_dim", 3072),
        facs_feature_dim=config.get("facs_feature_dim", 512),
        projection_dim=config.get("projection_dim", 512),
        loss_type=config.get("loss_type", "cosine"),
        lambda_repa=config.get("lambda_repa", 0.1),
        device=device,
    )


def _load_facs_model(model_path: str, device: torch.device) -> Optional[Any]:
    """
    Load a frozen FACS teacher encoder from checkpoint.

    TODO: Implement actual model loading. The model should:
    1. Accept video frames as input: [B*T, C, H, W]
    2. Output per-frame features: [B*T, D_facs]
    3. Be completely frozen (requires_grad=False for all params)

    Candidate models for implementation:
    - OpenFace AU extractor (C++ with Python binding)
    - LibreFace (PyTorch, AU detection)
    - MARLIN / VideoMAE fine-tuned on AffectNet
    - Custom DINOv2 fine-tuned on FACS AU detection

    Example:
        model = torch.load(model_path, map_location=device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
    """
    return None
