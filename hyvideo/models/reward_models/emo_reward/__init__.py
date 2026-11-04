"""
EmoReward: Agent-Guided Hierarchical Reward and Dynamic
Representation Alignment for Fine-Grained Emotional Video Generation.

This module provides:
1. emotion_taxonomy: 60+ emotion categories with FACS AU and muscle mappings
2. react_reward: ReAct paradigm hierarchical reward (Plan-and-Solve → Act → Reflect)
3. affect_repa: AG-AffectREPA for selective representation alignment with FACS teacher
4. openface_adapter: OpenFace 3.0 integration for AU/Emotion/Gaze extraction
"""

from .emotion_taxonomy import (
    EMOTION_TAXONOMY,
    EMOTION_BY_ID,
    EMOTION_BY_NAME_EN,
    EMOTION_BY_NAME_CN,
    FACS_ACTION_UNITS,
    AU_TO_MUSCLE_REGION,
    BODY_REGION_GROUPS,
    EmotionFamily,
    EmotionCategory,
    ActionUnit,
    get_emotion,
    get_required_aus_for_emotion,
    get_muscle_regions_for_emotion,
    build_emotion_au_matrix,
    get_emotions_by_family,
    get_all_emotion_names,
)

from .react_reward import (
    EmoRewardScorer,
    EmotionPlanner,
    PhysicalObserver,
    ReflectionRewarder,
    HierarchicalReward,
    emo_reward_score,
)

from .affect_repa import (
    get_facs_features,
    AGAffectREPALoss,
    AffectREPAIntegration,
    SpatioTemporalErrorMask,
    build_affect_repa,
)

from .openface_adapter import (
    OpenFaceExtractor,
    OpenFaceVideoAnalyzer,
    build_openface_au_extractor,
    AU_INDEX_TO_ID,
    AU_ID_TO_INDEX,
)

__all__ = [
    # emotion_taxonomy
    "EMOTION_TAXONOMY",
    "EMOTION_BY_ID",
    "FACS_ACTION_UNITS",
    "AU_TO_MUSCLE_REGION",
    "BODY_REGION_GROUPS",
    "EmotionFamily",
    "EmotionCategory",
    "ActionUnit",
    "get_emotion",
    "get_required_aus_for_emotion",
    "get_muscle_regions_for_emotion",
    "build_emotion_au_matrix",
    "get_all_emotion_names",
    # react_reward
    "EmoRewardScorer",
    "emo_reward_score",
    # affect_repa
    "get_facs_features",
    "AGAffectREPALoss",
    "AffectREPAIntegration",
    "build_affect_repa",
    # openface_adapter
    "OpenFaceExtractor",
    "OpenFaceVideoAnalyzer",
    "build_openface_au_extractor",
    "AU_INDEX_TO_ID",
    "AU_ID_TO_INDEX",
]
