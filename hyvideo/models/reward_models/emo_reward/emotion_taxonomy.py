"""
EmoReward: 60-Category Fine-Grained Emotion Taxonomy
with FACS Action Unit Mapping and Anatomical Muscle Groups.

Based on:
- Plutchik's Wheel of Emotions (8 primary emotions + derivatives)
- Cowen & Keltner's 27-28 Semantic Space Theory
- Shaver & Parrott's Tree-structured Taxonomy (3-level hierarchy)
- FACS (Facial Action Coding System) by Ekman & Friesen
- Hume AI's 53+ dimensional facial expression model
- GoEmotions 28-category fine-grained taxonomy

Reference: '情绪分类体系报告-Research.pdf' Section III
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class EmotionFamily(Enum):
    """Eight functional emotion family groups."""
    BASIC_NEGATIVE = "basic_high_arousal_negative"
    BASIC_POSITIVE = "basic_high_arousal_positive"
    COGNITIVE = "cognitive_epistemic"
    SOCIAL = "social_affiliative_empathetic"
    SELF_CONSCIOUS = "self_conscious_evaluative"
    AESTHETIC = "aesthetic_transcendent"
    ANTICIPATORY = "anticipatory_goal_oriented"
    PHYSIOLOGICAL = "physiological_low_arousal"


class ValenceType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class ArousalLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ActionUnit:
    """FACS Action Unit definition with anatomical muscle binding."""
    au_id: int
    name: str
    muscle: str
    description: str


@dataclass
class EmotionCategory:
    """Single emotion category with full FACS and body marker annotation."""
    emotion_id: int
    name_en: str
    name_cn: str
    family: EmotionFamily
    valence: ValenceType
    arousal: ArousalLevel
    required_aus: List[int]
    optional_aus: List[int] = field(default_factory=list)
    body_markers: List[str] = field(default_factory=list)
    temporal_dynamics: str = ""
    disambiguation_notes: str = ""


# ---------------------------------------------------------------------------
# FACS Action Unit Registry (anatomical ground truth)
# ---------------------------------------------------------------------------
FACS_ACTION_UNITS: Dict[int, ActionUnit] = {
    1:  ActionUnit(1,  "Inner Brow Raise",      "Frontalis (pars medialis)",        "Inner eyebrow pulled upward"),
    2:  ActionUnit(2,  "Outer Brow Raise",      "Frontalis (pars lateralis)",       "Outer eyebrow pulled upward"),
    4:  ActionUnit(4,  "Brow Lowerer",          "Corrugator supercilii / Depressor supercilii", "Eyebrows drawn together and down"),
    5:  ActionUnit(5,  "Upper Lid Raise",       "Levator palpebrae superioris",     "Upper eyelid raised, more sclera visible"),
    6:  ActionUnit(6,  "Cheek Raise",           "Orbicularis oculi (pars orbitalis)","Cheek pushed upward, crow's feet appear"),
    7:  ActionUnit(7,  "Lid Tightener",         "Orbicularis oculi (pars palpebralis)","Lower eyelid tightened"),
    9:  ActionUnit(9,  "Nose Wrinkler",         "Levator labii superioris alaeque nasi","Nose bridge wrinkled, nostrils pulled up"),
    10: ActionUnit(10, "Upper Lip Raise",       "Levator labii superioris",         "Upper lip raised to expose upper teeth"),
    12: ActionUnit(12, "Lip Corner Pull",       "Zygomaticus major",                "Lip corners pulled upward obliquely (smile)"),
    14: ActionUnit(14, "Dimpler",               "Buccinator",                       "Lip corners tightened, dimple appears"),
    15: ActionUnit(15, "Lip Corner Depress",    "Depressor anguli oris (Triangularis)","Lip corners pulled straight down"),
    16: ActionUnit(16, "Lower Lip Depress",     "Depressor labii inferioris",       "Lower lip pulled down, lower teeth exposed"),
    17: ActionUnit(17, "Chin Raise",            "Mentalis",                         "Chin boss pushed up, chin wrinkles"),
    20: ActionUnit(20, "Lip Stretch",           "Risorius / Platysma",              "Lips stretched horizontally"),
    22: ActionUnit(22, "Lip Funneler",          "Orbicularis oris",                 "Lips pushed forward into funnel shape"),
    23: ActionUnit(23, "Lip Tightener",         "Orbicularis oris",                 "Lips tightened, reddish area narrowed"),
    24: ActionUnit(24, "Lip Pressor",           "Orbicularis oris",                 "Lips pressed together firmly"),
    25: ActionUnit(25, "Lips Part",             "Depressor labii / Mentalis",       "Lips separate, teeth may not show"),
    26: ActionUnit(26, "Jaw Drop",              "Masseter / Internal Pterygoid / Digastric","Jaw pulled down, mouth opens"),
    27: ActionUnit(27, "Mouth Stretch",         "Pterygoids / Digastric",           "Jaw maximally opened"),
    28: ActionUnit(28, "Lip Suck",              "Mentalis",                         "Lips rolled inward over teeth"),
    31: ActionUnit(31, "Jaw Clench",            "Masseter / Temporalis",            "Teeth clenched, jaw muscles bulge"),
    32: ActionUnit(32, "Lip Bite",              "Orbicularis oris / Mentalis",      "Lower lip bitten by upper teeth"),
    38: ActionUnit(38, "Nostril Dilate",        "Nasalis (pars alaris)",            "Nostrils widen"),
    41: ActionUnit(41, "Lid Droop",             "Levator palpebrae (relaxation)",   "Upper eyelid droops, partially covers iris"),
    43: ActionUnit(43, "Eyes Closed",           "Orbicularis oculi (pars palpebralis)","Eyes deliberately closed tight"),
    44: ActionUnit(44, "Squint",                "Orbicularis oculi / Corrugator",   "Eyelids narrowed in evaluation"),
    45: ActionUnit(45, "Blink",                 "Orbicularis oculi",                "Rapid open-close of eyelids"),
    46: ActionUnit(46, "Wink",                  "Orbicularis oculi (unilateral)",   "One eye closes while other stays open"),
    51: ActionUnit(51, "Head Turn Left",        "Sternocleidomastoid",              "Head rotated to the left"),
    52: ActionUnit(52, "Head Turn Right",       "Sternocleidomastoid",              "Head rotated to the right"),
    53: ActionUnit(53, "Head Up",               "Splenius capitis / Trapezius",     "Head tilted backward, chin raised"),
    54: ActionUnit(54, "Head Down",             "Sternocleidomastoid / Longus colli","Head tilted forward, chin lowered"),
    55: ActionUnit(55, "Head Tilt Left",        "Sternocleidomastoid (unilateral)", "Head tilted to the left shoulder"),
    56: ActionUnit(56, "Head Tilt Right",       "Sternocleidomastoid (unilateral)", "Head tilted to the right shoulder"),
    57: ActionUnit(57, "Head Forward",          "Longus colli / Sternocleidomastoid","Head thrust forward"),
    58: ActionUnit(58, "Head Back",             "Splenius capitis / Trapezius",     "Head pulled backward"),
}


# ---------------------------------------------------------------------------
# Muscle Group Lookup (maps AU to muscle region for REPA masking)
# ---------------------------------------------------------------------------
AU_TO_MUSCLE_REGION: Dict[int, str] = {
    1: "forehead_medial", 2: "forehead_lateral", 4: "brow_glabellar",
    5: "upper_eyelid", 6: "periorbital", 7: "lower_eyelid",
    9: "nasal_bridge", 10: "upper_lip", 12: "zygomatic",
    14: "buccal", 15: "mouth_corner", 16: "lower_lip",
    17: "chin", 20: "platysma_lateral", 22: "orbicularis_oris",
    23: "orbicularis_oris", 24: "orbicularis_oris", 25: "lip_region",
    26: "jaw", 27: "jaw", 28: "lip_region",
    31: "jaw_masseter", 32: "lip_region", 38: "nasal_alar",
    41: "upper_eyelid", 43: "periorbital", 44: "periorbital",
    45: "periorbital", 46: "periorbital",
    51: "neck_sternocleidomastoid", 52: "neck_sternocleidomastoid",
    53: "neck_posterior", 54: "neck_anterior",
    55: "neck_lateral", 56: "neck_lateral",
    57: "neck_anterior", 58: "neck_posterior",
}


BODY_REGION_GROUPS = {
    "face_upper": ["forehead_medial", "forehead_lateral", "brow_glabellar",
                    "upper_eyelid", "periorbital", "lower_eyelid"],
    "face_mid": ["nasal_bridge", "nasal_alar", "zygomatic", "buccal"],
    "face_lower": ["upper_lip", "lower_lip", "mouth_corner", "lip_region",
                    "orbicularis_oris", "chin", "platysma_lateral",
                    "jaw", "jaw_masseter"],
    "head_neck": ["neck_sternocleidomastoid", "neck_posterior",
                   "neck_anterior", "neck_lateral"],
    "torso": ["chest_pectoralis", "shoulder_deltoid_trapezius",
              "spine_erector", "abdomen"],
    "upper_limb": ["arm_bicep_tricep", "forearm", "hand_fingers"],
    "lower_limb": ["thigh_quadriceps", "calf", "foot"],
}


# ---------------------------------------------------------------------------
# 60 Emotion Categories (organized by 8 functional families)
# ---------------------------------------------------------------------------

EMOTION_TAXONOMY: List[EmotionCategory] = [
    # =====================================================================
    # Family 1: Basic & High-Arousal Negative States (3.1)
    # =====================================================================
    EmotionCategory(
        emotion_id=1, name_en="Anger", name_cn="愤怒",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[4, 7, 23],
        optional_aus=[27],
        body_markers=["trunk_forward_lean", "fists_clenched", "invading_personal_space"],
        temporal_dynamics="sustained_tension",
    ),
    EmotionCategory(
        emotion_id=2, name_en="Fear", name_cn="恐惧",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[1, 2, 4, 5, 20],
        body_markers=["body_recoil", "hands_up_protective", "gaze_aversion"],
        temporal_dynamics="sudden_onset",
    ),
    EmotionCategory(
        emotion_id=3, name_en="Horror", name_cn="惊恐/骇惧",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.EXTREME,
        required_aus=[5, 26, 27],
        body_markers=["body_freeze", "prolonged_jaw_drop", "trembling"],
        temporal_dynamics="freeze_response_sustained",
    ),
    EmotionCategory(
        emotion_id=4, name_en="Disgust", name_cn="厌恶",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[9, 10],
        optional_aus=[16],
        body_markers=["head_aversion", "trunk_recoil", "gagging_motion"],
        temporal_dynamics="rapid_onset_sustained",
    ),
    EmotionCategory(
        emotion_id=5, name_en="Sadness", name_cn="悲伤",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.LOW,
        required_aus=[1, 4, 15],
        optional_aus=[17],
        body_markers=["body_slumped", "gaze_downcast", "muscle_tone_loss"],
        temporal_dynamics="gradual_onset_sustained",
    ),
    EmotionCategory(
        emotion_id=6, name_en="Distress", name_cn="苦恼/极度痛苦",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.EXTREME,
        required_aus=[6, 43, 25],
        optional_aus=[26],
        body_markers=["hands_clutching_head", "body_rocking", "uncoordinated_limbs"],
        temporal_dynamics="explosive_burst",
    ),
    EmotionCategory(
        emotion_id=7, name_en="Physical Pain", name_cn="生理疼痛",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[4, 6, 43, 20],
        body_markers=["guarding_injured_area", "body_writhing", "teeth_clenching"],
        temporal_dynamics="acute_onset",
    ),
    EmotionCategory(
        emotion_id=8, name_en="Contempt", name_cn="轻蔑",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.LOW,
        required_aus=[14],
        body_markers=["chin_raised", "gaze_downward_at_target", "head_tilt_back"],
        temporal_dynamics="sustained_asymmetric",
        disambiguation_notes="Only asymmetric basic facial expression in FACS (unilateral AU14)",
    ),
    EmotionCategory(
        emotion_id=9, name_en="Hate/Appalled", name_cn="仇恨/憎骇",
        family=EmotionFamily.BASIC_NEGATIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[4, 9],
        optional_aus=[10],
        body_markers=["body_rigid_cold", "gaze_locked_hostile", "fists_ready"],
        temporal_dynamics="sustained_cold_tension",
        disambiguation_notes="Compound of anger + disgust; colder/more rigid than explosive anger",
    ),

    # =====================================================================
    # Family 2: Basic & High-Arousal Positive States (3.2)
    # =====================================================================
    EmotionCategory(
        emotion_id=10, name_en="Joy/Happiness", name_cn="快乐/愉悦",
        family=EmotionFamily.BASIC_POSITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[6, 12],
        body_markers=["bouncing", "clapping", "hugging"],
        temporal_dynamics="sustained_warmth",
        disambiguation_notes="Must be Duchenne smile (AU6+12); AU12 alone = polite/social smile",
    ),
    EmotionCategory(
        emotion_id=11, name_en="Excitement", name_cn="兴奋",
        family=EmotionFamily.BASIC_POSITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.HIGH,
        required_aus=[5],
        optional_aus=[12],
        body_markers=["body_forward_lean", "jumping", "arm_pumping", "fast_movement"],
        temporal_dynamics="rapid_escalation",
    ),
    EmotionCategory(
        emotion_id=12, name_en="Ecstasy", name_cn="狂喜",
        family=EmotionFamily.BASIC_POSITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.EXTREME,
        required_aus=[6, 12],
        body_markers=["body_expansion_extreme", "tears_of_joy", "collapse_kneel"],
        temporal_dynamics="physiological_overload",
        disambiguation_notes="Peak joy with physiological overflow (crying from joy, loss of balance)",
    ),
    EmotionCategory(
        emotion_id=13, name_en="Amusement", name_cn="逗趣/被逗乐",
        family=EmotionFamily.BASIC_POSITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[12],
        body_markers=["shoulder_shaking_laughter", "head_tilt_back", "torso_tremor"],
        temporal_dynamics="rhythmic_bursts",
        disambiguation_notes="Must include trunk/shoulder tremor from laughter respiration",
    ),
    EmotionCategory(
        emotion_id=14, name_en="Triumph", name_cn="胜利",
        family=EmotionFamily.BASIC_POSITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.EXTREME,
        required_aus=[27],
        body_markers=["arms_raised_V", "chest_expanded", "head_thrown_back", "roaring"],
        temporal_dynamics="explosive_dominance_display",
        disambiguation_notes="Face shows dominance-roar (AU27) rather than smile; body is key marker",
    ),
    EmotionCategory(
        emotion_id=15, name_en="Romance", name_cn="浪漫/情爱",
        family=EmotionFamily.BASIC_POSITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[12],
        optional_aus=[6],
        body_markers=["intimate_proximity", "mutual_gaze_prolonged", "gentle_touch"],
        temporal_dynamics="slow_sustained",
    ),
    EmotionCategory(
        emotion_id=16, name_en="Contentment/Satisfaction", name_cn="内容满足",
        family=EmotionFamily.BASIC_POSITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.LOW,
        required_aus=[12],
        body_markers=["body_recline", "muscle_relaxation", "self_soothing_touch"],
        temporal_dynamics="sustained_low_energy",
        disambiguation_notes="Closed-lip AU12 without AU6; body reclines with released tension",
    ),
    EmotionCategory(
        emotion_id=17, name_en="Relief", name_cn="释然/如释重负",
        family=EmotionFamily.BASIC_POSITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[],
        optional_aus=[12],
        body_markers=["deep_exhalation_sigh", "shoulder_drop", "chest_deflation"],
        temporal_dynamics="cliff_drop_release",
        disambiguation_notes="Core is physiological transition: tense → fully relaxed (must show AU4→release)",
    ),

    # =====================================================================
    # Family 3: Cognitive & Epistemic States (3.3)
    # =====================================================================
    EmotionCategory(
        emotion_id=18, name_en="Concentration", name_cn="专注/聚精会神",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.NEUTRAL, arousal=ArousalLevel.MEDIUM,
        required_aus=[],
        body_markers=["gaze_fixation_stable", "blink_rate_low", "body_still", "forward_lean_slight"],
        temporal_dynamics="sustained_static",
    ),
    EmotionCategory(
        emotion_id=19, name_en="Determination", name_cn="决心",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.NEUTRAL, arousal=ArousalLevel.HIGH,
        required_aus=[],
        optional_aus=[31, 32],
        body_markers=["jaw_clench", "deep_breath", "fists_grip", "chin_tuck"],
        temporal_dynamics="building_tension",
        disambiguation_notes="vs Concentration: determination requires muscular loading/coiling, not just stillness",
    ),
    EmotionCategory(
        emotion_id=20, name_en="Contemplation", name_cn="沉思",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.NEUTRAL, arousal=ArousalLevel.LOW,
        required_aus=[],
        body_markers=["gaze_defocused", "head_tilt", "chin_rest_on_hand", "lip_touch"],
        temporal_dynamics="slow_drift",
    ),
    EmotionCategory(
        emotion_id=21, name_en="Confusion", name_cn="困惑",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[4],
        body_markers=["head_tilt_asymmetric", "gaze_scanning", "single_brow_raise"],
        temporal_dynamics="fluctuating",
    ),
    EmotionCategory(
        emotion_id=22, name_en="Doubt", name_cn="怀疑",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.LOW,
        required_aus=[44],
        body_markers=["arms_crossed", "body_lean_back", "squinting"],
        temporal_dynamics="sustained_guarded",
    ),
    EmotionCategory(
        emotion_id=23, name_en="Realization", name_cn="恍然大悟",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.HIGH,
        required_aus=[1, 2, 5],
        body_markers=["gesture_freeze_midair", "mouth_slight_open"],
        temporal_dynamics="sudden_transition_tense_to_open",
        disambiguation_notes="Must show rapid transition from furrowed/tense to open/relaxed",
    ),
    EmotionCategory(
        emotion_id=24, name_en="Interest", name_cn="兴趣",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[],
        optional_aus=[5],
        body_markers=["trunk_forward_lean", "gaze_tracking_target", "reaching_gesture"],
        temporal_dynamics="approach_oriented",
    ),
    EmotionCategory(
        emotion_id=25, name_en="Boredom", name_cn="无聊",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.LOW,
        required_aus=[41],
        body_markers=["body_slump_no_gravity_resistance", "head_prop_on_hand",
                       "yawning", "fidgeting", "clock_checking"],
        temporal_dynamics="sustained_low_tone",
    ),
    EmotionCategory(
        emotion_id=26, name_en="Surprise (Positive)", name_cn="惊讶（正面）",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.HIGH,
        required_aus=[1, 2, 5, 26],
        body_markers=["hands_cover_mouth", "open_body_posture"],
        temporal_dynamics="sub_second_then_joy",
        disambiguation_notes="Must transition to joy (AU6+12) within ~1 second",
    ),
    EmotionCategory(
        emotion_id=27, name_en="Surprise (Negative)", name_cn="惊讶（负面）",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[1, 2, 5],
        optional_aus=[4],
        body_markers=["defensive_recoil", "steering_grip", "protective_posture"],
        temporal_dynamics="sub_second_then_fear_or_anger",
    ),
    EmotionCategory(
        emotion_id=28, name_en="Fascination", name_cn="着迷/痴迷",
        family=EmotionFamily.COGNITIVE,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.HIGH,
        required_aus=[5],
        body_markers=["extreme_proximity_to_object", "breath_shallow",
                       "total_environmental_isolation"],
        temporal_dynamics="sustained_absorption",
    ),

    # =====================================================================
    # Family 4: Social, Affiliative & Empathetic States (3.4)
    # =====================================================================
    EmotionCategory(
        emotion_id=29, name_en="Love/Affection", name_cn="爱/情感",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[6, 12],
        body_markers=["intimate_distance_0_45cm", "gentle_stroking", "soft_gaze_non_threatening"],
        temporal_dynamics="slow_sustained_warmth",
    ),
    EmotionCategory(
        emotion_id=30, name_en="Adoration", name_cn="倾慕/溺爱",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.HIGH,
        required_aus=[6, 12],
        body_markers=["upward_gaze", "hands_clasped_prayer", "body_sway"],
        temporal_dynamics="sustained_worship",
    ),
    EmotionCategory(
        emotion_id=31, name_en="Admiration", name_cn="钦佩/赞赏",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[],
        body_markers=["emphatic_nodding", "standing_ovation", "clapping", "direct_respectful_gaze"],
        temporal_dynamics="building_then_sustained",
    ),
    EmotionCategory(
        emotion_id=32, name_en="Sympathy", name_cn="同情/怜悯",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.MIXED, arousal=ArousalLevel.MEDIUM,
        required_aus=[1],
        body_markers=["trunk_forward_lean", "gentle_pat", "head_tilt"],
        temporal_dynamics="gradual_warm",
        disambiguation_notes="vs Empathic Pain: sympathy lacks mirror physiological distress",
    ),
    EmotionCategory(
        emotion_id=33, name_en="Empathic Pain", name_cn="同情痛/共情痛",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[4, 6, 9, 20],
        body_markers=["wincing", "inhale_sharp", "eyes_cover", "body_shrink"],
        temporal_dynamics="reflexive_mirror_burst",
        disambiguation_notes="Must show mirror physiological pain response (wincing/flinching)",
    ),
    EmotionCategory(
        emotion_id=34, name_en="Gratitude", name_cn="感激",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[12],
        optional_aus=[6],
        body_markers=["bowing", "hand_grasp_tight", "tearful_eyes", "verbal_thanks"],
        temporal_dynamics="sustained_reciprocal",
    ),
    EmotionCategory(
        emotion_id=35, name_en="Connectedness", name_cn="联结感/羁绊",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.LOW,
        required_aus=[],
        body_markers=["group_synchrony", "hand_holding", "shoulder_to_shoulder",
                       "facial_defense_fully_released"],
        temporal_dynamics="sustained_collective",
    ),
    EmotionCategory(
        emotion_id=36, name_en="Awkwardness", name_cn="尴尬",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[12],
        body_markers=["gaze_aversion_rapid", "self_soothing_scratching",
                       "stiff_non_duchenne_smile"],
        temporal_dynamics="jittery_unstable",
        disambiguation_notes="AU12 present but NO AU6 → non-Duchenne (fake) smile",
    ),
    EmotionCategory(
        emotion_id=37, name_en="Compassion", name_cn="怜爱",
        family=EmotionFamily.SOCIAL,
        valence=ValenceType.MIXED, arousal=ArousalLevel.MEDIUM,
        required_aus=[1],
        body_markers=["gentle_rescue_action", "cradling", "extremely_soft_touch"],
        temporal_dynamics="sustained_active_care",
        disambiguation_notes="Includes sympathy face (inverted-V brow) PLUS concrete helping actions",
    ),

    # =====================================================================
    # Family 5: Self-Conscious & Evaluative States (3.5)
    # =====================================================================
    EmotionCategory(
        emotion_id=38, name_en="Pride", name_cn="自豪",
        family=EmotionFamily.SELF_CONSCIOUS,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[53],
        optional_aus=[12],
        body_markers=["chest_expanded", "chin_raised", "arms_akimbo_or_open"],
        temporal_dynamics="sustained_dominance",
        disambiguation_notes="vs Joy: pride requires trunk expansion + chin raise (dominance display)",
    ),
    EmotionCategory(
        emotion_id=39, name_en="Shame", name_cn="羞耻",
        family=EmotionFamily.SELF_CONSCIOUS,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[54],
        body_markers=["body_shrink_minimize", "gaze_downward_fixed",
                       "face_covering", "shoulders_collapsed"],
        temporal_dynamics="sustained_withdrawal",
    ),
    EmotionCategory(
        emotion_id=40, name_en="Guilt", name_cn="内疚",
        family=EmotionFamily.SELF_CONSCIOUS,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[1, 4],
        body_markers=["hand_wringing", "tentative_glances", "fidgeting"],
        temporal_dynamics="sustained_restless",
        disambiguation_notes="vs Shame: guilt shows reparative restlessness, not full withdrawal",
    ),
    EmotionCategory(
        emotion_id=41, name_en="Embarrassment", name_cn="难堪/窘迫",
        family=EmotionFamily.SELF_CONSCIOUS,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[],
        body_markers=["blushing", "gaze_rapid_downward", "lip_bite", "smirk"],
        temporal_dynamics="sudden_onset_acute",
    ),
    EmotionCategory(
        emotion_id=42, name_en="Envy", name_cn="羡慕",
        family=EmotionFamily.SELF_CONSCIOUS,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[],
        body_markers=["gaze_locked_on_desired_object", "jaw_tighten_subtle",
                       "lip_press"],
        temporal_dynamics="sustained_fixation",
        disambiguation_notes="Binary (self vs object); no third-party threat. vs Jealousy which is ternary",
    ),
    EmotionCategory(
        emotion_id=43, name_en="Jealousy", name_cn="嫉妒",
        family=EmotionFamily.SELF_CONSCIOUS,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[4],
        body_markers=["triangular_gaze_tracking", "arms_crossed_defensive",
                       "body_blocking_between_partner_and_rival"],
        temporal_dynamics="agitated_vigilant",
        disambiguation_notes="Ternary (self + partner + rival); gaze triangulates between partner and rival",
    ),
    EmotionCategory(
        emotion_id=44, name_en="Remorse", name_cn="悔恨/懊悔",
        family=EmotionFamily.SELF_CONSCIOUS,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.EXTREME,
        required_aus=[1, 4, 15],
        body_markers=["self_hitting", "hair_pulling", "body_curled_in_corner"],
        temporal_dynamics="intense_self_punishment",
    ),

    # =====================================================================
    # Family 6: Aesthetic & Transcendent States (3.6)
    # =====================================================================
    EmotionCategory(
        emotion_id=45, name_en="Awe", name_cn="敬畏",
        family=EmotionFamily.AESTHETIC,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.EXTREME,
        required_aus=[5, 26],
        optional_aus=[27],
        body_markers=["body_absolute_freeze", "breath_held", "head_tilt_up"],
        temporal_dynamics="instantaneous_freeze",
        disambiguation_notes="vs Aesthetic Appreciation (relaxed) and Entrancement (blank). Awe = jaw drop + freeze",
    ),
    EmotionCategory(
        emotion_id=46, name_en="Aesthetic Appreciation", name_cn="审美欣赏",
        family=EmotionFamily.AESTHETIC,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.LOW,
        required_aus=[],
        optional_aus=[12],
        body_markers=["smooth_eye_scanning", "relaxed_face", "stable_posture"],
        temporal_dynamics="sustained_calm_pleasure",
    ),
    EmotionCategory(
        emotion_id=47, name_en="Entrancement", name_cn="入迷/出神",
        family=EmotionFamily.AESTHETIC,
        valence=ValenceType.NEUTRAL, arousal=ArousalLevel.LOW,
        required_aus=[],
        body_markers=["blink_near_zero", "body_motion_zero", "blank_stare",
                       "muscle_fully_relaxed"],
        temporal_dynamics="sustained_trance",
    ),
    EmotionCategory(
        emotion_id=48, name_en="Wonder", name_cn="奇叹/惊奇",
        family=EmotionFamily.AESTHETIC,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.HIGH,
        required_aus=[5],
        body_markers=["pointing_gesture", "approach_movement", "gasp_inhale"],
        temporal_dynamics="excited_approach",
        disambiguation_notes="vs Awe: wonder is approach-oriented with active gestures; awe is immobilized",
    ),
    EmotionCategory(
        emotion_id=49, name_en="Nostalgia", name_cn="怀旧",
        family=EmotionFamily.AESTHETIC,
        valence=ValenceType.MIXED, arousal=ArousalLevel.LOW,
        required_aus=[12, 4],
        body_markers=["defocused_gaze_distance", "deep_sigh", "tearful_eyes"],
        temporal_dynamics="slow_bittersweet",
        disambiguation_notes="Must show BOTH positive (faint smile AU12) AND negative (furrowed brow AU4) cues",
    ),
    EmotionCategory(
        emotion_id=50, name_en="Inspiration/Feeling Moved", name_cn="启发/感动",
        family=EmotionFamily.AESTHETIC,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.HIGH,
        required_aus=[],
        body_markers=["tears_not_from_sadness", "chest_expansion", "fists_clasped_to_chest",
                       "emphatic_nodding", "eyes_bright"],
        temporal_dynamics="building_to_peak",
    ),
    EmotionCategory(
        emotion_id=51, name_en="Cringe", name_cn="尴尬癌/替人抠脚",
        family=EmotionFamily.AESTHETIC,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[43],
        body_markers=["trunk_recoil", "face_cover_peek_through_fingers",
                       "neck_retract", "hissing_sound"],
        temporal_dynamics="reflexive_withdrawal",
    ),

    # =====================================================================
    # Family 7: Anticipatory & Goal-Oriented States (3.7)
    # =====================================================================
    EmotionCategory(
        emotion_id=52, name_en="Anticipation", name_cn="期待",
        family=EmotionFamily.ANTICIPATORY,
        valence=ValenceType.NEUTRAL, arousal=ArousalLevel.HIGH,
        required_aus=[],
        body_markers=["high_muscle_tension", "penetrating_focused_gaze",
                       "body_forward_coiled"],
        temporal_dynamics="coiled_spring",
    ),
    EmotionCategory(
        emotion_id=53, name_en="Craving", name_cn="渴求/馋",
        family=EmotionFamily.ANTICIPATORY,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[],
        body_markers=["swallowing_visible", "lip_licking", "body_lean_toward_object"],
        temporal_dynamics="escalating_approach",
    ),
    EmotionCategory(
        emotion_id=54, name_en="Sexual Desire", name_cn="性欲",
        family=EmotionFamily.ANTICIPATORY,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.HIGH,
        required_aus=[25],
        body_markers=["pupil_dilation", "gaze_eyes_to_lips_scan",
                       "lip_bite", "breath_rate_increased"],
        temporal_dynamics="escalating_intimate",
    ),
    EmotionCategory(
        emotion_id=55, name_en="Anxiety/Nervousness", name_cn="焦虑/紧张",
        family=EmotionFamily.ANTICIPATORY,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[],
        body_markers=["leg_bouncing", "nail_biting", "skin_picking",
                       "rapid_unfocused_gaze_scanning", "rigid_posture"],
        temporal_dynamics="high_frequency_micro_movements",
    ),
    EmotionCategory(
        emotion_id=56, name_en="Hope/Eagerness", name_cn="渴望/祈愿",
        family=EmotionFamily.ANTICIPATORY,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[],
        body_markers=["prayer_hands", "body_forward_lean", "lip_bite_anticipatory"],
        temporal_dynamics="tense_hopeful",
    ),
    EmotionCategory(
        emotion_id=57, name_en="Frustration", name_cn="沮丧/受挫",
        family=EmotionFamily.ANTICIPATORY,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.HIGH,
        required_aus=[],
        body_markers=["table_slam", "object_throw", "heavy_sigh",
                       "hair_pulling_frustrated", "face_reddened"],
        temporal_dynamics="explosive_micro_bursts",
    ),
    EmotionCategory(
        emotion_id=58, name_en="Disappointment", name_cn="失望",
        family=EmotionFamily.ANTICIPATORY,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.MEDIUM,
        required_aus=[],
        optional_aus=[54],
        body_markers=["instant_muscle_tone_collapse", "deep_sigh",
                       "body_slump_from_expansion"],
        temporal_dynamics="dramatic_deflation",
    ),

    # =====================================================================
    # Family 8: Physiological & Low-Arousal Background States (3.8)
    # =====================================================================
    EmotionCategory(
        emotion_id=59, name_en="Tiredness/Fatigue", name_cn="疲惫/疲倦",
        family=EmotionFamily.PHYSIOLOGICAL,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.LOW,
        required_aus=[41],
        body_markers=["yawning_frequent", "head_nodding_off",
                       "anti_gravity_muscle_failure"],
        temporal_dynamics="sustained_declining",
    ),
    EmotionCategory(
        emotion_id=60, name_en="Calmness", name_cn="平静",
        family=EmotionFamily.PHYSIOLOGICAL,
        valence=ValenceType.NEUTRAL, arousal=ArousalLevel.LOW,
        required_aus=[],
        body_markers=["rhythmic_breathing", "all_facial_muscles_relaxed",
                       "slow_smooth_movement"],
        temporal_dynamics="baseline_steady_state",
    ),
    EmotionCategory(
        emotion_id=61, name_en="Serenity", name_cn="宁静/恬静",
        family=EmotionFamily.PHYSIOLOGICAL,
        valence=ValenceType.POSITIVE, arousal=ArousalLevel.LOW,
        required_aus=[6, 12],
        body_markers=["deep_inhale_environmental", "gentle_duchenne_smile",
                       "open_unguarded_posture"],
        temporal_dynamics="sustained_gentle_positive",
    ),
    EmotionCategory(
        emotion_id=62, name_en="Melancholy", name_cn="忧郁/惆怅",
        family=EmotionFamily.PHYSIOLOGICAL,
        valence=ValenceType.NEGATIVE, arousal=ArousalLevel.LOW,
        required_aus=[],
        body_markers=["psychomotor_retardation", "vacant_stare",
                       "overall_facial_droop"],
        temporal_dynamics="sustained_deep_slow",
    ),
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

EMOTION_BY_ID: Dict[int, EmotionCategory] = {e.emotion_id: e for e in EMOTION_TAXONOMY}
EMOTION_BY_NAME_EN: Dict[str, EmotionCategory] = {e.name_en.lower(): e for e in EMOTION_TAXONOMY}
EMOTION_BY_NAME_CN: Dict[str, EmotionCategory] = {e.name_cn: e for e in EMOTION_TAXONOMY}


def get_emotion(identifier) -> Optional[EmotionCategory]:
    """Lookup by id (int), English name (str), or Chinese name (str)."""
    if isinstance(identifier, int):
        return EMOTION_BY_ID.get(identifier)
    if isinstance(identifier, str):
        lower = identifier.strip().lower()
        result = EMOTION_BY_NAME_EN.get(lower)
        if result is None:
            result = EMOTION_BY_NAME_CN.get(identifier.strip())
        return result
    return None


def get_required_aus_for_emotion(emotion_id: int) -> List[ActionUnit]:
    """Return list of ActionUnit objects required for the given emotion."""
    emo = EMOTION_BY_ID.get(emotion_id)
    if emo is None:
        return []
    return [FACS_ACTION_UNITS[au] for au in emo.required_aus if au in FACS_ACTION_UNITS]


def get_muscle_regions_for_emotion(emotion_id: int) -> List[str]:
    """Return the anatomical muscle regions activated by the given emotion."""
    emo = EMOTION_BY_ID.get(emotion_id)
    if emo is None:
        return []
    regions = set()
    for au in emo.required_aus + emo.optional_aus:
        region = AU_TO_MUSCLE_REGION.get(au)
        if region:
            regions.add(region)
    return sorted(regions)


def build_emotion_au_matrix() -> Dict[int, Dict[str, float]]:
    """Build a binary matrix: emotion_id -> {au_id_str: 1.0 if required, 0.5 if optional}."""
    matrix = {}
    for emo in EMOTION_TAXONOMY:
        row = {}
        for au in emo.required_aus:
            row[str(au)] = 1.0
        for au in emo.optional_aus:
            row[str(au)] = 0.5
        matrix[emo.emotion_id] = row
    return matrix


def get_emotions_by_family(family: EmotionFamily) -> List[EmotionCategory]:
    return [e for e in EMOTION_TAXONOMY if e.family == family]


def get_all_emotion_names() -> List[Tuple[int, str, str]]:
    """Return list of (id, name_en, name_cn) for all 60+ emotions."""
    return [(e.emotion_id, e.name_en, e.name_cn) for e in EMOTION_TAXONOMY]
