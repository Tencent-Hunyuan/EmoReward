"""
Prompt templates for EmoReward ReAct Reward Loop.

Provides structured prompts for the Plan-and-Solve (emotion analysis) and
Reflect (macro reward scoring) phases, designed for Qwen2-VL chat format.

Architecture mapping:
    Plan-and-Solve (VLM)  →  build_plan_messages()
    Reflect (VLM)         →  build_reflect_messages()
"""

import json
from typing import Dict, List, Any, Optional

# ============================================================================
# Plan-and-Solve: Emotion Identification + AU Specification
# ============================================================================

PLAN_SYSTEM_PROMPT = (
    "You are an expert in human emotional expression and the Facial Action Coding System (FACS). "
    "Your task is to analyze video generation prompts and identify target emotions with their "
    "corresponding anatomical physical indicators. "
    "Respond with EXACTLY ONE JSON object and NOTHING else: no markdown fences, no prose, "
    "no commentary before or after the JSON. Output must start with '{' and end with '}'."
)

PLAN_USER_TEMPLATE = """Analyze the following video generation prompt and identify the primary emotion being depicted. Then specify the expected facial Action Units (AUs) and body language markers.

**Video generation prompt:**
"{prompt}"

**Instructions:**
1. Identify the single most prominent target emotion from the prompt.
2. List the FACS Action Units expected to be activated, with intensity from 0.0 to 1.0.
3. List expected body language markers (e.g., "trunk_forward_lean", "arms_raised", "shoulders_hunched").
4. Estimate the temporal window (as fraction of video duration, 0.0-1.0) where the emotion peak should occur.
5. Estimate the target affect as VAD: valence and arousal, each a float in [0.0, 1.0]
   (valence: 0=very negative, 1=very positive; arousal: 0=very calm, 1=very activated).

**Available AU IDs:** 1 (Inner Brow Raise), 2 (Outer Brow Raise), 4 (Brow Lowerer), 6 (Cheek Raiser), 9 (Nose Wrinkler), 12 (Lip Corner Puller), 25 (Lips Part), 26 (Jaw Drop)

Output EXACTLY this JSON object and nothing else (no ```` ``` ```` fences, no extra text):
{{
  "emotion_name": "<English emotion name>",
  "au_expectations": {{"<AU_ID>": <intensity 0-1>, ...}},
  "body_expectations": ["<marker1>", "<marker2>"],
  "temporal_window": [<start_ratio 0-1>, <end_ratio 0-1>],
  "target_vad": [<valence 0-1>, <arousal 0-1>],
  "confidence": <0.0 to 1.0>
}}"""


def build_plan_messages(prompt: str) -> List[Dict[str, Any]]:
    """Build Qwen2-VL chat messages for the Plan-and-Solve phase."""
    return [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PLAN_USER_TEMPLATE.format(prompt=prompt)},
            ],
        },
    ]


# ============================================================================
# Reflect: Macro Reward Scoring from AU Observations
# ============================================================================

REFLECT_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing generated video emotional fidelity. "
    "You receive objective FACS measurement data (NOT raw video) and judge how well "
    "the generated content matches the intended emotion. Always respond in valid JSON."
)

REFLECT_USER_TEMPLATE = """Evaluate the emotional fidelity of a generated video based on the following physical measurements.

**Original prompt:** "{prompt}"
**Target emotion:** {emotion_name}

**Expected AU activations (from plan):**
{expected_aus_text}

**Observed AU data (averaged across {num_frames} analyzed frames):**
{observed_aus_text}

**Detection statistics:**
- Face detection rate: {face_rate:.1%}
- Required AUs detected rate: {au_compliance:.1%}
- Frames analyzed: {num_frames}

**Evaluation criteria:**
1. Do the observed AUs match the expected pattern for {emotion_name}?
2. Are the AU intensities in a reasonable range for this emotion?
3. Is the face detection rate adequate for reliable assessment?
4. Overall, does the physical evidence support the intended emotion?

Score the macro-level emotional coherence from 0.0 (completely wrong emotion) to 1.0 (perfect emotional expression).

Respond with ONLY a JSON object:
```json
{{
  "macro_score": <0.0 to 1.0>,
  "reasoning": "<brief 1-2 sentence explanation>"
}}
```"""


def build_reflect_messages(
    prompt: str,
    plan,
    obs_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build Qwen2-VL chat messages for the Reflect phase."""
    emotion_name = plan.target_emotion.name_en if hasattr(plan, 'target_emotion') else "Unknown"

    expected_lines = []
    if hasattr(plan, 'expected_au_activations'):
        for au_id, intensity in plan.expected_au_activations.items():
            expected_lines.append(f"  AU{au_id}: {intensity:.2f}")
    expected_aus_text = "\n".join(expected_lines) if expected_lines else "  (none specified)"

    observed_lines = []
    au_ts = obs_summary.get("au_timeseries", {})
    for au_id, series in au_ts.items():
        if series:
            avg_val = sum(v for _, v in series) / len(series)
            observed_lines.append(f"  AU{au_id}: {avg_val:.2f} (over {len(series)} frames)")
    observed_aus_text = "\n".join(observed_lines) if observed_lines else "  (no AU data available)"

    face_rate = obs_summary.get("face_detection_rate", 0.0)
    num_frames = obs_summary.get("num_frames", 0)

    required_aus = set(plan.target_emotion.required_aus) if hasattr(plan, 'target_emotion') else set()
    if required_aus and au_ts:
        detected_required = sum(
            1 for au_id in required_aus
            if au_ts.get(au_id) and any(v > 0.3 for _, v in au_ts[au_id])
        )
        au_compliance = detected_required / max(len(required_aus), 1)
    else:
        au_compliance = 0.0

    text = REFLECT_USER_TEMPLATE.format(
        prompt=prompt,
        emotion_name=emotion_name,
        expected_aus_text=expected_aus_text,
        observed_aus_text=observed_aus_text,
        face_rate=face_rate,
        au_compliance=au_compliance,
        num_frames=num_frames,
    )

    return [
        {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    ]


# ============================================================================
# Response Parsers
# ============================================================================

def _candidate_json_objects(response_text: str) -> List[Dict[str, Any]]:
    """Robustly pull all balanced top-level {...} JSON objects from possibly
    polluted gateway text (trailing prose, multiple blocks, ``` fences)."""
    try:
        from .gemini_planner import extract_json_objects
        objs = extract_json_objects(response_text)
        if objs:
            return objs
    except Exception:
        pass
    # local fallback: fence strip + first/last brace
    s = (response_text or "").strip()
    if "```json" in s:
        s = s.split("```json")[-1].split("```")[0].strip()
    elif "```" in s:
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].split("```")[0].strip()
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        try:
            obj = json.loads(s[a:b + 1])
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            pass
    return []


def parse_plan_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse a (possibly polluted) VLM/Gemini plan response into a structured dict.

    Tolerant to markdown fences, leading/trailing prose, multiple JSON blocks,
    and trailing pollution. Picks the first JSON object that carries the
    plan-shaped fields. ``temporal_window`` is kept as the model emitted it
    (fractions 0..1 OR seconds); the caller (_vlm_plan) normalizes against the
    video duration.
    """
    candidates = _candidate_json_objects(response_text)
    if not candidates:
        return None
    # prefer an object that actually looks like a plan
    data = None
    for obj in candidates:
        if any(k in obj for k in ("emotion_name", "au_expectations", "target_vad")):
            data = obj
            break
    if data is None:
        data = candidates[0]

    try:
        au_expectations: Dict[int, float] = {}
        for key, val in (data.get("au_expectations", {}) or {}).items():
            try:
                au_expectations[int(key)] = float(val)
            except Exception:
                continue

        tw = data.get("temporal_window", [0.2, 0.8])
        try:
            temporal_window = (float(tw[0]), float(tw[1]))
        except Exception:
            temporal_window = (0.2, 0.8)

        result: Dict[str, Any] = {
            "emotion_name": str(data.get("emotion_name", "")),
            "au_expectations": au_expectations,
            "body_expectations": list(data.get("body_expectations", []) or []),
            "temporal_window": temporal_window,
            "confidence": float(data.get("confidence", 0.8)),
        }

        vad = data.get("target_vad")
        if isinstance(vad, (list, tuple)) and len(vad) == 2:
            try:
                result["target_vad"] = (
                    max(0.0, min(1.0, float(vad[0]))),
                    max(0.0, min(1.0, float(vad[1]))),
                )
            except Exception:
                pass
        return result
    except Exception:
        return None


def parse_reflect_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse VLM reflect response into structured dict."""
    try:
        cleaned = response_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            cleaned = cleaned[start:end]

        data = json.loads(cleaned)
        score = float(data.get("macro_score", 0.5))
        score = max(0.0, min(1.0, score))
        return {
            "macro_score": score,
            "reasoning": str(data.get("reasoning", "")),
        }
    except Exception:
        return None
