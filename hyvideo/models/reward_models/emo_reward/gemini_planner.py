"""Fault-tolerant Gemini planner client for EmoReward.

Talks to an OpenAI-style evaluation gateway (HMAC-signed, model_marker scheme):

  - HMAC-signed calls to a configurable data-eval gateway;
  - weighted multi-app-id pool with per-call random selection;
  - two-level fault tolerance:
      (1) network-level retry with exponential backoff (`call_data_eval_with_retry`);
      (2) answer-level retry with exponential backoff + jitter when the body is
          empty / non-JSON, aborting early on permanent SAFETY refusals;
  - large output / thinking budgets so the JSON body is not truncated;
  - robust JSON extraction tolerant to gateway "answer pollution" (trailing
    unrelated text or multiple blocks).

The Plan backend is optional and off by default (EMO_REWARD_PLAN_ENABLED=False).
All of it is env-overridable so a one-key launch script can configure it without
touching argparse plumbing; no endpoint or credential is shipped in this repo:

  EMO_REWARD_PLAN_API_HOST            gateway host (required to use the backend)
  EMO_REWARD_PLAN_API_APP_IDS         comma list of app ids
  EMO_REWARD_PLAN_API_APP_WEIGHTS     comma list of weights (len must match app ids)
  EMO_REWARD_PLAN_API_KEY_<APP_ID>    per-app secret key
  EMO_REWARD_PLAN_MAX_RETRIES         answer-level retries incl. first try (default 6)
  EMO_REWARD_PLAN_NET_RETRIES         network-level retries per attempt (default 5)
  EMO_REWARD_PLAN_MAX_OUTPUT_TOKENS   generationConfig.maxOutputTokens (default 8192)
  EMO_REWARD_PLAN_THINKING_BUDGET     generationConfig.thinkingConfig.thinkingBudget (default 2048)
  EMO_REWARD_PLAN_TIMEOUT_SEC         per-request HTTP timeout (default 300)
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

API_VERSION = "v2.03"
_SOURCE = "t2i"

# model alias -> gateway model_marker (kept in sync with the rubrics reference)
_MODEL_MARKERS = {
    "gemini-2.5-pro": "api_naci_default_gemini-2.5-pro",
    "gemini-2.5-flash": "api_naci_default_gemini-2.5-flash",
    "gemini-3-pro": "api_naci_default_gemini-3.1-pro-preview",
    "gemini-3-1-pro": "api_naci_default_gemini-3.1-pro-preview",
    "gemini-3-flash": "api_naci_default_gemini-3-flash-preview",
    "gemini-3-1-flash": "api_google_gemini-3.1-flash-lite-preview",
}

# Gateway credentials are provided at runtime via environment variables — no keys
# are shipped in the open-source repo. Set, for each app id:
#   EMO_REWARD_PLAN_API_APP_IDS="<id1>,<id2>"
#   EMO_REWARD_PLAN_API_KEY_<id1>="..."   EMO_REWARD_PLAN_API_KEY_<id2>="..."
# (Only needed when the Gemini planner backend is enabled; it is off by default.)
_DEFAULT_APP_IDS: List[str] = []
_DEFAULT_APP_KEYS: Dict[str, str] = {}
_DEFAULT_APP_WEIGHTS = [0.5, 0.5]

_REASON_TYPES = frozenset({"reasoning", "thought", "thinking"})
_SKIP_TYPES = frozenset({"image_url", "image", "inline_data", "tool_call", "tool_use"})
_SAFETY_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "OTHER"})


# ---------------------------------------------------------------------------
# Lightweight success-rate telemetry (per process)
# ---------------------------------------------------------------------------
class _PlannerStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.ok = 0
        self.empty = 0
        self.safety = 0
        self.net_err = 0

    def record(self, *, ok=False, empty=False, safety=False, net_err=False) -> None:
        with self._lock:
            self.calls += 1
            self.ok += int(ok)
            self.empty += int(empty)
            self.safety += int(safety)
            self.net_err += int(net_err)
            n = self.calls
        # log a running success rate every 25 calls
        if n % 25 == 0:
            logger.info(
                "[EmoReward-Gemini] planner success-rate %d/%d (%.1f%%) | empty=%d safety=%d net_err=%d",
                self.ok, n, 100.0 * self.ok / max(n, 1), self.empty, self.safety, self.net_err,
            )


STATS = _PlannerStats()


def _hmac_auth(source: str, secret_id: str, secret_key: str) -> Tuple[str, str]:
    dt = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    sign_str = f"date: {dt}\nsource: {source}"
    sig = base64.b64encode(
        hmac.new(secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
    ).decode()
    auth = (
        f'hmac id="{secret_id}", algorithm="hmac-sha1", '
        f'headers="date source", signature="{sig}"'
    )
    return auth, dt


def _default_host() -> str:
    # The optional Plan backend talks to an OpenAI-style gateway. There is no
    # public default: set EMO_REWARD_PLAN_API_HOST to your gateway endpoint.
    return os.getenv("EMO_REWARD_PLAN_API_HOST", "").strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _content_value(item: Dict[str, Any]) -> Optional[str]:
    for key in ("value", "text"):
        val = item.get(key)
        if val not in (None, ""):
            return str(val)
    return None


def _extract_answer_blocks(response_dict: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (body_text, chain_of_thought). Body = last non-reasoning text block."""
    raw = response_dict.get("answer")
    if raw is None:
        return None, None
    if isinstance(raw, str):
        text = raw.strip()
        return (text or None), None
    if not isinstance(raw, list):
        return None, None

    cot = None
    body: List[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        value = _content_value(item)
        if value is None or item_type in _SKIP_TYPES:
            continue
        if item_type in _REASON_TYPES:
            cot = value
        else:
            body.append(value)
    return (body[-1] if body else None), cot


def _all_text_blocks(response_dict: Dict[str, Any]) -> List[str]:
    """All non-reasoning text blocks (used by robust JSON extraction fallback)."""
    raw = response_dict.get("answer")
    out: List[str] = []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            t = str(item.get("type") or "").strip().lower()
            if t in _SKIP_TYPES:
                continue
            v = _content_value(item)
            if v:
                out.append(v)
    return out


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Robustly pull *all* balanced top-level {...} JSON objects from text.

    Tolerant to: ```json fences, leading/trailing prose, multiple objects,
    and trailing pollution after a valid object.
    """
    if not text:
        return []
    s = text.strip()
    # strip a single fenced block if present
    if "```" in s:
        # prefer the json-tagged fence content if any
        import re as _re
        m = _re.search(r"```(?:json)?\s*(.+?)```", s, flags=_re.DOTALL)
        if m:
            s = m.group(1).strip()
    objects: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except Exception:
                        pass
                    start = -1
    return objects


class _GatewayApi:
    def __init__(self, host: str, user: str, apikey: str, timeout: int = 300):
        self.host = host
        self.user = user
        self.apikey = apikey
        self.timeout = timeout

    def headers(self) -> Dict[str, str]:
        sign, dt = _hmac_auth(_SOURCE, self.user, self.apikey)
        return {
            "Apiversion": API_VERSION,
            "Authorization": sign,
            "Date": dt,
            "Source": _SOURCE,
        }

    def build_payload(self, messages, model: str, system_prompt: str) -> Tuple[Dict[str, Any], str]:
        if model not in _MODEL_MARKERS:
            raise ValueError(f"Unsupported Gemini model for planner: {model}")
        request_id = str(uuid.uuid4())
        max_out = _env_int("EMO_REWARD_PLAN_MAX_OUTPUT_TOKENS", 8192)
        think = _env_int("EMO_REWARD_PLAN_THINKING_BUDGET", 2048)
        gen_cfg: Dict[str, Any] = {"maxOutputTokens": max_out, "audioTimestamp": False}
        if think > 0:
            gen_cfg["thinkingConfig"] = {"include_thoughts": True, "thinkingBudget": think}
        else:
            gen_cfg["thinkingConfig"] = {"include_thoughts": False, "thinkingBudget": 0}
        payload = {
            "request_id": request_id,
            "model_marker": _MODEL_MARKERS[model],
            "system": system_prompt,
            "messages": messages,
            "params": {"generationConfig": gen_cfg},
            "timeout": _env_int("EMO_REWARD_PLAN_TIMEOUT_SEC", 300),
        }
        return payload, request_id

    def call_data_eval(self, messages, model: str, system_prompt: str) -> Tuple[requests.Response, str]:
        payload, request_id = self.build_payload(messages, model, system_prompt)
        response = requests.post(
            url=self.host + "/api/v1/data_eval",
            headers=self.headers(),
            json=payload,
            timeout=self.timeout,
        )
        return response, request_id

    def call_data_eval_with_retry(self, messages, model: str, system_prompt: str, max_retries: int = 5):
        last_exc = None
        for i in range(max_retries):
            try:
                return self.call_data_eval(messages, model, system_prompt)
            except Exception as e:  # network / connection / timeout errors
                last_exc = e
                logger.warning("[EmoReward-Gemini] network call failed: %s, retry %d/%d", e, i + 1, max_retries)
                time.sleep(min(60.0, 2 ** i + random.uniform(0, 2)))
        raise RuntimeError(f"call_data_eval_with_retry failed after {max_retries} retries: {last_exc}")


def _build_gateway_pool(timeout: int) -> Tuple[List[_GatewayApi], List[float]]:
    host = _default_host()
    app_ids = [x.strip() for x in os.getenv("EMO_REWARD_PLAN_API_APP_IDS", ",".join(_DEFAULT_APP_IDS)).split(",") if x.strip()]

    apis: List[_GatewayApi] = []
    for app_id in app_ids:
        app_key = os.getenv(f"EMO_REWARD_PLAN_API_KEY_{app_id}") or _DEFAULT_APP_KEYS.get(app_id)
        if not app_key:
            continue
        apis.append(_GatewayApi(host=host, user=app_id, apikey=app_key, timeout=timeout))
    if not apis:
        raise RuntimeError("No valid EmoReward planner gateway credentials configured")

    weights_raw = os.getenv("EMO_REWARD_PLAN_API_APP_WEIGHTS", "")
    if weights_raw:
        try:
            weights = [float(x.strip()) for x in weights_raw.split(",") if x.strip()]
        except Exception:
            weights = []
    else:
        weights = list(_DEFAULT_APP_WEIGHTS)
    if len(weights) != len(apis):
        weights = [1.0] * len(apis)
    return apis, weights


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if isinstance(content, str):
            normalized.append({"role": role, "content": [{"type": "text", "value": content}]})
            continue
        norm_items: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type") or "text") == "text":
                    text = item.get("value", item.get("text", ""))
                    norm_items.append({"type": "text", "value": str(text)})
        normalized.append({"role": role, "content": norm_items or [{"type": "text", "value": ""}]})
    return normalized


def _backoff(attempt: int, base: float, cap: float) -> float:
    return min(cap, base * (2 ** attempt) + random.uniform(0.0, 3.0))


def call_gemini_text(
    *,
    messages: List[Dict[str, Any]],
    system_prompt: str,
    model: str = "gemini-2.5-flash",
    max_answer_retries: Optional[int] = None,
    answer_retry_base_sec: float = 4.0,
    answer_retry_max_sec: float = 90.0,
) -> Tuple[Optional[str], Optional[str], str, str]:
    """Call the gateway and return (answer, cot, raw_response_json, request_id).

    answer is the last non-reasoning text block; None after exhausting retries.
    """
    if max_answer_retries is None:
        max_answer_retries = _env_int("EMO_REWARD_PLAN_MAX_RETRIES", 6)
    net_retries = _env_int("EMO_REWARD_PLAN_NET_RETRIES", 5)
    timeout = _env_int("EMO_REWARD_PLAN_TIMEOUT_SEC", 300)

    apis, weights = _build_gateway_pool(timeout)
    norm_messages = _normalize_messages(messages)

    last_response = "{}"
    last_request_id = ""
    last_cot = None

    for attempt in range(max_answer_retries):
        api = random.choices(apis, weights=weights, k=1)[0]
        try:
            response, request_id = api.call_data_eval_with_retry(
                norm_messages, model, system_prompt, max_retries=net_retries
            )
        except Exception as e:
            STATS.record(net_err=True)
            last_response = f"network_error: {e}"
            if attempt < max_answer_retries - 1:
                time.sleep(_backoff(attempt, answer_retry_base_sec, answer_retry_max_sec))
            continue
        last_request_id = request_id

        try:
            response_dict = response.json()
        except Exception:
            response_dict = None

        if isinstance(response_dict, dict):
            last_response = json.dumps(response_dict, ensure_ascii=False)
            answer, cot = _extract_answer_blocks(response_dict)
            last_cot = cot
            if answer and answer.strip():
                STATS.record(ok=True)
                return answer, cot, last_response, request_id
            finish_reason = response_dict.get("finish_reason") or response_dict.get("finishReason")
            if finish_reason and str(finish_reason).upper() in _SAFETY_REASONS:
                STATS.record(safety=True)
                return None, last_cot, last_response, last_request_id
        else:
            last_response = (getattr(response, "text", "") or "")[:4000]

        if attempt < max_answer_retries - 1:
            time.sleep(_backoff(attempt, answer_retry_base_sec, answer_retry_max_sec))

    STATS.record(empty=True)
    return None, last_cot, last_response, last_request_id


def get_planner_stats() -> Dict[str, int]:
    return {
        "calls": STATS.calls,
        "ok": STATS.ok,
        "empty": STATS.empty,
        "safety": STATS.safety,
        "net_err": STATS.net_err,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ans, cot, raw, rid = call_gemini_text(
        messages=[{"role": "user", "content": "Return ONLY this JSON: {\"ok\": true}"}],
        system_prompt="You output only valid JSON.",
        model=os.getenv("EMO_REWARD_PLAN_GEMINI_MODEL", "gemini-2.5-flash"),
    )
    print("request_id=", rid)
    print("answer=", ans)
    print("json_objects=", extract_json_objects(ans or ""))
