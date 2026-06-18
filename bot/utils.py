from __future__ import annotations

import json
import re


def clean_llm_json(raw: str) -> str:
    cleaned = raw.strip()
    match = re.search(r"[{\[]", cleaned)
    if match:
        cleaned = cleaned[match.start() :]
    last = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if last != -1:
        cleaned = cleaned[: last + 1]
    return cleaned


def parse_agent_config(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            result = json.loads(raw)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    return {}
