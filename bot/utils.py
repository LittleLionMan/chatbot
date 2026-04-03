from __future__ import annotations
import json
import re


def clean_llm_json(raw: str) -> str:
    return re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()


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
