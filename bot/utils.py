import re


def clean_llm_json(raw: str) -> str:
    return re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
