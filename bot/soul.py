from __future__ import annotations
import os
from pathlib import Path
from bot.config import BOT_NAME

_BASE_DIR = Path(__file__).parent.parent


def _load(filename: str) -> str:
    path = _BASE_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{filename} nicht gefunden. Kopiere {filename}.example zu {filename} und passe ihn an."
        )


def _render(template: str) -> str:
    return template.replace("{{BOT_NAME}}", BOT_NAME)


SOUL: str = _render(_load("soul.md"))
BEHAVIOR_RULES: str = _load("behavior.md")
