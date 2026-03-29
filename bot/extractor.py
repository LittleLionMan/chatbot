from __future__ import annotations
import json
import logging
import re
import asyncpg
from bot import brain, memory

logger = logging.getLogger(__name__)

_EXPLICIT_TRIGGERS = re.compile(
    r"^(merk dir|merke dir|remember|behalte|speicher(e)?)[:\s]+(.+)$",
    re.IGNORECASE,
)

_MAX_FACT_LENGTH = 120

_EXTRACTOR_SYSTEM = """Du extrahierst dauerhaft relevante Fakten über eine Person aus einem Gesprächsausschnitt.

Regeln:
- Antworte NUR mit einem JSON-Array von Strings. Kein anderer Text, keine Erklärungen, keine Markdown-Backticks.
- Jeder String ist ein einzelner sachlicher Fakt über die Person, maximal 100 Zeichen.
- Nur Fakten die dauerhaft relevant sind: Name, Beruf, Wohnort, Präferenzen, wichtige Lebenssituationen.
- Keine temporären Aussagen ("ist heute müde"), keine Meinungen, keine Spekulationen.
- Keine Inhalte aus dem Gesprächskontext selbst als Fakten übernehmen — nur über die Person.
- Wenn keine dauerhaft relevanten Fakten vorhanden sind, antworte mit: []
- Maximal 3 Fakten pro Aufruf.

Beispiel-Output: ["Wohnt in Berlin", "Arbeitet als Softwareentwickler", "Hat zwei Kinder"]"""


def _sanitize_fact(raw: str) -> str | None:
    fact = raw.strip()
    if not fact or len(fact) > _MAX_FACT_LENGTH:
        return None
    if any(c in fact for c in ["\n", "\r", "{"]):
        return None
    return fact


async def extract_and_store_automatic(
    pool: asyncpg.Pool,
    user_id: int,
    display_name: str,
    conversation_snippet: str,
) -> None:
    try:
        raw = await brain.chat(
            system=_EXTRACTOR_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Person: {display_name}\n\nGespräch:\n{conversation_snippet}",
                }
            ],
            max_tokens=256,
        )
        parsed = json.loads(raw.strip())
        if not isinstance(parsed, list):
            return
        existing = await memory.get_memories(pool, "user", user_id, limit=50)
        existing_lower = {e.lower() for e in existing}
        for item in parsed[:3]:
            if not isinstance(item, str):
                continue
            fact = _sanitize_fact(item)
            if fact is None:
                continue
            if fact.lower() in existing_lower:
                continue
            await memory.add_memory(pool, "user", user_id, fact)
            logger.info("Auto-memory stored for user %d: %s", user_id, fact)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Auto-extraction failed for user %d: %s", user_id, e)


def parse_explicit_trigger(text: str) -> str | None:
    match = _EXPLICIT_TRIGGERS.match(text.strip())
    if not match:
        return None
    return _sanitize_fact(match.group(3))


async def handle_explicit_memory(
    pool: asyncpg.Pool,
    user_id: int,
    text: str,
) -> str | None:
    fact = parse_explicit_trigger(text)
    if fact is None:
        return None
    existing = await memory.get_memories(pool, "user", user_id, limit=50)
    if fact.lower() in {e.lower() for e in existing}:
        return "Weiß ich schon."
    await memory.add_memory(pool, "user", user_id, fact)
    return "Gemerkt."
