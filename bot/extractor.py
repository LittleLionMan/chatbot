from __future__ import annotations
import json
import logging
import re
import asyncpg
from bot import brain, memory

logger = logging.getLogger(__name__)

_EXPLICIT_USER_TRIGGERS = re.compile(
    r"^(merk dir|merke dir|remember|behalte|speicher(e)?)[:\s]+(.+)$",
    re.IGNORECASE,
)

_EXPLICIT_BOT_TRIGGERS = re.compile(
    r"^(du bist|du kannst|du weißt|du hast|ihr seid|hier bist du|in dieser gruppe bist du)[:\s,]+(.+)$",
    re.IGNORECASE,
)

_MAX_FACT_LENGTH = 120

_USER_EXTRACTOR_SYSTEM = """Du extrahierst dauerhaft relevante Fakten über eine Person aus einem Gesprächsausschnitt.

Regeln:
- Antworte NUR mit einem JSON-Array von Strings. Kein anderer Text, keine Erklärungen, keine Markdown-Backticks.
- Jeder String ist ein einzelner sachlicher Fakt über die Person, maximal 100 Zeichen.
- Nur Fakten die dauerhaft relevant sind: Name, Beruf, Wohnort, Präferenzen, wichtige Lebenssituationen.
- Keine temporären Aussagen ("ist heute müde"), keine Meinungen, keine Spekulationen.
- Keine Inhalte aus dem Gesprächskontext selbst als Fakten übernehmen — nur über die Person.
- Wenn keine dauerhaft relevanten Fakten vorhanden sind, antworte mit: []
- Maximal 3 Fakten pro Aufruf.

Beispiel-Output: ["Wohnt in Berlin", "Arbeitet als Softwareentwickler", "Hat zwei Kinder"]"""

_REACTION_EXTRACTOR_SYSTEM = """Du analysierst ob eine Nutzerreaktion auf eine Bot-Antwort dauerhaft relevante Rückschlüsse über den Bot selbst erlaubt.

Regeln:
- Antworte NUR mit einem JSON-Array von Strings. Kein anderer Text, keine Erklärungen, keine Markdown-Backticks.
- Jeder String beschreibt einen dauerhaft relevanten Fakt über den Bot aus Sicht der Gruppe, maximal 100 Zeichen.
- Nur extrahieren wenn die Reaktion klar positiv ("gut", "genau", "perfekt", "hilfreich") oder klar negativ ("falsch", "daneben", "nicht hilfreich") ist.
- Formuliere als neutrale Beobachtung: "Gruppe findet Bot gut in X", "Bot lag bei Y daneben".
- Keine temporären oder zufälligen Reaktionen speichern.
- Wenn keine relevante Rückschlüsse möglich sind, antworte mit: []
- Maximal 2 Fakten pro Aufruf.

Beispiel-Output: ["Gruppe schätzt kurze prägnante Antworten bei Technikfragen"]"""


def _sanitize_fact(raw: str) -> str | None:
    fact = raw.strip()
    if not fact or len(fact) > _MAX_FACT_LENGTH:
        return None
    if any(c in fact for c in ["\n", "\r", "{"]):
        return None
    return fact


async def _store_if_new(pool: asyncpg.Pool, subject_type: str, subject_id: int, facts: list[str]) -> None:
    existing = await memory.get_memories(pool, subject_type, subject_id, limit=50)
    existing_lower = {e.lower() for e in existing}
    for fact in facts:
        sanitized = _sanitize_fact(fact)
        if sanitized is None:
            continue
        if sanitized.lower() in existing_lower:
            continue
        await memory.add_memory(pool, subject_type, subject_id, sanitized)
        logger.info("Memory stored [%s/%d]: %s", subject_type, subject_id, sanitized)


async def _extract_via_llm(system: str, content: str) -> list[str]:
    raw = await brain.chat(system=system, messages=[{"role": "user", "content": content}], max_tokens=256)
    parsed = json.loads(raw.strip())
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str)]


async def extract_and_store_automatic(
    pool: asyncpg.Pool,
    user_id: int,
    display_name: str,
    conversation_snippet: str,
) -> None:
    try:
        facts = await _extract_via_llm(
            _USER_EXTRACTOR_SYSTEM,
            f"Person: {display_name}\n\nGespräch:\n{conversation_snippet}",
        )
        await _store_if_new(pool, "user", user_id, facts[:3])
    except Exception as e:
        logger.warning("Auto user-extraction failed for user %d: %s", user_id, e)


async def extract_reaction_about_bot(
    pool: asyncpg.Pool,
    group_id: int,
    bot_response: str,
    user_reaction: str,
) -> None:
    try:
        facts = await _extract_via_llm(
            _REACTION_EXTRACTOR_SYSTEM,
            f"Bot-Antwort: {bot_response}\n\nNutzerreaktion: {user_reaction}",
        )
        await _store_if_new(pool, "bot", group_id, facts[:2])
    except Exception as e:
        logger.warning("Reaction extraction failed for group %d: %s", group_id, e)


def _parse_explicit_trigger(text: str, pattern: re.Pattern, group_index: int) -> str | None:
    match = pattern.match(text.strip())
    if not match:
        return None
    return _sanitize_fact(match.group(group_index))


async def handle_explicit_memory(
    pool: asyncpg.Pool,
    user_id: int,
    group_id: int | None,
    text: str,
) -> str | None:
    user_fact = _parse_explicit_trigger(text, _EXPLICIT_USER_TRIGGERS, 3)
    if user_fact is not None:
        existing = await memory.get_memories(pool, "user", user_id, limit=50)
        if user_fact.lower() in {e.lower() for e in existing}:
            return "Weiß ich schon."
        await memory.add_memory(pool, "user", user_id, user_fact)
        return "Gemerkt."

    if group_id is not None:
        bot_fact = _parse_explicit_trigger(text, _EXPLICIT_BOT_TRIGGERS, 2)
        if bot_fact is not None:
            existing = await memory.get_memories(pool, "bot", group_id, limit=50)
            if bot_fact.lower() in {e.lower() for e in existing}:
                return "Weiß ich schon."
            await memory.add_memory(pool, "bot", group_id, bot_fact)
            return "Gemerkt."

    return None
