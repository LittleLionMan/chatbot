from __future__ import annotations
import json
import logging
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import asyncpg
from bot import brain, memory
from bot.utils import clean_llm_json

logger = logging.getLogger(__name__)

_EXPLICIT_USER_TRIGGERS = re.compile(
    r"^(merk dir|merke dir|remember|behalte|speicher(e)?)[:\s]+(.+)$",
    re.IGNORECASE,
)

_EXPLICIT_BOT_TRIGGERS = re.compile(
    r"^(du bist|du kannst|du weißt|du hast|ihr seid|hier bist du|in dieser gruppe bist du)[:\s,]+(.+)$",
    re.IGNORECASE,
)

_TIMEZONE_TRIGGER = re.compile(
    r"^(meine zeitzone ist|my timezone is|zeitzone)[:\s]+(\S+)$",
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

_REFLECTION_SYSTEM = """Du bist Bob. Nach einer Interaktion schreibst du dir kurze Notizen — ehrliche subjektive Beobachtungen, keine Faktensammlung.

Regeln:
- Antworte NUR mit einem JSON-Array von Objekten. Kein anderer Text, keine Erklärungen, keine Markdown-Backticks.
- Jedes Objekt hat zwei Felder:
  - "text": Die Beobachtung in der Ich-Perspektive, maximal 120 Zeichen.
  - "target": "user" wenn die Beobachtung eine bestimmte Person betrifft, "group" wenn sie die Gruppe als Ganzes betrifft.
- Nur schreiben wenn die Interaktion wirklich etwas Bemerkenswertes hatte — ein Muster, eine Überraschung, eine Spannung.
- Erlaubt: Eindrücke, Muster, Überraschungen, Spannungen zwischen Personen.
- Nicht erlaubt: reine Fakten die besser als User-Memory passen, Bewertungen ohne Substanz.
- Wenn nichts Bemerkenswertes: antworte mit []
- Maximal 2 Einträge pro Aufruf.

Beispiel-Output: [{"text": "Habe das Gefühl dass Lisa Kritik besser annimmt wenn sie als Frage verpackt ist", "target": "user"}, {"text": "Die Gruppe diskutiert Technik enthusiastisch aber wird bei politischen Implikationen schnell defensiv", "target": "group"}]"""

_REFLECTION_DECISION_SYSTEM = """Entscheide ob eine Interaktion bemerkenswert genug ist für eine persönliche Reflexionsnotiz.
Antworte ausschließlich mit dem Wort 'ja' oder dem Wort 'nein'. Keine anderen Wörter, keine Erklärungen.
Bemerkenswert: Überraschung, erkennbares Muster, unerwartete Reaktion, interessante Spannung, etwas das beim nächsten Gespräch nützlich sein könnte.
Nicht bemerkenswert: reine Faktenfragen, kurze Bestätigungen, Small Talk ohne Substanz.
Beispiele: "User reagiert defensiv auf Kritik" → ja, "User fragt nach Uhrzeit" → nein."""


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
    parsed = json.loads(clean_llm_json(raw))
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str)]


async def _should_reflect(conversation_snippet: str) -> bool:
    try:
        decision = await brain.chat(
            system=_REFLECTION_DECISION_SYSTEM,
            messages=[{"role": "user", "content": conversation_snippet}],
            max_tokens=5,
        )
        return decision.strip().lower().startswith("ja")
    except Exception as e:
        logger.warning("Reflection decision failed: %s", e)
        return False


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


async def extract_and_store_reflection(
    pool: asyncpg.Pool,
    group_id: int,
    user_id: int,
    conversation_snippet: str,
) -> None:
    try:
        if not await _should_reflect(conversation_snippet):
            return
        raw = await brain.chat(
            system=_REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": f"Interaktion:\n{conversation_snippet}"}],
            max_tokens=256,
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, list):
            return
        for item in parsed[:2]:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            target = item.get("target", "user")
            sanitized = _sanitize_fact(text)
            if sanitized is None:
                continue
            subject_id = user_id if target == "user" else group_id
            await _store_if_new(pool, "reflection", subject_id, sanitized)
            logger.info("Reflection stored [%s/%d]: %s", target, subject_id, sanitized)
    except Exception as e:
        logger.warning("Reflection extraction failed: %s", e)


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
    tz_match = _TIMEZONE_TRIGGER.match(text.strip())
    if tz_match is not None:
        tz_str = tz_match.group(2).strip()
        try:
            ZoneInfo(tz_str)
        except ZoneInfoNotFoundError:
            return f"Unbekannte Zeitzone: {tz_str}. Gültige Beispiele: Europe/Berlin, UTC, America/New_York."
        await memory.set_user_timezone(pool, user_id, tz_str)
        return f"Zeitzone gesetzt: {tz_str}."

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
