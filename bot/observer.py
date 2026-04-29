from __future__ import annotations
import logging
from datetime import datetime, timezone
import asyncpg
from bot import brain, memory
from bot.models import CAPABILITY_SIMPLE_TASKS, CAPABILITY_CHAT
from bot import config

logger = logging.getLogger(__name__)

_OBSERVER_SYSTEM = """Du beobachtest einen Gesprächsverlauf und komprimierst ihn in ein dichtes, datiertes Beobachtungslog.

Regeln:
- Antworte NUR mit einem mehrzeiligen Text, eine Beobachtung pro Zeile.
- Jede Zeile beginnt mit einer Priorität und einem Zeitstempel:
  🔴 HH:MM Wichtige Fakten, Entscheidungen, explizite Aussagen über Personen oder Vorhaben
  🟡 HH:MM Potenziell relevante Details, Stimmungen, Themen
  🟢 HH:MM Reiner Kontext, der vollständig ist aber wenig Gewicht hat
- Schreibe kompakt — ein Fakt pro Zeile, kein Fließtext.
- Behalte konkrete Namen, Zahlen, Daten und Entscheidungen bei.
- Lass Small Talk und bedeutungslose Wechsel weg.
- Kein Präambel, keine Erklärungen, direkt mit der ersten Beobachtung beginnen."""

_REFLECTOR_SYSTEM = """Du reorganisierst und verdichtest ein bestehendes Beobachtungslog.

Regeln:
- Antworte NUR mit dem überarbeiteten Log, eine Beobachtung pro Zeile.
- Format: 🔴/🟡/🟢 HH:MM Inhalt
- Kombiniere thematisch verwandte Einträge zu einem einzigen dichteren Eintrag.
- Entferne Redundanzen und überholte Informationen (z.B. ein Plan der bereits ausgeführt wurde).
- Behalte alle konkreten Fakten, Namen, Zahlen und Entscheidungen bei.
- Hochpriorisiere (🔴) was für zukünftige Gespräche am relevantesten ist.
- Kein Präambel, keine Erklärungen, direkt mit der ersten Beobachtung beginnen."""


def _format_messages_for_observer(messages: list[dict]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = "Bot" if msg["role"] == "assistant" else "User"
        ts = msg["created_at"]
        if hasattr(ts, "strftime"):
            time_str = ts.strftime("%H:%M")
        else:
            time_str = "??"
        lines.append(f"[{time_str}] {role}: {msg['content'][:300]}")
    return "\n".join(lines)


def _format_observations_for_reflector(observations: list[dict]) -> str:
    lines: list[str] = []
    for obs in observations:
        lines.append(obs["content"])
    return "\n".join(lines)


async def run_observer(pool: asyncpg.Pool, chat_id: int) -> bool:
    last_observed = await memory.get_last_observed_at(pool, chat_id)

    if last_observed == datetime.min.replace(tzinfo=None):
        since = datetime.min.replace(tzinfo=None)
    else:
        since = last_observed

    unobserved = await memory.count_unobserved_messages(pool, chat_id, since)
    if unobserved < config.BOT_OBSERVER_THRESHOLD:
        return False

    messages = await memory.get_messages_since(pool, chat_id, since)
    if not messages:
        return False

    logger.info("observer: chat %d has %d unobserved messages, compressing", chat_id, len(messages))

    formatted = _format_messages_for_observer(messages)
    now = datetime.now(timezone.utc)
    date_prefix = f"Datum: {now.strftime('%Y-%m-%d')}\n\n"

    try:
        raw = await brain.chat(
            system=_OBSERVER_SYSTEM,
            messages=[{"role": "user", "content": date_prefix + formatted}],
            capability=CAPABILITY_SIMPLE_TASKS,
            caller=f"observer:{chat_id}",
        )
    except Exception as e:
        logger.warning("observer: LLM call failed for chat %d: %s", chat_id, e)
        return False

    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        logger.warning("observer: no observations generated for chat %d", chat_id)
        return False

    observation_block = "\n".join(lines)
    await memory.add_memory(
        pool,
        subject_type="chat",
        subject_id=chat_id,
        content=observation_block,
        memory_type="observation",
        observed_at=now,
    )
    logger.info("observer: stored %d observation lines for chat %d", len(lines), chat_id)
    return True


async def run_reflector(pool: asyncpg.Pool, chat_id: int) -> bool:
    count = await memory.count_observations(pool, chat_id)
    if count < config.BOT_REFLECTOR_THRESHOLD:
        return False

    observations = await memory.get_observations(pool, chat_id, limit=count)
    if not observations:
        return False

    logger.info("reflector: chat %d has %d observation blocks, condensing", chat_id, count)

    formatted = _format_observations_for_reflector(observations)

    try:
        raw = await brain.chat(
            system=_REFLECTOR_SYSTEM,
            messages=[{"role": "user", "content": formatted}],
            capability=CAPABILITY_CHAT,
            caller=f"reflector:{chat_id}",
        )
    except Exception as e:
        logger.warning("reflector: LLM call failed for chat %d: %s", chat_id, e)
        return False

    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        logger.warning("reflector: no condensed observations for chat %d", chat_id)
        return False

    await memory.mark_observations_compressed(pool, chat_id)

    condensed_block = "\n".join(lines)
    await memory.add_memory(
        pool,
        subject_type="chat",
        subject_id=chat_id,
        content=condensed_block,
        memory_type="observation",
        observed_at=datetime.now(timezone.utc),
        is_compressed=True,
    )
    logger.info("reflector: condensed %d blocks into 1 for chat %d", count, chat_id)
    return True


async def get_observation_context(pool: asyncpg.Pool, chat_id: int) -> str:
    observations = await memory.get_observations(pool, chat_id, limit=5)
    if not observations:
        return ""
    blocks = [obs["content"] for obs in reversed(observations)]
    return "\n".join(blocks)
