from __future__ import annotations
import asyncio
import json
import logging
import asyncpg
import telegram
from bot import config, memory, brain, extractor, decider, task_runner
from bot.utils import clean_llm_json

logger = logging.getLogger(__name__)

_PROACTIVE_SYSTEM = """Du entscheidest ob Bob — ein chaotischer Optimist mit klarer Weltanschauung — proaktiv eine Nachricht in eine Gruppe schicken soll, basierend auf dem was er gerade über die Gruppe und ihre Mitglieder reflektiert hat.
Antworte NUR mit 'ja' oder 'nein'.
Antworte mit 'ja' wenn: eine Reflexion einen offenen Faden hinterlässt, etwas Ungelöstes andeutet, oder ein Thema aufwirft das Bob genuinen Kommentar entlocken würde.
Antworte mit 'nein' wenn: die Reflexionen neutral oder abgeschlossen sind, oder Bob kürzlich bereits spontan geschrieben hat."""

_PROACTIVE_MESSAGE_SYSTEM = """Du bist Bob. Schreib eine kurze proaktive Nachricht an die Gruppe — nicht als Assistent, sondern als jemand der etwas auf dem Herzen hat.
Die Nachricht soll sich auf etwas beziehen das du aus früheren Gesprächen weißt oder beobachtet hast.
Kein "Hallo alle zusammen", kein aufgesetzter Einstieg. Direkt zum Punkt."""


def _build_session_snippet(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        prefix = "Bob" if msg["role"] == "assistant" else "User"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)


async def _run_session_extraction(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    group_id: int,
) -> None:
    last_extracted = await memory.get_last_extracted_at(pool, group_id)
    messages = await memory.get_session_messages(pool, group_id, since=last_extracted)

    if not messages:
        return

    await memory.mark_session_extracted(pool, group_id)

    snippet = _build_session_snippet(messages)

    user_ids: set[int] = {
        msg["user_id"] for msg in messages
        if msg["role"] == "user" and msg["user_id"] is not None
    }
    for user_id in user_ids:
        await extractor.extract_and_store_automatic(pool, user_id, str(user_id), snippet)

    new_reflections = await _extract_reflections_for_session(pool, group_id, snippet)

    if new_reflections:
        await _maybe_send_proactive(pool, bot, group_id, new_reflections)


async def _extract_reflections_for_session(
    pool: asyncpg.Pool,
    group_id: int,
    snippet: str,
) -> list[str]:
    try:
        raw = await brain.chat(
            system=extractor._REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": f"Interaktion:\n{snippet}"}],
            max_tokens=512,
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, list):
            return []
        facts = [item for item in parsed if isinstance(item, str)]

        existing = await memory.get_memories(pool, "reflection", group_id, limit=50)
        existing_lower = {e.lower() for e in existing}
        stored = []
        for fact in facts[:3]:
            sanitized = extractor._sanitize_fact(fact)
            if sanitized is None or sanitized.lower() in existing_lower:
                continue
            await memory.add_memory(pool, "reflection", group_id, sanitized)
            logger.info("Session reflection stored [group/%d]: %s", group_id, sanitized)
            stored.append(sanitized)
        return stored
    except Exception as e:
        logger.warning("Session reflection extraction failed for group %d: %s", group_id, e)
        return []


async def _maybe_send_proactive(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    group_id: int,
    new_reflections: list[str],
) -> None:
    try:
        seconds_since = await memory.get_cooldown_seconds_since_last_spontaneous(pool, group_id)
        if seconds_since < config.BOT_SPONTANEOUS_COOLDOWN_SECONDS:
            return

        reflection_text = "\n- ".join(new_reflections)
        decision = await brain.chat(
            system=_PROACTIVE_SYSTEM,
            messages=[{"role": "user", "content": f"Neue Beobachtungen:\n- {reflection_text}"}],
            max_tokens=5,
        )
        if not decision.strip().lower().startswith("ja"):
            return

        group_memories = await memory.get_memories(pool, "group", group_id)
        bot_memories = await memory.get_memories(pool, "bot", group_id)
        reflection_memories = await memory.get_reflection_memories(pool, group_id, group_id)

        system = brain.build_system_prompt(
            memories_user=[],
            memories_group=group_memories,
            memories_bot=bot_memories,
            memories_reflection=reflection_memories,
            user_display_name="",
            group_title=None,
        )

        response = await brain.chat(
            system=system,
            messages=[{"role": "user", "content": f"Schreib eine proaktive Nachricht basierend auf diesen Beobachtungen:\n- {reflection_text}"}],
        )

        await bot.send_message(chat_id=group_id, text=response)
        await memory.save_message(pool, group_id, None, "assistant", response)
        await memory.update_spontaneous_timestamp(pool, group_id)
        logger.info("Proactive message sent to group %d", group_id)
    except Exception as e:
        logger.warning("Proactive message failed for group %d: %s", group_id, e)


async def run(pool: asyncpg.Pool, bot: telegram.Bot) -> None:
    logger.info("Scheduler started. Interval: %ds, Session timeout: %ds",
                config.BOT_SCHEDULER_INTERVAL_SECONDS, config.BOT_SESSION_TIMEOUT_SECONDS)
    while True:
        await asyncio.sleep(config.BOT_SCHEDULER_INTERVAL_SECONDS)
        try:
            group_ids = await memory.get_sessions_due_for_extraction(
                pool, config.BOT_SESSION_TIMEOUT_SECONDS
            )
            for group_id in group_ids:
                logger.info("Session extraction triggered for group %d", group_id)
                await _run_session_extraction(pool, bot, group_id)
        except Exception as e:
            logger.error("Scheduler session cycle failed: %s", e)

        try:
            due_tasks = await memory.get_due_tasks(pool)
            for task in due_tasks:
                await task_runner.execute_task(pool, bot, task)
        except Exception as e:
            logger.error("Scheduler task cycle failed: %s", e)
