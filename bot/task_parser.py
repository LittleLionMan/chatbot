from __future__ import annotations
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from croniter import croniter
import asyncpg
from bot import brain, config, memory
from bot.utils import clean_llm_json

logger = logging.getLogger(__name__)

_TASK_PARSER_SYSTEM = """Du extrahierst einen wiederkehrenden Auftrag aus einer Nutzeranfrage.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "description": Was soll getan werden? Klarer Aufgabentext in einem Satz, maximal 200 Zeichen.
- "schedule": Wann und wie oft? Als Cron-Expression (5 Felder: Minute Stunde Tag Monat Wochentag). Beispiele: täglich um 9 Uhr = "0 9 * * *", stündlich = "0 * * * *", montags um 8 = "0 8 * * 1".
- "target": "same" wenn das Ergebnis in denselben Chat soll, "dm" wenn es als Privatnachricht an den User gehen soll.

Wenn kein sinnvoller Zeitplan erkennbar ist, setze schedule auf null.

Beispiel-Output:
{"description": "Günstige Grafikkarten auf Sekundärmärkten suchen und auflisten", "schedule": "0 9 * * *", "target": "same"}"""

_STOP_PARSER_SYSTEM = """Du analysierst eine Nutzeranfrage die einen oder mehrere aktive Tasks stoppen möchte.

Dir wird eine Liste aktiver Tasks gegeben. Antworte NUR mit einem JSON-Array von Task-IDs (integers) die gestoppt werden sollen.
Wenn keine passenden Tasks gefunden werden: [].
Kein anderer Text, keine Markdown-Backticks.

Beispiel: [3, 7]"""

_LIST_TRIGGER_SYSTEM = """Fragt der Nutzer nach seinen aktiven Aufgaben oder Tasks?
Antworte NUR mit 'ja' oder 'nein'."""

_CREATE_TRIGGER_SYSTEM = """Möchte der Nutzer eine neue wiederkehrende Aufgabe erstellen?
Kriterien: Die Nachricht beschreibt etwas das regelmäßig oder zu einem bestimmten Zeitpunkt automatisch erledigt werden soll.
Antworte NUR mit 'ja' oder 'nein'."""


async def is_task_creation(text: str) -> bool:
    try:
        result = await brain.chat(
            system=_CREATE_TRIGGER_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=5,
        )
        return result.strip().lower().startswith("ja")
    except Exception as e:
        logger.warning("Task creation detection failed: %s", e)
        return False


async def is_task_list_request(text: str) -> bool:
    try:
        result = await brain.chat(
            system=_LIST_TRIGGER_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=5,
        )
        return result.strip().lower().startswith("ja")
    except Exception as e:
        logger.warning("Task list detection failed: %s", e)
        return False


async def parse_task(
    text: str,
    user_id: int,
    source_chat_id: int,
    pool: asyncpg.Pool,
) -> dict | None:
    try:
        raw = await brain.chat(
            system=_TASK_PARSER_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=256,
        )
        logger.debug("Task parser raw LLM output: %r", raw)
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            return None

        description = parsed.get("description", "").strip()
        schedule = parsed.get("schedule")
        target = parsed.get("target", "same")

        if not description or not schedule:
            return None

        if not croniter.is_valid(schedule):
            logger.warning("Invalid cron expression from LLM: %s", schedule)
            return None

        tz_str = await memory.get_user_timezone(pool, user_id)
        try:
            tz = ZoneInfo(tz_str)
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)
        cron = croniter(schedule, now)
        next_run_naive = cron.get_next(datetime)
        if next_run_naive.tzinfo is None:
            next_run_aware = next_run_naive.replace(tzinfo=tz)
        else:
            next_run_aware = next_run_naive
        next_run_local = next_run_aware.astimezone(tz)

        target_chat_id = user_id if target == "dm" else source_chat_id

        return {
            "description": description,
            "schedule": schedule,
            "target_chat_id": target_chat_id,
            "next_run_at": next_run_local,
        }
    except Exception as e:
        logger.warning("Task parsing failed: %s", e)
        return None


async def parse_stop_request(
    text: str,
    active_tasks: list[dict],
) -> list[int]:
    if not active_tasks:
        return []
    try:
        task_list = "\n".join(
            f"ID {t['id']}: {t['description']} ({t['schedule']})"
            for t in active_tasks
        )
        raw = await brain.chat(
            system=_STOP_PARSER_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Aktive Tasks:\n{task_list}\n\nNutzeranfrage: {text}",
            }],
            max_tokens=64,
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, int)]
    except Exception as e:
        logger.warning("Stop request parsing failed: %s", e)
        return []


def next_run_after(schedule: str, timezone: str) -> datetime:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(config.BOT_DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    next_run_naive = croniter(schedule, now).get_next(datetime)
    if next_run_naive.tzinfo is None:
        next_run_aware = next_run_naive.replace(tzinfo=tz)
    else:
        next_run_aware = next_run_naive
    return next_run_aware.astimezone(tz)
