from __future__ import annotations
import logging
import asyncpg
import telegram
from bot import brain, memory
from bot.task_parser import next_run_after
from bot.soul import SOUL
from bot.models import CAPABILITY_BALANCED, CAPABILITY_SEARCH

logger = logging.getLogger(__name__)

_TASK_EXECUTION_SYSTEM = f"""{SOUL}

Du führst gerade einen automatischen Auftrag aus. Liefere das Ergebnis direkt und prägnant — keine Einleitung wie "Hier ist dein Ergebnis", kein Abschluss wie "Ich hoffe das hilft". Einfach das Ergebnis."""


async def execute_task(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    task: dict,
) -> None:
    task_id: int = task["id"]
    user_id: int = task["user_id"]
    target_chat_id: int = task["target_chat_id"]
    description: str = task["description"]
    schedule: str = task["schedule"]

    logger.info("Executing task %d for user %d: %s", task_id, user_id, description)

    try:
        response = await brain.chat(
            system=_TASK_EXECUTION_SYSTEM,
            messages=[{"role": "user", "content": description}],
            max_tokens=2048,
            use_web_search=True,
            capability=CAPABILITY_SEARCH,
        )

        await bot.send_message(chat_id=target_chat_id, text=response)

        tz = await memory.get_user_timezone(pool, user_id)
        next_run = next_run_after(schedule, tz)
        await memory.update_task_run(pool, task_id, next_run)

        logger.info("Task %d done. Next run: %s", task_id, next_run.isoformat())
    except Exception as e:
        logger.error("Task %d execution failed: %s", task_id, e)
