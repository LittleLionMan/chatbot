from __future__ import annotations
import logging
import anthropic
import asyncpg
import telegram
from bot import brain, memory
from bot.agent_parser import next_agent_run_after
from bot.utils import parse_agent_config

logger = logging.getLogger(__name__)

_AGENT_RUN_SYSTEM = """Du führst einen automatischen Beobachtungsauftrag aus.
Dir werden die Anweisung des Agenten und sein aktueller Gedächtnisstand übergeben.
Führe die Anweisung aus. Liefere das Ergebnis direkt und prägnant — keine Einleitung, kein Abschluss.
Wenn es im Vergleich zum letzten Stand nichts Neues oder Relevantes gibt, antworte mit dem exakten Text: KEINE_AENDERUNG"""

_MAX_SUMMARY_CHARS = 800


def _build_relay_system(agent_name: str) -> str:
    return f"""Du bist Bob. Formuliere den folgenden Agenten-Bericht als kurze Nachricht in der dritten Person.
Beispiele: "{agent_name} meldet: ...", "{agent_name} hat etwas gefunden: ...", "Laut {agent_name}: ..."
Keine Einleitung, kein Abschluss — nur die Weiterleitung in Bobs Stimme.
Behalte alle konkreten Fakten, Preise und Links vollständig bei."""


def _build_run_prompt(config_data: dict, state: dict[str, str]) -> str:
    instruction = config_data.get("instruction", "")
    state_keys: list[str] = config_data.get("state_keys", ["last_run_summary"])

    relevant_state: dict[str, str] = {}
    for k in state_keys:
        if k in state and state[k]:
            value = state[k]
            if k == "last_run_summary" and len(value) > _MAX_SUMMARY_CHARS:
                value = value[:_MAX_SUMMARY_CHARS] + "… [gekürzt]"
            relevant_state[k] = value

    if not relevant_state:
        return f"Anweisung: {instruction}\n\nErster Lauf — kein vorheriger Stand vorhanden."

    state_lines = "\n".join(f"{k}: {v}" for k, v in relevant_state.items())
    return f"Anweisung: {instruction}\n\nAktueller Stand aus letztem Lauf:\n{state_lines}"


def _has_relevant_change(previous_summary: str, new_result: str) -> bool:
    if not previous_summary:
        return True
    return new_result.strip() != "KEINE_AENDERUNG"


async def execute_agent(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    agent: dict,
) -> None:
    agent_id: int = agent["id"]
    user_id: int = agent["user_id"]
    target_chat_id: int = agent["target_chat_id"]
    name: str = agent["name"]
    config_data: dict = parse_agent_config(agent["config"])
    schedule: str = agent["schedule"]

    logger.info("Executing agent %d (%s) for user %d", agent_id, name, user_id)

    try:
        state = await memory.get_agent_state(pool, agent_id)
        state_keys: list[str] = config_data.get("state_keys", ["last_run_summary"])

        for key in state_keys:
            if key not in state:
                state[key] = ""

        previous_summary = state.get("last_run_summary", "")
        prompt = _build_run_prompt(config_data, state)

        raw_result = await brain.chat(
            system=_AGENT_RUN_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            use_web_search=True,
            web_search_max_uses=2,
        )

        relevant = _has_relevant_change(previous_summary, raw_result)

        state["last_run_summary"] = raw_result if raw_result.strip() != "KEINE_AENDERUNG" else previous_summary
        await memory.set_agent_state(pool, agent_id, state)

        if relevant and raw_result.strip() != "KEINE_AENDERUNG":
            relay_system = _build_relay_system(name)
            message_text = await brain.chat(
                system=relay_system,
                messages=[{"role": "user", "content": raw_result}],
                max_tokens=1024,
            )
            await bot.send_message(chat_id=target_chat_id, text=message_text)
            await memory.add_memory(pool, "agent", agent_id, raw_result[:200])
            logger.info("Agent %d (%s) reported change.", agent_id, name)
        else:
            logger.info("Agent %d (%s): no relevant change, skipping report.", agent_id, name)

        tz = await memory.get_user_timezone(pool, user_id)
        next_run = next_agent_run_after(schedule, tz)
        await memory.update_agent_run(pool, agent_id, next_run)
        logger.info("Agent %d done. Next run: %s", agent_id, next_run.isoformat())

    except anthropic.RateLimitError as e:
        logger.error("Agent %d (%s) hit rate limit, not updating next_run: %s", agent_id, name, e)

    except Exception as e:
        logger.error("Agent %d (%s) execution failed: %s", agent_id, name, e)
        try:
            tz = await memory.get_user_timezone(pool, user_id)
            next_run = next_agent_run_after(schedule, tz)
            await memory.update_agent_run(pool, agent_id, next_run)
            logger.info("Agent %d next_run updated after error. Next: %s", agent_id, next_run.isoformat())
        except Exception as inner_e:
            logger.error("Agent %d failed to update next_run after error: %s", agent_id, inner_e)
