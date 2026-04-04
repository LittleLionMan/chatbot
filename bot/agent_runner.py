from __future__ import annotations
import json
import logging
import anthropic
import asyncpg
import telegram
from bot import brain, memory
from bot.agent_parser import next_agent_run_after
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)

_AGENT_RUN_SYSTEM = """Du führst einen automatischen Beobachtungsauftrag aus.
Dir werden die Anweisung des Agenten, sein aktueller Gedächtnisstand und verfügbare Daten übergeben.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "report": Dein Bericht in natürlicher Sprache. Wenn es nichts Neues gibt: "KEINE_AENDERUNG".
- "notify_user": true wenn der User benachrichtigt werden soll, false sonst.
- "tool_calls": Liste von Tool-Aufrufen die nach diesem Lauf ausgeführt werden sollen. Leer wenn keine nötig.

Verfügbare Tools in tool_calls:
- {"tool": "db_write", "namespace": "...", "key": "...", "value": "..."} — speichert einen Wert
- {"tool": "db_read", "namespace": "...", "key": "..."} — liest einen Wert (wird im nächsten Lauf als Kontext übergeben)
- {"tool": "db_query", "namespace": "..."} — liest alle Einträge eines Namespaces
- {"tool": "trigger_agent", "target_agent_name": "...", "payload": {...}, "delay_minutes": 0} — löst einen anderen Agenten aus, optional zeitverzögert
- {"tool": "notify_user", "message": "..."} — sendet eine direkte Nachricht (alternative zu notify_user: true)

Beispiel-Output:
{"report": "AAPL hat heute Quartalszahlen veröffentlicht. KGV jetzt bei 28.", "notify_user": true, "tool_calls": [{"tool": "db_write", "namespace": "companies", "key": "AAPL:last_check", "value": "2025-04-04"}, {"tool": "trigger_agent", "target_agent_name": "Analyst", "payload": {"ticker": "AAPL"}}]}"""

_MAX_SUMMARY_CHARS = 800


def _build_relay_system(agent_name: str) -> str:
    return f"""Du bist Bob. Formuliere den folgenden Agenten-Bericht als kurze Nachricht in der dritten Person.
Beispiele: "{agent_name} meldet: ...", "{agent_name} hat etwas gefunden: ...", "Laut {agent_name}: ..."
Keine Einleitung, kein Abschluss — nur die Weiterleitung in Bobs Stimme.
Behalte alle konkreten Fakten, Preise und Links vollständig bei."""


def _build_run_prompt(
    config_data: dict,
    state: dict[str, str],
    injected_data: dict[str, str],
) -> str:
    instruction = config_data.get("instruction", "")
    state_keys: list[str] = config_data.get("state_keys", ["last_run_summary"])

    relevant_state: dict[str, str] = {}
    for k in state_keys:
        if k in state and state[k]:
            value = state[k]
            if k == "last_run_summary" and len(value) > _MAX_SUMMARY_CHARS:
                value = value[:_MAX_SUMMARY_CHARS] + "… [gekürzt]"
            relevant_state[k] = value

    parts = [f"Anweisung: {instruction}"]

    if not relevant_state:
        parts.append("Erster Lauf — kein vorheriger Stand vorhanden.")
    else:
        state_lines = "\n".join(f"{k}: {v}" for k, v in relevant_state.items())
        parts.append(f"Aktueller Stand aus letztem Lauf:\n{state_lines}")

    if injected_data:
        data_lines = "\n".join(f"{k}: {v}" for k, v in injected_data.items())
        parts.append(f"Zusätzliche Daten aus DB:\n{data_lines}")

    return "\n\n".join(parts)


async def _execute_tool_calls(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    agent_id: int,
    target_chat_id: int,
    tool_calls: list[dict],
) -> None:
    for call in tool_calls:
        tool = call.get("tool")
        try:
            if tool == "db_write":
                await memory.write_agent_data(
                    pool,
                    agent_id,
                    call["namespace"],
                    call["key"],
                    str(call["value"]),
                )
                logger.info("Agent %d db_write: %s/%s", agent_id, call["namespace"], call["key"])

            elif tool == "db_read":
                pass

            elif tool == "db_query":
                pass

            elif tool == "trigger_agent":
                target_name: str = call.get("target_agent_name", "")
                payload: dict = call.get("payload", {})
                delay: int = int(call.get("delay_minutes", 0))
                if target_name:
                    await memory.enqueue_agent_trigger(pool, agent_id, target_name, payload, delay)
                    logger.info("Agent %d queued trigger for: %s", agent_id, target_name)

            elif tool == "notify_user":
                msg = call.get("message", "")
                if msg:
                    await bot.send_message(chat_id=target_chat_id, text=msg)
                    logger.info("Agent %d notify_user sent direct message", agent_id)

            else:
                logger.warning("Agent %d unknown tool: %s", agent_id, tool)

        except Exception as e:
            logger.error("Agent %d tool %s failed: %s", agent_id, tool, e)


async def _load_trigger_payload_data(
    pool: asyncpg.Pool,
    payload: dict,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for k, v in payload.items():
        result[f"trigger_payload.{k}"] = str(v)
    return result


async def execute_agent(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    agent: dict,
    trigger_payload: dict | None = None,
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

        injected_data: dict[str, str] = {}
        if trigger_payload:
            injected_data = await _load_trigger_payload_data(pool, trigger_payload)

        previous_summary = state.get("last_run_summary", "")
        prompt = _build_run_prompt(config_data, state, injected_data)

        raw_result = await brain.chat(
            system=_AGENT_RUN_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            use_web_search=True,
            web_search_max_uses=1,
            caller=f"agent:{name}",
            pool=pool,
        )

        try:
            parsed_output = json.loads(clean_llm_json(raw_result))
        except Exception:
            parsed_output = {"report": raw_result, "notify_user": True, "tool_calls": []}

        report: str = parsed_output.get("report", "")
        notify_user: bool = parsed_output.get("notify_user", False)
        tool_calls: list[dict] = parsed_output.get("tool_calls", [])

        if report and report.strip() != "KEINE_AENDERUNG":
            state["last_run_summary"] = report
        elif not report or report.strip() == "KEINE_AENDERUNG":
            notify_user = False

        await memory.set_agent_state(pool, agent_id, state)

        if tool_calls:
            await _execute_tool_calls(pool, bot, agent_id, target_chat_id, tool_calls)

        has_notify_tool = any(c.get("tool") == "notify_user" for c in tool_calls)

        if notify_user and not has_notify_tool and report and report.strip() != "KEINE_AENDERUNG":
            relay_system = _build_relay_system(name)
            message_text = await brain.chat(
                system=relay_system,
                messages=[{"role": "user", "content": report}],
                max_tokens=1024,
                caller=f"agent_relay:{name}",
                pool=pool,
            )
            await bot.send_message(chat_id=target_chat_id, text=message_text)
            await memory.add_memory(pool, "agent", agent_id, report[:200])
            logger.info("Agent %d (%s) reported change.", agent_id, name)
        else:
            logger.info("Agent %d (%s): no relevant change or notify suppressed.", agent_id, name)

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
