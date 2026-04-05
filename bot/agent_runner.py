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

_AGENT_WORK_SYSTEM = """Du führst einen automatischen Beobachtungsauftrag aus.
Dir werden die Anweisung des Agenten, sein aktueller Gedächtnisstand und alle relevanten Daten aus der Datenbank übergeben.

Führe die Anweisung vollständig aus. Denke laut nach, recherchiere, analysiere.
Schreibe dein Ergebnis als klaren Fließtext — was du gefunden hast, was sich geändert hat, was gespeichert werden soll und warum.
Wichtig: Beschreibe explizit welche Daten du speichern möchtest und unter welchem Key."""

_AGENT_STRUCTURE_SYSTEM = """Du strukturierst das Ergebnis eines Agenten-Laufs in ein JSON-Objekt.

Dir wird das Arbeits-Ergebnis des Agenten übergeben. Extrahiere daraus die strukturierten Ausgaben.

Antworte ausschließlich mit rohem JSON. Der erste Charakter muss { sein, der letzte }.

Felder:
- "report": Zusammenfassung des Laufs in 1-3 Sätzen. "KEINE_AENDERUNG" wenn nichts Relevantes passiert ist.
- "notify_user": true wenn der User benachrichtigt werden soll, false sonst.
- "state_updates": Dict mit Key-Value-Paaren die im Agent-State gespeichert werden. Für kompakte persistente Daten: Listen, Flags, kurze Zusammenfassungen die andere Agents lesen sollen. Alle Werte müssen Strings sein.
- "tool_calls": Liste aller Tool-Aufrufe für große Dokumente oder Koordination.

Verfügbare Tools:
- {"tool": "db_write", "namespace": "...", "key": "...", "value": "..."} — nur für große Dokumente: vollständige Analysen, lange Berichte. Nicht für kompakte Listen.
- {"tool": "trigger_agent", "target_agent_name": "...", "payload": {...}, "delay_minutes": 0} — löst einen anderen Agenten aus.
- {"tool": "notify_user", "message": "..."} — sendet eine Nachricht an den User.

Wann state_updates, wann db_write:
- state_updates: Watchlists, Ticker-Listen, Flags, kurze Statusinfos — alles was andere Agents via State lesen sollen. Niemals last_run_summary in state_updates — der Runner setzt ihn automatisch aus report.
- db_write: Fundamentalanalysen, lange Berichte, Dokumente — alles was zu groß für den State ist

report ist die Zusammenfassung für den User und muss immer befüllt sein wenn etwas passiert ist — auch wenn state_updates gesetzt werden. Ohne report kein Telegram-Feedback.

Beispiel:
{"report": "3 neue Unternehmen gefunden und zur Watchlist hinzugefügt.", "notify_user": true, "state_updates": {"known_companies": "ORA, ENPH, BE"}, "tool_calls": [{"tool": "notify_user", "message": "3 neue Unternehmen in der Watchlist."}]}"""

_MAX_SUMMARY_CHARS = 800


def _build_relay_system(agent_name: str) -> str:
    return f"""Du bist Bob. Formuliere den folgenden Agenten-Bericht als kurze Nachricht in der dritten Person.
Beginne die Nachricht immer mit dem Namen des Agenten.
Beispiele: "{agent_name} meldet: ...", "{agent_name} hat etwas gefunden: ...", "Laut {agent_name}: ..."
Keine Einleitung, kein Abschluss — nur die Weiterleitung in Bobs Stimme.
Behalte alle konkreten Fakten, Preise und Links vollständig bei."""


def _resolve_template(template: str, trigger_payload: dict[str, str]) -> str:
    for k, v in trigger_payload.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    return template


async def _load_data_reads(
    pool: asyncpg.Pool,
    agent_id: int,
    data_reads: list[dict],
    trigger_payload: dict[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for read in data_reads:
        read_type = read.get("type", "namespace")
        agent_name = read.get("agent_name")

        if read_type == "state":
            if not agent_name:
                logger.warning("Agent %d data_read type=state missing agent_name, skipping", agent_id)
                continue
            state = await memory.get_agent_state_by_name(pool, agent_name)
            logger.warning("Agent %d data_read state from '%s': %r", agent_id, agent_name, state)
            if state is None:
                logger.warning("Agent %d data_read: agent '%s' not found or has no state", agent_id, agent_name)
                continue
            combined = "\n".join(f"{k}: {v}" for k, v in state.items() if k != "last_run_summary")
            if combined:
                result[f"state:{agent_name}"] = combined
                logger.info("Agent %d pre-loaded state from '%s' (%d keys)", agent_id, agent_name, len(state))
            else:
                logger.warning("Agent %d data_read state from '%s' was empty after filtering", agent_id, agent_name)

        else:
            namespace = _resolve_template(read.get("namespace", ""), trigger_payload)
            key = read.get("key", "")

            target_agent_id: int = agent_id
            if agent_name:
                resolved = await memory.get_agent_id_by_name(pool, agent_name)
                if resolved is None:
                    logger.warning("Agent %d data_read: agent_name '%s' not found, skipping", agent_id, agent_name)
                    continue
                target_agent_id = resolved

            if not key:
                rows = await memory.query_agent_data(pool, namespace, agent_id=target_agent_id)
                if rows:
                    combined = "\n".join(f"{r['key']}: {r['value']}" for r in rows)
                    label = f"db:{agent_name or 'self'}:{namespace}"
                    result[label] = combined
                    logger.info("Agent %d pre-loaded namespace %s from agent_id %d (%d entries)", agent_id, namespace, target_agent_id, len(rows))
            else:
                key = _resolve_template(key, trigger_payload)
                value = await memory.read_agent_data(pool, target_agent_id, namespace, key)
                if value is not None:
                    label = f"db:{agent_name or 'self'}:{namespace}:{key}"
                    result[label] = value
                    logger.info("Agent %d pre-loaded %s/%s from agent_id %d", agent_id, namespace, key, target_agent_id)

    return result


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
        parts.append(f"Daten aus der Datenbank:\n{data_lines}")

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

            elif tool == "trigger_agent":
                target_name: str = call.get("target_agent_name", "")
                payload: dict = call.get("payload", {})
                delay: int = int(call.get("delay_minutes", 0))
                if target_name:
                    await memory.enqueue_agent_trigger(pool, agent_id, target_name, payload, delay)
                    logger.info("Agent %d queued trigger for: %s (delay: %dm)", agent_id, target_name, delay)

            elif tool == "notify_user":
                msg = call.get("message", "")
                if msg:
                    await bot.send_message(chat_id=target_chat_id, text=msg)
                    logger.info("Agent %d notify_user sent direct message", agent_id)

            else:
                logger.warning("Agent %d unknown tool: %s", agent_id, tool)

        except Exception as e:
            logger.error("Agent %d tool %s failed: %s", agent_id, tool, e)


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

        flat_payload: dict[str, str] = {k: str(v) for k, v in (trigger_payload or {}).items()}

        injected_data: dict[str, str] = {}
        for k, v in flat_payload.items():
            injected_data[f"trigger_payload.{k}"] = v

        data_reads: list[dict] = config_data.get("data_reads", [])
        if data_reads:
            db_data = await _load_data_reads(pool, agent_id, data_reads, flat_payload)
            injected_data.update(db_data)

        prompt = _build_run_prompt(config_data, state, injected_data)

        work_result = await brain.chat(
            system=_AGENT_WORK_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            use_web_search=True,
            web_search_max_uses=1,
            caller=f"agent_work:{name}",
            pool=pool,
        )

        logger.debug("Agent %d (%s) work result: %r", agent_id, name, work_result[:200])

        raw_structured = await brain.chat(
            system=_AGENT_STRUCTURE_SYSTEM,
            messages=[{"role": "user", "content": work_result}],
            max_tokens=2048,
            use_web_search=False,
            caller=f"agent_structure:{name}",
            pool=pool,
        )

        try:
            parsed_output = json.loads(clean_llm_json(raw_structured))
        except Exception:
            logger.warning("Agent %d (%s): structure JSON parse failed: %r", agent_id, name, raw_structured[:300])
            parsed_output = {"report": work_result, "notify_user": True, "tool_calls": []}

        report: str = parsed_output.get("report", "")
        notify_user: bool = parsed_output.get("notify_user", False)
        tool_calls: list[dict] = parsed_output.get("tool_calls", [])
        state_updates: dict[str, str] = parsed_output.get("state_updates", {})

        if report and report.strip() != "KEINE_AENDERUNG":
            state["last_run_summary"] = report
        else:
            notify_user = False

        for k, v in state_updates.items():
            if k == "last_run_summary":
                continue
            state[k] = str(v)
            logger.info("Agent %d state_update: %s = %r", agent_id, k, str(v)[:80])

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
