from __future__ import annotations
import json
import logging
import asyncpg
import telegram
from bot import brain, memory
from bot.agent_parser import next_agent_run_after
from bot.brain import ProviderRateLimitError
from bot.models import CAPABILITY_FAST, CAPABILITY_BALANCED, CAPABILITY_SEARCH, CAPABILITY_REASONING, CAPABILITY_DEEP_REASONING, CAPABILITY_CODING
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)

_AGENT_STRUCTURE_SYSTEM = """Du strukturierst das Ergebnis eines Agenten-Laufs in ein JSON-Objekt.

Dir wird das Arbeits-Ergebnis des Agenten übergeben. Extrahiere daraus die strukturierten Ausgaben.

Antworte ausschließlich mit rohem JSON. Der erste Charakter muss { sein, der letzte }.

Felder:
- "report": Zusammenfassung des Laufs in maximal 3 kurzen Sätzen. "KEINE_AENDERUNG" wenn nichts Relevantes passiert ist.
- "notify_user": true wenn der User benachrichtigt werden soll, false sonst.
- "state_updates": Dict mit Key-Value-Paaren die im Agent-State gespeichert werden. Für kompakte persistente Daten: Listen, Flags, kurze Zusammenfassungen die andere Agents lesen sollen. Alle Werte müssen Strings sein.
- "tool_calls": Liste aller Tool-Aufrufe für große Dokumente oder Koordination.

Verfügbare Tools:
- {"tool": "db_write", "namespace": "...", "key": "...", "value": "..."} — für kurze Werte die sicher in JSON passen (URLs, Datum, kurze Statusmeldungen).
- {"tool": "db_write_from_work", "namespace": "...", "key": "...", "source_key": "..."} — für lange Texte, Analysen, Berichte. Das "source_key"-Feld referenziert einen Pipeline-Context-Key dessen Inhalt gespeichert werden soll — nicht den work_result. Wenn der work_result eine Speicheranweisung mit source_key enthält (z.B. "Speichern: db_write_from_work namespace=X key=Y source_key=full_analysis"), MUSS dieser Tool-Call mit exakt diesem source_key erzeugt werden. Wenn verfügbare Pipeline-Context-Keys am Ende des work_result aufgelistet sind, zeigen diese welche source_keys verfügbar sind — nutze sie.
- {"tool": "trigger_agent", "target_agent_name": "...", "payload": {...}, "delay_minutes": 0} — löst einen anderen Agenten aus. Wichtig: target_agent_name exakt so schreiben wie in der Agenten-Instruction genannt — keine Underscores statt Leerzeichen, keine Veränderung der Groß-/Kleinschreibung. Beispiel: "Jim Cramer" nicht "jim_cramer", "Gecko" nicht "gecko".
- {"tool": "notify_user", "message": "..."} — sendet eine Nachricht an den User.

Wann state_updates, wann db_write, wann db_write_from_work:
- state_updates: Watchlists, Ticker-Listen, Flags, kurze Statusinfos — alles was andere Agents via State lesen sollen. Niemals last_run_summary in state_updates — der Runner setzt ihn automatisch aus report. Beispiele für typische State-Keys: analyses_overview, fundamentalanalyse_pending, known_companies.
- db_write: kurze Werte (URLs, Datum, kurze Statusmeldungen) — nur wenn der Wert in einem JSON-String sicher passt.
- db_write_from_work: alle langen Texte — Analysen, Berichte, Markdown-Dokumente. Kein JSON-Escaping nötig.

Wichtig: state_updates werden IMMER gesetzt wenn der Work-Result State-Änderungen beschreibt — unabhängig davon ob report "KEINE_AENDERUNG" ist. Ein "KEINE_AENDERUNG"-Report bedeutet nur dass der User nicht benachrichtigt wird, nicht dass State-Updates weggelassen werden.

report ist die Zusammenfassung für den User und muss immer befüllt sein wenn etwas passiert ist — auch wenn state_updates gesetzt werden. Ohne report kein Telegram-Feedback.

Beispiel:
{"report": "3 neue Unternehmen gefunden und zur Watchlist hinzugefügt.", "notify_user": true, "state_updates": {"known_companies": "ORA, ENPH, BE"}, "tool_calls": [{"tool": "notify_user", "message": "3 neue Unternehmen in der Watchlist."}]}"""

_MAX_SUMMARY_CHARS = 800


def _build_relay_system(agent_name: str) -> str:
    return f"""Du bist {agent_name}. Formuliere den folgenden Bericht als direkte Nachricht in der ersten Person.
Beginne immer mit "{agent_name}:" — nie mit "Bob" oder einem anderen Namen.
Keine Einleitung, kein Abschluss — nur die Nachricht direkt.
Behalte alle konkreten Fakten, Kurse, Kursziele und Empfehlungen vollständig bei."""


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
                    label = read.get("as") or f"db:{agent_name or 'self'}:{namespace}"
                    result[label] = combined
                    logger.info("Agent %d pre-loaded namespace %s from agent_id %d (%d entries)", agent_id, namespace, target_agent_id, len(rows))
            else:
                key = _resolve_template(key, trigger_payload)
                value = await memory.read_agent_data(pool, target_agent_id, namespace, key)
                if value is not None:
                    label = read.get("as") or f"db:{agent_name or 'self'}:{namespace}:{key}"
                    result[label] = value
                    logger.info("Agent %d pre-loaded %s/%s from agent_id %d", agent_id, namespace, key, target_agent_id)

    return result


async def _execute_tool_calls(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    agent_id: int,
    target_chat_id: int,
    tool_calls: list[dict],
    work_result: str = "",
    pipeline_context: dict[str, str] | None = None,
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

            elif tool == "db_write_from_work":
                source_key: str | None = call.get("source_key")
                if source_key and pipeline_context and source_key in pipeline_context:
                    content = pipeline_context[source_key]
                    logger.info("Agent %d db_write_from_work (source_key=%s): %s/%s (%d chars)", agent_id, source_key, call["namespace"], call["key"], len(content))
                else:
                    content = work_result
                    if source_key:
                        logger.warning("Agent %d db_write_from_work: source_key '%s' not found in context, falling back to work_result", agent_id, source_key)
                    logger.info("Agent %d db_write_from_work: %s/%s (%d chars)", agent_id, call["namespace"], call["key"], len(content))
                await memory.write_agent_data(
                    pool,
                    agent_id,
                    call["namespace"],
                    call["key"],
                    content,
                )

            elif tool == "notify_user":
                msg = call.get("message", "")
                if msg:
                    await bot.send_message(chat_id=target_chat_id, text=msg)
                    logger.info("Agent %d notify_user sent direct message", agent_id)

            else:
                logger.warning("Agent %d unknown tool: %s", agent_id, tool)

        except Exception as e:
            logger.error("Agent %d tool %s failed: %s", agent_id, tool, e)


def _resolve_pipeline_template(template: str, context: dict[str, str]) -> str:
    for k, v in context.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    return template


_AGENT_OUTPUT_SYSTEM = """Du strukturierst das Ergebnis eines Agenten-Laufs in ein JSON-Objekt.
Antworte ausschließlich mit rohem JSON. Der erste Charakter muss { sein, der letzte }.

Felder:
- "report": Zusammenfassung in maximal 3 kurzen Sätzen. "KEINE_AENDERUNG" wenn nichts Relevantes passiert ist.
- "notify_user": true wenn der User benachrichtigt werden soll, false sonst.
- "state_updates": Dict mit Key-Value-Paaren für den Agent-State. Alle Werte müssen Strings sein. Niemals "last_run_summary" setzen.
- "tool_calls": Liste der Tool-Aufrufe.

Verfügbare Tools:
- {"tool": "db_write", "namespace": "...", "key": "...", "value": "..."} — kurze Werte (URLs, Datum, Statusmeldungen).
- {"tool": "db_write_from_work", "namespace": "...", "key": "...", "source_key": "..."} — lange Texte. source_key referenziert einen Pipeline-Context-Key. Nur verwenden wenn im Prompt explizit angewiesen.
- {"tool": "trigger_agent", "target_agent_name": "...", "payload": {...}, "delay_minutes": 0} — target_agent_name exakt wie in der Instruction genannt.
- {"tool": "notify_user", "message": "..."} — Nachricht an den User.

Wichtig: Erzeuge NUR Tool-Calls die im Prompt explizit angewiesen werden. Keine eigenen Entscheidungen über was gespeichert oder getriggert werden soll."""

_ROUTER_SYSTEM = """Du entscheidest welcher Ausführungspfad gilt.
Antworte NUR mit einem einzigen Wort. Befolge die Entscheidungslogik im Router-Prompt exakt und in der angegebenen Reihenfolge.
Wenn keine Bedingung zutrifft: antworte mit 'normal'."""


def _expand_pipeline_template(
    config_data: dict,
    state: dict[str, str],
    injected_data: dict[str, str],
) -> list[dict]:
    template = config_data.get("pipeline_template")
    if not template:
        return []

    source = template.get("source", "state")
    foreach_key = template.get("foreach", "")
    split_by = template.get("split_by", ",")
    batch_size = int(template.get("batch_size", 1))
    step_template = template.get("step", {})
    aggregate_key = template.get("aggregate_key", "template_results")
    only_if_route = template.get("only_if_route")
    step_time_range: str | None = template.get("time_range")
    step_categories: str | None = template.get("categories")
    foreach_items: list[str] = template.get("foreach_items", [])

    if source == "static":
        items = foreach_items
    elif source == "state":
        raw = state.get(foreach_key, "")
        items = [i.strip() for i in raw.split(split_by) if i.strip()]
    elif source == "injected":
        raw = injected_data.get(foreach_key, "")
        items = [i.strip() for i in raw.split(split_by) if i.strip()]
    else:
        items = []

    if not items:
        logger.info("Pipeline template foreach '%s' (source=%s) produced no items", foreach_key, source)
        return []

    batches: list[list[str]] = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])

    expanded: list[dict] = []
    all_output_keys: list[str] = []

    for batch in batches:
        item_str = ", ".join(batch)
        safe_id = item_str.lower().replace(" ", "_").replace("/", "_").replace("<", "lt").replace(">", "gt")[:40]

        step = {}
        for k, v in step_template.items():
            if isinstance(v, str):
                step[k] = v.replace("{{item}}", item_str).replace("{{item_id}}", safe_id)
            else:
                step[k] = v

        step["id"] = step.get("id", f"template_{safe_id}").replace("{{item}}", item_str).replace("{{item_id}}", safe_id)
        step["output_key"] = step.get("output_key", f"result_{safe_id}").replace("{{item}}", item_str).replace("{{item_id}}", safe_id)

        if only_if_route:
            step["only_if_route"] = only_if_route
        if step_time_range:
            step["time_range"] = step_time_range
        if step_categories:
            step["categories"] = step_categories

        all_output_keys.append(step["output_key"])
        expanded.append(step)
        logger.info("Pipeline template expanded step '%s' for item(s): %s", step["id"], item_str)

    config_data["_template_output_keys"] = all_output_keys
    config_data["_template_aggregate_key"] = aggregate_key

    return expanded


async def _execute_pipeline(
    pool: asyncpg.Pool,
    agent_id: int,
    name: str,
    pipeline: list[dict],
    state: dict[str, str],
    injected_data: dict[str, str],
    config_data: dict,
) -> tuple[str, dict[str, str]]:
    context: dict[str, str] = {}
    context.update({k: v for k, v in state.items() if v})
    context.update(injected_data)

    instruction = config_data.get("instruction", "")
    agent_system = f"Gesamtauftrag des Agenten:\n{instruction}"

    pipeline_before: list[dict] = config_data.get("pipeline", [])
    template_steps: list[dict] = _expand_pipeline_template(config_data, state, injected_data)
    pipeline_after: list[dict] = config_data.get("pipeline_after_template", [])

    full_pipeline = pipeline_before + template_steps + pipeline_after
    if not full_pipeline:
        full_pipeline = pipeline

    aggregate_key: str = config_data.get("_template_aggregate_key", "")
    template_output_keys: list[str] = config_data.get("_template_output_keys", [])

    active_route: str | None = None
    last_output = ""

    for step in full_pipeline:
        step_id: str = step["id"]
        capability: str = step["capability"]
        prompt_template: str = step["prompt_template"]
        output_key: str = step["output_key"]
        is_router: bool = step.get("is_router", False)
        only_if_route: list[str] | str | None = step.get("only_if_route")

        if only_if_route is not None and active_route is not None:
            allowed = [only_if_route] if isinstance(only_if_route, str) else only_if_route
            if active_route not in allowed:
                logger.info("Agent %d (%s) pipeline step '%s' skipped (route=%s, allowed=%s)", agent_id, name, step_id, active_route, allowed)
                continue

        prompt = _resolve_pipeline_template(prompt_template, context)

        is_output_step: bool = step.get("is_output", False)

        if is_output_step:
            try:
                output_raw = await brain.chat(
                    system=_AGENT_OUTPUT_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    capability=CAPABILITY_FAST,
                    caller=f"agent_output:{name}",
                    pool=pool,
                )
                context[output_key] = output_raw
                last_output = output_raw
                logger.info("Agent %d (%s) output step '%s' done (%d chars)", agent_id, name, step_id, len(output_raw))
            except Exception as e:
                logger.error("Agent %d (%s) output step '%s' failed: %s", agent_id, name, step_id, e)
                raise
            continue

        if is_router:
            try:
                payload_keys = {k: v for k, v in context.items() if k.startswith("trigger_payload.")}
                payload_summary = "\n".join(f"  {k} = {v}" for k, v in payload_keys.items()) if payload_keys else "  (kein Payload vorhanden)"
                state_summary = "\n".join(f"  {k} = {str(v)[:80]}" for k, v in context.items() if not k.startswith("trigger_payload.") and not k.startswith("state:") and not k.startswith("db:"))

                router_context = (
                    f"Trigger-Payload:\n{payload_summary}\n\n"
                    f"Aktueller State:\n{state_summary}\n\n"
                    f"Router-Prompt: {prompt}"
                )
                route_output = await brain.chat(
                    system=_ROUTER_SYSTEM,
                    messages=[{"role": "user", "content": router_context}],
                    max_tokens=20,
                    capability=CAPABILITY_FAST,
                    caller=f"agent_router:{name}",
                    pool=pool,
                )
                active_route = route_output.strip().lower()
                context[output_key] = active_route
                last_output = active_route
                logger.info("Agent %d (%s) router decided route: '%s' (payload_keys: %s)", agent_id, name, active_route, list(payload_keys.keys()))
            except Exception as e:
                logger.error("Agent %d (%s) router failed: %s", agent_id, name, e)
                raise
            continue

        is_search_step = capability == CAPABILITY_SEARCH
        is_finance_step = capability == "finance"
        use_web_search = is_search_step
        web_search_max_uses = 3 if is_search_step else None
        search_time_range: str | None = step.get("time_range") if is_search_step else None
        search_categories: str | None = step.get("categories") if is_search_step else None

        if is_finance_step:
            from bot import finance as _finance
            ticker_key = step.get("ticker_key", "selected_ticker")
            ticker = context.get(ticker_key, "").strip()
            if not ticker:
                logger.warning("Agent %d (%s) finance step '%s': no ticker in context key '%s'", agent_id, name, step_id, ticker_key)
                context[output_key] = "Kein Ticker verfügbar."
                last_output = context[output_key]
                continue
            logger.info("Agent %d (%s) pipeline step '%s' [finance] ticker=%s", agent_id, name, step_id, ticker)
            finance_result = await _finance.get_quote_summary(ticker)
            if not finance_result:
                logger.warning("Agent %d (%s) finance step '%s': no data for %s", agent_id, name, step_id, ticker)
                finance_result = f"Keine Finanzdaten für {ticker} verfügbar."
            context[output_key] = finance_result
            last_output = finance_result
            logger.info("Agent %d (%s) step '%s' done (%d chars output)", agent_id, name, step_id, len(finance_result))
            continue

        search_queries: list[str] | None = None
        if is_search_step:
            raw_query = step.get("search_query", "")
            if raw_query:
                resolved_query = _resolve_pipeline_template(raw_query, context)
                search_queries = [resolved_query]
                logger.info("Agent %d (%s) step '%s' using search_query: %r", agent_id, name, step_id, resolved_query)
            else:
                search_queries = [prompt]

        force_model: str | None = None
        if is_search_step:
            from bot.models import select_model_for_provider
            from bot import search as _search
            if not await _search.is_available():
                force_model = select_model_for_provider(capability, "anthropic")

        logger.info(
            "Agent %d (%s) pipeline step '%s' [%s]%s%s",
            agent_id, name, step_id, capability,
            f" → {force_model}" if force_model else "",
            f" time_range={search_time_range}" if search_time_range else "",
        )

        try:
            step_output = await brain.chat(
                system=agent_system,
                messages=[{"role": "user", "content": prompt}],
                use_web_search=use_web_search,
                web_search_max_uses=web_search_max_uses,
                search_queries=search_queries,
                search_time_range=search_time_range,
                search_categories=search_categories,
                capability=capability,
                force_model=force_model,
                caller=f"agent_pipeline:{name}:{step_id}",
                pool=pool,
            )
        except Exception as e:
            logger.error("Agent %d (%s) pipeline step '%s' failed: %s", agent_id, name, step_id, e)
            raise

        if is_search_step and not step_output:
            context[output_key] = "Keine aktuellen Suchergebnisse verfügbar."
            last_output = context[output_key]
            logger.info("Agent %d (%s) step '%s' skipped — no search results", agent_id, name, step_id)
        else:
            context[output_key] = step_output
            last_output = step_output
            logger.info(
                "Agent %d (%s) step '%s' done (%d chars output)",
                agent_id, name, step_id, len(step_output),
            )

        if aggregate_key and output_key in template_output_keys:
            existing = context.get(aggregate_key, "")
            context[aggregate_key] = f"{existing}\n\n---\n\n{output_key}:\n{step_output}" if existing else f"{output_key}:\n{step_output}"

    has_output_step = any(s.get("is_output", False) for s in full_pipeline)
    return last_output, context, has_output_step


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

    work_capability: str = config_data.get("work_capability", CAPABILITY_BALANCED)

    logger.info("Executing agent %d (%s) for user %d with capability=%s", agent_id, name, user_id, work_capability)

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

        pipeline: list[dict] = config_data.get("pipeline", [])

        work_result, pipeline_context, has_output_step = await _execute_pipeline(
            pool, agent_id, name, pipeline, state, injected_data, config_data,
        )

        logger.warning("Agent %d (%s) work result: %r", agent_id, name, work_result[:300])

        if has_output_step:
            try:
                parsed_output = json.loads(clean_llm_json(work_result))
                logger.warning("Agent %d (%s) output parsed directly from pipeline", agent_id, name)
            except Exception:
                logger.warning("Agent %d (%s): output step JSON parse failed: %r", agent_id, name, work_result[:300])
                parsed_output = {"report": work_result, "notify_user": True, "tool_calls": []}
        else:
            raw_structured = await brain.chat(
                system=_AGENT_STRUCTURE_SYSTEM,
                messages=[{"role": "user", "content": work_result}],
                capability=CAPABILITY_FAST,
                caller=f"agent_structure:{name}",
                pool=pool,
            )
            logger.warning("Agent %d (%s) structure result: %r", agent_id, name, raw_structured[:300])
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
            await _execute_tool_calls(pool, bot, agent_id, target_chat_id, tool_calls, work_result, pipeline_context)

        has_notify_tool = any(c.get("tool") == "notify_user" for c in tool_calls)

        if notify_user and not has_notify_tool and report and report.strip() != "KEINE_AENDERUNG":
            relay_system = _build_relay_system(name)
            message_text = await brain.chat(
                system=relay_system,
                messages=[{"role": "user", "content": report}],
                capability=CAPABILITY_FAST,
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

    except ProviderRateLimitError as e:
        logger.error("Agent %d (%s) hit rate limit on provider %s, not updating next_run: %s", agent_id, name, e.provider, e)

    except Exception as e:
        logger.error("Agent %d (%s) execution failed: %s", agent_id, name, e)
        try:
            tz = await memory.get_user_timezone(pool, user_id)
            next_run = next_agent_run_after(schedule, tz)
            await memory.update_agent_run(pool, agent_id, next_run)
            logger.info("Agent %d next_run updated after error. Next: %s", agent_id, next_run.isoformat())
        except Exception as inner_e:
            logger.error("Agent %d failed to update next_run after error: %s", agent_id, inner_e)
