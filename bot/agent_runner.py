from __future__ import annotations
import json
import logging
import re
import statistics
from typing import Callable, Awaitable
import io
import httpx
import openpyxl
import asyncpg
import telegram
from bot import brain, memory
from bot.agent_parser import next_agent_run_after
from bot.brain import ProviderRateLimitError
from bot.models import CAPABILITY_SIMPLE_TASKS, CAPABILITY_CHAT, CAPABILITY_REASONING, CAPABILITY_DEEP_REASONING
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)


_RELAY_SYSTEM_TEMPLATE = """Du bist {name}. Formuliere den folgenden Bericht als direkte Nachricht in der ersten Person.
Beginne immer mit "{name}:" — nie mit "Bob" oder einem anderen Namen.
Keine Einleitung, kein Abschluss — nur die Nachricht direkt.
Behalte alle konkreten Fakten vollständig bei."""


def _build_relay_system(agent_name: str) -> str:
    return _RELAY_SYSTEM_TEMPLATE.format(name=agent_name)


def _resolve_template(template: str, context: dict[str, str]) -> str:
    for k, v in context.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    for match in re.findall(r"\{\{([^}]+)\}\}", template):
        if "." in match:
            resolved = _get(context, match)
            template = template.replace(f"{{{{{match}}}}}", resolved)
    return template


def _get(context: dict[str, str], key: str, default: str = "") -> str:
    if key in context:
        return context[key]
    if "." not in key:
        return default
    parts = key.split(".", 1)
    raw = context.get(parts[0], "")
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
        result = parsed
        for part in parts[1].split("."):
            if isinstance(result, dict):
                result = result.get(part)
            else:
                return default
        if result is None:
            return default
        if isinstance(result, bool):
            return "true" if result else "false"
        return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    except Exception:
        return default


def _set(context: dict[str, str], key: str, value: str) -> None:
    context[key] = value


def _parse_json_list(raw: str) -> list:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts


def _to_str_set(items: list) -> set[str]:
    return {json.dumps(i, ensure_ascii=False) if isinstance(i, (dict, list)) else str(i) for i in items}


def _evaluate_condition(condition: str, context: dict[str, str]) -> bool:
    match = re.match(r"^([\w.]+)\s*(==|!=|>|<|>=|<=)\s*(.*)$", condition.strip())
    if not match:
        logger.warning("router_match: unparseable condition %r", condition)
        return False

    lhs_key, op, rhs_raw = match.group(1), match.group(2), match.group(3).strip()
    lhs_raw = context.get(lhs_key.replace(".", "_"), context.get(lhs_key, ""))
    rhs = rhs_raw.strip("'\"")
    if rhs_raw.lower() in ("null", "") or rhs == "":
        rhs = None

    if op == "==" and rhs is None:
        return lhs_raw == "" or lhs_raw is None
    if op == "!=" and rhs is None:
        return lhs_raw not in ("", None)

    try:
        lhs = float(lhs_raw)
        rhs_cmp = float(rhs)
        return {"==": lhs == rhs_cmp, "!=": lhs != rhs_cmp,
                ">": lhs > rhs_cmp, "<": lhs < rhs_cmp,
                ">=": lhs >= rhs_cmp, "<=": lhs <= rhs_cmp}[op]
    except (ValueError, TypeError):
        return {"==": lhs_raw == rhs, "!=": lhs_raw != rhs}.get(op, False)


async def _handle_router_match(
    step: dict,
    context: dict[str, str],
    **_,
) -> str:
    rules: list[dict] = step.get("rules", [])
    default: str = step.get("default", "default")
    for rule in rules:
        condition: str = rule.get("if", "")
        route: str = rule.get("then", default)
        if _evaluate_condition(condition, context):
            logger.info("router_match: condition %r → route %r", condition, route)
            return route
    logger.info("router_match: no condition matched → default %r", default)
    return default


async def _handle_router_llm(
    step: dict,
    context: dict[str, str],
    agent_system: str,
    pool: asyncpg.Pool,
    name: str,
    **_,
) -> str:
    _ROUTER_SYSTEM = """Du entscheidest welcher Ausführungspfad gilt.
Antworte NUR mit einem einzigen Wort. Befolge die Entscheidungslogik im Prompt exakt.
Wenn keine Bedingung zutrifft: antworte mit 'normal'."""

    prompt = _resolve_template(step.get("prompt", ""), context)
    result = await brain.chat(
        system=_ROUTER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        capability=CAPABILITY_SIMPLE_TASKS,
        caller=f"agent_router:{name}",
        pool=pool,
    )
    return result.strip().lower()


_LLM_CAPABILITY_MAP: dict[str, str] = {
    "llm_extract": CAPABILITY_SIMPLE_TASKS,
    "llm_decide": CAPABILITY_REASONING,
    "llm_summarize": CAPABILITY_CHAT,
    "llm_analyze": CAPABILITY_DEEP_REASONING,
}

_EXTRACT_SYSTEM = """Du extrahierst strukturierte Daten. Antworte ausschließlich mit rohem JSON.
Der erste Charakter muss { sein, der letzte }. Kein anderer Text."""

_DECIDE_SYSTEM = """Du bewertest und entscheidest. Antworte ausschließlich mit rohem JSON.
Der erste Charakter muss { sein, der letzte }. Kein anderer Text."""

_SUMMARIZE_SYSTEM = """Du fasst zusammen. Schreibe kompaktes Markdown, maximal 300 Wörter, nur Fakten.
Das Ergebnis wird von einem anderen Modell weiterverarbeitet."""

_ANALYZE_SYSTEM = """Du erstellst tiefgehende Analysen aus heterogenen Quellen. Antworte ausschließlich mit rohem JSON.
Der erste Charakter muss { sein, der letzte }. Kein anderer Text.
Wäge Informationen aus verschiedenen Quellen gegeneinander ab, erkenne Widersprüche, ziehe mehrstufige Schlussfolgerungen und bilde ein begründetes Urteil."""

_LLM_SYSTEM_MAP: dict[str, str] = {
    "llm_extract": _EXTRACT_SYSTEM,
    "llm_decide": _DECIDE_SYSTEM,
    "llm_summarize": _SUMMARIZE_SYSTEM,
    "llm_analyze": _ANALYZE_SYSTEM,
}

_OUTPUT_SYSTEM = """Du strukturierst das Ergebnis eines Agenten-Laufs in ein JSON-Objekt.
Antworte ausschließlich mit rohem JSON. Der erste Charakter muss { sein, der letzte }.
Felder:
- "report": Zusammenfassung in maximal 3 kurzen Sätzen. "KEINE_AENDERUNG" wenn nichts Relevantes passiert ist.
- "notify_user": true wenn der User benachrichtigt werden soll, false sonst.
- "state_updates": immer leeres Dict {} — State wird per state_write Steps gepflegt.
- "tool_calls": Liste der Tool-Aufrufe.
Verfügbare Tools:
- {"tool": "notify_user", "message": "..."} — Nachricht an den User.
- {"tool": "trigger_agent", "target_agent_name": "...", "payload": {...}, "delay_minutes": 0}
Erzeuge NUR Tool-Calls die im Prompt explizit angewiesen werden."""


async def _handle_llm_step(
    step: dict,
    step_type: str,
    context: dict[str, str],
    agent_system: str,
    pool: asyncpg.Pool,
    name: str,
    **_,
) -> str:
    is_output = step.get("is_output", False)
    system = _OUTPUT_SYSTEM if is_output else _LLM_SYSTEM_MAP[step_type]
    capability = CAPABILITY_SIMPLE_TASKS if is_output else _LLM_CAPABILITY_MAP[step_type]
    prompt = _resolve_template(step.get("prompt", ""), context)
    result = await brain.chat(
        system=system if is_output else f"{agent_system}\n\n{system}",
        messages=[{"role": "user", "content": prompt}],
        capability=capability,
        caller=f"agent_{step_type}:{name}",
        pool=pool,
    )
    logger.info("agent %s step %r (%s): %d chars → %s", name, step["id"], step_type, len(result), result[:120])
    return result


async def _handle_web_search(
    step: dict,
    context: dict[str, str],
    agent_system: str,
    pool: asyncpg.Pool,
    name: str,
    **_,
) -> str:
    from bot import search as _search
    from bot.models import select_model_for_provider

    query_template = step.get("query_template", "")
    query = _resolve_template(query_template, context)
    time_range: str | None = step.get("time_range")
    categories: str | None = step.get("categories")
    prompt = _resolve_template(step.get("prompt", "Fasse die Suchergebnisse zusammen."), context)

    if not await _search.is_available():
        force_model = select_model_for_provider(CAPABILITY_CHAT, "anthropic")
        result = await brain.chat(
            system=agent_system,
            messages=[{"role": "user", "content": prompt}],
            use_web_search=True,
            capability=CAPABILITY_CHAT,
            force_model=force_model,
            caller=f"agent_web_search:{name}",
            pool=pool,
        )
        return result

    search_result = await _search.search(query, time_range=time_range, categories=categories)
    if not search_result:
        logger.info("agent %s web_search %r: no results", name, query)
        return ""

    augmented_prompt = f"{prompt}\n\n[Suchergebnisse für '{query}']\n\n{search_result}"
    result = await brain.chat(
        system=agent_system,
        messages=[{"role": "user", "content": augmented_prompt}],
        capability=CAPABILITY_CHAT,
        caller=f"agent_web_search:{name}",
        pool=pool,
    )
    logger.info("agent %s web_search %r: %d chars", name, query, len(result))
    return result


async def _handle_finance(
    step: dict,
    context: dict[str, str],
    **_,
) -> str:
    from bot import finance as _finance

    ticker_key = step.get("ticker_key", "selected_ticker")
    ticker = context.get(ticker_key, "").strip()
    if not ticker:
        logger.warning("finance step: no ticker in context key %r", ticker_key)
        return ""

    result = await _finance.get_quote_summary(ticker)
    logger.info("finance step: fetched %s (%d chars)", ticker, len(result))
    return result


async def _handle_finance_search(
    step: dict,
    context: dict[str, str],
    **_,
) -> str:
    from bot import finance as _finance

    query_key = step.get("query_key", "selected_isin")
    query = _get(context, query_key).strip()
    if not query:
        logger.warning("finance_search step: no query in context key %r", query_key)
        return "{}"

    result = await _finance.search_ticker(query)
    if result is None:
        logger.info("finance_search: no result for %r", query)
        return "{}"

    logger.info("finance_search: found %s (%s) for %r", result.get("symbol"), result.get("longname"), query)
    return json.dumps(result, ensure_ascii=False)


async def _handle_state_read(
    step: dict,
    context: dict[str, str],
    state: dict[str, str],
    **_,
) -> str:
    key: str = step["key"]
    default: str = step.get("default", "")
    return state.get(key, default)


async def _handle_state_write(
    step: dict,
    context: dict[str, str],
    state: dict[str, str],
    **_,
) -> str:
    key: str = step["key"]
    source_key: str = step["source_key"]
    value = _get(context, source_key)
    state[key] = value
    logger.info("state_write: %r = %d chars", key, len(value))
    return value


async def _handle_state_init(
    step: dict,
    context: dict[str, str],
    state: dict[str, str],
    pool: asyncpg.Pool,
    agent_id: int,
    **_,
) -> str:
    key: str = step["key"]
    source_key: str = step["source_key"]
    if key in state and state[key] not in ("", "[]", "{}", "null"):
        logger.info("state_init: key %r already exists, skipping", key)
        return state[key]
    value = _get(context, source_key)
    if not value or value in ("[]", "{}", "null", ""):
        logger.info("state_init: source_key %r is empty, skipping write", source_key)
        return state.get(key, step.get("default", "[]"))
    state[key] = value
    logger.info("state_init: initialized %r (%d chars)", key, len(value))
    return value


async def _handle_data_read(
    step: dict,
    context: dict[str, str],
    pool: asyncpg.Pool,
    agent_id: int,
    **_,
) -> str:
    namespace: str = step["namespace"]
    key_template: str = step.get("key_template", "")
    key = _resolve_template(key_template, context)
    value = await memory.read_agent_data(pool, agent_id, namespace, key)
    return value or step.get("default", "")


async def _handle_data_write(
    step: dict,
    context: dict[str, str],
    pool: asyncpg.Pool,
    agent_id: int,
    **_,
) -> str:
    namespace: str = step["namespace"]
    key_template: str = step.get("key_template", "")
    key = _resolve_template(key_template, context)
    source_key: str = step["source_key"]
    raw = _get(context, source_key)
    value = clean_llm_json(raw) if raw.strip().startswith("```") else raw
    await memory.write_agent_data(pool, agent_id, namespace, key, value)
    logger.info("data_write: %s/%s (%d chars)", namespace, key, len(value))
    return value


async def _handle_data_read_external(
    step: dict,
    context: dict[str, str],
    pool: asyncpg.Pool,
    **_,
) -> str:
    agent_name: str = step["agent_name"]
    namespace: str = step["namespace"]
    key_template: str = step.get("key_template", "")
    key = _resolve_template(key_template, context)
    target_id = await memory.get_agent_id_by_name(pool, agent_name)
    if target_id is None:
        logger.warning("data_read_external: agent %r not found", agent_name)
        return step.get("default", "")
    value = await memory.read_agent_data(pool, target_id, namespace, key)
    return value or step.get("default", "")


async def _handle_data_write_external(
    step: dict,
    context: dict[str, str],
    pool: asyncpg.Pool,
    **_,
) -> str:
    agent_name: str = step["agent_name"]
    namespace: str = step["namespace"]
    key_template: str = step.get("key_template", "")
    key = _resolve_template(key_template, context)
    source_key: str = step["source_key"]
    value = _get(context, source_key)
    target_id = await memory.get_agent_id_by_name(pool, agent_name)
    if target_id is None:
        logger.warning("data_write_external: agent %r not found", agent_name)
        return ""
    await memory.write_agent_data(pool, target_id, namespace, key, value)
    logger.info("data_write_external: %s → %s/%s (%d chars)", agent_name, namespace, key, len(value))
    return value


async def _handle_state_read_external(
    step: dict,
    context: dict[str, str],
    pool: asyncpg.Pool,
    **_,
) -> str:
    agent_name: str = step["agent_name"]
    key: str = step["key"]
    state = await memory.get_agent_state_by_name(pool, agent_name)
    if state is None:
        logger.warning("state_read_external: agent %r not found", agent_name)
        return step.get("default", "")
    return state.get(key, step.get("default", ""))


async def _handle_state_write_external(
    step: dict,
    context: dict[str, str],
    pool: asyncpg.Pool,
    **_,
) -> str:
    agent_name: str = step["agent_name"]
    key: str = step["key"]
    source_key: str = step["source_key"]
    value = _get(context, source_key)
    target_id = await memory.get_agent_id_by_name(pool, agent_name)
    if target_id is None:
        logger.warning("state_write_external: agent %r not found", agent_name)
        return ""
    await memory.set_agent_state(pool, target_id, {key: value})
    logger.info("state_write_external: %s[%r] = %d chars", agent_name, key, len(value))
    return value


def _matches_item_filter(item: object, f: dict) -> bool:
    field: str = f.get("field", "")
    operator: str = f.get("operator", "equals")
    value: str = str(f.get("value", ""))

    if isinstance(item, dict):
        cell = str(item.get(field, "")).strip()
    else:
        cell = str(item).strip()

    if operator == "equals": return cell == value
    if operator == "not_equals": return cell != value
    if operator == "contains": return value.lower() in cell.lower()
    if operator == "not_contains": return value.lower() not in cell.lower()
    if operator == "not_empty": return cell != ""
    if operator == "empty": return cell == ""
    if operator == "starts_with": return cell.lower().startswith(value.lower())
    if operator == "ends_with": return cell.lower().endswith(value.lower())
    if operator == "gt":
        try: return float(cell) > float(value)
        except (ValueError, TypeError): return False
    if operator == "lt":
        try: return float(cell) < float(value)
        except (ValueError, TypeError): return False
    logger.warning("transform filter: unknown operator %r", operator)
    return True


def _transform_array_push(step: dict, context: dict[str, str]) -> str:
    value_key: str = step["value_key"]
    group_key: str = step["group_key"]
    target_key: str = step["target_key"]
    max_items: int = int(step.get("max_items", 500))
    value = _get(context, value_key)
    group = _get(context, group_key)
    if not value or not group:
        logger.warning("transform array_push: value_key %r or group_key %r is empty", value_key, group_key)
        return context.get(target_key, "{}")
    raw = context.get(target_key, "{}")
    try:
        result: dict[str, list] = json.loads(raw)
        if not isinstance(result, dict):
            result = {}
    except Exception:
        result = {}
    result.setdefault(group, [])
    try:
        entry: float | str = float(value)
    except (ValueError, TypeError):
        entry = value
    result[group].append(entry)
    if len(result[group]) > max_items:
        result[group] = result[group][-max_items:]
    logger.info("transform array_push: %s[%r] now %d entries", target_key, group, len(result[group]))
    return json.dumps(result, ensure_ascii=False)


def _transform_map_field(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    field: str = step["field"]
    output_format: str = step.get("output_format", "json_array")
    raw = _get(context, source_key)
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError
    except Exception:
        logger.warning("transform map_field: source_key %r is not a valid JSON array", source_key)
        return "[]"
    values = [
        str(item[field]).strip()
        for item in data
        if isinstance(item, dict) and field in item and item[field] is not None and str(item[field]).strip()
    ]
    if output_format == "comma_list":
        return ",".join(values)
    return json.dumps(values, ensure_ascii=False)


def _transform_filter(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    filters: list[dict] = step.get("filters", [])
    raw = _get(context, source_key)
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError
    except Exception:
        logger.warning("transform filter: source_key %r is not a valid JSON array", source_key)
        return "[]"
    result = [item for item in data if all(_matches_item_filter(item, f) for f in filters)]
    logger.info("transform filter: %d → %d items", len(data), len(result))
    return json.dumps(result, ensure_ascii=False)


def _transform_first(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    raw = _get(context, source_key)
    items = _parse_json_list(raw)
    if not items:
        return step.get("default", "")
    first = items[0]
    return json.dumps(first, ensure_ascii=False) if isinstance(first, (dict, list)) else str(first)


def _transform_slice(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    start: int = int(step.get("start", 0))
    end: int | None = step.get("end")
    raw = _get(context, source_key)
    items = _parse_json_list(raw)
    sliced = items[start:end] if end is not None else items[start:]
    logger.info("transform slice: [%s:%s] → %d items", start, end, len(sliced))
    return json.dumps(sliced, ensure_ascii=False)


def _transform_diff(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    subtract_key: str = step["subtract_key"]
    output_format: str = step.get("output_format", "json_array")
    source_items = _parse_json_list(_get(context, source_key))
    subtract_items = _parse_json_list(_get(context, subtract_key))
    subtract_set = _to_str_set(subtract_items)
    result = [
        item for item in source_items
        if (json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)) not in subtract_set
    ]
    logger.info("transform diff: %d - %d = %d items", len(source_items), len(subtract_items), len(result))
    if output_format == "comma_list":
        return ",".join(str(i) for i in result)
    return json.dumps(result, ensure_ascii=False)


def _transform_intersect(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    other_key: str = step["other_key"]
    output_format: str = step.get("output_format", "json_array")
    source_items = _parse_json_list(_get(context, source_key))
    other_set = _to_str_set(_parse_json_list(_get(context, other_key)))
    result = [
        item for item in source_items
        if (json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)) in other_set
    ]
    logger.info("transform intersect: %d items", len(result))
    if output_format == "comma_list":
        return ",".join(str(i) for i in result)
    return json.dumps(result, ensure_ascii=False)


def _transform_union(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    other_key: str = step["other_key"]
    output_format: str = step.get("output_format", "json_array")
    source_items = _parse_json_list(_get(context, source_key))
    other_items = _parse_json_list(_get(context, other_key))
    seen: set[str] = set()
    result: list = []
    for item in source_items + other_items:
        key = json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    logger.info("transform union: %d + %d = %d unique items", len(source_items), len(other_items), len(result))
    if output_format == "comma_list":
        return ",".join(str(i) for i in result)
    return json.dumps(result, ensure_ascii=False)


def _transform_list_append(step: dict, context: dict[str, str]) -> str:
    value_key: str = step["value_key"]
    target_key: str = step["target_key"]
    max_items: int = int(step.get("max_items", 10000))
    output_format: str = step.get("output_format", "json_array")
    value = _get(context, value_key)
    if not value:
        logger.warning("transform list_append: value_key %r is empty", value_key)
        return context.get(target_key, "[]")
    items = _parse_json_list(context.get(target_key, "[]"))
    if value not in items:
        items.append(value)
    if len(items) > max_items:
        items = items[-max_items:]
    logger.info("transform list_append: %s now %d entries", target_key, len(items))
    if output_format == "comma_list":
        return ",".join(str(i) for i in items)
    return json.dumps(items, ensure_ascii=False)


def _transform_count(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    items = _parse_json_list(_get(context, source_key))
    return str(len(items))


def _transform_group_by(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    group_field: str = step["group_field"]
    value_field: str | None = step.get("value_field")
    max_items: int = int(step.get("max_items", 500))
    raw = _get(context, source_key)
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError
    except Exception:
        logger.warning("transform group_by: source_key %r is not a valid JSON array", source_key)
        return "{}"
    existing_raw = context.get(step.get("target_key", ""), "{}")
    try:
        result: dict[str, list] = json.loads(existing_raw)
        if not isinstance(result, dict):
            result = {}
    except Exception:
        result = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        group = str(item.get(group_field, "")).strip()
        if not group:
            continue
        value = item.get(value_field) if value_field else item
        if value is None:
            continue
        result.setdefault(group, [])
        entry = value
        if value_field:
            try:
                entry = float(value)
            except (ValueError, TypeError):
                entry = value
        result[group].append(entry)
        if len(result[group]) > max_items:
            result[group] = result[group][-max_items:]
    logger.info("transform group_by: %d groups", len(result))
    return json.dumps(result, ensure_ascii=False)


def _transform_flatten(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    output_format: str = step.get("output_format", "json_array")
    raw = _get(context, source_key)
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError
    except Exception:
        logger.warning("transform flatten: source_key %r is not a valid JSON array", source_key)
        return "[]"
    result: list = []
    for item in data:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    logger.info("transform flatten: → %d items", len(result))
    if output_format == "comma_list":
        return ",".join(str(i) for i in result)
    return json.dumps(result, ensure_ascii=False)


def _transform_sort(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    field: str | None = step.get("field")
    reverse: bool = bool(step.get("reverse", False))
    raw = _get(context, source_key)
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError
    except Exception:
        logger.warning("transform sort: source_key %r is not a valid JSON array", source_key)
        return "[]"
    try:
        if field:
            data.sort(key=lambda x: x.get(field, "") if isinstance(x, dict) else "", reverse=reverse)
        else:
            data.sort(reverse=reverse)
    except Exception as e:
        logger.warning("transform sort: failed: %s", e)
    return json.dumps(data, ensure_ascii=False)


def _transform_statistics(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    model_key: str | None = step.get("model_key")
    functions: list[str] = step.get("functions", ["count", "mean", "median", "q1", "q3", "iqr", "lower_bound", "upper_bound"])
    multiplier: float = float(step.get("multiplier", 1.5))
    raw = _get(context, source_key)
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        logger.warning("transform statistics: source %r is not a valid dict", source_key)
        return "{}"

    def _calc(prices: list) -> dict:
        floats = sorted([float(p) for p in prices if p is not None])
        count = len(floats)
        result: dict = {"count": count}
        if "mean" in functions: result["mean"] = round(statistics.mean(floats), 2) if count else None
        if "median" in functions: result["median"] = round(statistics.median(floats), 2) if count else None
        if "std_dev" in functions: result["std_dev"] = round(statistics.stdev(floats), 2) if count >= 2 else None
        if "min" in functions: result["min"] = round(min(floats), 2) if count else None
        if "max" in functions: result["max"] = round(max(floats), 2) if count else None
        if any(f in functions for f in ["q1", "q3", "iqr", "lower_bound", "upper_bound"]):
            if count >= 4:
                q1 = statistics.quantiles(floats, n=4)[0]
                q3 = statistics.quantiles(floats, n=4)[2]
                iqr = q3 - q1
                if "q1" in functions: result["q1"] = round(q1, 2)
                if "q3" in functions: result["q3"] = round(q3, 2)
                if "iqr" in functions: result["iqr"] = round(iqr, 2)
                if "lower_bound" in functions: result["lower_bound"] = round(q1 - multiplier * iqr, 2)
                if "upper_bound" in functions: result["upper_bound"] = round(q3 + multiplier * iqr, 2)
            else:
                for f in ["q1", "q3", "iqr", "lower_bound", "upper_bound"]:
                    if f in functions: result[f] = None
        return result

    if model_key:
        model = _get(context, model_key)
        if not model:
            logger.warning("transform statistics: model_key %r is empty", model_key)
            return "{}"
        prices = data.get(model, [])
        result = _calc(prices)
        logger.info("transform statistics: model=%r count=%d", model, result["count"])
        return json.dumps(result, ensure_ascii=False)

    result: dict[str, dict] = {}
    for model, prices in data.items():
        result[model] = _calc(prices)
    return json.dumps(result, ensure_ascii=False)


def _transform_json_path(step: dict, context: dict[str, str]) -> str:
    source_key: str = step["source_key"]
    path: str = step["path"]
    raw = context.get(source_key, "")
    try:
        data = json.loads(raw)
        for part in path.split("."):
            if part.isdigit():
                data = data[int(part)]
            else:
                data = data[part]
        return str(data)
    except Exception:
        logger.warning("transform json_path: could not extract %r from %r", path, source_key)
        return step.get("default", "")


def _transform_xml_extract(step: dict, context: dict[str, str]) -> str:
    import xml.etree.ElementTree as ET

    source_key: str = step["source_key"]
    xpath: str = step["xpath"]
    attribute: str | None = step.get("attribute")
    raw = _get(context, source_key)
    if not raw:
        logger.warning("transform xml_extract: source_key %r is empty", source_key)
        return step.get("default", "")
    try:
        namespaces: dict[str, str] = {}
        for _, (prefix, uri) in ET.iterparse(io.StringIO(raw), events=["start-ns"]):
            if prefix:
                namespaces[prefix] = uri
            else:
                namespaces["ns"] = uri
        root = ET.fromstring(raw)
        elements = root.findall(xpath, namespaces) if namespaces else root.findall(xpath)
        if not elements:
            logger.warning("transform xml_extract: xpath %r found no elements", xpath)
            return step.get("default", "")
        el = elements[0]
        result = el.get(attribute, step.get("default", "")) if attribute else (el.text or step.get("default", ""))
        return str(result).strip()
    except Exception as e:
        logger.warning("transform xml_extract: failed for xpath %r: %s", xpath, e)
        return step.get("default", "")


def _transform_regex_extract(step: dict, context: dict[str, str]) -> str:
    import re as _re
    source_key: str = step["source_key"]
    pattern: str = step["pattern"]
    group: int = int(step.get("group", 1))
    raw = context.get(source_key, "")
    try:
        match = _re.search(pattern, raw)
        if not match:
            logger.warning("transform regex_extract: pattern %r found no match", pattern)
            return step.get("default", "")
        return match.group(group).strip()
    except Exception as e:
        logger.warning("transform regex_extract: failed for pattern %r: %s", pattern, e)
        return step.get("default", "")


def _transform_arithmetic(step: dict, context: dict[str, str]) -> str:
    import re as _re
    expression: str = step.get("expression", "")
    default: str = step.get("default", "")
    tokens = _re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*", expression)
    resolved = expression
    for token in sorted(tokens, key=len, reverse=True):
        val = _get(context, token)
        if val == "":
            logger.warning("transform arithmetic: variable %r not found in context", token)
            return default
        try:
            float(val)
        except ValueError:
            logger.warning("transform arithmetic: variable %r is not numeric: %r", token, val)
            return default
        resolved = resolved.replace(token, val)
    allowed = set("0123456789+-*/()., \t")
    if not all(c in allowed for c in resolved):
        logger.warning("transform arithmetic: expression contains invalid characters: %r", resolved)
        return default
    try:
        result = eval(resolved, {"__builtins__": {}}, {})  # noqa: S307
        rounded = step.get("round")
        if rounded is not None:
            result = round(float(result), int(rounded))
        return str(result)
    except Exception as e:
        logger.warning("transform arithmetic: eval failed for %r: %s", resolved, e)
        return default


def _transform_compare(step: dict, context: dict[str, str]) -> str:
    left_key: str = step.get("left_key", "")
    right_key: str = step.get("right_key", "")
    operator: str = step.get("operator", "<=")
    output_true: str = step.get("output_true", "true")
    output_false: str = step.get("output_false", "false")
    left_raw = _get(context, left_key)
    right_raw = _get(context, right_key)
    try:
        left = float(left_raw)
        right = float(right_raw)
    except (ValueError, TypeError):
        logger.warning("transform compare: could not parse %r=%r or %r=%r as float", left_key, left_raw, right_key, right_raw)
        return output_false
    result = (
        left < right if operator == "<" else
        left <= right if operator == "<=" else
        left > right if operator == ">" else
        left >= right if operator == ">=" else
        left == right if operator == "==" else
        left != right if operator == "!=" else
        False
    )
    logger.info("transform compare: %s %s %s → %s", left, operator, right, result)
    return output_true if result else output_false


_TRANSFORM_OPERATIONS: dict[str, Callable[[dict, dict[str, str]], str]] = {
    "array_push": _transform_array_push,
    "map_field": _transform_map_field,
    "filter": _transform_filter,
    "first": _transform_first,
    "slice": _transform_slice,
    "diff": _transform_diff,
    "intersect": _transform_intersect,
    "union": _transform_union,
    "list_append": _transform_list_append,
    "count": _transform_count,
    "group_by": _transform_group_by,
    "flatten": _transform_flatten,
    "sort": _transform_sort,
    "statistics": _transform_statistics,
    "json_path": _transform_json_path,
    "json_extract": _transform_json_path,
    "xml_extract": _transform_xml_extract,
    "regex_extract": _transform_regex_extract,
    "arithmetic": _transform_arithmetic,
    "compare": _transform_compare,
}


async def _handle_transform(
    step: dict,
    context: dict[str, str],
    **_,
) -> str:
    operation: str = step.get("operation", "")
    handler = _TRANSFORM_OPERATIONS.get(operation)
    if handler is None:
        logger.warning("transform: unknown operation %r", operation)
        return ""
    return handler(step, context)


async def _handle_http_fetch(
    step: dict,
    context: dict[str, str],
    **_,
) -> str:
    url_template: str = step.get("url_template") or step.get("url", "")
    url = _resolve_template(url_template, context)
    if not url:
        logger.warning("http_fetch: no url configured")
        return step.get("default", "")

    method: str = step.get("method", "GET").upper()
    headers: dict[str, str] = step.get("headers", {})
    timeout: float = float(step.get("timeout", 15.0))
    body: str | None = step.get("body")
    resolved_headers = {k: _resolve_template(v, context) for k, v in headers.items()}

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            if method == "GET":
                resp = await client.get(url, headers=resolved_headers)
            elif method == "POST":
                resp = await client.post(url, headers=resolved_headers, content=body)
            else:
                logger.warning("http_fetch: unsupported method %r", method)
                return step.get("default", "")
            resp.raise_for_status()
            result = resp.text
            logger.info("http_fetch: %s %s → %d (%d chars)", method, url[:80], resp.status_code, len(result))
            return result
    except httpx.HTTPStatusError as e:
        logger.warning("http_fetch: HTTP error %d for %s", e.response.status_code, url[:80])
        return step.get("default", "")
    except Exception as e:
        logger.warning("http_fetch: failed for %s: %s", url[:80], e)
        return step.get("default", "")


async def _handle_xlsx_fetch(
    step: dict,
    context: dict[str, str],
    **_,
) -> str:
    url_template: str = step.get("url_template") or step.get("url", "")
    url = _resolve_template(url_template, context)
    if not url:
        logger.warning("xlsx_fetch: no url configured")
        return step.get("default", "[]")

    sheet_ref = step.get("sheet", 0)
    columns: list[str] = step.get("columns", [])
    row_filter: dict | None = step.get("filter")
    row_filters: list[dict] = step.get("filters", [])
    timeout: float = float(step.get("timeout", 30.0))

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "BobAgent/1.0"})
            resp.raise_for_status()
            raw_bytes = resp.content
            logger.info("xlsx_fetch: GET %s → %d (%d bytes)", url[:80], resp.status_code, len(raw_bytes))

        wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), read_only=True, data_only=True)
        ws = wb.worksheets[sheet_ref] if isinstance(sheet_ref, int) else wb[sheet_ref]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return "[]"

        header = [str(c).strip() if c is not None else f"col_{i}" for i, c in enumerate(rows[0])]

        if columns:
            col_indices = [header.index(col) for col in columns if col in header]
            use_cols = [col for col in columns if col in header]
        else:
            col_indices = list(range(len(header)))
            use_cols = header

        def _cell_str(val: object) -> str:
            return str(val).strip() if val is not None else ""

        def _matches_xlsx_filter(row: tuple, f: dict) -> bool:
            col_name: str = f.get("column", "")
            operator: str = f.get("operator", "equals")
            value: str = str(f.get("value", ""))
            if col_name not in header:
                return True
            col_idx = header.index(col_name)
            cell = _cell_str(row[col_idx] if col_idx < len(row) else None)
            if operator == "equals": return cell == value
            if operator == "not_equals": return cell != value
            if operator == "contains": return value.lower() in cell.lower()
            if operator == "not_contains": return value.lower() not in cell.lower()
            if operator == "not_empty": return cell != ""
            if operator == "empty": return cell == ""
            if operator == "starts_with": return cell.lower().startswith(value.lower())
            if operator == "ends_with": return cell.lower().endswith(value.lower())
            return True

        all_filters: list[dict] = list(row_filters)
        if row_filter and not row_filters:
            all_filters = [{"column": row_filter["column"], "operator": "equals", "value": str(row_filter["value"])}]

        result: list[dict] = []
        for row in rows[1:]:
            if all_filters and not all(_matches_xlsx_filter(row, f) for f in all_filters):
                continue
            record: dict[str, str] = {}
            for i, idx in enumerate(col_indices):
                val = row[idx] if idx < len(row) else None
                record[use_cols[i]] = _cell_str(val)
            result.append(record)

        wb.close()
        logger.info("xlsx_fetch: extracted %d rows from %s", len(result), url[:60])
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.warning("xlsx_fetch: failed for %s: %s", url[:80], e)
        return step.get("default", "[]")


async def _handle_trigger_agent(
    step: dict,
    context: dict[str, str],
    pool: asyncpg.Pool,
    agent_id: int,
    **_,
) -> str:
    target_name: str = _resolve_template(step.get("target_agent_name", ""), context)
    delay: int = int(step.get("delay_minutes", 0))
    payload_template: dict = step.get("payload", {})
    payload = {k: _resolve_template(str(v), context) for k, v in payload_template.items()}
    if target_name:
        await memory.enqueue_agent_trigger(pool, agent_id, target_name, payload, delay)
        logger.info("trigger_agent: queued %r (delay: %dm)", target_name, delay)
    return ""


async def _handle_notify_user(
    step: dict,
    context: dict[str, str],
    bot: telegram.Bot,
    target_chat_id: int,
    **_,
) -> str:
    condition_key: str | None = step.get("condition_key")
    if condition_key:
        condition_val = _get(context, condition_key)
        try:
            parsed = json.loads(condition_val)
            if not parsed:
                logger.info("notify_user: condition_key %r is empty/falsy, skipping", condition_key)
                return ""
        except Exception:
            if not condition_val:
                logger.info("notify_user: condition_key %r is empty, skipping", condition_key)
                return ""

    message_template: str = step.get("message_template", "")
    source_key: str | None = step.get("source_key")
    message = _get(context, source_key) if source_key else _resolve_template(message_template, context)

    if message:
        await bot.send_message(chat_id=target_chat_id, text=message)
        logger.info("notify_user: sent %d chars", len(message))
    return ""


StepHandler = Callable[..., Awaitable[str]]

_STEP_HANDLERS: dict[str, StepHandler] = {
    "router_match": _handle_router_match,
    "router_llm": _handle_router_llm,
    "llm_extract": _handle_llm_step,
    "llm_decide": _handle_llm_step,
    "llm_summarize": _handle_llm_step,
    "llm_analyze": _handle_llm_step,
    "web_search": _handle_web_search,
    "finance": _handle_finance,
    "finance_search": _handle_finance_search,
    "http_fetch": _handle_http_fetch,
    "xlsx_fetch": _handle_xlsx_fetch,
    "state_read": _handle_state_read,
    "state_write": _handle_state_write,
    "state_init": _handle_state_init,
    "state_read_external": _handle_state_read_external,
    "state_write_external": _handle_state_write_external,
    "data_read": _handle_data_read,
    "data_write": _handle_data_write,
    "data_read_external": _handle_data_read_external,
    "data_write_external": _handle_data_write_external,
    "transform": _handle_transform,
    "trigger_agent": _handle_trigger_agent,
    "notify_user": _handle_notify_user,
}


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
            if tool == "notify_user":
                msg = call.get("message", "")
                if msg:
                    sent = await bot.send_message(chat_id=target_chat_id, text=msg)
                    try:
                        await memory.save_agent_notification(
                            pool,
                            message_id=sent.message_id,
                            chat_id=target_chat_id,
                            agent_id=agent_id,
                            notification_type="tool_notify",
                            payload_summary={"summary": msg[:200]},
                        )
                    except Exception as notif_err:
                        logger.warning("tool notify_user: failed to save notification: %s", notif_err)
                    logger.info("tool notify_user: sent")
            elif tool == "trigger_agent":
                target_name: str = call.get("target_agent_name", "")
                payload: dict = call.get("payload", {})
                delay: int = int(call.get("delay_minutes", 0))
                if target_name:
                    await memory.enqueue_agent_trigger(pool, agent_id, target_name, payload, delay)
                    logger.info("tool trigger_agent: queued %r", target_name)
            else:
                logger.warning("unknown tool_call: %r", tool)
        except Exception as e:
            logger.error("tool_call %r failed: %s", tool, e)


def _route_allows(step: dict, active_route: str | None) -> bool:
    only_if_route = step.get("only_if_route")
    if only_if_route is None:
        return True
    if active_route is None:
        return False
    allowed = [only_if_route] if isinstance(only_if_route, str) else only_if_route
    return active_route in allowed


def _key_condition_allows(step: dict, context: dict[str, str]) -> bool:
    only_if_key = step.get("only_if_key")
    if only_if_key is None:
        return True
    key: str = only_if_key.get("key", "")
    expected: str = str(only_if_key.get("value", "true"))
    actual = _get(context, key)
    return actual == expected


async def _execute_pipeline(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    agent_id: int,
    name: str,
    steps: list[dict],
    state: dict[str, str],
    trigger_payload: dict[str, str],
    config_data: dict,
    target_chat_id: int = 0,
) -> tuple[str, bool]:
    context: dict[str, str] = {}
    context.update({k: v for k, v in state.items() if v})
    for k, v in trigger_payload.items():
        context[f"trigger_payload.{k}"] = str(v)
        context[k] = str(v)

    agent_system = f"Gesamtauftrag des Agenten:\n{config_data.get('instruction', '')}"
    active_route: str | None = None
    output_step_result: str = ""
    has_output_step = False

    shared_kwargs = dict(
        pool=pool,
        bot=bot,
        agent_id=agent_id,
        name=name,
        state=state,
        agent_system=agent_system,
        target_chat_id=target_chat_id,
    )

    for step in steps:
        step_id: str = step.get("id", "?")
        step_type: str = step.get("type", "")
        output_key: str = step.get("output_key", "")

        if not _route_allows(step, active_route):
            logger.info("agent %s step %r skipped (route=%s)", name, step_id, active_route)
            continue

        if not _key_condition_allows(step, context):
            only_if_key = step.get("only_if_key", {})
            logger.info("agent %s step %r skipped (key condition: %s != %s)",
                       name, step_id, only_if_key.get("key"), only_if_key.get("value"))
            continue

        handler = _STEP_HANDLERS.get(step_type)
        if handler is None:
            logger.warning("agent %s step %r: unknown type %r", name, step_id, step_type)
            continue

        try:
            result = await handler(step=step, step_type=step_type, context=context, **shared_kwargs)
        except Exception as e:
            logger.error("agent %s step %r failed: %s", name, step_id, e)
            raise

        if step_type in ("router_match", "router_llm"):
            active_route = result
            logger.info("agent %s route set to %r", name, active_route)

        if output_key:
            _set(context, output_key, result)

        if step.get("is_output"):
            output_step_result = result
            has_output_step = True

    return output_step_result, has_output_step


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
                continue
            state = await memory.get_agent_state_by_name(pool, agent_name)
            if state is None:
                continue
            combined = "\n".join(f"{k}: {v}" for k, v in state.items() if k != "last_run_summary")
            if combined:
                result[f"state:{agent_name}"] = combined
        else:
            namespace = read.get("namespace", "")
            key = read.get("key", "")
            target_agent_id = agent_id
            if agent_name:
                resolved = await memory.get_agent_id_by_name(pool, agent_name)
                if resolved is None:
                    continue
                target_agent_id = resolved
            if not key:
                rows = await memory.query_agent_data(pool, namespace, agent_id=target_agent_id)
                if rows:
                    combined = "\n".join(f"{r['key']}: {r['value']}" for r in rows)
                    label = read.get("as") or f"db:{agent_name or 'self'}:{namespace}"
                    result[label] = combined
            else:
                resolved_key = key
                for k, v in trigger_payload.items():
                    resolved_key = resolved_key.replace(f"{{{{{k}}}}}", v)
                value = await memory.read_agent_data(pool, target_agent_id, namespace, resolved_key)
                if value is not None:
                    label = read.get("as") or f"db:{agent_name or 'self'}:{namespace}:{resolved_key}"
                    result[label] = value
    return result


_FALLBACK_STRUCTURE_SYSTEM = """Du strukturierst das Ergebnis eines Agenten-Laufs in ein JSON-Objekt.
Antworte ausschließlich mit rohem JSON. Der erste Charakter muss { sein, der letzte }.
Felder:
- "report": Zusammenfassung in maximal 3 kurzen Sätzen. "KEINE_AENDERUNG" wenn nichts Relevantes passiert ist.
- "notify_user": true wenn der User benachrichtigt werden soll, false sonst.
- "tool_calls": Liste der Tool-Aufrufe. Verfügbare Tools:
  {"tool": "notify_user", "message": "..."}, {"tool": "trigger_agent", "target_agent_name": "...", "payload": {}}"""


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

    logger.info("executing agent %d (%s)", agent_id, name)

    try:
        state = await memory.get_agent_state(pool, agent_id)
        flat_payload: dict[str, str] = {k: str(v) for k, v in (trigger_payload or {}).items()}

        data_reads: list[dict] = config_data.get("data_reads", [])
        if data_reads:
            injected = await _load_data_reads(pool, agent_id, data_reads, flat_payload)
            flat_payload.update(injected)

        steps: list[dict] = (
            config_data.get("steps")
            or config_data.get("pipeline", []) + config_data.get("pipeline_after_template", [])
        )
        output_step_result, has_output_step = await _execute_pipeline(
            pool=pool, bot=bot, agent_id=agent_id, name=name, steps=steps,
            state=state, trigger_payload=flat_payload, config_data=config_data,
            target_chat_id=target_chat_id,
        )

        if has_output_step:
            try:
                parsed_output = json.loads(clean_llm_json(output_step_result))
            except Exception:
                logger.warning("agent %s output step JSON parse failed: %r", name, output_step_result[:300])
                parsed_output = {"report": output_step_result, "notify_user": True, "tool_calls": []}
        else:
            raw_structured = await brain.chat(
                system=_FALLBACK_STRUCTURE_SYSTEM,
                messages=[{"role": "user", "content": output_step_result or "Keine Ausgabe."}],
                capability=CAPABILITY_SIMPLE_TASKS,
                caller=f"agent_structure:{name}",
                pool=pool,
            )
            try:
                parsed_output = json.loads(clean_llm_json(raw_structured))
            except Exception:
                parsed_output = {"report": raw_structured, "notify_user": True, "tool_calls": []}

        report: str = parsed_output.get("report", "")
        notify_user: bool = parsed_output.get("notify_user", False)
        tool_calls: list[dict] = parsed_output.get("tool_calls", [])

        if report and report.strip() != "KEINE_AENDERUNG":
            state["last_run_summary"] = report
        else:
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
                    capability=CAPABILITY_SIMPLE_TASKS,
                    caller=f"agent_relay:{name}",
                    pool=pool,
                )
                sent = await bot.send_message(chat_id=target_chat_id, text=message_text)
                try:
                    await memory.save_agent_notification(
                        pool,
                        message_id=sent.message_id,
                        chat_id=target_chat_id,
                        agent_id=agent_id,
                        notification_type="report",
                        payload_summary={"summary": report[:200], "run_type": "scheduled"},
                    )
                except Exception as notif_err:
                    logger.warning("agent %s: failed to save notification: %s", name, notif_err)
                await memory.add_memory(pool, "agent", agent_id, report[:200])
                logger.info("agent %s notified user", name)
        else:
            logger.info("agent %s: no change or notify suppressed", name)

        if schedule:
            tz = await memory.get_user_timezone(pool, user_id)
            next_run = next_agent_run_after(schedule, tz)
            await memory.update_agent_run(pool, agent_id, next_run)
            logger.info("agent %d done. next run: %s", agent_id, next_run.isoformat())
        else:
            await pool.execute("UPDATE agents SET last_run_at = NOW() WHERE id = $1", agent_id)
            logger.info("agent %d done. trigger-only, no next run.", agent_id)

    except ProviderRateLimitError as e:
        logger.error("agent %d rate limit on %s", agent_id, e.provider)
    except Exception as e:
        logger.error("agent %d execution failed: %s", agent_id, e)
        try:
            if schedule:
                tz = await memory.get_user_timezone(pool, user_id)
                next_run = next_agent_run_after(schedule, tz)
                await memory.update_agent_run(pool, agent_id, next_run)
        except Exception as inner_e:
            logger.error("agent %d failed to update next_run: %s", agent_id, inner_e)
