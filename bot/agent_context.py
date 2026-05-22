from __future__ import annotations
import json
import logging
import asyncpg

from bot import brain, memory
from bot.models import CAPABILITY_SIMPLE_TASKS
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)

_RELEVANCE_SYSTEM = """Du entscheidest welche gespeicherten Daten eines Agents für eine Nutzeranfrage relevant sind.

Du bekommst:
- Die Nutzeranfrage
- Verfügbare State-Keys mit Vorschau
- Verfügbare Data-Namespaces mit Keys

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "state_keys": Liste der relevanten State-Keys (leer wenn keine relevant)
- "data_items": Liste von Objekten mit "namespace" und "key" (leer wenn keine relevant)
- "has_relevant_data": true wenn mindestens ein Item relevant ist

Sei konservativ — lade nur was direkt zur Frage passt. Große Listen oder akkumulierte Rohdaten sind nur relevant wenn explizit danach gefragt wird."""


async def load_shallow(
    pool: asyncpg.Pool,
    agent: dict,
) -> dict:
    state_keys = await memory.get_agent_state_keys_with_preview(pool, agent["id"])
    last_run_summary = next(
        (k["preview"] for k in state_keys if k["key"] == "last_run_summary"), None
    )
    return {
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "last_run_summary": last_run_summary,
        "current_rating": agent.get("current_rating"),
        "current_rating_note": agent.get("current_rating_note"),
        "schedule": agent.get("schedule"),
        "last_run_at": agent.get("last_run_at"),
        "state_keys": [k["key"] for k in state_keys if k["key"] != "last_run_summary"],
        "load_depth": "shallow",
    }


async def load_deep(
    pool: asyncpg.Pool,
    agent: dict,
    user_query: str,
) -> dict:
    state_keys_with_preview = await memory.get_agent_state_keys_with_preview(pool, agent["id"])
    namespaces = await memory.get_agent_data_namespaces(pool, agent["id"])

    data_keys: dict[str, list[str]] = {}
    for ns in namespaces:
        keys = await memory.get_agent_data_keys_in_namespace(pool, agent["id"], ns)
        data_keys[ns] = keys

    has_anything = bool(state_keys_with_preview) or bool(namespaces)
    if not has_anything:
        return {
            "agent_id": agent["id"],
            "agent_name": agent["name"],
            "load_depth": "deep",
            "has_data": False,
            "message": f"{agent['name']} speichert keine abfragbaren Daten.",
        }

    state_summary = "\n".join(
        f"- {k['key']}: {k['preview']} ({k['length']} Zeichen)"
        for k in state_keys_with_preview
    )
    data_summary = "\n".join(
        f"- {ns}: {', '.join(keys)}"
        for ns, keys in data_keys.items()
    )

    content = (
        f"Nutzeranfrage: {user_query}\n\n"
        f"Verfügbare State-Keys:\n{state_summary or '(keine)'}\n\n"
        f"Verfügbare Data-Namespaces/Keys:\n{data_summary or '(keine)'}"
    )

    try:
        raw = await brain.chat(
            system=_RELEVANCE_SYSTEM,
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="agent_context:relevance",
        )
        relevance = json.loads(clean_llm_json(raw))
    except Exception as e:
        logger.warning("agent_context: relevance check failed: %s", e)
        relevance = {"state_keys": [], "data_items": [], "has_relevant_data": False}

    if not relevance.get("has_relevant_data"):
        config_data = parse_agent_config(agent.get("config", {}))
        steps = config_data.get("steps", [])
        writes = [
            s.get("key") or s.get("key_template", "")
            for s in steps
            if s.get("type") in ("state_write", "data_write")
        ]
        return {
            "agent_id": agent["id"],
            "agent_name": agent["name"],
            "load_depth": "deep",
            "has_data": False,
            "message": (
                f"{agent['name']} speichert zur aktuellen Frage keine abfragbaren Daten."
                + (f" Der Agent schreibt: {', '.join(w for w in writes if w)}." if writes else "")
            ),
        }

    loaded_state: dict[str, str] = {}
    for key in relevance.get("state_keys", []):
        state = await memory.get_agent_state(pool, agent["id"])
        if key in state:
            loaded_state[key] = state[key]

    loaded_data: dict[str, str] = {}
    for item in relevance.get("data_items", []):
        ns = item.get("namespace", "")
        key = item.get("key", "")
        if ns and key:
            value = await memory.get_agent_data_by_namespace_and_key(pool, agent["id"], ns, key)
            if value:
                loaded_data[f"{ns}/{key}"] = value

    last_run_summary = next(
        (k["preview"] for k in state_keys_with_preview if k["key"] == "last_run_summary"),
        None,
    )

    return {
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "load_depth": "deep",
        "has_data": True,
        "last_run_summary": last_run_summary,
        "current_rating": agent.get("current_rating"),
        "schedule": agent.get("schedule"),
        "loaded_state": loaded_state,
        "loaded_data": loaded_data,
    }


def format_for_system_prompt(ctx: dict) -> str:
    if not ctx:
        return ""

    name = ctx.get("agent_name", "Agent")
    lines: list[str] = [f"## Kontext: Agent {name}"]

    if ctx.get("load_depth") == "deep" and not ctx.get("has_data"):
        lines.append(ctx.get("message", f"{name} hat keine abfragbaren Daten."))
        return "\n".join(lines)

    if ctx.get("last_run_summary"):
        lines.append(f"Letzter Lauf: {ctx['last_run_summary']}")

    if ctx.get("current_rating"):
        note = f" — {ctx['current_rating_note']}" if ctx.get("current_rating_note") else ""
        lines.append(f"Bewertung: {ctx['current_rating']}{note}")

    if ctx.get("schedule"):
        lines.append(f"Zeitplan: {ctx['schedule']}")

    if ctx.get("load_depth") == "shallow" and ctx.get("state_keys"):
        lines.append(f"Verfügbare State-Keys: {', '.join(ctx['state_keys'])}")

    loaded_state: dict[str, str] = ctx.get("loaded_state", {})
    if loaded_state:
        lines.append("\nGeladener State:")
        for key, value in loaded_state.items():
            preview = value[:500] + "…" if len(value) > 500 else value
            lines.append(f"  {key}: {preview}")

    loaded_data: dict[str, str] = ctx.get("loaded_data", {})
    if loaded_data:
        lines.append("\nGeladene Daten:")
        for path, value in loaded_data.items():
            preview = value[:2000] + "…" if len(value) > 2000 else value
            lines.append(f"  [{path}]\n{preview}")

    return "\n".join(lines)


_DEEP_LOAD_TRIGGERS = {
    "zeig", "zeige", "was hat", "was hast", "was hat gefunden", "letzten",
    "analyse", "angebot", "bericht", "report", "ergebnis", "gefunden",
    "gespeichert", "daten", "inhalt", "ausgabe", "liste",
}


def needs_deep_load(text: str) -> bool:
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in _DEEP_LOAD_TRIGGERS)
