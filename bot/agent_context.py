from __future__ import annotations

import json
import logging

import asyncpg

from bot import brain, memory
from bot.agent_skills import summarize_steps
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


_MEMBERSHIP_SYSTEM = """Du entscheidest ob eine neue Nutzernachricht zu einem laufenden Gespräch über einen Agenten gehört.

Du bekommst:
- Den Namen des Agents um den es im laufenden Gespräch geht
- Eine Zusammenfassung des bisherigen Gesprächsverlaufs
- Die neue Nutzernachricht

Antworte NUR mit einem einzigen Wort, kein anderer Text:
- "continue" wenn die Nachricht das laufende Gespräch über den Agenten fortsetzt (Rückfrage, Reaktion, Zustimmung, Widerspruch, Vertiefung).
- "closed" wenn die Nachricht das Thema ohne Ergebnis abschließt (z.B. "lass gut sein", "passt schon", "vergiss es", "egal", "schon ok").
- "unrelated" wenn die Nachricht sich auf etwas völlig anderes bezieht das mit dem Agenten nichts zu tun hat.

Im Zweifel zwischen continue und unrelated: continue."""


async def classify_talk_membership(
    agent_name: str,
    conversation_summary: str,
    new_message: str,
) -> str:
    content = (
        f"Agent: {agent_name}\n\n"
        f"Bisheriges Gespräch:\n{conversation_summary}\n\n"
        f"Neue Nachricht: {new_message}"
    )
    try:
        raw = await brain.chat(
            system=_MEMBERSHIP_SYSTEM,
            messages=[{"role": "user", "content": content}],
            max_tokens=5,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="agent_context:membership",
        )
        verdict = raw.strip().lower()
        if verdict.startswith("closed"):
            return "closed"
        if verdict.startswith("unrelated"):
            return "unrelated"
        return "continue"
    except Exception as e:
        logger.warning("agent_context: membership check failed: %s", e)
        return "continue"


def _summarize_pipeline_for_prompt(config_data: dict) -> str:
    steps = config_data.get("steps") or (
        config_data.get("pipeline", []) + config_data.get("pipeline_after_template", [])
    )
    if not steps:
        return ""

    lines: list[str] = [
        f"Pipeline-Übersicht: {summarize_steps(steps)}",
        "",
        "Steps im Detail:",
    ]
    for s in steps:
        step_id = s.get("id", "?")
        step_type = s.get("type", "?")
        detail = f"- {step_id} ({step_type})"
        if s.get("operation"):
            detail += f" op={s['operation']}"
        if s.get("output_key"):
            detail += f" → {s['output_key']}"
        lines.append(detail)
        prompt = s.get("prompt")
        if prompt and step_type in (
            "llm_extract",
            "llm_decide",
            "llm_summarize",
            "llm_analyze",
            "router_llm",
        ):
            preview = prompt[:300] + "…" if len(prompt) > 300 else prompt
            lines.append(f"    Prompt: {preview}")
    return "\n".join(lines)


async def load(
    pool: asyncpg.Pool,
    agent: dict,
    user_query: str,
) -> dict:
    config_data = parse_agent_config(agent.get("config", {}))
    pipeline_summary = _summarize_pipeline_for_prompt(config_data)

    state_keys_with_preview = await memory.get_agent_state_keys_with_preview(
        pool, agent["id"]
    )
    namespaces = await memory.get_agent_data_namespaces(pool, agent["id"])

    data_keys: dict[str, list[str]] = {}
    for ns in namespaces:
        keys = await memory.get_agent_data_keys_in_namespace(pool, agent["id"], ns)
        data_keys[ns] = keys

    last_run_summary = next(
        (
            k["preview"]
            for k in state_keys_with_preview
            if k["key"] == "last_run_summary"
        ),
        None,
    )

    base: dict = {
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "instruction": config_data.get("instruction", ""),
        "pipeline_summary": pipeline_summary,
        "last_run_summary": last_run_summary,
        "current_rating": agent.get("current_rating"),
        "current_rating_note": agent.get("current_rating_note"),
        "schedule": agent.get("schedule"),
        "last_run_at": agent.get("last_run_at"),
    }

    has_anything = bool(state_keys_with_preview) or bool(namespaces)
    if not has_anything:
        base["has_data"] = False
        base["message"] = f"{agent['name']} speichert keine abfragbaren Daten."
        return base

    state_summary = "\n".join(
        f"- {k['key']}: {k['preview']} ({k['length']} Zeichen)"
        for k in state_keys_with_preview
    )
    data_summary = "\n".join(
        f"- {ns}: {', '.join(keys)}" for ns, keys in data_keys.items()
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
        steps = config_data.get("steps", [])
        writes = [
            s.get("key") or s.get("key_template", "")
            for s in steps
            if s.get("type") in ("state_write", "data_write")
        ]
        base["has_data"] = False
        base["message"] = (
            f"{agent['name']} speichert zur aktuellen Frage keine abfragbaren Daten."
            + (
                f" Der Agent schreibt: {', '.join(w for w in writes if w)}."
                if writes
                else ""
            )
        )
        return base

    loaded_state: dict[str, str] = {}
    state = await memory.get_agent_state(pool, agent["id"])
    for key in relevance.get("state_keys", []):
        if key in state:
            loaded_state[key] = state[key]

    loaded_data: dict[str, str] = {}
    for item in relevance.get("data_items", []):
        ns = item.get("namespace", "")
        key = item.get("key", "")
        if ns and key:
            value = await memory.get_agent_data_by_namespace_and_key(
                pool, agent["id"], ns, key
            )
            if value:
                loaded_data[f"{ns}/{key}"] = value

    base["has_data"] = True
    base["loaded_state"] = loaded_state
    base["loaded_data"] = loaded_data
    return base


def format_for_system_prompt(ctx: dict) -> str:
    if not ctx:
        return ""

    name = ctx.get("agent_name", "Agent")
    lines: list[str] = [f"## Kontext: Agent {name}"]

    if ctx.get("instruction"):
        lines.append(f"Auftrag: {ctx['instruction']}")

    if ctx.get("pipeline_summary"):
        lines.append("")
        lines.append(ctx["pipeline_summary"])

    if ctx.get("last_run_summary"):
        lines.append(f"\nLetzter Lauf: {ctx['last_run_summary']}")

    if ctx.get("current_rating"):
        note = (
            f" — {ctx['current_rating_note']}" if ctx.get("current_rating_note") else ""
        )
        lines.append(f"Bewertung: {ctx['current_rating']}{note}")

    if ctx.get("schedule"):
        lines.append(f"Zeitplan: {ctx['schedule']}")

    if not ctx.get("has_data"):
        lines.append("")
        lines.append(ctx.get("message", f"{name} hat keine abfragbaren Daten."))
        return "\n".join(lines)

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
