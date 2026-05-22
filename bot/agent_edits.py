from __future__ import annotations
import json
import logging
import asyncpg
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from bot import brain, memory
from bot.models import CAPABILITY_REASONING
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)


def confirmation_keyboard(confirmation_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Bestätigen", callback_data=f"confirm:yes:{confirmation_id}"),
            InlineKeyboardButton("Abbrechen", callback_data=f"confirm:no:{confirmation_id}"),
            InlineKeyboardButton("Anpassen", callback_data=f"confirm:adjust:{confirmation_id}"),
        ]
    ])


def rollback_keyboard(confirmation_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Rückgängig", callback_data=f"confirm:rollback:{confirmation_id}")]
    ])


_DATA_EDIT_SYSTEM = """Du führst einen präzisen Edit an einem gespeicherten Dokument durch.

Du bekommst:
- Die Nutzeranfrage (was soll geändert werden)
- Den aktuellen Dokumentinhalt

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "found": true wenn die zu ändernde Stelle gefunden wurde
- "change_description": Kurze Beschreibung was geändert wird (1 Satz)
- "original_excerpt": Die Originalstelle (max 200 Zeichen) die geändert wird
- "new_content": Der vollständige neue Dokumentinhalt nach der Änderung
- "affected_section": Welcher Abschnitt/Bereich betroffen ist"""


_STEP_PATCH_SYSTEM = """Du identifizierst welcher Pipeline-Step ein beschriebenes Problem verursacht und formulierst einen chirurgischen Prompt-Patch.

Du bekommst:
- Die Nutzeranfrage (welches Problem, was soll anders sein)
- Die aktuelle Pipeline mit allen Steps

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "found": true wenn ein betroffener Step identifiziert wurde
- "step_id": ID des betroffenen Steps
- "step_type": Typ des betroffenen Steps
- "problem_description": Was am aktuellen Prompt das Problem verursacht (1 Satz)
- "patch_description": Was am Prompt geändert wird (1 Satz)
- "original_prompt": Der aktuelle Prompt des Steps
- "new_prompt": Der neue Prompt mit der Ergänzung/Änderung
- "confidence": float 0.0-1.0 wie sicher du dir bist dass dies der richtige Step ist"""


_PREFERENCE_SYSTEM = """Du extrahierst eine neue Nutzerpräferenz aus einer Feedback-Nachricht.

Du bekommst:
- Die Feedback-Nachricht
- Bestehende Präferenzen (zur Konflikterkennung)

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "type": "hard_constraint" (immer ablehnen) oder "soft_preference" (bevorzugen/benachteiligen)
- "rule": Kurzer snake_case Bezeichner
- "description": Menschenlesbare Beschreibung der Präferenz
- "conflict_with": Liste von rule-Bezeichnern bestehender Präferenzen die widersprechen (leer wenn kein Konflikt)
- "conflict_description": Beschreibung des Konflikts wenn vorhanden"""


async def prepare_data_edit(
    pool: asyncpg.Pool,
    agent: dict,
    user_query: str,
    namespace: str,
    key: str,
) -> dict | None:
    current_value = await memory.get_agent_data_by_namespace_and_key(
        pool, agent["id"], namespace, key
    )
    if not current_value:
        return None

    content = f"Anfrage: {user_query}\n\nAktueller Inhalt:\n{current_value}"
    try:
        raw = await brain.chat(
            system=_DATA_EDIT_SYSTEM,
            messages=[{"role": "user", "content": content}],
            capability=CAPABILITY_REASONING,
            caller="agent_edits:data_edit",
        )
        result = json.loads(clean_llm_json(raw))
    except Exception as e:
        logger.warning("agent_edits: data edit preparation failed: %s", e)
        return None

    if not result.get("found"):
        return None

    return {
        "edit_type": "data_edit",
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "namespace": namespace,
        "key": key,
        "original_value": current_value,
        "new_value": result.get("new_content", ""),
        "change_description": result.get("change_description", ""),
        "original_excerpt": result.get("original_excerpt", ""),
        "affected_section": result.get("affected_section", ""),
    }


async def prepare_step_patch(
    pool: asyncpg.Pool,
    agent: dict,
    user_query: str,
) -> dict | None:
    config_data = parse_agent_config(agent.get("config", {}))
    steps = config_data.get("steps", [])
    if not steps:
        return None

    steps_summary = json.dumps(steps, ensure_ascii=False, indent=2)
    content = f"Anfrage: {user_query}\n\nAktuelle Pipeline:\n{steps_summary}"

    try:
        raw = await brain.chat(
            system=_STEP_PATCH_SYSTEM,
            messages=[{"role": "user", "content": content}],
            capability=CAPABILITY_REASONING,
            caller="agent_edits:step_patch",
        )
        result = json.loads(clean_llm_json(raw))
    except Exception as e:
        logger.warning("agent_edits: step patch preparation failed: %s", e)
        return None

    if not result.get("found"):
        return None

    step_id = result.get("step_id", "")
    original_step = next((s for s in steps if s.get("id") == step_id), None)
    if not original_step:
        return None

    new_steps = []
    for s in steps:
        if s.get("id") == step_id:
            patched = dict(s)
            patched["prompt"] = result.get("new_prompt", s.get("prompt", ""))
            new_steps.append(patched)
        else:
            new_steps.append(s)

    return {
        "edit_type": "step_patch",
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "step_id": step_id,
        "step_type": result.get("step_type", ""),
        "problem_description": result.get("problem_description", ""),
        "patch_description": result.get("patch_description", ""),
        "original_prompt": result.get("original_prompt", ""),
        "new_prompt": result.get("new_prompt", ""),
        "original_steps": steps,
        "new_steps": new_steps,
        "confidence": result.get("confidence", 0.0),
    }


async def prepare_preference(
    pool: asyncpg.Pool,
    agent: dict,
    user_query: str,
) -> dict | None:
    state = await memory.get_agent_state(pool, agent["id"])
    existing_prefs_raw = state.get("user_preferences", "[]")
    try:
        existing_prefs: list[dict] = json.loads(existing_prefs_raw)
    except Exception:
        existing_prefs = []

    existing_summary = json.dumps(existing_prefs, ensure_ascii=False)
    content = f"Feedback: {user_query}\n\nBestehende Präferenzen: {existing_summary}"

    try:
        raw = await brain.chat(
            system=_PREFERENCE_SYSTEM,
            messages=[{"role": "user", "content": content}],
            capability=CAPABILITY_REASONING,
            caller="agent_edits:preference",
        )
        result = json.loads(clean_llm_json(raw))
    except Exception as e:
        logger.warning("agent_edits: preference preparation failed: %s", e)
        return None

    return {
        "edit_type": "preference",
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "preference_type": result.get("type", "hard_constraint"),
        "rule": result.get("rule", ""),
        "description": result.get("description", ""),
        "conflicts": result.get("conflict_with", []),
        "conflict_description": result.get("conflict_description", ""),
        "existing_preferences": existing_prefs,
    }


def format_confirmation_message(edit_payload: dict) -> str:
    edit_type = edit_payload.get("edit_type", "")
    name = edit_payload.get("agent_name", "Agent")

    if edit_type == "data_edit":
        ns = edit_payload.get("namespace", "")
        key = edit_payload.get("key", "")
        change = edit_payload.get("change_description", "")
        excerpt = edit_payload.get("original_excerpt", "")
        section = edit_payload.get("affected_section", "")
        msg = f"{name} — Data-Edit in {ns}/{key}"
        if section:
            msg += f" (Abschnitt: {section})"
        msg += f"\n\n{change}"
        if excerpt:
            msg += f"\n\nEntfernt wird: »{excerpt}«"
        return msg

    if edit_type == "step_patch":
        step_id = edit_payload.get("step_id", "")
        step_type = edit_payload.get("step_type", "")
        problem = edit_payload.get("problem_description", "")
        patch = edit_payload.get("patch_description", "")
        orig = edit_payload.get("original_prompt", "")[:200]
        new = edit_payload.get("new_prompt", "")[:200]
        confidence = edit_payload.get("confidence", 0.0)
        conf_str = f" (Konfidenz: {confidence:.0%})" if confidence < 0.8 else ""
        return (
            f"{name} — Step-Patch: {step_id} ({step_type}){conf_str}\n\n"
            f"Problem: {problem}\n"
            f"Änderung: {patch}\n\n"
            f"Aktueller Prompt: …{orig}…\n"
            f"Neuer Prompt: …{new}…"
        )

    if edit_type == "preference":
        pref_type = edit_payload.get("preference_type", "")
        desc = edit_payload.get("description", "")
        conflicts = edit_payload.get("conflicts", [])
        conflict_desc = edit_payload.get("conflict_description", "")
        msg = f"{name} — Neue Präferenz ({pref_type})\n\n{desc}"
        if conflicts:
            msg += f"\n\n⚠️ Konflikt mit bestehender Regel: {conflict_desc}\nSoll die bestehende Regel ersetzt werden?"
        return msg

    return "Unbekannter Edit-Typ."


async def execute_edit(
    pool: asyncpg.Pool,
    edit_payload: dict,
) -> str:
    edit_type = edit_payload.get("edit_type", "")
    agent_id = edit_payload.get("agent_id")
    name = edit_payload.get("agent_name", "Agent")

    if edit_type == "data_edit":
        ns = edit_payload["namespace"]
        key = edit_payload["key"]
        new_value = edit_payload["new_value"]
        original_value = edit_payload["original_value"]

        try:
            from bot import agent_skills as _skills
            await _skills.record_pipeline_edit(
                pool, agent_id,
                edit_payload.get("agent_type", "unknown"),
                f"data_edit:{ns}/{key}",
                [{"type": "data", "namespace": ns, "key": key, "value": original_value}],
                [{"type": "data", "namespace": ns, "key": key, "value": new_value}],
                session_id=f"data_edit_{agent_id}_{ns}_{key}",
            )
        except Exception:
            pass

        await memory.write_agent_data(pool, agent_id, ns, key, new_value)
        logger.info("agent_edits: data_edit executed for agent %d %s/%s", agent_id, ns, key)
        return f"Erledigt. {name}: {ns}/{key} wurde aktualisiert."

    if edit_type == "step_patch":
        step_id = edit_payload["step_id"]
        new_steps = edit_payload["new_steps"]
        original_steps = edit_payload["original_steps"]
        agent_type = edit_payload.get("agent_type", "unknown")

        try:
            from bot import agent_skills as _skills
            orig_step = next((s for s in original_steps if s.get("id") == step_id), None)
            new_step = next((s for s in new_steps if s.get("id") == step_id), None)
            if orig_step and new_step:
                await _skills.record_pipeline_edit(
                    pool, agent_id, agent_type,
                    edit_payload.get("problem_description", ""),
                    [orig_step], [new_step],
                    session_id=f"step_patch_{agent_id}_{step_id}",
                )
        except Exception:
            pass

        row = await pool.fetchrow("SELECT config FROM agents WHERE id = $1", agent_id)
        if not row:
            return "Agent nicht gefunden."
        config_data = parse_agent_config(row["config"])
        config_data["steps"] = new_steps
        await memory.update_agent_config(pool, agent_id, config_data)

        try:
            from bot import agent_skills as _skills
            await _skills.mark_dirty(pool)
        except Exception:
            pass

        logger.info("agent_edits: step_patch executed for agent %d step %s", agent_id, step_id)
        return f"Erledigt. {name}: Step {step_id} wurde angepasst."

    if edit_type == "preference":
        rule = edit_payload["rule"]
        description = edit_payload["description"]
        pref_type = edit_payload["preference_type"]
        existing = edit_payload.get("existing_preferences", [])
        conflicts = edit_payload.get("conflicts", [])

        updated = [p for p in existing if p.get("rule") not in conflicts]
        from datetime import datetime, timezone
        updated.append({
            "type": pref_type,
            "rule": rule,
            "description": description,
            "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "confirmed": True,
        })

        await memory.set_agent_state(pool, agent_id, {
            "user_preferences": json.dumps(updated, ensure_ascii=False)
        })
        logger.info("agent_edits: preference added for agent %d: %s", agent_id, rule)

        replaced = f" ({len(conflicts)} Regel(n) ersetzt)" if conflicts else ""
        return f"Erledigt. {name}: Präferenz '{description}' wurde gespeichert{replaced}."

    return "Unbekannter Edit-Typ."


async def rollback_edit(
    pool: asyncpg.Pool,
    edit_payload: dict,
) -> str:
    edit_type = edit_payload.get("edit_type", "")
    agent_id = edit_payload.get("agent_id")
    name = edit_payload.get("agent_name", "Agent")

    if edit_type == "data_edit":
        ns = edit_payload["namespace"]
        key = edit_payload["key"]
        original_value = edit_payload["original_value"]
        await memory.write_agent_data(pool, agent_id, ns, key, original_value)
        return f"Rückgängig. {name}: {ns}/{key} wurde wiederhergestellt."

    if edit_type == "step_patch":
        original_steps = edit_payload["original_steps"]
        row = await pool.fetchrow("SELECT config FROM agents WHERE id = $1", agent_id)
        if not row:
            return "Agent nicht gefunden."
        config_data = parse_agent_config(row["config"])
        config_data["steps"] = original_steps
        await memory.update_agent_config(pool, agent_id, config_data)
        return f"Rückgängig. {name}: Step wurde auf den vorherigen Stand zurückgesetzt."

    if edit_type == "preference":
        existing = edit_payload.get("existing_preferences", [])
        await memory.set_agent_state(pool, agent_id, {
            "user_preferences": json.dumps(existing, ensure_ascii=False)
        })
        return f"Rückgängig. {name}: Präferenz wurde entfernt."

    return "Rollback nicht möglich."
