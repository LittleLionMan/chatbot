from __future__ import annotations
import logging
import asyncpg
import telegram
from bot import brain, memory
from bot.agent_parser import next_agent_run_after
from bot.soul import SOUL

logger = logging.getLogger(__name__)

_DIFF_SYSTEM = """Entscheide ob ein neues Ergebnis relevante Änderungen oder neue Erkenntnisse gegenüber dem letzten Stand enthält.
Antworte ausschließlich mit dem Wort 'ja' oder dem Wort 'nein'. Keine anderen Wörter, keine Erklärungen.
Relevant: neue Treffer, Zustandsänderungen, neue Ergebnisse, veränderte Lage, aktualisierte Versionen.
Nicht relevant: identische Ergebnisse, minimale Formulierungsunterschiede, leere Ergebnisse.
Beispiele: "Container nginx ist down" bei vorherigem "Alle Container laufen" → ja, "Alle Container laufen" bei vorherigem "Alle Container laufen" → nein, "Neues Script mit verbesserter Fehlerbehandlung" bei vorherigem "Script v1" → ja."""

_AGENT_RUN_SYSTEM = f"""{SOUL}

Du führst gerade einen automatischen Auftrag für einen persistenten Agenten aus.
Dir werden die Anweisung des Agenten und sein aktueller Gedächtnisstand übergeben.
Führe die Anweisung aus. Liefere das Ergebnis direkt und prägnant — keine Einleitung, kein Abschluss.
Wenn es im Vergleich zum letzten Stand nichts Neues oder Relevantes gibt, antworte mit dem exakten Text: KEINE_AENDERUNG"""


def _build_relay_system(agent_name: str) -> str:
    return f"""{SOUL}

Du hast gerade einen Bericht von deinem Agenten {agent_name} erhalten.
Formuliere diesen Bericht als kurze Nachricht in der dritten Person — Bob spricht über {agent_name}, nicht als {agent_name}.
Beispiele: "{agent_name} meldet: ...", "{agent_name} hat etwas gefunden: ...", "Laut {agent_name}: ..."
Keine Einleitung, kein Abschluss, kein Kommentar von dir — nur die Weiterleitung des Berichts in Bobs Stimme.
Behalte alle konkreten Fakten, Werte und Ergebnisse aus dem Bericht vollständig bei."""


def _build_run_prompt(config_data: dict, state: dict[str, str]) -> str:
    instruction = config_data.get("instruction", "")
    state_keys: list[str] = config_data.get("state_keys", ["last_run_summary"])

    relevant_state = {k: state[k] for k in state_keys if k in state and state[k]}

    if not relevant_state:
        return f"Anweisung: {instruction}\n\nErster Lauf — kein vorheriger Stand vorhanden."

    state_lines = "\n".join(f"{k}: {v}" for k, v in relevant_state.items())
    return f"Anweisung: {instruction}\n\nAktueller Stand aus letztem Lauf:\n{state_lines}"


async def _has_relevant_change(previous_summary: str, new_result: str) -> bool:
    if not previous_summary:
        return True
    if new_result.strip() == "KEINE_AENDERUNG":
        return False
    try:
        decision = await brain.chat(
            system=_DIFF_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Letzter Stand:\n{previous_summary}\n\nNeues Ergebnis:\n{new_result}",
            }],
            max_tokens=5,
        )
        return decision.strip().lower().startswith("ja")
    except Exception as e:
        logger.warning("Diff check failed for agent, defaulting to report: %s", e)
        return True


async def execute_agent(
    pool: asyncpg.Pool,
    bot: telegram.Bot,
    agent: dict,
) -> None:
    agent_id: int = agent["id"]
    user_id: int = agent["user_id"]
    target_chat_id: int = agent["target_chat_id"]
    name: str = agent["name"]
    config_data: dict = agent["config"] if isinstance(agent["config"], dict) else {}
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
        )

        relevant = await _has_relevant_change(previous_summary, raw_result)

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
    except Exception as e:
        logger.error("Agent %d (%s) execution failed: %s", agent_id, name, e)
