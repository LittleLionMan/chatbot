from __future__ import annotations
import json
import logging
import random
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from croniter import croniter
import asyncpg
from bot import brain, config, memory
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)

_NAMES_BY_TOPIC = {
    "gpu": ["Linus", "Jensen"],
    "finance": ["Gordon", "Warren", "Floyd"],
    "news": ["Wolf", "Peter", "Anna"],
    "research": ["Ada", "Grace", "Alan"],
    "market": ["Gordon", "Floyd", "Morgan"],
    "monitoring": ["Argus", "HAL", "Watcher"],
    "coding": ["Dennis", "Guido", "Ada"],
    "default": ["Iris", "Hermes", "Scout", "Remy"],
}

_AGENT_PARSER_SYSTEM = """Du extrahierst einen persistenten Agenten aus einer Nutzeranfrage.
Ein Agent läuft nach Plan, erinnert sich an frühere Ergebnisse und handelt nur wenn sich etwas Relevantes ändert oder eine wiederkehrende Aufgabe ansteht.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "instruction": Was soll der Agent tun? Vollständige, eigenständige Anweisung in natürlicher Sprache. So formulieren dass der Agent sie ohne weiteren Kontext ausführen kann. Maximal 400 Zeichen.
- "state_keys": Liste von Schlüsseln die der Agent zwischen Läufen im Gedächtnis behalten soll. Immer enthalten: "last_run_summary". Weitere nach Bedarf.
- "data_reads": Liste von Datenbank-Lesevorgängen die vor jedem Lauf automatisch ausgeführt werden. Zwei Typen:
  - {"type": "state", "agent_name": "..."} — liest den kompletten State eines anderen Agenten. Nutze das wenn dieser Agent die primäre Datenquelle ist.
  - {"type": "namespace", "namespace": "...", "agent_name": "..."} — liest einen DB-Namespace eines anderen Agenten. Optional "key" für einzelne Einträge, Template-Variablen wie {{trigger_payload.ticker}} möglich.
  Leer wenn der Agent keine fremden Daten braucht.
- "type": Kurzes Schlagwort für den Bereich. Beispiele: "monitoring", "research", "coding", "finance", "news", "market".
- "schedule": Cron-Expression (5 Felder). Beispiele: stündlich = "0 * * * *", täglich um 9 = "0 9 * * *", montags = "0 9 * * 1".
- "target": "same" für denselben Chat, "dm" für Privatnachricht.
- "wants_name": true wenn der User einen Namen erwähnt oder explizit fragt, false sonst.
- "suggested_name": Konkreter Name wenn der User einen nennt, sonst null.

Wenn kein sinnvoller Zeitplan erkennbar ist, setze schedule auf null.

Beispiele:

Eingabe: "Überwache meine Docker Container stündlich und sag mir wenn einer down ist"
Output: {"instruction": "Prüfe den Status der laufenden Docker Container. Vergleiche mit dem letzten bekannten Zustand. Melde nur wenn sich etwas verändert hat.", "state_keys": ["last_run_summary", "known_container_states"], "data_reads": [], "type": "monitoring", "schedule": "0 * * * *", "target": "same", "wants_name": false, "suggested_name": null}

Eingabe: "Erstelle täglich um 9 für jeweils ein Unternehmen aus Gordons Liste eine Fundamentalanalyse"
Output: {"instruction": "Lies Gordons Unternehmensliste. Wähle ein Unternehmen ohne Fundamentalanalyse und erstelle eine vollständige Analyse.", "state_keys": ["last_run_summary", "analyzed_tickers"], "data_reads": [{"type": "state", "agent_name": "Gordon"}], "type": "finance", "schedule": "0 9 * * *", "target": "same", "wants_name": false, "suggested_name": null}

Eingabe: "Analysiere täglich um 10 das Unternehmen das per Trigger-Payload als ticker übergeben wird"
Output: {"instruction": "Erstelle eine Fundamentalanalyse für den per trigger_payload.ticker übergebenen Ticker.", "state_keys": ["last_run_summary"], "data_reads": [{"namespace": "companies", "key": "{{trigger_payload.ticker}}:criteria_match"}], "type": "finance", "schedule": "0 10 * * *", "target": "same", "wants_name": false, "suggested_name": null}

Eingabe: "Beobachte RTX 4060 Ti Preise täglich unter 220€, nenn ihn Linus"
Output: {"instruction": "Suche nach Angeboten für RTX 4060 Ti unter 220€ auf deutschen Sekundärmarkt-Plattformen. Vergleiche mit bekannten Fundstücken. Melde nur neue Treffer oder relevante Preisänderungen.", "state_keys": ["last_run_summary", "known_listings", "price_baseline"], "data_reads": [], "type": "research", "schedule": "0 9 * * *", "target": "same", "wants_name": true, "suggested_name": "Linus"}"""

_NAME_RESOLUTION_SYSTEM = """Identifiziere welcher Agent aus der Liste gemeint ist.
Antworte NUR mit der ID des Agenten als Integer, kein anderer Text.
Wenn kein Agent eindeutig zuzuordnen ist, antworte mit 0.
Beispiel: 3"""

_AGENT_TALK_SYSTEM = """Du bist Bob. Ein Nutzer spricht direkt mit einem deiner laufenden Agenten oder fragt nach ihm.

Dir werden Name, Konfiguration, aktueller State und bisherige Beobachtungen des Agenten übergeben.

Mögliche Anfragen:
- Statusabfrage ("Wie läuft X?", "Was hat X gefunden?") → fasse State und Beobachtungen zusammen
- Konfigurationsänderung (Suchgebiet, Häufigkeit, Inhalt, Kriterien) → bestätige knapp, gib das vollständige neue config-Objekt zurück: ```config\n{...}\n```
- Umbenennung ("nenn ihn X", "er soll jetzt Y heißen") → bestätige knapp, gib den neuen Namen zurück: ```name\nNeuerName\n```
- Kombination aus mehreren Änderungen → alle zutreffenden Blöcke zurückgeben
- Allgemeine Ansprache → antworte im Stil von Bob

Wenn du die Konfiguration änderst, gib immer das vollständige neue config-Objekt zurück — alle Felder, nicht nur die geänderten.
Das config-Objekt hat die Felder: instruction, state_keys, data_reads, type."""


def _pick_name_for_topic(topic_type: str) -> str:
    candidates = _NAMES_BY_TOPIC.get(topic_type.lower(), _NAMES_BY_TOPIC["default"])
    return random.choice(candidates)


async def resolve_agent_by_text(
    text: str,
    active_agents: list[dict],
) -> dict | None:
    if not active_agents:
        return None
    agent_list = "\n".join(
        f"ID {a['id']}: {a['name']} — {parse_agent_config(a['config']).get('instruction', '')[:80]}"
        for a in active_agents
    )
    try:
        raw = await brain.chat(
            system=_NAME_RESOLUTION_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Agenten:\n{agent_list}\n\nNutzeranfrage: {text}",
            }],
            max_tokens=10,
        )
        resolved_id = int(raw.strip())
        if resolved_id == 0:
            return None
        return next((a for a in active_agents if a["id"] == resolved_id), None)
    except Exception as e:
        logger.warning("Agent name resolution failed: %s", e)
        return None


async def parse_agent_creation(
    text: str,
    user_id: int,
    source_chat_id: int,
    pool: asyncpg.Pool,
) -> dict | None:
    try:
        raw = await brain.chat(
            system=_AGENT_PARSER_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=600,
        )
        logger.debug("Agent parser raw LLM output: %r", raw)
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            return None

        schedule = parsed.get("schedule")
        if not schedule or not croniter.is_valid(schedule):
            logger.warning("Invalid or missing cron from agent parser: %s", schedule)
            return None

        instruction = parsed.get("instruction", "").strip()
        if not instruction:
            return None

        tz_str = await memory.get_user_timezone(pool, user_id)
        try:
            tz = ZoneInfo(tz_str)
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)
        next_run_local = croniter(schedule, now).get_next(datetime)
        next_run_utc = next_run_local.astimezone(ZoneInfo("UTC"))

        target_chat_id = user_id if parsed.get("target") == "dm" else source_chat_id

        agent_config = {
            "instruction": instruction,
            "state_keys": parsed.get("state_keys", ["last_run_summary"]),
            "data_reads": parsed.get("data_reads", []),
            "type": parsed.get("type", "default"),
        }

        raw_suggested: str | None = parsed.get("suggested_name")
        if raw_suggested and raw_suggested.strip().lower() == config.BOT_NAME.lower():
            raw_suggested = None

        return {
            "config": agent_config,
            "schedule": schedule,
            "target_chat_id": target_chat_id,
            "next_run_at": next_run_utc,
            "next_run_display": next_run_local,
            "wants_name": bool(parsed.get("wants_name", False)),
            "suggested_name": raw_suggested,
        }
    except Exception as e:
        logger.warning("Agent parsing failed: %s", e)
        return None


async def handle_agent_talk(
    text: str,
    agent: dict,
    state: dict[str, str],
    agent_memories: list[str],
) -> tuple[str, dict | None, str | None]:
    config_data = parse_agent_config(agent["config"])
    state_summary = "\n".join(f"{k}: {v}" for k, v in state.items()) if state else "noch kein State"
    memories_summary = "\n- ".join(agent_memories) if agent_memories else "noch keine Beobachtungen"

    context = (
        f"Agent: {agent['name']}\n"
        f"Konfiguration: {json.dumps(config_data, ensure_ascii=False)}\n\n"
        f"Aktueller State:\n{state_summary}\n\n"
        f"Bisherige Beobachtungen:\n- {memories_summary}"
    )

    try:
        response = await brain.chat(
            system=_AGENT_TALK_SYSTEM,
            messages=[
                {"role": "user", "content": f"{context}\n\nNutzeranfrage: {text}"},
            ],
            max_tokens=2048,
        )
    except Exception as e:
        logger.warning("Agent talk LLM call failed: %s", e)
        return "Konnte den Agenten nicht befragen.", None, None

    new_config: dict | None = None
    new_name: str | None = None

    if "```config" in response:
        try:
            start = response.index("```config") + len("```config")
            end = response.index("```", start)
            new_config = json.loads(response[start:end].strip())
            response = response[:response.index("```config")].strip()
        except Exception as e:
            logger.warning("Config extraction from agent talk response failed: %s", e)

    if "```name" in response:
        try:
            start = response.index("```name") + len("```name")
            end = response.index("```", start)
            new_name = response[start:end].strip()
            response = response[:response.index("```name")].strip()
        except Exception as e:
            logger.warning("Name extraction from agent talk response failed: %s", e)

    return response, new_config, new_name


def next_agent_run_after(schedule: str, timezone: str) -> datetime:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(config.BOT_DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    next_run_local = croniter(schedule, now).get_next(datetime)
    return next_run_local.astimezone(ZoneInfo("UTC"))
