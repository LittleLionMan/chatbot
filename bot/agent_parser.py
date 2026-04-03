from __future__ import annotations
import json
import logging
import random
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from croniter import croniter
import asyncpg
from bot import brain, config, memory
from bot.utils import clean_llm_json

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
- "state_keys": Liste von Schlüsseln die der Agent zwischen Läufen im Gedächtnis behalten soll. Immer enthalten: "last_run_summary". Weitere nach Bedarf — was muss der Agent wissen um beim nächsten Lauf sinnvoll weiterzumachen? Beispiele: "known_container_states", "last_script_version", "price_baseline", "open_issues".
- "type": Kurzes Schlagwort für den Bereich. Beispiele: "monitoring", "research", "coding", "finance", "news", "market". Dient nur der Namenswahl.
- "schedule": Cron-Expression (5 Felder). Beispiele: stündlich = "0 * * * *", täglich um 9 = "0 9 * * *", montags = "0 9 * * 1".
- "target": "same" für denselben Chat, "dm" für Privatnachricht.
- "wants_name": true wenn der User einen Namen erwähnt oder explizit fragt, false sonst.
- "suggested_name": Konkreter Name wenn der User einen nennt, sonst null.

Wenn kein sinnvoller Zeitplan erkennbar ist, setze schedule auf null.

Beispiele:

Eingabe: "Überwache meine Docker Container stündlich und sag mir wenn einer down ist"
Output: {"instruction": "Prüfe den Status der laufenden Docker Container. Vergleiche mit dem letzten bekannten Zustand. Melde nur wenn sich etwas verändert hat — Container die neu down oder up sind.", "state_keys": ["last_run_summary", "known_container_states"], "type": "monitoring", "schedule": "0 * * * *", "target": "same", "wants_name": false, "suggested_name": null}

Eingabe: "Schreib mir jeden Montag ein kleines Python-Script das die größten Dateien in /var/log auflistet, und verbessere es wenn du Verbesserungspotenzial siehst"
Output: {"instruction": "Erstelle oder verbessere ein Python-Script das die 10 größten Dateien in /var/log auflistet und ihre Größe leserlich formatiert. Wenn bereits eine Version existiert, prüfe ob Verbesserungen sinnvoll sind und liefere nur eine neue Version wenn ja.", "state_keys": ["last_run_summary", "current_script"], "type": "coding", "schedule": "0 9 * * 1", "target": "same", "wants_name": false, "suggested_name": null}

Eingabe: "Beobachte RTX 4060 Ti Preise täglich unter 220€, nenn ihn Linus"
Output: {"instruction": "Suche nach Angeboten für RTX 4060 Ti unter 220€ auf deutschen Sekundärmarkt-Plattformen. Vergleiche mit bekannten Fundstücken. Melde nur neue Treffer oder relevante Preisänderungen.", "state_keys": ["last_run_summary", "known_listings", "price_baseline"], "type": "research", "schedule": "0 9 * * *", "target": "same", "wants_name": true, "suggested_name": "Linus"}"""

_CREATE_TRIGGER_SYSTEM = """Entscheide ob der Nutzer einen neuen persistenten Agenten erstellen möchte.
Ein Agent läuft nach Plan, hat Gedächtnis zwischen den Läufen und handelt nur bei relevanten Änderungen oder wiederkehrenden Aufgaben.
Antworte ausschließlich mit dem Wort 'ja' oder dem Wort 'nein'. Keine anderen Wörter, keine Erklärungen.
Erkennungsmerkmale: "beobachte", "verfolge", "überwache", "halte mich auf dem Laufenden", "melde wenn", "analysiere laufend", "schreib mir regelmäßig", "halte X aktuell".
Kein Agent: einmalige Aufgabe, stateless Task ohne Vergleichsbedarf, einfache Erinnerung.
Beispiele: "Überwache meine Docker Container" → ja, "Verfolge GPU-Preise" → ja, "Halte mein Deployment-Script aktuell" → ja, "Erinnere mich jeden Montag" → nein, "Such mir jetzt nach RTX" → nein."""

_STOP_TRIGGER_SYSTEM = """Entscheide ob der Nutzer einen laufenden Agenten stoppen oder deaktivieren möchte.
Antworte ausschließlich mit dem Wort 'ja' oder dem Wort 'nein'. Keine anderen Wörter, keine Erklärungen.
Beispiele: "Stopp Linus" → ja, "Deaktiviere den GPU-Agenten" → ja, "Argus soll aufhören" → ja, "Erstelle einen Agenten" → nein."""

_LIST_TRIGGER_SYSTEM = """Entscheide ob der Nutzer seine bereits existierenden Agenten auflisten möchte.
Antworte ausschließlich mit dem Wort 'ja' oder dem Wort 'nein'. Keine anderen Wörter, keine Erklärungen.
Nur 'ja' wenn der Nutzer explizit nach einer Liste oder einem Überblick seiner laufenden Agenten fragt.
Beispiele: "Zeig meine Agenten" → ja, "Welche Agenten laufen" → ja, "Was für Agenten habe ich" → ja.
Beispiele für nein: "Was macht Linus" → nein, "Stopp Gordon" → nein, "Beobachte täglich den Markt" → nein, "Erstelle einen Agenten" → nein."""

_RENAME_TRIGGER_SYSTEM = """Entscheide ob der Nutzer einen Agenten umbenennen möchte.
Antworte ausschließlich mit dem Wort 'ja' oder dem Wort 'nein'. Keine anderen Wörter, keine Erklärungen.
Beispiele: "Nenn Linus jetzt Max" → ja, "Benenne den GPU-Agenten um in Torsten" → ja, "Stopp Linus" → nein."""

_TALK_TRIGGER_SYSTEM = """Entscheide ob der Nutzer mit einem bestimmten Agenten sprechen oder ihn direkt ansprechen möchte.
Das umfasst: nach dem Status fragen, Konfiguration ändern, bisherige Ergebnisse abfragen, oder den Agenten direkt beim Namen ansprechen.
Antworte ausschließlich mit dem Wort 'ja' oder dem Wort 'nein'. Keine anderen Wörter, keine Erklärungen.
Beispiele: "Wie läuft Linus?" → ja, "Argus, konzentriere dich nur auf den nginx Container" → ja, "Was hat Gordon bisher beobachtet?" → ja, "Zeig meine Agenten" → nein, "Stopp Linus" → nein."""

_NAME_RESOLUTION_SYSTEM = """Identifiziere welcher Agent aus der Liste gemeint ist.
Antworte NUR mit der ID des Agenten als Integer, kein anderer Text.
Wenn kein Agent eindeutig zuzuordnen ist, antworte mit 0.
Beispiel: 3"""

_RENAME_PARSER_SYSTEM = """Extrahiere aus der Nutzeranfrage den neuen Namen für den Agenten.
Antworte NUR mit dem neuen Namen, kein anderer Text, keine Anführungszeichen.
Wenn kein neuer Name erkennbar ist, antworte mit dem Wort UNBEKANNT."""

_AGENT_TALK_SYSTEM = """Du bist Bob. Ein Nutzer spricht direkt mit einem deiner laufenden Agenten oder fragt nach ihm.

Dir werden Name, Konfiguration, aktueller State und bisherige Beobachtungen des Agenten übergeben.

Mögliche Anfragen:
- Statusabfrage ("Wie läuft X?", "Was hat X gefunden?") → fasse State und Beobachtungen zusammen
- Konfigurationsänderung ("X, konzentriere dich ab jetzt auf Y") → bestätige die Änderung knapp, gib die neue config als JSON-Block zurück: ```config\n{...}\n```
- Allgemeine Ansprache → antworte im Stil von Bob, aus der Perspektive des Agenten

Wenn du die Konfiguration änderst, gib immer das vollständige neue config-Objekt zurück — alle Felder, nicht nur die geänderten.
Das config-Objekt hat die Felder: instruction, state_keys, type."""


def _pick_name_for_topic(topic_type: str) -> str:
    candidates = _NAMES_BY_TOPIC.get(topic_type.lower(), _NAMES_BY_TOPIC["default"])
    return random.choice(candidates)


async def _binary(system: str, text: str) -> bool:
    try:
        result = await brain.chat(
            system=system,
            messages=[{"role": "user", "content": text}],
            max_tokens=5,
        )
        return result.strip().lower().startswith("ja")
    except Exception as e:
        logger.warning("Agent binary check failed: %s", e)
        return False


async def is_agent_creation(text: str) -> bool:
    return await _binary(_CREATE_TRIGGER_SYSTEM, text)


async def is_agent_stop_request(text: str) -> bool:
    return await _binary(_STOP_TRIGGER_SYSTEM, text)


async def is_agent_list_request(text: str) -> bool:
    return await _binary(_LIST_TRIGGER_SYSTEM, text)


async def is_agent_rename_request(text: str) -> bool:
    return await _binary(_RENAME_TRIGGER_SYSTEM, text)


async def is_agent_talk(text: str) -> bool:
    return await _binary(_TALK_TRIGGER_SYSTEM, text)


async def resolve_agent_by_text(
    text: str,
    active_agents: list[dict],
) -> dict | None:
    if not active_agents:
        return None
    agent_list = "\n".join(
        f"ID {a['id']}: {a['name']} — {a['config'].get('instruction', '')[:80]}"
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
            "type": parsed.get("type", "default"),
        }

        return {
            "config": agent_config,
            "schedule": schedule,
            "target_chat_id": target_chat_id,
            "next_run_at": next_run_utc,
            "next_run_display": next_run_local,
            "wants_name": bool(parsed.get("wants_name", False)),
            "suggested_name": parsed.get("suggested_name"),
        }
    except Exception as e:
        logger.warning("Agent parsing failed: %s", e)
        return None


async def parse_rename_request(text: str) -> str | None:
    try:
        raw = await brain.chat(
            system=_RENAME_PARSER_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=30,
        )
        name = raw.strip()
        if name.upper() == "UNBEKANNT" or not name:
            return None
        return name
    except Exception as e:
        logger.warning("Rename parsing failed: %s", e)
        return None


async def handle_agent_talk(
    text: str,
    agent: dict,
    state: dict[str, str],
    agent_memories: list[str],
) -> tuple[str, dict | None]:
    config_data = agent["config"] if isinstance(agent["config"], dict) else {}
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
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning("Agent talk LLM call failed: %s", e)
        return "Konnte den Agenten nicht befragen.", None

    new_config: dict | None = None
    if "```config" in response:
        try:
            start = response.index("```config") + len("```config")
            end = response.index("```", start)
            raw_config = response[start:end].strip()
            new_config = json.loads(raw_config)
            response = response[:response.index("```config")].strip()
        except Exception as e:
            logger.warning("Config extraction from agent talk response failed: %s", e)

    return response, new_config


def next_agent_run_after(schedule: str, timezone: str) -> datetime:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(config.BOT_DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    next_run_local = croniter(schedule, now).get_next(datetime)
    return next_run_local.astimezone(ZoneInfo("UTC"))
