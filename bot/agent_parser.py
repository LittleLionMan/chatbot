from __future__ import annotations
import json
import logging
import random
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from croniter import croniter
import asyncpg
from bot import brain, config, memory
from bot.models import (
    CAPABILITY_FAST,
    CAPABILITY_BALANCED,
    CAPABILITY_SEARCH,
    CAPABILITY_REASONING,
    CAPABILITY_DEEP_REASONING,
    CAPABILITY_CODING,
)
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

_CAPABILITY_CLASSIFIER_SYSTEM = """Analysiere diese Agent-Instruction und bestimme welche primäre Fähigkeit der ausführende LLM-Call benötigt.

Antworte NUR mit einem dieser Werte, kein anderer Text:
- fast: einfache Statusprüfungen, Ja/Nein-Entscheidungen, kurze Transformationen ohne eigenes Urteil
- balanced: moderate Analyse, Zusammenfassungen, normaler Informationsabruf
- search: Web-Recherche, Nachrichtenauswertung, große Mengen Text zusammenfassen und bewerten
- reasoning: Analysen mit mehreren Abhängigkeiten, Bewertungen die Urteilsvermögen erfordern
- deep_reasoning: komplexe mehrstufige Schlussfolgerungen mit vielen Interdependenzen — Fundamentalanalysen, strategische Systembewertungen, Entscheidungen mit langfristigen Konsequenzen die schwer rückgängig zu machen sind
- coding: Code schreiben, debuggen, Codebasen analysieren

Wähle deep_reasoning nur wenn die Aufgabe wirklich von tiefem Reasoning profitiert — nicht als Default für alles Komplexe.

Beispiele:
"Überwache Docker Container und melde wenn einer down ist" → fast
"Suche täglich nach GPU-Angeboten unter 300€ auf Secondhand-Plattformen" → search
"Prüfe aktuelle Nachrichten zu Unternehmen auf der Watchlist" → search
"Erstelle vollständige Fundamentalanalysen inkl. Bilanzqualität, Marktposition und Kursziel" → deep_reasoning
"Bewerte ob neue Quartalszahlen die bestehende Analyse verändern" → reasoning
"Schreibe und teste neue API-Endpoints für das Projekt" → coding
"Fasse den täglichen Wetterbericht zusammen" → balanced"""

_NAME_RESOLUTION_SYSTEM = """Identifiziere welcher Agent aus der Liste gemeint ist.
Antworte NUR mit der ID des Agenten als Integer, kein anderer Text.
Wenn kein Agent eindeutig zuzuordnen ist, antworte mit 0.
Beispiel: 3"""

_AGENT_TALK_SYSTEM = """Du bist Bob. Ein Nutzer fragt nach einem deiner laufenden Agenten oder möchte dessen Konfiguration ändern.

Du sprichst ÜBER den Agenten — du bist nicht der Agent und schlüpfst nicht in seine Rolle.

Dir werden Name, Konfiguration, aktueller State und bisherige Beobachtungen des Agenten übergeben.

Mögliche Anfragen:
- Statusabfrage ("Wie läuft X?", "Was hat X gefunden?") → fasse State und Beobachtungen in Bobs Stimme zusammen, z.B. "Gecko hat bisher 12 Unternehmen gefunden..."
- Inhaltliche Konfigurationsänderung (Suchkriterien, Häufigkeit, Fokus, Instruktion) → bestätige knapp was geändert wird, gib das vollständige neue config-Objekt zurück: ```config\n{...}\n```
- Umbenennung ("nenn ihn X", "er soll jetzt Y heißen") → bestätige knapp, gib den neuen Namen zurück: ```name\nNeuerName\n```
- Kombination aus mehreren Änderungen → alle zutreffenden Blöcke zurückgeben
- Allgemeine Frage über den Agenten → antworte in Bobs Stimme, nicht in der des Agenten

Wenn du die Konfiguration änderst, gib immer das vollständige neue config-Objekt zurück — alle Felder, nicht nur die geänderten.
Das config-Objekt hat die Felder: instruction, state_keys, data_reads, type, work_capability.

Wichtig: Technische Meta-Operationen wie Capability-Klassifizierung oder Pipeline-Generierung werden separat behandelt — du musst sie hier nicht zurückgeben."""


def _pick_name_for_topic(topic_type: str) -> str:
    candidates = _NAMES_BY_TOPIC.get(topic_type.lower(), _NAMES_BY_TOPIC["default"])
    return random.choice(candidates)


async def _classify_work_capability(instruction: str) -> str:
    valid = {CAPABILITY_FAST, CAPABILITY_BALANCED, CAPABILITY_SEARCH, CAPABILITY_REASONING, CAPABILITY_DEEP_REASONING, CAPABILITY_CODING}
    try:
        raw = await brain.chat(
            system=_CAPABILITY_CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": instruction}],
            max_tokens=20,
            capability=CAPABILITY_FAST,
        )
        result = raw.strip().lower()
        if result in valid:
            return result
        logger.warning("Capability classifier returned unknown value %r, falling back to balanced", result)
        return CAPABILITY_BALANCED
    except Exception as e:
        logger.warning("Capability classification failed: %s", e)
        return CAPABILITY_BALANCED


_PIPELINE_GENERATOR_SYSTEM = """Du entwirfst eine Ausführungs-Pipeline für einen Agenten.

Die Konfiguration besteht aus drei optionalen Teilen die kombiniert werden:
1. "pipeline": Feste Steps die immer in dieser Reihenfolge laufen (Router, State-Reads, etc.)
2. "pipeline_template": Ein wiederholbarer Step-Pattern für variable Datenlisten
3. "pipeline_after_template": Feste Steps die nach den Template-Steps laufen (Analyse, Trigger, etc.)

Antworte NUR mit einem JSON-Objekt mit diesen optionalen Feldern. Kein anderer Text, keine Markdown-Backticks.

Felder in "pipeline" und "pipeline_after_template" — jeder Step hat:
- "id": Eindeutiger snake_case Bezeichner
- "capability": "fast", "search", "reasoning", "deep_reasoning"
- "prompt_template": Anweisung für diesen Teilschritt. Vorherige Step-Outputs als {{output_key}} verfügbar. State-Variablen als {{key}}, trigger_payload als {{trigger_payload.key}}.
- "output_key": Unter welchem Key der Output gespeichert wird
- "is_router": true nur für Router-Steps
- "only_if_route": String oder Liste — Step nur bei diesem Route-Wert ausführen

Felder in "pipeline_template":
- "source": "state" (aus Agent-State), "injected" (aus data_reads), "static" (feste Liste in Config)
- "foreach": Key-Name im State/injected (bei source=state/injected)
- "foreach_items": Feste Liste von Strings (bei source=static)
- "split_by": Trennzeichen bei source=state/injected (Standard: ",")
- "batch_size": Items pro Step (1 = ein Step pro Item)
- "aggregate_key": Unter diesem Key werden alle Template-Outputs gesammelt und sind im nächsten Step verfügbar
- "only_if_route": Optionaler Route-Filter für alle Template-Steps
- "step": Step-Template mit {{item}} als Platzhalter für den aktuellen Wert, {{item_id}} als URL-sicherer Bezeichner

Regeln:
- Nur pipeline_template verwenden wenn die Anzahl der Items zur Laufzeit variabel ist (aus State) oder zur Erstellungszeit fix aber mehrere Search-Steps rechtfertigt (static)
- Die vollständige Agenten-Instruction wird jedem Step als Kontext mitgegeben — Prompts müssen sie nicht wiederholen
- Search-Steps enden immer mit: "Fasse als kompaktes Markdown zusammen — maximal 300 Wörter, nur Fakten. Das Ergebnis wird von einem anderen Modell weiterverarbeitet."
- Router immer als ersten Step in "pipeline" wenn die Instruction Modi beschreibt
- Der letzte Step in pipeline_after_template ist immer reasoning/deep_reasoning

Beispiel Jim Cramer (variable Ticker aus State, zwei Modi):
{
  "pipeline": [
    {"id": "router", "capability": "fast", "is_router": true, "prompt_template": "Prüfe ob trigger_payload.watch_ticker vorhanden ist. Falls ja: antworte mit 'trigger'. Falls nein: antworte mit 'normal'.", "output_key": "route"},
    {"id": "update_watch_trigger", "capability": "fast", "only_if_route": "trigger", "prompt_template": "Aktualisiere watch_triggers für {{trigger_payload.watch_ticker}}.", "output_key": "trigger_result"}
  ],
  "pipeline_template": {
    "source": "state",
    "foreach": "fundamentalanalyse_vorhanden",
    "split_by": ",",
    "batch_size": 1,
    "aggregate_key": "all_news",
    "only_if_route": "normal",
    "step": {
      "id": "search_{{item_id}}",
      "capability": "search",
      "prompt_template": "Suche nach aktuellen Nachrichten für {{item}} der letzten 7 Tage. Nur signifikante Ereignisse. Fasse als kompaktes Markdown zusammen — maximal 300 Wörter, nur Fakten. Das Ergebnis wird von einem anderen Modell weiterverarbeitet.",
      "output_key": "news_{{item_id}}"
    }
  },
  "pipeline_after_template": [
    {"id": "search_macro", "capability": "search", "only_if_route": "normal", "prompt_template": "Suche nach aktuellen Makro-Ereignissen. Fasse als kompaktes Markdown zusammen — maximal 300 Wörter, nur Fakten. Das Ergebnis wird von einem anderen Modell weiterverarbeitet.", "output_key": "macro_news"},
    {"id": "analyze_and_trigger", "capability": "deep_reasoning", "only_if_route": "normal", "prompt_template": "Analysiere alle Nachrichten: {{all_news}} und Makro: {{macro_news}}. Vergleiche mit last_news. Entscheide über Trigger.", "output_key": "final_result"}
  ]
}

Beispiel GPU-Agent (feste Items, kein Router):
{
  "pipeline_template": {
    "source": "static",
    "foreach_items": ["RTX 4060 Ti 16GB <220€", "RTX 9070 XT 16GB <500€", "RTX 3090 24GB <600€"],
    "batch_size": 1,
    "aggregate_key": "all_listings",
    "step": {
      "id": "search_{{item_id}}",
      "capability": "search",
      "prompt_template": "Suche nach aktuellen Angeboten für {{item}} auf deutschen Secondhand-Plattformen. Fasse als kompaktes Markdown zusammen — maximal 300 Wörter, nur Fakten. Das Ergebnis wird von einem anderen Modell weiterverarbeitet.",
      "output_key": "listings_{{item_id}}"
    }
  },
  "pipeline_after_template": [
    {"id": "analyze", "capability": "reasoning", "prompt_template": "Analysiere alle gefundenen Angebote: {{all_listings}}. Vergleiche mit known_listings und price_baseline.", "output_key": "final_result"}
  ]
}"""


_PIPELINE_CAPABILITIES = {CAPABILITY_SEARCH, CAPABILITY_REASONING, CAPABILITY_DEEP_REASONING, CAPABILITY_CODING}


async def _generate_pipeline(instruction: str, work_capability: str, state_keys: list[str]) -> dict | None:
    if work_capability not in _PIPELINE_CAPABILITIES:
        return None
    try:
        state_hint = f"Verfügbare State-Variablen: {', '.join(state_keys)}" if state_keys else ""
        content = f"Agent-Instruction: {instruction}"
        if state_hint:
            content += f"\n{state_hint}"
        raw = await brain.chat(
            system=_PIPELINE_GENERATOR_SYSTEM,
            messages=[{"role": "user", "content": content}],
            capability=CAPABILITY_REASONING,
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            logger.warning("Pipeline generator returned non-dict")
            return None

        has_pipeline = isinstance(parsed.get("pipeline"), list)
        has_template = isinstance(parsed.get("pipeline_template"), dict)
        has_after = isinstance(parsed.get("pipeline_after_template"), list)

        if not has_pipeline and not has_template and not has_after:
            logger.warning("Pipeline generator returned empty structure")
            return None

        logger.info(
            "Pipeline generated: %d fixed steps, template=%s, %d after-steps",
            len(parsed.get("pipeline", [])),
            "yes" if has_template else "no",
            len(parsed.get("pipeline_after_template", [])),
        )
        return parsed
    except Exception as e:
        logger.warning("Pipeline generation failed: %s", e)
        return None


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
            capability=CAPABILITY_FAST,
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
            capability=CAPABILITY_BALANCED,
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

        work_capability = await _classify_work_capability(instruction)
        logger.info("Agent work_capability classified as: %s", work_capability)

        state_keys: list[str] = parsed.get("state_keys", ["last_run_summary"])
        pipeline_result = await _generate_pipeline(instruction, work_capability, state_keys)
        if pipeline_result:
            steps_count = len(pipeline_result.get("pipeline", [])) + len(pipeline_result.get("pipeline_after_template", []))
            has_template = bool(pipeline_result.get("pipeline_template"))
            logger.info("Agent pipeline generated: %d fixed steps, template=%s", steps_count, has_template)

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
            "state_keys": state_keys,
            "data_reads": parsed.get("data_reads", []),
            "type": parsed.get("type", "default"),
            "work_capability": work_capability,
        }
        if pipeline_result:
            if pipeline_result.get("pipeline"):
                agent_config["pipeline"] = pipeline_result["pipeline"]
            if pipeline_result.get("pipeline_template"):
                agent_config["pipeline_template"] = pipeline_result["pipeline_template"]
            if pipeline_result.get("pipeline_after_template"):
                agent_config["pipeline_after_template"] = pipeline_result["pipeline_after_template"]

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
            messages=[{"role": "user", "content": f"{context}\n\nNutzeranfrage: {text}"}],
            capability=CAPABILITY_BALANCED,
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
