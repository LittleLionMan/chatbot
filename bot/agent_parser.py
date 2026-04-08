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
    CAPABILITY_MULTIMODAL,
    CAPABILITY_LONG_CONTEXT,
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


_PIPELINE_GENERATOR_SYSTEM = """Du entwirfst eine Ausführungs-Pipeline für einen Agenten der Web-Recherche betreibt oder tiefes Reasoning braucht.

Die Pipeline besteht aus sequenziellen Steps. Jeder Step bekommt den Output der vorherigen Steps als Template-Variablen.

Antworte NUR mit einem JSON-Array von Steps, kein anderer Text, keine Markdown-Backticks.

Jeder Step hat:
- "id": Eindeutiger snake_case Bezeichner (z.B. "search_kleinanzeigen", "search_ebay", "analyze")
- "capability": Einer von: "search", "reasoning", "deep_reasoning"
- "prompt_template": Die vollständige Anweisung für diesen Step. Vorherige Step-Outputs sind als {{step_id}} verfügbar. Search-Steps enden immer mit: "Fasse deine Ergebnisse als kompaktes Markdown zusammen — maximal 300 Wörter, nur Fakten, keine Einleitungen. Das Ergebnis wird von einem anderen Modell weiterverarbeitet."
- "output_key": Unter welchem Key der Output gespeichert wird — wird als {{output_key}} in späteren Steps verfügbar

Regeln:
- Search-Steps: je eine klar abgegrenzte Quelle oder Suchfrage pro Step. Maximal 5 Search-Steps.
- Der letzte Step ist immer ein "reasoning" oder "deep_reasoning" Step der alle Search-Outputs zusammenführt und das finale Ergebnis produziert.
- Template-Variablen aus dem Agent-State oder trigger_payload sind als {{variable_name}} verfügbar.
- Wenn die Instruction keine Web-Recherche braucht sondern nur tiefes Reasoning: ein einziger "deep_reasoning" Step reicht.

Beispiel für einen Research-Agent:
Input: "Suche täglich nach RTX 4060 Ti Angeboten unter 220€ auf deutschen Plattformen"

Output:
[
  {"id": "search_kleinanzeigen", "capability": "search", "prompt_template": "Suche nach aktuellen Angeboten für RTX 4060 Ti unter 220€ auf Kleinanzeigen.de. Fasse deine Ergebnisse als kompaktes Markdown zusammen — maximal 300 Wörter, nur Fakten, keine Einleitungen. Das Ergebnis wird von einem anderen Modell weiterverarbeitet.", "output_key": "search_kleinanzeigen"},
  {"id": "search_ebay", "capability": "search", "prompt_template": "Suche nach aktuellen Angeboten für RTX 4060 Ti unter 220€ auf eBay Kleinanzeigen und eBay.de. Fasse deine Ergebnisse als kompaktes Markdown zusammen — maximal 300 Wörter, nur Fakten, keine Einleitungen. Das Ergebnis wird von einem anderen Modell weiterverarbeitet.", "output_key": "search_ebay"},
  {"id": "analyze", "capability": "reasoning", "prompt_template": "Analysiere diese Suchergebnisse auf echte Deals für eine RTX 4060 Ti unter 220€. Bekannte Angebote aus früheren Läufen: {{known_listings}}\n\nKleinanzeigen:\n{{search_kleinanzeigen}}\n\neBay:\n{{search_ebay}}\n\nIdentifiziere nur neue Angebote. Bewerte Preis, Zustand und Verkäufer-Reputation.", "output_key": "final_result"}
]

Beispiel für einen Analyse-Agent ohne Search:
Input: "Erstelle Fundamentalanalysen für Unternehmen aus der Watchlist"

Output:
[
  {"id": "analyze", "capability": "deep_reasoning", "prompt_template": "Erstelle eine vollständige Fundamentalanalyse für {{trigger_payload.ticker}}: Geschäftsmodell, Marktposition, Bilanzqualität, Management, Wachstumstreiber, Risiken. Schließe mit Kauf/Halten/Verkauf-Empfehlung und Kursziel ab.", "output_key": "final_result"}
]"""


_PIPELINE_CAPABILITIES = {CAPABILITY_SEARCH, CAPABILITY_REASONING, CAPABILITY_DEEP_REASONING, CAPABILITY_CODING, CAPABILITY_MULTIMODAL, CAPABILITY_LONG_CONTEXT}


async def _generate_pipeline(instruction: str, work_capability: str, state_keys: list[str]) -> list[dict] | None:
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
        if not isinstance(parsed, list) or not parsed:
            logger.warning("Pipeline generator returned invalid structure")
            return None
        required_keys = {"id", "capability", "prompt_template", "output_key"}
        for step in parsed:
            if not isinstance(step, dict) or not required_keys.issubset(step.keys()):
                logger.warning("Pipeline step missing required keys: %r", step)
                return None
        logger.info("Pipeline generated with %d steps: %s", len(parsed), [s["id"] for s in parsed])
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
        pipeline = await _generate_pipeline(instruction, work_capability, state_keys)
        if pipeline:
            logger.info("Agent will run with pipeline (%d steps)", len(pipeline))

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
        if pipeline:
            agent_config["pipeline"] = pipeline

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
