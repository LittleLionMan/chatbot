from __future__ import annotations
import json
import logging
import asyncpg
from bot import brain
from bot.models import CAPABILITY_SIMPLE_TASKS

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """Klassifiziere die Nutzeranfrage in genau eine der folgenden Kategorien. Antworte ausschließlich mit dem Kategorie-Namen, kein anderer Text.

Kategorien:
- agent_system: Nutzer beschreibt mehrere koordinierte Aufgaben die zusammen ein System bilden — Sammeln + Analysieren, Beobachten + Melden + Aktualisieren, mehrere abhängige Schritte. Erkennungsmerkmale: mehrere Verben mit impliziter Reihenfolge oder Abhängigkeit, "und dann", "falls", "für jedes gefundene", "lasse laufen".
- agent_create: Nutzer möchte einen einzelnen persistenten Agenten erstellen. Erkennungsmerkmale: "beobachte", "verfolge", "überwache", "halte mich auf dem Laufenden", "melde wenn", "analysiere laufend". Nur wenn es klar eine einzelne Aufgabe ist.
- agent_trigger: Nutzer möchte einen existierenden Agenten JETZT oder einmalig außer der Reihe ausführen. Erkennungsmerkmale: "jetzt", "sofort", "einmal", "starte", "lauf", "außerhalb des Schedules", "außer der Reihe", Agentenname + expliziter Ausführungsbefehl. Nur wenn klar eine Ausführung gemeint ist — nicht bei Abfragen oder Statusfragen.
- agent_config: Nutzer möchte technische Meta-Eigenschaften eines Agenten ändern die kein Gespräch erfordern — Capability neu klassifizieren, Pipeline generieren oder regenerieren, work_capability direkt setzen. Erkennungsmerkmale: "analysiere seine Capability", "klassifiziere neu", "generiere eine Pipeline", "setze work_capability", "optimiere seinen Workflow".
- agent_stop: Nutzer möchte einen laufenden Agenten stoppen oder deaktivieren.
- agent_list: Nutzer möchte explizit eine Liste seiner laufenden Agenten sehen.
- monitor_create: Nutzer möchte einen Monitor-Service einrichten der einen Agenten automatisch triggert. Erkennungsmerkmale: "richte den RSS-Monitor ein", "richte einen Monitor ein", "überwache News für", "erstelle einen News-Monitor", "richte Benachrichtigungen ein für". Nur wenn explizit ein Monitor-Service gemeint ist, nicht ein normaler Agent.
- agent_talk: Nutzer möchte die Konfiguration oder Instruktion eines Agenten inhaltlich ändern, fragt nach seinem Status, oder möchte gespeicherte Ergebnisse und Daten abfragen. Erkennungsmerkmale: "mach das in Zukunft so", "ändere dein Suchkriterium", "wie läuft X", "was hat X gefunden", "zeig mir", "was weißt du über", "gib mir den Bericht", "wie schätzt du ein", "was ist deine Meinung zu". NICHT wenn eine sofortige Ausführung gemeint ist.
- task_create: Nutzer möchte eine neue stateless wiederkehrende Aufgabe erstellen. Jeder Lauf ist unabhängig, kein Vergleich mit früheren Ergebnissen.
- task_stop: Nutzer möchte eine wiederkehrende Aufgabe beenden oder löschen.
- task_list: Nutzer möchte seine aktiven Aufgaben sehen.
- none: Keine der obigen Kategorien — normale Unterhaltung, Frage, einmalige Anfrage.

Trennlinie agent_trigger vs agent_talk:
- Ausführung jetzt/einmalig → agent_trigger
- Inhaltliche Konfigurationsänderung für zukünftige Läufe → agent_talk
- Technische Meta-Änderung (Capability, Pipeline) → agent_config
- Abfrage von gespeicherten Daten, Status oder Ergebnissen → agent_talk
  ("zeig mir", "was weißt du über", "wie schätzt du ein", "gib mir den Bericht", "was hast du gefunden")

Beispiele:
"Lass Gecko jetzt laufen" → agent_trigger
"Starte den Agenten außer der Reihe" → agent_trigger
"Lauf mal kurz durch" → agent_trigger
"Führe eine einmalige Analyse durch" → agent_trigger
"Wie läuft der Agent?" → agent_talk
"Was hast du bisher gefunden?" → agent_talk
"Zeig mir deinen letzten Bericht" → agent_talk
"Was weißt du über das Thema?" → agent_talk
"Gib mir eine Zusammenfassung deiner Ergebnisse" → agent_talk
"Wie schätzt du die aktuelle Lage ein?" → agent_talk
"Ändere dein Suchkriterium auf Small Caps" → agent_talk
"Analysiere in Zukunft auch Dividendenrendite" → agent_talk
"Analysiere deine Capability neu" → agent_config
"Generiere eine Pipeline" → agent_config
"Setze work_capability auf deep_reasoning" → agent_config
"Sammle Unternehmen nach Kriterien, analysiere jedes und beobachte News" → agent_system
"Überwache meine Docker Container stündlich" → agent_create
"Erinnere mich jeden Montag an den Standup" → task_create
"Stopp den Agenten" → agent_stop
"Zeig meine Agenten" → agent_list
"Was läuft gerade" → task_list
"Was denkst du über KI?" → none"""

_TRIGGER_PAYLOAD_SYSTEM = """Extrahiere aus einer Nutzeranfrage den Agentennamen und alle relevanten Parameter als JSON.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "agent_name": Name des Agenten der ausgeführt werden soll.
- "payload": Dict mit allen relevanten Parametern aus dem Text. Leer wenn keine Parameter genannt werden.

Beispiele:
"Jordan, analysiere BE neu" → {"agent_name": "Jordan", "payload": {"ticker": "BE"}}
"Lass Gecko jetzt laufen" → {"agent_name": "Gecko", "payload": {}}
"Jordan soll EOSE überprüfen, aktueller Kurs $5.70" → {"agent_name": "Jordan", "payload": {"ticker": "EOSE", "reason": "aktueller Kurs $5.70"}}
"Jim Cramer, prüf mal die News zu ORA" → {"agent_name": "Jim Cramer", "payload": {"ticker": "ORA"}}"""

_AGENT_CONFIG_SYSTEM = """Extrahiere aus einer Nutzeranfrage den Agentennamen und welche Konfigurationsoperation gewünscht ist.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "agent_name": Name des Agenten.
- "reclassify_capability": true wenn Capability neu klassifiziert werden soll, false sonst.
- "regenerate_pipeline": true wenn Pipeline neu generiert werden soll, false sonst.
- "set_capability": Direkt gesetzter Capability-Wert wenn der Nutzer einen expliziten Wert nennt (z.B. "chat", "reasoning", "deep_reasoning"), sonst null.

Beispiele:
"Gecko, analysiere deine Capability neu" → {"agent_name": "Gecko", "reclassify_capability": true, "regenerate_pipeline": false, "set_capability": null}
"Jordan, generiere eine Pipeline" → {"agent_name": "Jordan", "reclassify_capability": false, "regenerate_pipeline": true, "set_capability": null}
"Gecko, analysiere Capability und generiere Pipeline" → {"agent_name": "Gecko", "reclassify_capability": true, "regenerate_pipeline": true, "set_capability": null}
"Jordan, setze work_capability auf deep_reasoning" → {"agent_name": "Jordan", "reclassify_capability": false, "regenerate_pipeline": false, "set_capability": "deep_reasoning"}"""


_MONITOR_CREATE_SYSTEM = """Extrahiere aus einer Nutzeranfrage die Parameter für einen neuen Monitor-Service.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "monitor_type": Typ des Monitors. Aktuell verfügbar: "rss"
- "source_agent": Name des Agents dessen State als Watchlist genutzt wird (z.B. "Jordan")
- "source_state_key": State-Key der die zu überwachenden Items enthält (z.B. "analyses_overview")
- "source_format": Format des State-Werts. Werte: "pipe_delimited_overview" (TICKER|NAME|...|DATUM), "comma_list" (kommagetrennte Liste), "pipe_name_map" (TICKER|NAME pro Zeile)
- "target_agent": Name des Agents der getriggert werden soll (z.B. "Jim Cramer")
- "name": Beschreibender Name für diesen Monitor
- "poll_interval_seconds": Polling-Intervall in Sekunden (Standard: 900)

Beispiele:
"Richte den RSS-Monitor für Jim Cramer ein" → {"monitor_type": "rss", "source_agent": "Jordan", "source_state_key": "analyses_overview", "source_format": "pipe_delimited_overview", "target_agent": "Jim Cramer", "name": "Finanz-News für Jim Cramer", "poll_interval_seconds": 900}
"Erstelle einen News-Monitor der Jen-Hsun Huang triggert wenn es News zu seinen Watchlist-Modellen gibt" → {"monitor_type": "rss", "source_agent": "Jen-Hsun Huang", "source_state_key": "watched_models", "source_format": "comma_list", "target_agent": "Jen-Hsun Huang", "name": "GPU-News für Jen-Hsun Huang", "poll_interval_seconds": 900}"""


async def extract_monitor_create_params(text: str, pool: asyncpg.Pool) -> dict:
    try:
        raw = await brain.chat(
            system=_MONITOR_CREATE_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=256,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="monitor_create_extractor",
            pool=pool,
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except Exception as e:
        logger.warning("Monitor create extraction failed: %s", e)
        return {}


async def classify(
    text: str,
    pool: asyncpg.Pool,
    has_active_agents: bool = False,
    has_active_tasks: bool = False,
) -> str:
    context_hints: list[str] = []
    if not has_active_agents:
        context_hints.append("Der Nutzer hat keine aktiven Agenten — agent_stop, agent_list, agent_talk, agent_trigger und agent_config sind daher unwahrscheinlich.")
    if not has_active_tasks:
        context_hints.append("Der Nutzer hat keine aktiven Aufgaben — task_stop und task_list sind daher unwahrscheinlich.")

    content = text
    if context_hints:
        content = "\n".join(context_hints) + "\n\nNutzeranfrage: " + text

    try:
        result = await brain.chat(
            system=_CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": content}],
            max_tokens=20,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="intent_classifier",
            pool=pool,
        )
        intent = result.strip().lower()
        valid = {"agent_system", "agent_create", "agent_trigger", "agent_stop", "agent_list",
                 "agent_talk", "agent_config", "monitor_create",
                 "task_create", "task_stop", "task_list", "none"}
        if intent not in valid:
            logger.warning("Classifier returned unknown intent %r, falling back to none", intent)
            return "none"
        logger.debug("classify(%r) -> %s", text[:50], intent)
        return intent
    except Exception as e:
        logger.warning("Intent classification failed: %s", e)
        return "none"


async def extract_trigger_payload(text: str, pool: asyncpg.Pool) -> dict:
    try:
        raw = await brain.chat(
            system=_TRIGGER_PAYLOAD_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=256,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="trigger_payload_extractor",
            pool=pool,
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {"agent_name": "", "payload": {}}
        return parsed
    except Exception as e:
        logger.warning("Trigger payload extraction failed: %s", e)
        return {"agent_name": "", "payload": {}}


async def extract_agent_config_request(text: str, pool: asyncpg.Pool) -> dict:
    try:
        raw = await brain.chat(
            system=_AGENT_CONFIG_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=128,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="agent_config_extractor",
            pool=pool,
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {"agent_name": "", "reclassify_capability": False, "regenerate_pipeline": False, "set_capability": None}
        return parsed
    except Exception as e:
        logger.warning("Agent config extraction failed: %s", e)
        return {"agent_name": "", "reclassify_capability": False, "regenerate_pipeline": False, "set_capability": None}
