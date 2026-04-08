from __future__ import annotations
import json
import logging
import asyncpg
from bot import brain
from bot.models import CAPABILITY_FAST

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """Klassifiziere die Nutzeranfrage in genau eine der folgenden Kategorien. Antworte ausschließlich mit dem Kategorie-Namen, kein anderer Text.

Kategorien:
- agent_system: Nutzer beschreibt mehrere koordinierte Aufgaben die zusammen ein System bilden — Sammeln + Analysieren, Beobachten + Melden + Aktualisieren, mehrere abhängige Schritte. Erkennungsmerkmale: mehrere Verben mit impliziter Reihenfolge oder Abhängigkeit, "und dann", "falls", "für jedes gefundene", "lasse laufen".
- agent_create: Nutzer möchte einen einzelnen persistenten Agenten erstellen. Erkennungsmerkmale: "beobachte", "verfolge", "überwache", "halte mich auf dem Laufenden", "melde wenn", "analysiere laufend". Nur wenn es klar eine einzelne Aufgabe ist.
- agent_trigger: Nutzer möchte einen existierenden Agenten JETZT oder einmalig außer der Reihe ausführen — unabhängig davon ob ein konkreter Auftrag dabei ist. Erkennungsmerkmale: "jetzt", "sofort", "einmal", "starte", "lauf", "außerhalb des Schedules", "außer der Reihe", Agentenname + einmalige Aufgabe oder direkter Ausführungsbefehl. Auch ohne explizites "jetzt" wenn klar eine Ausführung gemeint ist.
- agent_config: Nutzer möchte technische Meta-Eigenschaften eines Agenten ändern die kein Gespräch erfordern — Capability neu klassifizieren, Pipeline generieren oder regenerieren, work_capability direkt setzen. Erkennungsmerkmale: "analysiere seine Capability", "klassifiziere neu", "generiere eine Pipeline", "setze work_capability", "optimiere seinen Workflow".
- agent_stop: Nutzer möchte einen laufenden Agenten stoppen oder deaktivieren.
- agent_list: Nutzer möchte explizit eine Liste seiner laufenden Agenten sehen.
- agent_talk: Nutzer möchte die Konfiguration oder Instruktion eines Agenten inhaltlich ändern, oder fragt nach seinem Status. Erkennungsmerkmale: "mach das in Zukunft so", "ändere dein Suchkriterium", "analysiere in Zukunft auch X", "wie läuft X", "was hat X gefunden", "zeig mir den Status". NICHT wenn eine sofortige Ausführung gemeint ist.
- task_create: Nutzer möchte eine neue stateless wiederkehrende Aufgabe erstellen. Jeder Lauf ist unabhängig, kein Vergleich mit früheren Ergebnissen.
- task_stop: Nutzer möchte eine wiederkehrende Aufgabe beenden oder löschen.
- task_list: Nutzer möchte seine aktiven Aufgaben sehen.
- none: Keine der obigen Kategorien — normale Unterhaltung, Frage, einmalige Anfrage.

Trennlinie agent_trigger vs agent_talk:
- Ausführung jetzt/einmalig → agent_trigger
- Inhaltliche Konfigurationsänderung für zukünftige Läufe → agent_talk
- Technische Meta-Änderung (Capability, Pipeline) → agent_config

Beispiele:
"Jordan, analysiere BE neu" → agent_trigger
"Lass Gecko jetzt laufen" → agent_trigger
"Gecko, starte außerhalb des Schedules" → agent_trigger
"Jordan soll EOSE überprüfen, aktueller Kurs $5.70" → agent_trigger
"Gecko, starte eine Suche genau jetzt" → agent_trigger
"Wie läuft Jordan?" → agent_talk
"Jordan, ändere dein Suchkriterium auf Small Caps" → agent_talk
"Jordan, analysiere in Zukunft auch Dividendenrendite" → agent_talk
"Gecko, analysiere deine Capability neu" → agent_config
"Jordan, generiere eine Pipeline" → agent_config
"Gecko, setze work_capability auf search" → agent_config
"Sammle Unternehmen nach Kriterien, analysiere jedes und beobachte News" → agent_system
"Überwache meine Docker Container stündlich" → agent_create
"Erinnere mich jeden Montag an den Standup" → task_create
"Stopp Linus" → agent_stop
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
- "set_capability": Direkt gesetzter Capability-Wert wenn der Nutzer einen expliziten Wert nennt (z.B. "search", "reasoning", "deep_reasoning"), sonst null.

Beispiele:
"Gecko, analysiere deine Capability neu" → {"agent_name": "Gecko", "reclassify_capability": true, "regenerate_pipeline": false, "set_capability": null}
"Jordan, generiere eine Pipeline" → {"agent_name": "Jordan", "reclassify_capability": false, "regenerate_pipeline": true, "set_capability": null}
"Gecko, analysiere Capability und generiere Pipeline" → {"agent_name": "Gecko", "reclassify_capability": true, "regenerate_pipeline": true, "set_capability": null}
"Jordan, setze work_capability auf deep_reasoning" → {"agent_name": "Jordan", "reclassify_capability": false, "regenerate_pipeline": false, "set_capability": "deep_reasoning"}"""


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
            capability=CAPABILITY_FAST,
            caller="intent_classifier",
            pool=pool,
        )
        intent = result.strip().lower()
        valid = {"agent_system", "agent_create", "agent_trigger", "agent_stop", "agent_list",
                 "agent_talk", "agent_config", "task_create", "task_stop", "task_list", "none"}
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
            capability=CAPABILITY_FAST,
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
            capability=CAPABILITY_FAST,
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
