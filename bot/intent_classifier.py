from __future__ import annotations
import logging
import asyncpg
from bot import brain

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """Klassifiziere die Nutzeranfrage in genau eine der folgenden Kategorien. Antworte ausschließlich mit dem Kategorie-Namen, kein anderer Text.

Kategorien:
- agent_system: Nutzer beschreibt mehrere koordinierte Aufgaben die zusammen ein System bilden — Sammeln + Analysieren, Beobachten + Melden + Aktualisieren, mehrere abhängige Schritte. Erkennungsmerkmale: mehrere Verben mit impliziter Reihenfolge oder Abhängigkeit, "und dann", "falls", "für jedes gefundene", "lasse laufen".
- agent_create: Nutzer möchte einen einzelnen persistenten Agenten erstellen. Erkennungsmerkmale: "beobachte", "verfolge", "überwache", "halte mich auf dem Laufenden", "melde wenn", "analysiere laufend". Nur wenn es klar eine einzelne Aufgabe ist.
- agent_trigger: Nutzer möchte einen existierenden Agenten jetzt sofort oder einmalig außer der Reihe ausführen — mit oder ohne spezifischen Auftrag. Erkennungsmerkmale: Agentenname + einmalige Aufgabe oder direkter Befehl, "analysiere jetzt", "prüfe mal", "lauf einmal durch", "schau dir X an", "überprüfe Y für Z".
- agent_stop: Nutzer möchte einen laufenden Agenten stoppen oder deaktivieren.
- agent_list: Nutzer möchte explizit eine Liste seiner laufenden Agenten sehen.
- agent_talk: Nutzer spricht direkt mit einem namentlich bekannten Agenten oder fragt nach ihm — Statusabfrage oder Konfigurationsänderung, aber kein einmaliger Ausführungsauftrag.
- task_create: Nutzer möchte eine neue stateless wiederkehrende Aufgabe erstellen. Jeder Lauf ist unabhängig, kein Vergleich mit früheren Ergebnissen.
- task_stop: Nutzer möchte eine wiederkehrende Aufgabe beenden oder löschen.
- task_list: Nutzer möchte seine aktiven Aufgaben sehen.
- none: Keine der obigen Kategorien — normale Unterhaltung, Frage, einmalige Anfrage.

Wichtig: agent_trigger wenn der Nutzer einen Agent einmalig mit einer konkreten Aufgabe beauftragen will. agent_talk wenn er nur Status oder Konfiguration will.

Beispiele:
"Jordan, analysiere BE neu" → agent_trigger
"Lass Gecko jetzt laufen" → agent_trigger
"Jordan soll EOSE überprüfen, aktueller Kurs $5.70" → agent_trigger
"Wie läuft Jordan?" → agent_talk
"Jordan, ändere dein Suchkriterium" → agent_talk
"Sammle Unternehmen nach Kriterien, analysiere jedes und beobachte News" → agent_system
"Überwache meine Docker Container stündlich" → agent_create
"Erinnere mich jeden Montag an den Standup" → task_create
"Stopp Linus" → agent_stop
"Zeig meine Agenten" → agent_list
"Bob, erstelle einen Agenten der täglich..." → agent_create
"Beende den Grafikkarten-Task" → task_stop
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


async def classify(
    text: str,
    pool: asyncpg.Pool,
    has_active_agents: bool = False,
    has_active_tasks: bool = False,
) -> str:
    context_hints: list[str] = []
    if not has_active_agents:
        context_hints.append("Der Nutzer hat keine aktiven Agenten — agent_stop, agent_list, agent_talk und agent_trigger sind daher unwahrscheinlich.")
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
            caller="intent_classifier",
            pool=pool,
            capability=CAPABILITY_FAST,
        )
        intent = result.strip().lower()
        valid = {"agent_system", "agent_create", "agent_trigger", "agent_stop", "agent_list", "agent_talk", "task_create", "task_stop", "task_list", "none"}
        if intent not in valid:
            logger.warning("Classifier returned unknown intent %r, falling back to none", intent)
            return "none"
        logger.debug("classify(%r) -> %s", text[:50], intent)
        return intent
    except Exception as e:
        logger.warning("Intent classification failed: %s", e)
        return "none"


async def extract_trigger_payload(text: str, pool: asyncpg.Pool) -> dict:
    import json
    from bot.utils import clean_llm_json
    try:
        raw = await brain.chat(
            system=_TRIGGER_PAYLOAD_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=256,
            caller="trigger_payload_extractor",
            pool=pool,
            capability=CAPABILITY_FAST,
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            return {"agent_name": "", "payload": {}}
        return parsed
    except Exception as e:
        logger.warning("Trigger payload extraction failed: %s", e)
        return {"agent_name": "", "payload": {}}
