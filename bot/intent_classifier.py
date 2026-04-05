from __future__ import annotations
import logging
import asyncpg
from bot import brain

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """Klassifiziere die Nutzeranfrage in genau eine der folgenden Kategorien. Antworte ausschließlich mit dem Kategorie-Namen, kein anderer Text.

Kategorien:
- agent_system: Nutzer beschreibt mehrere koordinierte Aufgaben die zusammen ein System bilden — Sammeln + Analysieren, Beobachten + Melden + Aktualisieren, mehrere abhängige Schritte. Erkennungsmerkmale: mehrere Verben mit impliziter Reihenfolge oder Abhängigkeit, "und dann", "falls", "für jedes gefundene", "lasse laufen".
- agent_create: Nutzer möchte einen einzelnen persistenten Agenten erstellen. Erkennungsmerkmale: "beobachte", "verfolge", "überwache", "halte mich auf dem Laufenden", "melde wenn", "analysiere laufend". Nur wenn es klar eine einzelne Aufgabe ist.
- agent_stop: Nutzer möchte einen laufenden Agenten stoppen oder deaktivieren.
- agent_list: Nutzer möchte explizit eine Liste seiner laufenden Agenten sehen.
- agent_talk: Nutzer spricht direkt mit einem namentlich bekannten Agenten oder fragt nach ihm. Voraussetzung: konkreter Agentenname eines existierenden Agenten. Der Bot-Name selbst ist kein Agentenname.
- task_create: Nutzer möchte eine neue stateless wiederkehrende Aufgabe erstellen. Jeder Lauf ist unabhängig, kein Vergleich mit früheren Ergebnissen.
- task_stop: Nutzer möchte eine wiederkehrende Aufgabe beenden oder löschen.
- task_list: Nutzer möchte seine aktiven Aufgaben sehen.
- none: Keine der obigen Kategorien — normale Unterhaltung, Frage, einmalige Anfrage.

Wichtig: agent_system wenn mehrere abhängige Aufgaben beschrieben werden. agent_create nur wenn eindeutig eine einzelne Aufgabe gemeint ist.

Beispiele:
"Sammle Unternehmen nach Kriterien, analysiere jedes und beobachte News" → agent_system
"Beobachte GPU-Preise, analysiere Funde und halte mich auf dem Laufenden" → agent_system
"Überwache meine Docker Container stündlich" → agent_create
"Erinnere mich jeden Montag an den Standup" → task_create
"Stopp Linus" → agent_stop
"Zeig meine Agenten" → agent_list
"Wie läuft Gordon?" → agent_talk
"Bob, erstelle einen Agenten der täglich..." → agent_create
"Beende den Grafikkarten-Task" → task_stop
"Was läuft gerade" → task_list
"Was denkst du über KI?" → none"""


async def classify(
    text: str,
    pool: asyncpg.Pool,
    has_active_agents: bool = False,
    has_active_tasks: bool = False,
) -> str:
    context_hints: list[str] = []
    if not has_active_agents:
        context_hints.append("Der Nutzer hat keine aktiven Agenten — agent_stop, agent_list und agent_talk sind daher unwahrscheinlich.")
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
        )
        intent = result.strip().lower()
        valid = {"agent_system", "agent_create", "agent_stop", "agent_list", "agent_talk", "task_create", "task_stop", "task_list", "none"}
        if intent not in valid:
            logger.warning("Classifier returned unknown intent %r, falling back to none", intent)
            return "none"
        logger.debug("classify(%r) -> %s", text[:50], intent)
        return intent
    except Exception as e:
        logger.warning("Intent classification failed: %s", e)
        return "none"
