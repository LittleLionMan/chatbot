from __future__ import annotations
import logging
import asyncpg
from bot import brain
from bot.config import BOT_NAME

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = f"""Klassifiziere die Nutzeranfrage in genau eine der folgenden Kategorien. Antworte ausschließlich mit dem Kategorie-Namen, kein anderer Text.

Kategorien:
- agent_create: Nutzer möchte einen neuen persistenten Agenten erstellen der mit Gedächtnis zwischen Läufen arbeitet. Erkennungsmerkmale: "beobachte", "verfolge", "überwache", "halte mich auf dem Laufenden", "melde wenn", "analysiere laufend", "schreib mir regelmäßig", "halte X aktuell".
- agent_stop: Nutzer möchte einen laufenden Agenten stoppen oder deaktivieren. Erkennungsmerkmale: Agentenname + "stopp", "deaktiviere", "soll aufhören".
- agent_list: Nutzer möchte explizit eine Liste seiner laufenden Agenten sehen. Erkennungsmerkmale: "zeig meine Agenten", "welche Agenten laufen", "was für Agenten habe ich".
- agent_talk: Nutzer spricht direkt mit einem namentlich bekannten Agenten oder fragt nach ihm. Voraussetzung: konkreter Agentenname oder eindeutige Referenz wie "dein GPU-Agent". Der Botname selbst ({BOT_NAME}) ist kein Agentenname. Beispiele: "Wie läuft Linus?", "Linus, konzentriere dich auf RTX 4090", "Was hat Gordon beobachtet?". Gegenbeispiel: "Bob, beobachte täglich um 9 Uhr..." -> agent_create.
- task_create: Nutzer möchte eine neue stateless wiederkehrende Aufgabe erstellen. Jeder Lauf ist unabhängig, kein Vergleich mit früheren Ergebnissen. Beispiele: "Erinnere mich jeden Montag", "Schreib täglich um 9 piep", "Such mir jeden Freitag neue Podcast-Folgen".
- task_stop: Nutzer möchte eine wiederkehrende Aufgabe beenden oder löschen.
- task_list: Nutzer möchte seine aktiven Aufgaben sehen.
- none: Keine der obigen Kategorien trifft zu — normale Unterhaltung, Frage, oder einmalige Anfrage.

Wichtig: agent_create und task_create schließen sich gegenseitig aus. Agent wenn Vergleich mit früheren Ergebnissen nötig ist. Task wenn jeder Lauf unabhängig ist.

Beispiele:
"Überwache meine Docker Container stündlich" → agent_create
"Erinnere mich jeden Montag an den Standup" → task_create
"Stopp Linus" → agent_stop
"Zeig meine Agenten" → agent_list
"Wie läuft Gordon?" → agent_talk
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
        valid = {"agent_create", "agent_stop", "agent_list", "agent_talk", "task_create", "task_stop", "task_list", "none"}
        if intent not in valid:
            logger.warning("Classifier returned unknown intent %r, falling back to none", intent)
            return "none"
        logger.debug("classify(%r) -> %s", text[:50], intent)
        return intent
    except Exception as e:
        logger.warning("Intent classification failed: %s", e)
        return "none"
