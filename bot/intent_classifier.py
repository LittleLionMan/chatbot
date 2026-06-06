from __future__ import annotations

import json
import logging
from typing import TypedDict

import asyncpg

from bot import brain
from bot.models import CAPABILITY_SIMPLE_TASKS
from bot.utils import clean_llm_json

logger = logging.getLogger(__name__)


class ClassifiedIntent(TypedDict):
    intent: str
    needs_search: bool
    wants_voice: bool


_CLASSIFIER_SYSTEM = """Klassifiziere die Nutzeranfrage. Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "intent": genau eine der Kategorien unten
- "needs_search": true wenn aktuelle Informationen aus dem Internet nötig sind — Preise, Kurse, Nachrichten, Wetter, aktuelle Ereignisse. False bei Meinungen, Konzepten, internen Abfragen.
- "wants_voice": true wenn der User explizit eine Sprachantwort fordert.

Intent-Kategorien — nur für explizite Aktionen die Bob nicht aus dem Gesprächskontext ableiten kann:

"agent_create" — User möchte einen neuen persistenten Agenten erstellen der nach Plan läuft oder auf Trigger reagiert. Eindeutige Signale: Zeitplan ("täglich", "jede Stunde", "wenn X passiert"), Monitoring-Aufgaben, Wörter wie "Agent", "überwache", "beobachte dauerhaft", "richte ein".

"agent_trigger" — User möchte etwas JETZT ausführen oder explizit stoppen. Signale: "starte jetzt", "führe aus", "stopp", "deaktiviere", "einmalig ausführen".

"scraper_create" — User möchte einen Scraper für eine externe Plattform einrichten. Signale: Plattformnamen (Kleinanzeigen, eBay, Immoscout etc.), "scrape", "neue Listings", "bei neuen Angeboten".

"monitor_create" — User möchte einen RSS- oder Feed-Monitor einrichten. Signale: "RSS", "Feed", "überwache News", "bei neuen Artikeln".

"task_create" — User möchte eine neue zustandslose wiederkehrende Aufgabe. Signale: klarer Zeitplan, aber keine Persistenz oder Gedächtnis zwischen Läufen nötig.

"task_stop" — User möchte eine bestehende wiederkehrende Aufgabe beenden.

"none" — alles andere. Normales Gespräch, Fragen, Reaktionen auf Agent-Outputs, Feedback, Statusabfragen, Konfigurationsänderungen, Umbenennung, Datenabruf — Bob löst das selbst aus dem Kontext.

Wichtigste Trennlinie:
"none" ist der Default. Nur wenn der User eine der sechs Aktionen oben EXPLIZIT anfordert, wird ein anderer Intent gesetzt. Im Zweifel: none."""

_VALID_INTENTS = {
    "agent_create",
    "agent_trigger",
    "scraper_create",
    "monitor_create",
    "task_create",
    "task_stop",
    "none",
}


async def classify(
    text: str,
    pool: asyncpg.Pool,
    has_active_agents: bool = False,
    has_active_tasks: bool = False,
) -> ClassifiedIntent:
    hints: list[str] = []
    if not has_active_agents:
        hints.append(
            "Der Nutzer hat keine aktiven Agenten — agent_trigger ist daher unwahrscheinlich."
        )
    if not has_active_tasks:
        hints.append(
            "Der Nutzer hat keine aktiven Aufgaben — task_stop ist daher unwahrscheinlich."
        )

    content = text
    if hints:
        content = "\n".join(hints) + "\n\nNutzeranfrage: " + text

    try:
        raw = await brain.chat(
            system=_CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="intent_classifier",
            pool=pool,
        )
        parsed = json.loads(clean_llm_json(raw))
        intent = parsed.get("intent", "none").strip().lower()
        if intent not in _VALID_INTENTS:
            logger.warning(
                "classifier returned unknown intent %r, falling back to none", intent
            )
            intent = "none"
        result: ClassifiedIntent = {
            "intent": intent,
            "needs_search": bool(parsed.get("needs_search", False)),
            "wants_voice": bool(parsed.get("wants_voice", False)),
        }
        logger.debug("classify(%r) → %s", text[:60], intent)
        return result
    except Exception as e:
        logger.warning("intent classification failed: %s", e)
        return {"intent": "none", "needs_search": True, "wants_voice": False}


_TRIGGER_PAYLOAD_SYSTEM = """Extrahiere aus einer Nutzeranfrage den Agentennamen und alle relevanten Parameter.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "agent_name": Name des Agenten.
- "action": "run" wenn ausführen, "stop" wenn stoppen.
- "payload": Dict mit allen inhaltlichen Parametern. Leer wenn keine genannt."""


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
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            return {"agent_name": "", "action": "run", "payload": {}}
        return parsed
    except Exception as e:
        logger.warning("trigger payload extraction failed: %s", e)
        return {"agent_name": "", "action": "run", "payload": {}}


_MONITOR_CREATE_SYSTEM = """Extrahiere aus einer Nutzeranfrage die Parameter für einen neuen RSS-Monitor.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Zwei Modi:

1. "static" — überwacht feste RSS-Feed-URLs:
{"monitor_type": "rss", "source": "static", "name": "...", "target_agent": "...", "feed_urls": ["https://..."], "keywords": [], "poll_interval_seconds": 3600}

2. "agent" — generiert Feed-URLs aus einer Watchlist im State eines anderen Agenten:
{"monitor_type": "rss", "source": "agent", "name": "...", "source_agent": "...", "source_state_key": "...", "source_format": "comma_list", "target_agent": "...", "feed_templates": ["https://...{query}..."], "keywords": [], "poll_interval_seconds": 900}"""


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
        parsed = json.loads(clean_llm_json(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        logger.warning("monitor create extraction failed: %s", e)
        return {}


_SCRAPER_CREATE_SYSTEM = """Extrahiere aus einer Nutzeranfrage die Parameter für einen neuen Scraper.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Verfügbare Plattformen: kleinanzeigen, ebay, reddit, immoscout, wggesucht, stepstone, linkedin

Felder:
- "platforms": Liste der Plattformen. Wenn keine explizit genannt: Gebrauchtwaren → ["kleinanzeigen", "ebay"], Wohnungen → ["immoscout", "wggesucht"], Jobs → ["stepstone", "linkedin"].
- "category": Kurzes Schlagwort (gpu, apartment, job, vehicle, furniture).
- "query": Optimierter Suchbegriff, 1-5 Wörter.
- "filters": Dict mit optionalen Filtern (price_min, price_max, location, city, rooms_min, sqm_min). Nur wenn explizit genannt.
- "target_agent": Name des Agenten der getriggert wird.
- "poll_interval_seconds": Standard 3600."""


async def extract_scraper_create_params(text: str, pool: asyncpg.Pool) -> dict:
    try:
        raw = await brain.chat(
            system=_SCRAPER_CREATE_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=256,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="scraper_create_extractor",
            pool=pool,
        )
        parsed = json.loads(clean_llm_json(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        logger.warning("scraper create extraction failed: %s", e)
        return {}
