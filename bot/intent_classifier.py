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
    edit_type: str | None


_CLASSIFIER_SYSTEM = """Klassifiziere die Nutzeranfrage. Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "intent": genau eine der Kategorien unten
- "needs_search": true wenn die Antwort aktuelle Informationen aus dem Internet erfordert — Preise, Kurse, Nachrichten, Wetter, aktuelle Ereignisse, Fakten die sich ändern. False bei internen Abfragen, Meinungen, Konzepten, Agent-Status.
- "wants_voice": true wenn der User explizit eine Sprachantwort anfordert ("vorlesen", "sprich", "red mal", "antworte als Sprachnachricht"). Sonst false.
- "edit_type": Nur bei agent_feedback. Einer von: "data_edit", "step_patch", "preference", null.

Intent-Kategorien:
- "agent_system": User beschreibt mehrere koordinierte Aufgaben die zusammen ein System bilden — mehrere abhängige Schritte, "sammle und analysiere dann", "für jedes gefundene", implizite Reihenfolge zwischen Agents.
- "agent_create": User möchte einen einzelnen persistenten Agenten erstellen der nach Plan läuft oder auf Trigger reagiert. Erkennungsmerkmale: "beobachte", "verfolge", "überwache", "halte mich auf dem Laufenden", "melde wenn", "analysiere laufend".
- "agent_trigger": User möchte etwas an einem Agenten JETZT ausführen — starten, stoppen, oder einmalig außer der Reihe ausführen. Erkennungsmerkmale: "jetzt", "sofort", "einmal", "starte", "lauf", "stopp", "deaktiviere", "außer der Reihe".
- "agent_talk": Alles andere rund um einen bestehenden Agenten — Statusabfragen, gespeicherte Daten abrufen, Konfiguration ändern, Pipeline neu generieren oder erstellen. Erkennungsmerkmale: "wie läuft", "was hat gefunden", "zeig mir", "ändere", "mach in Zukunft", "generiere Pipeline", "erstelle Pipeline für", "neue Pipeline für".
- "agent_feedback": User reagiert auf einen Agent-Output oder gibt Feedback zu einem konkreten Ergebnis. Erkennungsmerkmale: Ablehnung oder Korrektur eines Ergebnisses ("das nicht weil", "das ist falsch", "filtere raus", "nicht interessant"), Bezug auf ein konkretes Listing/Dokument/Analyse das ein Agent produziert hat, Edit-Wunsch an gespeicherten Inhalten oder Pipeline-Verhalten. edit_type: "data_edit" wenn gespeicherter Inhalt geändert werden soll, "step_patch" wenn Pipeline-Verhalten korrigiert werden soll, "preference" wenn ein neues Kriterium/Constraint formuliert wird, null wenn unklar.
- "agent_list": User möchte explizit eine Liste seiner laufenden Agenten sehen.
- "task_create": User möchte eine neue stateless wiederkehrende Aufgabe erstellen.
- "task_stop": User möchte eine wiederkehrende Aufgabe beenden.
- "task_list": User möchte seine aktiven Aufgaben sehen.
- "scraper_create": User möchte einen Scraper einrichten der eine externe Plattform kontinuierlich nach Listings durchsucht und einen Agenten bei neuen Funden triggert.
- "monitor_create": User möchte einen RSS-Monitor einrichten der Feeds überwacht und einen Agenten bei neuen Artikeln triggert. Erkennungsmerkmale: "überwache", "feed", "rss", "reddit", "monitor", "beobachte news".
- "none": Normale Unterhaltung, Frage, einmalige Anfrage ohne Zeitplan.

Trennlinien:
- agent_trigger vs agent_talk: Ausführung/Stopp JETZT → trigger. Änderung für zukünftige Läufe oder Abfrage → talk.
- agent_talk vs agent_feedback: agent_talk wenn nach Daten gefragt wird ("zeig mir", "was hat gefunden"). agent_feedback wenn auf einen konkreten Output reagiert wird ("das ist falsch", "das nicht weil", "filtere X raus").
- agent_talk vs agent_create: "Pipeline erstellen/generieren für [Name]" → immer agent_talk. agent_create nur wenn völlig neuer Agent ohne bestehenden Namen.
- agent_create vs task_create: Agent hat State, erinnert sich, vergleicht. Task ist zustandslos.
- agent_create vs scraper_create: Scraper durchsucht externe Plattformen. Agent wertet aus.
- agent_system vs agent_create: Mehrere abhängige Agents → system. Einer → create.

Beispiele:
{"intent": "agent_trigger", "needs_search": false, "wants_voice": false, "edit_type": null} # "Lass Jordan jetzt laufen"
{"intent": "agent_trigger", "needs_search": false, "wants_voice": false, "edit_type": null} # "Stopp den Agenten"
{"intent": "agent_talk", "needs_search": false, "wants_voice": false, "edit_type": null} # "Was hat Jordan bisher gefunden?"
{"intent": "agent_talk", "needs_search": false, "wants_voice": false, "edit_type": null} # "Ändere Jordans Suchkriterium auf Small Caps"
{"intent": "agent_talk", "needs_search": false, "wants_voice": false, "edit_type": null} # "Generiere eine neue Pipeline für Jordan"
{"intent": "agent_feedback", "needs_search": false, "wants_voice": false, "edit_type": "preference"} # "Das hat keinen Balkon, nicht interessant"
{"intent": "agent_feedback", "needs_search": false, "wants_voice": false, "edit_type": "step_patch"} # "Tauschgeschäfte sollen rausgefiltert werden Linus"
{"intent": "agent_feedback", "needs_search": false, "wants_voice": false, "edit_type": "data_edit"} # "Entferne diesen Satz aus der Analyse Jordan"
{"intent": "agent_create", "needs_search": false, "wants_voice": false, "edit_type": null} # "Überwache meine Docker Container stündlich"
{"intent": "agent_system", "needs_search": false, "wants_voice": false, "edit_type": null} # "Sammle täglich Unternehmen nach Kriterien, analysiere jeden Fund dann einzeln"
{"intent": "task_create", "needs_search": false, "wants_voice": false, "edit_type": null} # "Erinnere mich jeden Montag an den Standup"
{"intent": "scraper_create", "needs_search": false, "wants_voice": false, "edit_type": null} # "Richte einen Scraper auf Kleinanzeigen ein der GPUs sucht und Linus triggert"
{"intent": "none", "needs_search": true, "wants_voice": false, "edit_type": null} # "Was kostet Bitcoin gerade?"
{"intent": "none", "needs_search": false, "wants_voice": true, "edit_type": null} # "Kannst du das vorlesen?"
{"intent": "agent_list", "needs_search": false, "wants_voice": false, "edit_type": null} # "Zeig meine Agenten" """

_VALID_INTENTS = {
    "agent_system", "agent_create", "agent_trigger", "agent_talk", "agent_feedback",
    "task_create", "task_stop", "task_list", "agent_list",
    "scraper_create", "monitor_create", "none",
}

_DATA_EDIT_SIGNALS = {"entferne", "lösche", "ändere den satz", "streiche", "korrigiere den", "ersetze den"}
_STEP_PATCH_SIGNALS = {"filtere", "rausfiltern", "nicht einbeziehen", "dieser step", "der step", "pipeline"}
_PREFERENCE_SIGNALS = {"nicht interessant", "kein ", "keine ", "zu klein", "zu groß", "ohne ", "das nicht"}

_TRIGGER_PAYLOAD_SYSTEM = """Extrahiere aus einer Nutzeranfrage den Agentennamen und alle relevanten Parameter als JSON.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "agent_name": Name des Agenten.
- "action": "run" wenn der Agent ausgeführt werden soll, "stop" wenn er gestoppt werden soll.
- "payload": Dict mit allen relevanten Parametern aus dem Text. Leer wenn keine Parameter genannt werden.

Beispiele:
"Jordan, analysiere BE neu" → {"agent_name": "Jordan", "action": "run", "payload": {"ticker": "BE"}}
"Lass Gecko jetzt laufen" → {"agent_name": "Gecko", "action": "run", "payload": {}}
"Stopp den Scout" → {"agent_name": "Scout", "action": "stop", "payload": {}}"""

_AGENT_TALK_EXTRACTION_SYSTEM = """Extrahiere aus einer Nutzeranfrage den Agentennamen und welche Art von Anfrage gestellt wird.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "agent_name": Name des Agenten.
- "talk_type": Art der Anfrage.
  - "query": Statusabfrage oder Datenabruf ("wie läuft", "was hat gefunden", "zeig mir", "gib mir den Bericht")
  - "config_change": Inhaltliche Konfigurationsänderung ("ändere Suchkriterium", "mach in Zukunft", "fokussiere auf", "passe Instruction an")
  - "regenerate_pipeline": Pipeline neu generieren oder erstellen ("generiere Pipeline neu", "neue Pipeline", "pipeline regenerieren", "erstelle Pipeline", "erstelle eine neue Pipeline")
  - "rename": Umbenennung des Agenten

Beispiele:
"Wie läuft Jordan?" → {"agent_name": "Jordan", "talk_type": "query"}
"Was hat Scout gefunden?" → {"agent_name": "Scout", "talk_type": "query"}
"Jordan, ändere dein Suchkriterium auf Small Caps" → {"agent_name": "Jordan", "talk_type": "config_change"}
"Jordan, generiere eine neue Pipeline" → {"agent_name": "Jordan", "talk_type": "regenerate_pipeline"}
"Generiere Linus' Pipeline neu" → {"agent_name": "Linus", "talk_type": "regenerate_pipeline"}
"Benenne Scout in Hermes um" → {"agent_name": "Scout", "talk_type": "rename"}"""

_MONITOR_CREATE_SYSTEM = """Extrahiere aus einer Nutzeranfrage die Parameter für einen neuen RSS-Monitor.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Zwei Modi:

1. "static" — überwacht feste RSS-Feed-URLs direkt:
{
  "monitor_type": "rss",
  "source": "static",
  "name": "Beschreibender Name",
  "target_agent": "Name des Agents der getriggert werden soll",
  "feed_urls": ["https://...", "https://..."],
  "keywords": ["Keyword1", "Keyword2"],
  "poll_interval_seconds": 3600
}
keywords: optionale Filterliste — nur Artikel die mind. ein Keyword in Titel oder Text enthalten triggern den Agent. Leer = alle Artikel.

2. "agent" — generiert Feed-URLs dynamisch aus einer Watchlist im State eines anderen Agents:
{
  "monitor_type": "rss",
  "source": "agent",
  "name": "Beschreibender Name",
  "source_agent": "Name des Agents dessen State als Watchlist genutzt wird",
  "source_state_key": "State-Key der die Watchlist enthält",
  "source_format": "comma_list|pipe_delimited_overview|pipe_name_map",
  "target_agent": "Name des Agents der getriggert werden soll",
  "feed_templates": ["https://news.google.com/rss/search?q={query}"],
  "keywords": [],
  "poll_interval_seconds": 900
}

Erkennungsmerkmale für static: explizite Feed-URLs genannt, oder Suchwörter/Subbedits ohne Agenten-Watchlist.
Erkennungsmerkmale für agent: Watchlist aus einem anderen Agent, dynamische Suche basierend auf Liste.

Beispiele:
"Beobachte [Thema] und triggere [Agent] wenn neue Artikel erscheinen" →
{"monitor_type": "rss", "source": "static", "name": "...", "target_agent": "...", "feed_urls": ["https://..."], "keywords": ["..."], "poll_interval_seconds": 3600}

"Überwache News zu den Einträgen in [Agent]s Liste und triggere ihn" →
{"monitor_type": "rss", "source": "agent", "name": "...", "source_agent": "...", "source_state_key": "...", "source_format": "comma_list", "target_agent": "...", "feed_templates": ["https://...search?q={query}"], "keywords": [], "poll_interval_seconds": 900}"""

_SCRAPER_CREATE_SYSTEM = """Extrahiere aus einer Nutzeranfrage die Parameter für einen neuen Scraper-Config.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Verfügbare Plattformen: kleinanzeigen, ebay, reddit, immoscout, wggesucht, stepstone, linkedin

Felder:
- "platforms": Liste der zu durchsuchenden Plattformen. Wenn keine explizit genannt: für GPUs/Hardware ["kleinanzeigen", "ebay", "reddit"], für Wohnungen ["immoscout", "wggesucht"], für Jobs ["stepstone", "linkedin"].
- "category": Kurzes Schlagwort für die Kategorie. Beispiele: "gpu", "apartment", "job", "bike", "furniture".
- "query": Optimierte Suchanfrage für die Plattformen (1-5 Wörter).
- "filters": Dict mit optionalen Filtern. Mögliche Keys: price_min, price_max, location, city, rooms_min, sqm_min.
- "target_agent": Name des Agenten der bei neuen Listings getriggert werden soll.
- "poll_interval_seconds": Wie oft gescraped werden soll. Standard: 3600 (1h). Für zeitkritische Suchen: 1800 (30min).

Beispiele:
"Richte einen Scraper auf Kleinanzeigen und eBay ein der RTX 4090 sucht und Linus triggert" →
{"platforms": ["kleinanzeigen", "ebay"], "category": "gpu", "query": "RTX 4090", "filters": {}, "target_agent": "Linus", "poll_interval_seconds": 3600}

"Durchsuche Immoscout und WG-Gesucht stündlich nach 2-Zimmer-Wohnungen in München unter 1500€ für meinen Wohnungs-Agent" →
{"platforms": ["immoscout", "wggesucht"], "category": "apartment", "query": "2 Zimmer München", "filters": {"price_max": 1500, "city": "münchen", "rooms_min": 2}, "target_agent": "Wohnungs-Agent", "poll_interval_seconds": 3600}"""


def _infer_edit_type(text: str) -> str | None:
    text_lower = text.lower()
    if any(s in text_lower for s in _DATA_EDIT_SIGNALS):
        return "data_edit"
    if any(s in text_lower for s in _STEP_PATCH_SIGNALS):
        return "step_patch"
    if any(s in text_lower for s in _PREFERENCE_SIGNALS):
        return "preference"
    return None


async def classify(
    text: str,
    pool: asyncpg.Pool,
    has_active_agents: bool = False,
    has_active_tasks: bool = False,
    notification_context: dict | None = None,
) -> ClassifiedIntent:
    if notification_context and notification_context.get("notification_type") not in (
        "confirmation", "adjust_request"
    ):
        edit_type = _infer_edit_type(text)
        return {
            "intent": "agent_feedback",
            "needs_search": False,
            "wants_voice": False,
            "edit_type": edit_type,
        }

    context_hints: list[str] = []
    if not has_active_agents:
        context_hints.append("Der Nutzer hat keine aktiven Agenten — agent_trigger, agent_talk, agent_feedback und agent_list sind daher unwahrscheinlich.")
    if not has_active_tasks:
        context_hints.append("Der Nutzer hat keine aktiven Aufgaben — task_stop und task_list sind daher unwahrscheinlich.")

    content = text
    if context_hints:
        content = "\n".join(context_hints) + "\n\nNutzeranfrage: " + text

    try:
        raw = await brain.chat(
            system=_CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": content}],
            max_tokens=200,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="intent_classifier",
            pool=pool,
        )
        parsed = json.loads(clean_llm_json(raw))
        intent = parsed.get("intent", "none").strip().lower()
        if intent not in _VALID_INTENTS:
            logger.warning("classifier returned unknown intent %r, falling back to none", intent)
            intent = "none"
        edit_type: str | None = None
        if intent == "agent_feedback":
            edit_type = parsed.get("edit_type") or _infer_edit_type(text)
        result: ClassifiedIntent = {
            "intent": intent,
            "needs_search": bool(parsed.get("needs_search", False)),
            "wants_voice": bool(parsed.get("wants_voice", False)),
            "edit_type": edit_type,
        }
        logger.debug("classify(%r) → %s edit_type=%s", text[:50], intent, edit_type)
        return result
    except Exception as e:
        logger.warning("intent classification failed: %s", e)
        return {"intent": "none", "needs_search": True, "wants_voice": False, "edit_type": None}


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


async def extract_agent_talk(text: str, pool: asyncpg.Pool) -> dict:
    try:
        raw = await brain.chat(
            system=_AGENT_TALK_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=128,
            capability=CAPABILITY_SIMPLE_TASKS,
            caller="agent_talk_extractor",
            pool=pool,
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            return {"agent_name": "", "talk_type": "query"}
        return parsed
    except Exception as e:
        logger.warning("agent talk extraction failed: %s", e)
        return {"agent_name": "", "talk_type": "query"}


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
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except Exception as e:
        logger.warning("monitor create extraction failed: %s", e)
        return {}


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
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except Exception as e:
        logger.warning("scraper create extraction failed: %s", e)
        return {}
