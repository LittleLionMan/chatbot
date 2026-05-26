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
- "wants_voice": true wenn der User explizit eine Sprachantwort anfordert. Sonst false.
- "edit_type": Nur bei agent_feedback. Einer von: "data_edit", "step_patch", "preference", null.

Intent-Kategorien:
- "agent_system": User beschreibt mehrere koordinierte Aufgaben die zusammen ein System bilden — mehrere abhängige Schritte, implizite Reihenfolge zwischen Agents, "für jedes gefundene X tue Y".
- "agent_create": User möchte einen einzelnen neuen persistenten Agenten erstellen der nach Plan läuft oder auf Trigger reagiert. Nur wenn kein bestehender Agent gemeint ist.
- "agent_trigger": User möchte etwas JETZT ausführen — starten, stoppen, einmalig außer der Reihe. Zeitlich unmittelbar.
- "agent_talk": Alles rund um einen bestehenden Agenten ohne konkreten Output-Bezug — Statusabfragen, Daten abrufen, Konfiguration ändern, Pipeline neu generieren.
- "agent_feedback": User reagiert auf einen konkreten Agent-Output — korrigiert, bewertet, lehnt ab, formuliert einen Änderungswunsch der aus diesem Output hervorgeht.
- "agent_list": User möchte explizit alle laufenden Agenten sehen.
- "task_create": User möchte eine neue zustandslose wiederkehrende Aufgabe erstellen.
- "task_stop": User möchte eine wiederkehrende Aufgabe beenden.
- "task_list": User möchte aktive Aufgaben sehen.
- "scraper_create": User möchte einen Scraper einrichten der eine externe Plattform durchsucht und einen Agenten triggert.
- "monitor_create": User möchte einen RSS- oder Feed-Monitor einrichten.
- "none": Normale Unterhaltung, Frage, Reaktion ohne Änderungswunsch, einmalige Anfrage.

Trennlinien — diese sind entscheidend:

agent_trigger vs agent_talk:
Unmittelbare Ausführung oder Stopp → trigger. Abfrage oder Änderung für zukünftige Läufe → talk.

agent_talk vs agent_feedback:
agent_talk: User fragt nach dem Agenten selbst — seinem State, seinen Daten, seiner Konfiguration. Kein Bezug auf einen konkreten Output.
agent_feedback: User reagiert auf etwas das der Agent produziert hat — ein Ergebnis, ein Angebot, einen Bericht. Der Output ist der Ausgangspunkt.

agent_feedback edit_type — die drei Typen unterscheiden sich durch die Art des Änderungswunsches:
"data_edit": User möchte etwas in einem gespeicherten Dokument oder Datensatz des Agenten ändern — konkrete Inhalte, nicht Verhalten. Erkennbar daran dass eine bestimmte gespeicherte Information falsch oder veraltet ist.
"step_patch": User beschreibt ein systematisches Verhaltensproblem des Agenten das bei vielen Inputs auftreten würde — der Agent tut generell etwas falsch, eine Regel fehlt, eine Logik greift nicht. Der Fehler liegt im Prozess, nicht im Kriterium.
"preference": User formuliert ein neues Kriterium, eine Regel oder einen Constraint der zukünftig gelten soll — was er will oder nicht will. Der Agent hat nicht unbedingt einen Fehler gemacht, der User präzisiert seine Anforderungen.
null: Reaktion auf einen Output ohne klaren Änderungswunsch — Rückfrage, Bestätigung, allgemeines Feedback.

Grenzfälle:
- "Das ist falsch" allein → null (unklar was geändert werden soll)
- "Das passiert immer wenn X" → step_patch (systematisches Problem)
- "Ich will generell keine Y" → preference (neues Kriterium)
- "Dieser Eintrag stimmt nicht" → data_edit (konkreter gespeicherter Inhalt)
- Lob, Dank, neutrale Reaktion → none
- Folgefrage zum Output ("was noch?", "gibt es mehr?") → agent_talk
- "Kannst du das nochmal prüfen?" → agent_feedback, null (Bitte um Wiederholung, kein Edit)

agent_talk vs agent_create: "Pipeline für [bestehender Name]" → immer agent_talk.
agent_create vs task_create: Agent hat State und Gedächtnis. Task ist zustandslos.
agent_system vs agent_create: Mehrere abhängige Agents → system. Einer → create.

Wenn ein Agent-Output als Kontext mitgeliefert wird (Präfix "Agent hat folgendes gemeldet:"):
Nutze den Output-Inhalt um die Nutzerantwort einzuordnen. Der Output ist der Bezugspunkt — was sagt der User dazu?
- Korrektur, Ablehnung, neues Kriterium → agent_feedback
- Folgefrage zum Agenten oder seinen Daten → agent_talk
- Unmittelbare Aktion → agent_trigger
- Neutrale Reaktion, Dank, kein Handlungsbedarf → none"""

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

Prinzipien:
- agent_name exakt so übernehmen wie genannt, auch Spitznamen oder Kurzformen.
- payload enthält alle inhaltlichen Parameter die der User mitgibt — Ticker, URLs, Suchbegriffe, IDs, Zeiträume. Nie raten.
- action ist "stop" nur wenn der User explizit stoppen oder deaktivieren will."""

_AGENT_TALK_EXTRACTION_SYSTEM = """Extrahiere aus einer Nutzeranfrage den Agentennamen und welche Art von Anfrage gestellt wird.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "agent_name": Name des Agenten.
- "talk_type": Art der Anfrage.
  - "query": Statusabfrage oder Datenabruf — User will wissen was der Agent beobachtet, gefunden oder gespeichert hat.
  - "config_change": Inhaltliche Konfigurationsänderung — User möchte die Instruction oder das Verhalten des Agenten dauerhaft ändern.
  - "regenerate_pipeline": User möchte die Pipeline des Agenten neu generieren oder erstellen.
  - "rename": User möchte den Agenten umbenennen.

Prinzipien:
- agent_name exakt so übernehmen wie genannt.
- query wenn der User nach Ergebnissen, Status oder gespeicherten Daten fragt.
- config_change wenn der User eine inhaltliche Anpassung des Agenten beschreibt die dauerhaft gelten soll — nicht als Reaktion auf einen konkreten Output.
- regenerate_pipeline nur bei explizitem Wunsch nach Pipeline-Neugenerierung oder -Erstellung.
- rename wenn ein neuer Name für den Agenten genannt wird."""

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
- "platforms": Liste der zu durchsuchenden Plattformen. Wenn keine explizit genannt, wähle passende anhand der Kategorie: Gebrauchtwaren/Hardware → ["kleinanzeigen", "ebay"], Wohnungen → ["immoscout", "wggesucht"], Jobs → ["stepstone", "linkedin"].
- "category": Kurzes Schlagwort für die Kategorie (z.B. "electronics", "apartment", "job", "vehicle", "furniture").
- "query": Optimierte Suchanfrage für die Plattformen (1-5 Wörter, kein Fülltext).
- "filters": Dict mit optionalen Filtern. Mögliche Keys: price_min, price_max, location, city, rooms_min, sqm_min. Nur setzen wenn explizit genannt.
- "target_agent": Name des Agenten der bei neuen Listings getriggert werden soll.
- "poll_interval_seconds": Wie oft gescraped werden soll. Standard: 3600. Für zeitkritische Suchen: 1800.

Prinzipien:
- Plattformen aus dem Kontext ableiten wenn nicht explizit genannt — Kategorie und Zielgruppe entscheiden.
- query so kurz und präzise wie möglich, keine Plattformnamen oder Füllwörter.
- filters nur befüllen wenn der User konkrete Werte nennt — nie raten.
- target_agent exakt so übernehmen wie der User den Agenten nennt."""


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
    context_hints: list[str] = []
    if not has_active_agents:
        context_hints.append("Der Nutzer hat keine aktiven Agenten — agent_trigger, agent_talk, agent_feedback und agent_list sind daher unwahrscheinlich.")
    if not has_active_tasks:
        context_hints.append("Der Nutzer hat keine aktiven Aufgaben — task_stop und task_list sind daher unwahrscheinlich.")

    content = text
    if notification_context and notification_context.get("notification_type") not in (
        "confirmation", "adjust_request"
    ):
        notification_summary = (notification_context.get("payload_summary") or {}).get("summary", "")
        if notification_summary:
            content = f"Agent hat folgendes gemeldet:\n{notification_summary}\n\nNutzer antwortet: {text}"

    if context_hints:
        content = "\n".join(context_hints) + "\n\nNutzeranfrage: " + content

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
