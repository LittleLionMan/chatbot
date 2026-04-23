from __future__ import annotations
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
import asyncpg

from bot import brain, memory
from bot.agent_parser import _decompose_task, _generate_pipeline, _pick_name_for_topic
from bot.models import CAPABILITY_DEEP_REASONING, CAPABILITY_REASONING
from bot.utils import clean_llm_json

logger = logging.getLogger(__name__)

_AVAILABLE_STEP_TYPES = [
    "router_match", "router_llm",
    "llm_extract", "llm_decide", "llm_summarize",
    "web_search", "finance", "http_fetch",
    "state_read", "state_write",
    "state_read_external", "state_write_external",
    "data_read", "data_write",
    "data_read_external", "data_write_external",
    "transform",
    "trigger_agent", "notify_user",
]

_TRANSFORM_OPERATIONS = [
    "array_append", "iqr_bounds", "json_path", "xml_extract", "regex_extract",
]

_EXTERNAL_SERVICES = [
    "scraper (Kleinanzeigen, eBay, Immoscout, WG-Gesucht, StepStone, LinkedIn)",
    "rss_monitor (News-Feeds, Google News)",
    "finance_service (Börsenkurse via yfinance)",
    "stt (Speech-to-Text)",
    "tts (Text-to-Speech)",
]

_SCRAPER_PAYLOAD_SCHEMA = {
    "listing_id": "int — interne DB-ID des Listings",
    "platform": "str — z.B. 'kleinanzeigen' oder 'ebay'",
    "category": "str — z.B. 'gpu'",
    "title": "str — Titel des Listings",
    "price": "float | null — Preis in der Währung der Plattform",
    "currency": "str | null — z.B. 'EUR' oder 'USD'",
    "url": "str — Link zum Listing",
    "location": "str | null — Ort des Verkäufers",
    "condition": "str | null — z.B. 'very_good', 'good', 'acceptable'",
    "seller_rating": "float | null — Bewertung des Verkäufers",
    "attributes": "dict — plattformspezifische Zusatzfelder",
    "raw_text": "str — Beschreibungstext des Listings (max 2000 Zeichen)",
}

_RSS_MONITOR_PAYLOAD_SCHEMA = {
    "key": "str — Watchlist-Item (z.B. Ticker oder Suchbegriff)",
    "name": "str — Anzeigename des Items",
    "reason": "str — Artikeltitel + Quelle",
    "article_text": "str — Volltext des Artikels (max 8000 Zeichen)",
    "article_url": "str — Link zum Artikel",
    "published": "str — Veröffentlichungsdatum",
    "since_date": "str — Datum der letzten Analyse dieses Items",
}


_AVAILABLE_SCRAPER_PLATFORMS = [
    "kleinanzeigen", "ebay", "reddit",
    "immoscout", "wggesucht",
    "stepstone", "linkedin",
]

_PLAN_SYSTEM = f"""Du bist Bob. Du planst ein Agent-System für einen User.

Dir wird der bisherige Gesprächsverlauf übergeben — der ursprüngliche Prompt des Users und alle bisherigen Klärungsrunden.

Deine Aufgabe: analysiere ob du genug weißt um ein sinnvolles System zu bauen. Wenn nicht, stelle eine präzise Rückfrage. Wenn ja, präsentiere einen konkreten Plan.

Verfügbare Bausteine:
{json.dumps(_AVAILABLE_STEP_TYPES, ensure_ascii=False)}

Verfügbare Transform-Operationen (für deterministisches Parsing ohne LLM):
{json.dumps(_TRANSFORM_OPERATIONS, ensure_ascii=False)}

Externe Services die Agents triggern können:
{json.dumps(_EXTERNAL_SERVICES, ensure_ascii=False)}

Verfügbare Scraper-Plattformen (können sofort eingerichtet werden):
{json.dumps(_AVAILABLE_SCRAPER_PLATFORMS, ensure_ascii=False)}

Wenn ein Scraper einen Agent triggert, enthält der Trigger-Payload immer diese Felder:
{json.dumps(_SCRAPER_PAYLOAD_SCHEMA, ensure_ascii=False)}

Wenn ein RSS-Monitor einen Agent triggert, enthält der Trigger-Payload immer diese Felder:
{json.dumps(_RSS_MONITOR_PAYLOAD_SCHEMA, ensure_ascii=False)}

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Mögliche Status-Werte:

1. Du brauchst noch Information:
{{"status": "needs_clarification", "question": "eine einzelne präzise Rückfrage — nicht mehrere Fragen auf einmal"}}

2. Du hast genug und präsentierst einen Plan:
{{
  "status": "ready",
  "description": "Menschenlesbare Zusammenfassung in Bobs Stimme. Beschreibe jeden Agent mit Name, Aufgabe und Zeitplan/Trigger. Benenne explizit welcher Agent wessen Daten liest. Nenne Annahmen die du getroffen hast. Wenn RSS-Monitore geplant sind, nenne konkret welche Feeds du vorschlägst und warum — der User weiß oft nicht welche Feeds existieren. Wenn Scraper geplant sind, nenne welche Plattformen verfügbar sind und ob eine gewünschte Plattform noch nicht existiert. Schließe mit: Passt das so, oder soll ich etwas anpassen?",
  "agents": [
    {{
      "name": "thematisch passender menschlicher Name",
      "role": "ein Satz was dieser Agent tut",
      "schedule": "Cron-Expression oder null wenn trigger-only",
      "trigger": "womit wird der Agent getriggert, oder null"
    }}
  ],
  "monitors": [
    {{
      "name": "beschreibender Name",
      "source": "static oder agent",
      "target_agent": "Name des Agents der getriggert wird",
      "feed_urls": ["https://... (nur bei source=static)"],
      "feed_templates": ["https://...{{query}}... (nur bei source=agent)"],
      "source_agent": "Name des Quell-Agents (nur bei source=agent)",
      "source_state_key": "State-Key der Watchlist (nur bei source=agent)",
      "source_format": "comma_list (nur bei source=agent)",
      "keywords": ["optionale Filterkeywords"],
      "poll_interval_seconds": 3600
    }}
  ],
  "scrapers": [
    {{
      "platform": "eine der verfügbaren Plattformen",
      "category": "kurzes Schlagwort z.B. gpu, apartment, job",
      "query": "Suchbegriff 1-5 Wörter",
      "target_agent": "Name des Agents der getriggert wird",
      "filters": {{}},
      "poll_interval_seconds": 3600,
      "unavailable": false
    }}
  ],
  "assumptions": ["Annahme 1", "Annahme 2"],
  "missing_capabilities": ["Beschreibung was fehlt und warum — leer wenn alles abdeckbar"]
}}

3. User hat bestätigt:
{{"status": "confirmed"}}

Regeln für Rückfragen:
- Stelle eine Rückfrage nur wenn die fehlende Information das System grundlegend verändert und nicht sinnvoll angenommen werden kann
- Eine Rückfrage pro Runde, die wichtigste offene zuerst
- Frage nicht nach Dingen die du sinnvoll annehmen und als Annahme benennen kannst
- Irgendwann reicht es — präsentiere einen Plan und benenne was du angenommen hast

Regeln für den Plan:
- Eine Korrektur des Users ("nein, nicht täglich sondern wöchentlich") → direkt angepassten Plan zurückgeben
- Erkenne Bestätigungen: "ja", "gut", "mach es", "los", "ok", "passt", "stimmt so", "anlegen", "setze um"
- missing_capabilities nur wenn eine Teilaufgabe wirklich nicht mit verfügbaren Bausteinen abbildbar ist
- Strukturierte Daten von bekannten APIs oder Feeds → http_fetch + transform, kein LLM-Call nötig
- Scraper-Trigger-Payload ist bekannt — frage nicht danach wenn der User Scraper erwähnt

Regeln für RSS-Monitore:
- Wenn ein Agent kontinuierlich auf neue externe Inhalte reagieren soll (News, Artikel, Releases, Posts) → RSS-Monitor vorschlagen
- Nenne konkrete Feed-URLs aus deinem Wissen — der User weiß oft nicht welche Feeds existieren
- Wenn Keywords sinnvoll sind um irrelevante Artikel herauszufiltern → im Plan nennen
- Frage den User ob die vorgeschlagenen Feeds passen oder ob er andere bevorzugt

Regeln für Scraper:
- Wenn ein Agent auf neue Listings von Marktplätzen reagieren soll → Scraper vorschlagen
- Nur Plattformen aus der verfügbaren Liste können sofort eingerichtet werden
- Wenn eine gewünschte Plattform nicht verfügbar ist: setze unavailable=true und erkläre es in der description
- Scraper mit unavailable=true werden nicht angelegt, aber dem User erklärt
"""


async def plan(
    accumulated_context: str,
    pool: asyncpg.Pool,
    clarification_rounds: int = 0,
) -> dict:
    try:
        raw = await brain.chat(
            system=_PLAN_SYSTEM,
            messages=[{"role": "user", "content": accumulated_context}],
            capability=CAPABILITY_DEEP_REASONING,
            caller="agent_planner",
            pool=pool,
        )
        logger.debug("planner raw: %r", raw[:300])
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            logger.warning("planner returned non-dict")
            return _fallback_plan()

        status = parsed.get("status", "")
        if status not in ("needs_clarification", "ready", "confirmed"):
            logger.warning("planner returned unknown status %r", status)
            return _fallback_plan()

        return parsed
    except Exception as e:
        logger.warning("planner failed: %s", e)
        return _fallback_plan()


def _fallback_plan() -> dict:
    return {
        "status": "needs_clarification",
        "question": "Ich konnte den Auftrag nicht vollständig verstehen. Kannst du beschreiben was der Agent tun soll, wie oft er laufen soll, und was er am Ende ausgeben soll?",
    }


def format_plan_message(plan_result: dict) -> str:
    if plan_result["status"] == "needs_clarification":
        return plan_result.get("question", "Kannst du das etwas konkreter beschreiben?")

    if plan_result["status"] != "ready":
        return ""

    msg = plan_result.get("description", "")

    missing = plan_result.get("missing_capabilities", [])
    if missing:
        msg += "\n\nFür folgende Teilaufgaben fehlen mir noch Bausteine:\n"
        msg += "\n".join(f"— {m}" for m in missing)

    return msg


async def finalize(
    plan_result: dict,
    accumulated_context: str,
    user_id: int,
    source_chat_id: int,
    pool: asyncpg.Pool,
) -> list[dict] | None:
    agents_in_plan: list[dict] = plan_result.get("agents", [])
    if not agents_in_plan:
        logger.warning("finalize: no agents in plan")
        return None

    tz_str = await memory.get_user_timezone(pool, user_id)
    try:
        tz = ZoneInfo(tz_str)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    prepared: list[dict] = []

    for agent_meta in agents_in_plan:
        agent_name: str = agent_meta.get("name", "Agent")
        role: str = agent_meta.get("role", "")
        schedule_raw: str | None = agent_meta.get("schedule")

        instruction = await _build_instruction(
            agent_name=agent_name,
            role=role,
            accumulated_context=accumulated_context,
            all_agents=agents_in_plan,
            pool=pool,
        )
        if not instruction:
            logger.warning("finalize: could not build instruction for %s", agent_name)
            continue

        schedule: str | None = None
        if schedule_raw and croniter.is_valid(schedule_raw):
            schedule = schedule_raw

        next_run_utc: datetime | None = None
        next_run_local: datetime | None = None
        if schedule:
            next_run_local = croniter(schedule, now).get_next(datetime)
            next_run_utc = next_run_local.astimezone(ZoneInfo("UTC"))

        decomposition = await _decompose_task(instruction)
        if decomposition is None:
            logger.warning("finalize: decomposition failed for %s", agent_name)
            continue

        pipeline_result = await _generate_pipeline(instruction, decomposition)
        agent_type: str = decomposition.get("type", "default")

        agent_config: dict = {
            "instruction": instruction,
            "type": agent_type,
            "data_reads": [],
        }
        if pipeline_result:
            if pipeline_result.get("pipeline"):
                agent_config["pipeline"] = pipeline_result["pipeline"]
            if pipeline_result.get("pipeline_after_template"):
                agent_config["pipeline_after_template"] = pipeline_result["pipeline_after_template"]

        target_chat_id = source_chat_id

        prepared.append({
            "name": agent_name,
            "config": agent_config,
            "schedule": schedule,
            "next_run_at": next_run_utc,
            "next_run_display": next_run_local,
            "target_chat_id": target_chat_id,
        })
        logger.info(
            "finalize: prepared agent %s (type=%s, schedule=%s, steps=%d)",
            agent_name,
            agent_type,
            schedule or "trigger-only",
            len(agent_config.get("pipeline", [])) + len(agent_config.get("pipeline_after_template", [])),
        )

    return prepared if prepared else None


async def finalize_monitors(
    plan_result: dict,
    pool: asyncpg.Pool,
) -> list[dict]:
    monitors_in_plan: list[dict] = plan_result.get("monitors", [])
    created: list[dict] = []

    for mon in monitors_in_plan:
        source: str = mon.get("source", "static")
        target_agent: str = mon.get("target_agent", "")
        if not target_agent:
            logger.warning("finalize_monitors: monitor without target_agent, skipping")
            continue
        try:
            monitor_id = await memory.create_monitor_config(
                pool,
                monitor_type=mon.get("monitor_type", "rss"),
                name=mon.get("name", f"Monitor für {target_agent}"),
                source=source,
                target_agent=target_agent,
                feed_templates=mon.get("feed_urls", []) if source == "static" else mon.get("feed_templates", []),
                poll_interval_seconds=mon.get("poll_interval_seconds", 3600),
                source_agent=mon.get("source_agent", ""),
                source_state_key=mon.get("source_state_key", ""),
                source_format=mon.get("source_format", "comma_list"),
                keywords=mon.get("keywords", []),
            )
            created.append({"id": monitor_id, "name": mon.get("name", ""), "target_agent": target_agent})
            logger.info("finalize_monitors: created monitor %d for %s", monitor_id, target_agent)
        except Exception as e:
            logger.error("finalize_monitors: failed for %s: %s", target_agent, e)

    return created


_AVAILABLE_SCRAPER_PLATFORMS_SET = set(_AVAILABLE_SCRAPER_PLATFORMS)


async def finalize_scrapers(
    plan_result: dict,
    pool: asyncpg.Pool,
) -> tuple[list[dict], list[dict]]:
    scrapers_in_plan: list[dict] = plan_result.get("scrapers", [])
    created: list[dict] = []
    unavailable: list[dict] = []

    for scraper in scrapers_in_plan:
        platform: str = scraper.get("platform", "")
        target_agent: str = scraper.get("target_agent", "")

        if scraper.get("unavailable", False) or platform not in _AVAILABLE_SCRAPER_PLATFORMS_SET:
            unavailable.append({"platform": platform, "target_agent": target_agent})
            logger.info("finalize_scrapers: platform %s unavailable, skipping", platform)
            continue

        if not target_agent or not platform:
            logger.warning("finalize_scrapers: missing platform or target_agent, skipping")
            continue

        try:
            scraper_id = await memory.create_scraper_config(
                pool,
                platform=platform,
                category=scraper.get("category", ""),
                query=scraper.get("query", ""),
                target_agent=target_agent,
                filters=scraper.get("filters", {}),
                poll_interval_seconds=scraper.get("poll_interval_seconds", 3600),
            )
            created.append({"id": scraper_id, "platform": platform, "target_agent": target_agent})
            logger.info("finalize_scrapers: created scraper %d on %s for %s", scraper_id, platform, target_agent)
        except Exception as e:
            logger.error("finalize_scrapers: failed for %s/%s: %s", platform, target_agent, e)

    return created, unavailable


_INSTRUCTION_BUILDER_SYSTEM = """Du formulierst eine vollständige, eigenständige Instruction für einen einzelnen Agenten.

Du bekommst:
- Den Gesprächskontext der den Gesamtauftrag beschreibt
- Den Namen und die Rolle dieses Agenten
- Die anderen Agenten im System (für Abhängigkeiten)

Formuliere eine Instruction die:
- Vollständig und eigenständig ist — der Agent kann sie ohne weiteren Kontext ausführen
- Alle relevanten Details enthält: was er tut, womit er es tut, was er ausgibt
- Abhängigkeiten zu anderen Agenten explizit benennt (wessen State er liest, wen er triggert)
- Technische Details aus dem Kontext übernimmt (URLs, Schwellenwerte, Filterregeln)

Antworte NUR mit der Instruction als reiner Text. Kein JSON, keine Erklärungen."""


async def _build_instruction(
    agent_name: str,
    role: str,
    accumulated_context: str,
    all_agents: list[dict],
    pool: asyncpg.Pool,
) -> str:
    other_agents = [a for a in all_agents if a.get("name") != agent_name]
    other_agents_desc = "\n".join(
        f"- {a['name']}: {a.get('role', '')}" for a in other_agents
    ) if other_agents else "keine"

    content = (
        f"Gesamtkontext:\n{accumulated_context}\n\n"
        f"Dieser Agent: {agent_name}\nRolle: {role}\n\n"
        f"Andere Agents im System:\n{other_agents_desc}"
    )

    try:
        instruction = await brain.chat(
            system=_INSTRUCTION_BUILDER_SYSTEM,
            messages=[{"role": "user", "content": content}],
            capability=CAPABILITY_REASONING,
            caller=f"instruction_builder:{agent_name}",
            pool=pool,
        )
        return instruction.strip()
    except Exception as e:
        logger.warning("instruction builder failed for %s: %s", agent_name, e)
        return role
