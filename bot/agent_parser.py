from __future__ import annotations
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
import asyncpg

from bot import brain, config, memory
from bot.models import CAPABILITY_CHAT, CAPABILITY_REASONING, CAPABILITY_DEEP_REASONING
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)


# ── Task Decomposition ────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """Du analysierst einen Agenten-Auftrag und zerlegst ihn in Teilaufgaben.
Für jede Teilaufgabe bestimmst du den optimalen Baustein-Typ.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "type": Kurzes Schlagwort für den Bereich. Beispiele: monitoring, research, finance, news, market, coding.
- "subtasks": Liste von Teilaufgaben. Jede Teilaufgabe hat:
  - "id": snake_case Bezeichner
  - "description": Was passiert hier in einem Satz
  - "classification": Einer der verfügbaren Baustein-Typen (siehe unten)
  - "inputs": Liste von Eingaben (z.B. "trigger_payload.url", "state:baselines", "context:search_result")
  - "outputs": Liste von Ausgaben (z.B. "context:extracted", "state:baselines")
  - "operation": Nur für transform-Bausteine. Einer von: array_push, statistics, json_path, xml_extract, regex_extract, arithmetic, compare
  - "condition": Nur wenn diese Teilaufgabe nur unter bestimmten Bedingungen läuft. Freitext.
  - "route": Nur wenn diese Teilaufgabe nur auf einem bestimmten Route-Pfad läuft.

Verfügbare Baustein-Typen:

ROUTING:
- router_match: Deterministisches Routing auf Basis exakter Werte in trigger_payload oder context.
- router_llm: LLM-basiertes Routing wenn die Entscheidung Interpretation erfordert.

LLM — nur wenn Urteilsvermögen, Abstraktion oder Sprachverständnis nötig ist:
- llm_extract: Strukturierte Daten aus unstrukturiertem Text extrahieren. Gibt immer JSON zurück.
- llm_decide: Bewertung, Klassifikation oder Urteil mit Begründung. Gibt immer JSON zurück.
- llm_summarize: Zusammenfassung für Menschen oder als Input für weitere Steps.

DATENZUGRIFF — deterministisch:
- web_search: Websuche wenn die URL nicht bekannt ist oder die Ergebnisse variabel sind.
- finance: Börsenkurse und Finanzkennzahlen für einen Ticker.
- http_fetch: HTTP-Request an eine bekannte URL. Gibt den Response-Body als String zurück. Für strukturierte APIs, XML-Feeds, REST-Endpunkte.
- state_read / state_write: Einzelnen Key im eigenen Agent-State lesen oder schreiben.
- state_read_external / state_write_external: Key im State eines anderen Agenten lesen oder schreiben.
- data_read / data_write: Längere Dokumente im eigenen agent_data Namespace lesen oder schreiben.
- data_read_external / data_write_external: Längere Dokumente im agent_data eines anderen Agenten.

TRANSFORMATION — deterministisch, operiert auf Context-Werten:
- transform: Berechnung oder Strukturänderung auf bereits im Context vorhandenen Daten.
  Operationen: array_push, statistics, json_path, xml_extract, regex_extract, arithmetic, compare

KOORDINATION — deterministisch:
- trigger_agent: Anderen Agenten mit Payload anstoßen.
- notify_user: Nachricht direkt an den User senden.

ENTSCHEIDUNGSMATRIX:
Was ist deterministisch — verwende NIE ein LLM dafür:
- Routing auf strukturierten Feldern (type, id, url != null) → router_match
- Bekannte URL mit strukturiertem Response → http_fetch + transform
- Wert aus JSON extrahieren → transform(json_path)
- Wert aus XML extrahieren → transform(xml_extract)
- Wert aus Text per Regex → transform(regex_extract)
- Zahl/Wert in Liste einpflegen → state_read + transform(array_push) + state_write
- Werte an gruppiertes Array anhängen → transform(array_push)
- Statistiken auf gesammelten Zahlen → transform(statistics)
- Arithmetik zwischen zwei Werten (z.B. Währungsumrechnung) → transform(arithmetic)
- Numerischer Vergleich (z.B. Preis <= Schwellenwert) → transform(compare)
- Kurze Fakten im State speichern → state_write
- Lange Dokumente speichern → data_write
- Anderen Agent starten → trigger_agent
- User benachrichtigen → notify_user

Was braucht ein LLM:
- Unstrukturierter Text der verstanden werden muss → llm_extract
- Bewertung, Urteil, Entscheidung mit Begründung → llm_decide
- Zusammenfassung für Menschen → llm_summarize
- Websuche wenn URL nicht bekannt → web_search"""


async def _decompose_task(instruction: str) -> dict | None:
    try:
        raw = await brain.chat(
            system=_DECOMPOSE_SYSTEM,
            messages=[{"role": "user", "content": instruction}],
            capability=CAPABILITY_REASONING,
            caller="task_decomposition",
        )
        logger.debug("decompose raw: %r", raw[:300])
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict) or "subtasks" not in parsed:
            logger.warning("task decomposition returned unexpected structure")
            return None
        logger.info(
            "task decomposed: %d subtasks, type=%s",
            len(parsed["subtasks"]),
            parsed.get("type", "?"),
        )
        return parsed
    except Exception as e:
        logger.warning("task decomposition failed: %s", e)
        return None


# ── Pipeline Generator ────────────────────────────────────────────────────────

_PIPELINE_GENERATOR_SYSTEM = """Du übersetzt eine Aufgaben-Klassifikation in eine ausführbare Pipeline.

Du bekommst:
1. Die originale Instruction des Agenten
2. Eine strukturierte Klassifikation der Teilaufgaben mit Baustein-Typen

Deine Aufgabe ist ausschließlich Übersetzung — du erfindest keine neue Logik, du folgst der Klassifikation.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "steps": Alle Steps der Pipeline in Ausführungsreihenfolge

Step-Schemas nach Typ:

router_match:
{"id": "route", "type": "router_match", "rules": [{"if": "trigger_payload.X == 'value'", "then": "route_a"}, {"if": "trigger_payload.Y != null", "then": "route_b"}], "default": "idle", "output_key": "route"}

router_llm:
{"id": "route", "type": "router_llm", "prompt": "Entscheide welcher Pfad gilt.", "output_key": "route"}

llm_extract:
{"id": "extract", "type": "llm_extract", "prompt": "Extrahiere X aus {{source_key}}. Antworte NUR mit rohem JSON: {\"field\": \"...\"}", "output_key": "extracted", "only_if_route": "route_name"}

llm_decide:
{"id": "decide", "type": "llm_decide", "prompt": "Bewerte X anhand von {{data}}. Antworte NUR mit rohem JSON: {\"verdict\": \"...\", \"reason\": \"...\"}", "output_key": "decision", "only_if_route": "route_name"}

llm_summarize:
{"id": "summarize", "type": "llm_summarize", "prompt": "Fasse zusammen.", "output_key": "summary"}
{"id": "search_and_summarize", "type": "llm_summarize", "prompt": "Fasse die Suchergebnisse zusammen.", "search_query": "kurzer Suchbegriff 1-6 Wörter", "time_range": "day|week|month|year", "categories": "general|news|finance|it|science", "output_key": "summary"}

web_search:
{"id": "search", "type": "web_search", "query_template": "{{context_key}} relevante begriffe", "prompt": "Fasse zusammen.", "time_range": "week", "categories": "general", "output_key": "search_result"}

finance:
{"id": "get_quote", "type": "finance", "ticker_key": "selected_ticker", "output_key": "quote_data"}

http_fetch:
{"id": "fetch", "type": "http_fetch", "url": "https://example.com/api/data", "output_key": "raw_response", "default": ""}
{"id": "fetch", "type": "http_fetch", "url_template": "https://api.example.com/{{context_key}}", "headers": {"Accept": "application/xml"}, "timeout": 15.0, "output_key": "raw_response"}

state_read:
{"id": "read_data", "type": "state_read", "key": "my_key", "output_key": "data", "default": "{}"}

state_write:
{"id": "write_data", "type": "state_write", "key": "my_key", "source_key": "context_key_to_save"}

state_read_external / state_write_external:
{"id": "read_other", "type": "state_read_external", "agent_name": "OtherAgent", "key": "their_key", "output_key": "data", "default": ""}
{"id": "write_other", "type": "state_write_external", "agent_name": "OtherAgent", "key": "their_key", "source_key": "context_key"}

data_read / data_write:
{"id": "read_doc", "type": "data_read", "namespace": "my_namespace", "key_template": "{{context_key}}", "output_key": "document", "default": ""}
{"id": "write_doc", "type": "data_write", "namespace": "my_namespace", "key_template": "{{context_key}}", "source_key": "document_to_save"}

data_read_external / data_write_external:
{"id": "read_doc", "type": "data_read_external", "agent_name": "OtherAgent", "namespace": "their_namespace", "key_template": "{{context_key}}", "output_key": "document", "default": ""}
{"id": "write_doc", "type": "data_write_external", "agent_name": "OtherAgent", "namespace": "their_namespace", "key_template": "{{context_key}}", "source_key": "document"}

transform array_push:
{"id": "append", "type": "transform", "operation": "array_push", "value_key": "price_eur", "group_key": "extracted_model", "target_key": "historical_prices", "output_key": "historical_prices", "max_items": 500}
value_key: Context-Key mit dem anzuhängenden Wert. group_key: Context-Key mit dem Gruppennamen. target_key: Context-Key des bestehenden Dict {gruppe: [werte]}.

transform statistics:
{"id": "stats", "type": "transform", "operation": "statistics", "source_key": "historical_prices", "model_key": "extracted_model", "functions": ["q1", "q3", "iqr", "lower_bound"], "multiplier": 1.5, "output_key": "price_stats"}
Verfügbare functions: mean, median, std_dev, min, max, count, q1, q3, iqr, lower_bound, upper_bound. Mit model_key: gibt Stats nur für das angegebene Modell zurück — lower_bound/q1/q3 sind null wenn weniger als 4 Datensätze. Ohne model_key: gibt Dict aller Modelle zurück. multiplier gilt für lower_bound/upper_bound (Standard 1.5).

transform arithmetic:
{"id": "convert", "type": "transform", "operation": "arithmetic", "expression": "price / exchange_rate_eur_usd", "round": 2, "output_key": "price_eur", "default": ""}
Variablen im expression werden aus dem Context aufgelöst. Dot-Notation funktioniert (z.B. price_stats.lower_bound). Nur +, -, *, / und Klammern erlaubt.

transform compare:
{"id": "is_bargain", "type": "transform", "operation": "compare", "left_key": "price_eur", "right_key": "price_stats.lower_bound", "operator": "<=", "output_true": "true", "output_false": "false", "output_key": "is_bargain"}
operator: "<", "<=", ">", ">=", "==", "!=" — vergleicht zwei numerische Context-Werte.

transform json_path:
{"id": "extract_field", "type": "transform", "operation": "json_path", "source_key": "json_string", "path": "nested.field", "output_key": "value", "default": ""}

transform xml_extract:
{"id": "extract_xml", "type": "transform", "operation": "xml_extract", "source_key": "xml_string", "xpath": ".//ns:Element[@attr='value']", "attribute": "rate", "output_key": "value", "default": ""}
Hinweis: Namespaces werden automatisch erkannt. Default-Namespace wird als "ns:" registriert, benannte Prefixe bleiben erhalten (z.B. gesmes:). XPath muss entsprechende Prefixe verwenden.

transform regex_extract:
{"id": "extract_pattern", "type": "transform", "operation": "regex_extract", "source_key": "text", "pattern": "Pattern: ([\\d.]+)", "group": 1, "output_key": "value", "default": ""}

trigger_agent:
{"id": "trigger", "type": "trigger_agent", "target_agent_name": "TargetAgent", "payload": {"key": "{{context_key}}"}, "delay_minutes": 0}

notify_user:
{"id": "notify", "type": "notify_user", "source_key": "message_context_key"}

Output-Step — letzter Step jeder Route die ein Ergebnis produziert:
{"id": "output", "type": "llm_extract", "is_output": true, "prompt": "Erstelle aus {{result}} ein JSON-Objekt.\\n\\nRegeln:\\n- state_updates: immer {}\\n- notify_user: true nur wenn ...", "output_key": "output", "only_if_route": "route_name"}

STRUKTURREGELN:
- Der Output-Step hat immer is_output=true und type=llm_extract
- state_updates im Output-Step ist IMMER {} — State wird ausschließlich durch state_write Steps geschrieben
- LLM-Prompts für llm_extract/llm_decide enden mit "Antworte NUR mit rohem JSON: {Felder}"
- Jeder Step der Daten eines anderen Steps braucht — {{output_key}} als Template-Variable im prompt
- only_if_route weglassen wenn der Step auf allen Routen läuft
- only_if_key: {"key": "context_key", "value": "wert"} — Step läuft nur wenn Context-Key den Wert hat. Dot-Notation möglich. Nützlich um LLM-Steps deterministisch zu überspringen wenn ein vorheriger compare-Step false ergeben hat (z.B. only_if_key: {"key": "is_bargain", "value": "true"})
- Keine Steps erfinden die nicht in der Klassifikation stehen
- Dot-Notation für verschachtelte Felder: wenn ein LLM-Step {"merged_list": [...], "new_models": [...]} zurückgibt mit output_key "merge_result", kann ein nachfolgender Step source_key "merge_result.merged_list" oder condition_key "merge_result.new_models" verwenden — kein extra Transform-Step nötig"""


async def _generate_pipeline(
    instruction: str,
    decomposition: dict,
) -> dict | None:
    try:
        content = f"Instruction: {instruction}\n\nKlassifikation:\n{json.dumps(decomposition, ensure_ascii=False, indent=2)}"
        raw = await brain.chat(
            system=_PIPELINE_GENERATOR_SYSTEM,
            messages=[{"role": "user", "content": content}],
            capability=CAPABILITY_REASONING,
            caller="pipeline_generator",
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            logger.warning("pipeline generator returned non-dict")
            return None

        has_steps = isinstance(parsed.get("steps"), list)

        if not has_steps:
            logger.warning("pipeline generator returned empty structure")
            return None

        logger.info(
            "pipeline generated: %d steps",
            len(parsed.get("steps", [])),
        )
        return parsed
    except Exception as e:
        logger.warning("pipeline generation failed: %s", e)
        return None


# ── Agent Name Resolution ─────────────────────────────────────────────────────

_NAME_RESOLUTION_SYSTEM = """Identifiziere welcher Agent aus der Liste gemeint ist.
Antworte NUR mit der ID des Agenten als Integer, kein anderer Text.
Wenn kein Agent eindeutig zuzuordnen ist, antworte mit 0.
Beispiel: 3"""


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
            messages=[{"role": "user", "content": f"Agenten:\n{agent_list}\n\nNutzeranfrage: {text}"}],
            max_tokens=10,
            capability="simple_tasks",
        )
        resolved_id = int(raw.strip())
        if resolved_id == 0:
            return None
        return next((a for a in active_agents if a["id"] == resolved_id), None)
    except Exception as e:
        logger.warning("agent name resolution failed: %s", e)
        return None


# ── Agent Creation ────────────────────────────────────────────────────────────

_AGENT_PARSER_SYSTEM = """Du extrahierst einen persistenten Agenten aus einer Nutzeranfrage.
Ein Agent läuft nach Plan, erinnert sich an frühere Ergebnisse und handelt nur wenn sich etwas Relevantes ändert.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "instruction": Vollständige, eigenständige Anweisung in natürlicher Sprache.
- "schedule": Cron-Expression (5 Felder). Beispiele: stündlich = "0 * * * *", täglich um 9 = "0 9 * * *".
- "target": "same" für denselben Chat, "dm" für Privatnachricht.
- "wants_name": true wenn der User einen Namen erwähnt oder explizit fragt.
- "suggested_name": Konkreter Name wenn der User einen nennt, sonst null.
- "wants_monitor": true wenn ein RSS-Monitor sinnvoll wäre.
- "wants_scraper": true wenn ein Scraper-Service sinnvoll wäre.

Wenn kein sinnvoller Zeitplan erkennbar ist, setze schedule auf null."""


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
            capability=CAPABILITY_CHAT,
            caller="agent_parser",
        )
        logger.debug("agent parser raw: %r", raw[:200])
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            return None

        schedule_raw = parsed.get("schedule")
        schedule: str | None = None
        if schedule_raw and croniter.is_valid(schedule_raw):
            schedule = schedule_raw
        else:
            logger.info("no valid schedule — agent will be trigger-only")

        instruction = parsed.get("instruction", "").strip()
        if not instruction:
            return None

        decomposition = await _decompose_task(instruction)
        if decomposition is None:
            logger.warning("task decomposition failed for agent creation")
            return None

        pipeline_result = await _generate_pipeline(instruction, decomposition)
        agent_type: str = decomposition.get("type", "default")

        next_run_utc: datetime | None = None
        next_run_local: datetime | None = None

        if schedule:
            tz_str = await memory.get_user_timezone(pool, user_id)
            try:
                tz = ZoneInfo(tz_str)
            except ZoneInfoNotFoundError:
                tz = ZoneInfo("UTC")
            now = datetime.now(tz)
            next_run_local = croniter(schedule, now).get_next(datetime)
            next_run_utc = next_run_local.astimezone(ZoneInfo("UTC"))

        target_chat_id = user_id if parsed.get("target") == "dm" else source_chat_id

        agent_config: dict = {
            "instruction": instruction,
            "type": agent_type,
            "data_reads": [],
        }
        if pipeline_result:
            agent_config["steps"] = pipeline_result.get("steps", [])

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
            "wants_monitor": bool(parsed.get("wants_monitor", False)),
            "wants_scraper": bool(parsed.get("wants_scraper", False)),
        }
    except Exception as e:
        logger.warning("agent parsing failed: %s", e)
        return None


# ── Agent Talk ────────────────────────────────────────────────────────────────

_AGENT_TALK_SYSTEM = """Du bist Bob. Ein Nutzer fragt nach einem deiner laufenden Agenten oder möchte dessen Konfiguration ändern.

Du sprichst ÜBER den Agenten in Bobs Stimme — identifiziere dich immer mit dem Agenten-Namen.
Du bist nicht der Agent und schlüpfst nicht in seine Rolle.

Mögliche Anfragen:
- Statusabfrage → fasse State, Beobachtungen und gespeicherte Daten zusammen
- Inhaltliche Konfigurationsänderung → bestätige knapp was geändert wird, gib vollständiges neues config-Objekt zurück: ```config\\n{...}\\n```
- Umbenennung → bestätige knapp, gib neuen Namen zurück: ```name\\nNeuerName\\n```

Wenn du die Konfiguration änderst: gib das vollständige config-Objekt zurück mit ALLEN bestehenden Feldern.
Ändere NUR instruction und type — niemals steps direkt."""


async def handle_agent_talk(
    text: str,
    agent: dict,
    state: dict[str, str],
    agent_memories: list[str],
    pool: asyncpg.Pool | None = None,
) -> tuple[str, dict | None, str | None]:
    config_data = parse_agent_config(agent["config"])
    state_summary = "\n".join(f"{k}: {v}" for k, v in state.items()) if state else "noch kein State"
    memories_summary = "\n- ".join(agent_memories) if agent_memories else "noch keine Beobachtungen"

    data_summary = ""
    full_content_blocks: list[str] = []

    if pool is not None:
        try:
            data_rows = await memory.get_all_agent_data(pool, agent["id"])
            if data_rows:
                text_lower = text.lower()
                ns_lines: list[str] = []
                for row in data_rows[:50]:
                    key_lower = row["key"].lower()
                    if key_lower in text_lower or any(
                        word in text_lower for word in key_lower.replace("_", " ").replace(".", " ").split()
                        if len(word) > 3
                    ):
                        full_content_blocks.append(
                            f"[Vollständiger Inhalt — {row['namespace']}/{row['key']}]\n{row['value']}"
                        )
                    else:
                        preview = row["value"][:120] + "…" if len(row["value"]) > 120 else row["value"]
                        ns_lines.append(f"{row['namespace']}/{row['key']}: {preview}")
                data_summary = "\n".join(ns_lines)
        except Exception as e:
            logger.warning("failed to load agent data for talk: %s", e)

    context = (
        f"Agent: {agent['name']}\n"
        f"Konfiguration: {json.dumps(config_data, ensure_ascii=False)}\n\n"
        f"Aktueller State:\n{state_summary}\n\n"
        f"Bisherige Beobachtungen:\n- {memories_summary}"
        + (f"\n\nGespeicherte Daten:\n{data_summary}" if data_summary else "")
        + (f"\n\n{chr(10).join(full_content_blocks)}" if full_content_blocks else "")
    )

    try:
        response = await brain.chat(
            system=_AGENT_TALK_SYSTEM,
            messages=[{"role": "user", "content": f"{context}\n\nNutzeranfrage: {text}"}],
            capability=CAPABILITY_CHAT,
            caller="agent_talk",
        )
    except Exception as e:
        logger.warning("agent talk failed: %s", e)
        return "Konnte den Agenten nicht befragen.", None, None

    new_config: dict | None = None
    new_name: str | None = None

    if "```config" in response:
        try:
            start = response.index("```config") + len("```config")
            end = response.index("```", start)
            raw_config = json.loads(response[start:end].strip())
            if isinstance(raw_config, dict):
                new_config = raw_config
                if new_config.get("instruction") and new_config["instruction"] != config_data.get("instruction"):
                    decomposition = await _decompose_task(new_config["instruction"])
                    if decomposition:
                        new_pipeline = await _generate_pipeline(new_config["instruction"], decomposition)
                        if new_pipeline:
                            new_config["steps"] = new_pipeline.get("steps", [])
                            new_config.pop("pipeline", None)
                            new_config.pop("pipeline_after_template", None)
                            new_config["type"] = decomposition.get("type", config_data.get("type", "default"))
            response = response[:response.index("```config")].strip()
        except Exception as e:
            logger.warning("config extraction from agent talk failed: %s", e)

    if "```name" in response:
        try:
            start = response.index("```name") + len("```name")
            end = response.index("```", start)
            new_name = response[start:end].strip()
            response = response[:response.index("```name")].strip()
        except Exception as e:
            logger.warning("name extraction from agent talk failed: %s", e)

    return response, new_config, new_name


# ── Pipeline Regeneration (for handler.py) ────────────────────────────────────

async def regenerate_pipeline_for_agent(agent_config: dict) -> dict:
    instruction = agent_config.get("instruction", "")
    if not instruction:
        return agent_config

    decomposition = await _decompose_task(instruction)
    if decomposition is None:
        return agent_config

    pipeline_result = await _generate_pipeline(instruction, decomposition)
    if pipeline_result is None:
        return agent_config

    updated = dict(agent_config)
    updated["steps"] = pipeline_result.get("steps", [])
    updated.pop("pipeline", None)
    updated.pop("pipeline_after_template", None)
    updated["type"] = decomposition.get("type", agent_config.get("type", "default"))
    updated.pop("work_capability", None)
    return updated


# ── Scheduling Helper ─────────────────────────────────────────────────────────

def next_agent_run_after(schedule: str, timezone: str) -> datetime:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(config.BOT_DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    next_run_local = croniter(schedule, now).get_next(datetime)
    return next_run_local.astimezone(ZoneInfo("UTC"))
