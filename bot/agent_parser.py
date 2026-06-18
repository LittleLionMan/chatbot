from __future__ import annotations

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg
from croniter import croniter

from bot import agent_skills, brain, config
from bot.models import CAPABILITY_REASONING
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)


_DECOMPOSE_SYSTEM = """Du analysierst einen Agenten-Auftrag und zerlegst ihn in Teilaufgaben.
Für jede Teilaufgabe bestimmst du den optimalen Baustein-Typ.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "type": Kurzes Schlagwort für den Bereich. Beispiele: monitoring, research, finance, news, market, coding.
- "subtasks": Liste von Teilaufgaben. Jede Teilaufgabe hat:
  - "id": snake_case Bezeichner
  - "description": Was passiert hier in einem Satz
  - "classification": Einer der verfügbaren Baustein-Typen (siehe unten)
  - "inputs": Liste von Eingaben (z.B. "trigger_payload.url", "state:preferences", "context:search_result")
  - "outputs": Liste von Ausgaben (z.B. "context:extracted", "state:results")
  - "operation": Nur für transform-Bausteine. Einer von: array_push, group_by, statistics, json_path, xml_extract, regex_extract, arithmetic, compare
  - "condition": Nur wenn diese Teilaufgabe nur unter bestimmten Bedingungen läuft. Freitext.
  - "route": Nur wenn diese Teilaufgabe nur auf einem bestimmten Route-Pfad läuft.
  - "required": false wenn dieser Subtask optional ist und ein leeres Ergebnis die Pipeline NICHT abbrechen soll. Standard ist true. Setze required=false nur bei datenabrufenden Subtasks (web_search, http_fetch, finance, finance_search, xlsx_fetch), wenn mehrere gleichartige Abrufe parallel laufen und ein einzelner leerer Rückgabewert verkraftbar ist.
  - "parameters": Strukturierte Parameter die direkt aus der Instruction ableitbar sind. MUSS bei xlsx_fetch gesetzt werden — ohne parameters ist ein xlsx_fetch-Subtask unvollständig und ungültig. Bei xlsx_fetch enthält parameters zwingend: "columns" (alle in der Instruction genannten Spaltennamen als Liste) und "filters" (alle in der Instruction beschriebenen Filterbedingungen als Liste). Jede Filterbedingung hat "column", "operator" und optional "value". Verfügbare Operatoren: not_empty, empty, equals, not_equals, contains, not_contains, starts_with, ends_with. Ein xlsx_fetch-Subtask ohne parameters.filters ist ein Fehler.

PFLICHTREGELN FÜR URTEILENDE SUBTASKS:
Wenn ein Subtask ein inhaltliches Urteil trifft (Relevanz, Qualität, Akzeptanz, Eignung —
erkennbar an Klassifikation llm_decide oder llm_analyze):
- Sein inputs MUSS "state:preferences" enthalten
- Sein outputs MUSS "state:results" enthalten
- Vor diesem Subtask MUSS ein Subtask stehen der Präferenzen aus der Instruction extrahiert
  (classification: llm_extract, outputs: ["context:extracted_preferences"])
- Vor diesem Subtask MUSS ein Subtask stehen der diese Präferenzen in den State initialisiert
  (classification: state_init, inputs: ["context:extracted_preferences"], outputs: ["state:preferences"])
- Nach diesem Subtask MUSS ein Subtask stehen der Ergebnisse akkumuliert
  (classification: state_write, inputs: ["context:results"])

Verfügbare Baustein-Typen:

ROUTING:
- router_match: Deterministisches Routing auf Basis exakter Werte in trigger_payload oder context.
- router_llm: LLM-basiertes Routing wenn die Entscheidung Interpretation erfordert.

LLM — nur wenn Urteilsvermögen, Abstraktion oder Sprachverständnis nötig ist:
- llm_extract: Strukturierte Daten aus bekanntem Format extrahieren. Gibt immer JSON zurück. Capability: simple_tasks.
- llm_decide: Bewertung, Klassifikation oder Urteil mit Begründung. Gibt immer JSON zurück. Capability: reasoning.
- llm_summarize: Zusammenfassung für Menschen oder als Input für weitere Steps. Capability: chat.
- llm_analyze: Tiefgehende Analyse aus heterogenen Quellen mit mehrstufigem Schlussfolgern. Gibt immer JSON zurück. Capability: deep_reasoning.

DATENZUGRIFF — deterministisch:
- web_search: Websuche wenn die URL nicht bekannt ist oder die Ergebnisse variabel sind.
- finance: Börsenkurse und Finanzkennzahlen für einen bekannten Ticker-Symbol.
- finance_search: Ticker und Name eines Unternehmens über ISIN oder Suchbegriff deterministisch ermitteln.
- http_fetch: HTTP-Request an eine bekannte URL.
- xlsx_fetch: Lädt eine Excel-Datei (.xlsx) von einer URL und gibt ein JSON-Array der Zeilen zurück.
- state_read / state_write: Einzelnen Key im eigenen Agent-State lesen oder schreiben.
- state_init: Key im eigenen Agent-State schreiben — aber NUR wenn der Key noch nicht existiert.
- state_read_external / state_write_external: Key im State eines anderen Agenten lesen oder schreiben.
- data_read / data_write: Längere Dokumente im eigenen agent_data Namespace lesen oder schreiben.
- data_read_external / data_write_external: Längere Dokumente im agent_data eines anderen Agenten.

TRANSFORMATION — deterministisch, operiert auf Context-Werten:
- transform: Berechnung oder Strukturänderung auf bereits im Context vorhandenen Daten.

KOORDINATION — deterministisch:
- trigger_agent: Anderen Agenten mit Payload anstoßen.
- notify_user: Nachricht direkt an den User senden.

ENTSCHEIDUNGSMATRIX:
Was ist deterministisch — verwende NIE ein LLM dafür:
- Routing auf strukturierten Feldern → router_match
- ISIN oder Name → Ticker ermitteln → finance_search
- Excel-Datei (.xlsx) → xlsx_fetch
- Bekannte URL mit strukturiertem Response → http_fetch + transform
- Kurze Fakten im State → state_write
- Lange Dokumente → data_write
- Anderen Agent starten → trigger_agent
- User benachrichtigen → notify_user

Was braucht ein LLM:
- Unstrukturierter Text der verstanden werden muss → llm_extract
- Bewertung, Urteil, Entscheidung → llm_decide
- Zusammenfassung für Menschen → llm_summarize
- Websuche → web_search
- Komplexe Analyse aus heterogenen Quellen → llm_analyze"""


async def _decompose_task(
    instruction: str, pool: asyncpg.Pool | None = None
) -> dict | None:
    skill_context = ""
    if pool is not None:
        skill_context = await agent_skills.load_skill_context(pool)
    system = _DECOMPOSE_SYSTEM + ("\n\n" + skill_context if skill_context else "")
    try:
        raw = await brain.chat(
            system=system,
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


_PIPELINE_GENERATOR_SYSTEM = """Du übersetzt eine Aufgaben-Klassifikation in eine ausführbare Pipeline.

Du bekommst:
1. Die originale Instruction des Agenten
2. Eine strukturierte Klassifikation der Teilaufgaben mit Baustein-Typen

Deine Aufgabe ist ausschließlich Übersetzung — du erfindest keine neue Logik, du folgst der Klassifikation.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "steps": Alle Steps der Pipeline in Ausführungsreihenfolge

PFLICHTREGELN FÜR URTEILENDE STEPS:
Wenn ein Step vom Typ llm_decide oder llm_analyze ein inhaltliches Urteil trifft:

1. Ganz am Anfang der Pipeline MUSS stehen:
   a) llm_extract — extrahiert alle Präferenzen, Kriterien und Regeln aus der Instruction.
   b) state_init — schreibt extracted_preferences in den State, NUR wenn der Key noch nicht existiert.

2. Direkt vor dem llm_decide/llm_analyze Step MUSS stehen:
   state_read — liest den Präferenz-Key aus dem State.

3. Der llm_decide/llm_analyze Prompt MUSS die gelesenen Präferenzen einbinden.

4. Nach dem llm_decide/llm_analyze Step MUSS state_write folgen der Ergebnisse akkumuliert.

DATENABRUF UND ABBRUCH:
Steps die externe Daten holen (web_search, http_fetch, finance, finance_search, xlsx_fetch) brechen den
gesamten Lauf ab wenn sie ein leeres Ergebnis liefern — es sei denn, ein "default" ist gesetzt oder "required": false.
- Setze "required": false NUR wenn mehrere gleichartige Abruf-Steps parallel laufen und ein einzelner
  leerer Rückgabewert die Pipeline nicht sinnlos macht.
- Bei einem einzelnen, für den Lauf zwingenden Abruf: required weglassen (Default true).

Step-Schemas nach Typ:

router_match:
{"id": "route", "type": "router_match", "rules": [{"if": "trigger_payload.X == 'value'", "then": "route_a"}], "default": "idle", "output_key": "route"}

router_llm:
{"id": "route", "type": "router_llm", "prompt": "Entscheide welcher Pfad gilt.", "output_key": "route"}

llm_extract:
{"id": "extract", "type": "llm_extract", "prompt": "Extrahiere X aus {{source_key}}. Antworte NUR mit rohem JSON: {\"field\": \"...\"}", "output_key": "extracted", "only_if_route": "route_name"}

llm_decide:
{"id": "decide", "type": "llm_decide", "prompt": "Bewerte X anhand von {{data}}. Antworte NUR mit rohem JSON: {\"verdict\": true oder false}", "output_key": "decision"}

llm_analyze:
{"id": "analyze", "type": "llm_analyze", "prompt": "Analysiere X auf Basis von {{source_a}} und {{source_b}}. Antworte NUR mit rohem JSON: {\"field\": \"...\"}", "output_key": "analysis_result"}

llm_summarize:
{"id": "summarize", "type": "llm_summarize", "prompt": "Fasse zusammen.", "output_key": "summary"}

web_search:
{"id": "search", "type": "web_search", "query_template": "{{context_key}} relevante begriffe", "prompt": "Fasse zusammen.", "time_range": "week", "categories": "general", "output_key": "search_result", "required": false}

finance:
{"id": "get_quote", "type": "finance", "ticker_key": "selected_ticker", "output_key": "quote_data"}

finance_search:
{"id": "resolve_ticker", "type": "finance_search", "query_key": "selected_isin", "output_key": "company_info"}

http_fetch:
{"id": "fetch", "type": "http_fetch", "url": "https://example.com/api/data", "output_key": "raw_response", "default": ""}

xlsx_fetch:
{"id": "fetch_data", "type": "xlsx_fetch", "url": "https://example.com/data.xlsx", "sheet": 0, "columns": ["<spalte_1>"], "filters": [{"column": "<spalte>", "operator": "<operator>"}], "output_key": "<key>"}

state_read:
{"id": "read_data", "type": "state_read", "key": "my_key", "output_key": "data", "default": "[]"}

state_write:
{"id": "write_data", "type": "state_write", "key": "my_key", "source_key": "context_key_to_save"}

state_init:
{"id": "init_preferences", "type": "state_init", "key": "my_key", "source_key": "extracted_preferences", "default": "[]"}

state_read_external / state_write_external:
{"id": "read_other", "type": "state_read_external", "agent_name": "OtherAgent", "key": "their_key", "output_key": "data", "default": ""}
{"id": "write_other", "type": "state_write_external", "agent_name": "OtherAgent", "key": "their_key", "source_key": "context_key"}

data_read / data_write:
{"id": "read_doc", "type": "data_read", "namespace": "my_namespace", "key_template": "{{context_key}}", "output_key": "document", "default": ""}
{"id": "write_doc", "type": "data_write", "namespace": "my_namespace", "key_template": "{{context_key}}", "source_key": "document_to_save"}

data_read_external / data_write_external:
{"id": "read_doc", "type": "data_read_external", "agent_name": "OtherAgent", "namespace": "their_namespace", "key_template": "{{context_key}}", "output_key": "document", "default": ""}
{"id": "write_doc", "type": "data_write_external", "agent_name": "OtherAgent", "namespace": "their_namespace", "key_template": "{{context_key}}", "source_key": "document"}

transform:
{"id": "append", "type": "transform", "operation": "array_push", "value_key": "price_eur", "group_key": "extracted_model", "target_key": "historical_prices", "output_key": "historical_prices", "max_items": 500}

trigger_agent:
{"id": "trigger", "type": "trigger_agent", "target_agent_name": "TargetAgent", "payload": {"key": "{{context_key}}"}, "delay_minutes": 0}
Bedingt: {"id": "trigger", "type": "trigger_agent", "target_agent_name": "TargetAgent", "payload": {}, "only_if_key": {"key": "decision.verdict", "value": "true"}}

notify_user:
{"id": "notify", "type": "notify_user", "source_key": "message_context_key"}
Bedingt: {"id": "notify", "type": "notify_user", "source_key": "message_context_key", "only_if_key": {"key": "decision.verdict", "value": "true"}}

STRUKTURREGELN:
- only_if_key bevorzugen wenn die einzige Konsequenz ein bedingter trigger_agent oder notify_user ist.
- LLM-Prompts für llm_extract/llm_decide enden mit "Antworte NUR mit rohem JSON: {Felder}"
- only_if_route weglassen wenn der Step auf allen Routen läuft.
- trigger_agent Steps immer nach allen state_write/data_write Steps.
- xlsx_fetch nie durch http_fetch + transform ersetzen."""


async def _generate_pipeline(
    instruction: str,
    decomposition: dict,
    pool: asyncpg.Pool | None = None,
) -> dict | None:
    skill_context = ""
    if pool is not None:
        skill_context = await agent_skills.load_skill_context(pool)
    system = _PIPELINE_GENERATOR_SYSTEM + (
        "\n\n" + skill_context if skill_context else ""
    )
    try:
        content = f"Instruction: {instruction}\n\nKlassifikation:\n{json.dumps(decomposition, ensure_ascii=False, indent=2)}"
        raw = await brain.chat(
            system=system,
            messages=[{"role": "user", "content": content}],
            capability=CAPABILITY_REASONING,
            caller="pipeline_generator",
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            logger.warning("pipeline generator returned non-dict")
            return None
        if not isinstance(parsed.get("steps"), list):
            logger.warning("pipeline generator returned empty structure")
            return None
        logger.info("pipeline generated: %d steps", len(parsed.get("steps", [])))
        return parsed
    except Exception as e:
        logger.warning("pipeline generation failed: %s", e)
        return None


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
            messages=[
                {
                    "role": "user",
                    "content": f"Agenten:\n{agent_list}\n\nNutzeranfrage: {text}",
                }
            ],
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


def next_agent_run_after(schedule: str, timezone: str) -> datetime:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(config.BOT_DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    next_run_local = croniter(schedule, now).get_next(datetime)
    return next_run_local.astimezone(ZoneInfo("UTC"))
