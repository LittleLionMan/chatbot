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
  - "parameters": Strukturierte Parameter die direkt aus der Instruction ableitbar sind. MUSS bei xlsx_fetch gesetzt werden — ohne parameters ist ein xlsx_fetch-Subtask unvollständig und ungültig. Bei xlsx_fetch enthält parameters zwingend: "columns" (alle in der Instruction genannten Spaltennamen als Liste) und "filters" (alle in der Instruction beschriebenen Filterbedingungen als Liste). Jede Filterbedingung hat "column", "operator" und optional "value". Verfügbare Operatoren: not_empty, empty, equals, not_equals, contains, not_contains, starts_with, ends_with. Ein xlsx_fetch-Subtask ohne parameters.filters ist ein Fehler.

Verfügbare Baustein-Typen:

ROUTING:
- router_match: Deterministisches Routing auf Basis exakter Werte in trigger_payload oder context.
- router_llm: LLM-basiertes Routing wenn die Entscheidung Interpretation erfordert.

LLM — nur wenn Urteilsvermögen, Abstraktion oder Sprachverständnis nötig ist:
- llm_extract: Strukturierte Daten aus bekanntem Format extrahieren. Gibt immer JSON zurück. Capability: simple_tasks.
- llm_decide: Bewertung, Klassifikation oder Urteil mit Begründung. Gibt immer JSON zurück. Auch für Set-Operationen zwischen zwei bereits im Context vorhandenen Listen. Capability: reasoning.
- llm_summarize: Zusammenfassung für Menschen oder als Input für weitere Steps. Capability: chat.
- llm_analyze: Tiefgehende Analyse aus heterogenen Quellen mit mehrstufigem Schlussfolgern. Gibt immer JSON zurück. Verwenden wenn Informationen aus verschiedenen Quellen gegeneinander abgewogen werden müssen, Widersprüche erkannt werden müssen, oder ein begründetes Gesamturteil gebildet werden soll (z.B. Fundamentalanalyse, komplexe Risikobewertung, strategische Einschätzung). Capability: deep_reasoning.

DATENZUGRIFF — deterministisch:
- web_search: Websuche wenn die URL nicht bekannt ist oder die Ergebnisse variabel sind.
- finance: Börsenkurse und Finanzkennzahlen für einen Ticker.
- http_fetch: HTTP-Request an eine bekannte URL. Gibt den Response-Body als String zurück. Für strukturierte APIs, XML-Feeds, REST-Endpunkte. Nie für Excel-Dateien verwenden.
- xlsx_fetch: Lädt eine Excel-Datei (.xlsx) von einer URL und gibt ein JSON-Array der Zeilen zurück. Verwende dies immer wenn die Quelle eine .xlsx-Datei ist — nie http_fetch + transform für Excel.
- state_read / state_write: Einzelnen Key im eigenen Agent-State lesen oder schreiben.
- state_read_external / state_write_external: Key im State eines anderen Agenten lesen oder schreiben.
- data_read / data_write: Längere Dokumente im eigenen agent_data Namespace lesen oder schreiben.
- data_read_external / data_write_external: Längere Dokumente im agent_data eines anderen Agenten.

TRANSFORMATION — deterministisch, operiert auf Context-Werten:
- transform: Berechnung oder Strukturänderung auf bereits im Context vorhandenen Daten.
  Operationen: map_field, filter, first, slice, diff, intersect, union, list_append, count, group_by, flatten, sort, statistics, json_path, xml_extract, regex_extract, arithmetic, compare

  Selektion:
  - map_field: Einen bestimmten Key aus jedem Objekt eines JSON-Arrays extrahieren. Ergebnis: JSON-Array oder comma_list.
  - filter: JSON-Array nach Bedingungen filtern. filters-Array mit field, operator, value. Operatoren: equals, not_equals, contains, not_contains, not_empty, empty, starts_with, ends_with, gt, lt.
  - first: Erstes Element eines JSON-Arrays oder einer Liste zurückgeben.
  - slice: Teilmenge eines Arrays. Parameter: start, end (optional).
  - json_path: Einzelnes Feld aus einem JSON-Objekt extrahieren (Dot-Notation). Nicht für Listenoperationen.

  Mengenlehre (alle auf JSON-Arrays oder kommaseparierten Listen):
  - diff: Elemente die in source_key aber nicht in subtract_key sind.
  - intersect: Elemente die in beiden Listen source_key und other_key sind.
  - union: Beide Listen zusammenführen ohne Duplikate.
  - list_append: Einzelnen Wert an ein flaches Array anhängen ohne Duplikat.

  Aggregation:
  - count: Länge eines Arrays als String zurückgeben.
  - group_by: Objekte eines Arrays nach group_field gruppieren. value_field optional — wenn gesetzt, wird nur dieser Wert pro Gruppe gesammelt (für Statistiken), sonst das ganze Objekt. Ersetzt array_push vollständig.
  - statistics: Statistische Kennzahlen auf gruppierten numerischen Listen (erwartet Dict aus group_by).

  Transformation:
  - flatten: Verschachteltes Array flach machen (eine Ebene).
  - sort: Array sortieren. field optional für Objekt-Arrays, reverse für absteigende Reihenfolge.
  - xml_extract, regex_extract, arithmetic, compare: unverändert.

  Hinweis zu json_path: nur für einzelne Felder aus Objekten. Für Listenoperationen die passende Mengenoperation verwenden.

KOORDINATION — deterministisch:
- trigger_agent: Anderen Agenten mit Payload anstoßen. Muss immer nach allen state_write, state_write_external, data_write und data_write_external Steps stehen — nie davor, da der getriggerte Agent sonst auf veralteten State zugreift.
- notify_user: Nachricht direkt an den User senden.

ENTSCHEIDUNGSMATRIX:
Was ist deterministisch — verwende NIE ein LLM dafür:
- Routing auf strukturierten Feldern (type, id, url != null) → router_match
- Excel-Datei (.xlsx) von einer URL → xlsx_fetch (gibt direkt JSON-Array zurück). Filterbedingungen MÜSSEN als parameters.filters im Subtask stehen.
- Bekannte URL mit strukturiertem Response (API, XML, JSON) → http_fetch + transform
- Einen Key aus jedem Objekt eines JSON-Arrays extrahieren → transform(map_field)
- JSON-Array nach Bedingungen filtern → transform(filter)
- Erstes Element eines Arrays → transform(first)
- Teilmenge eines Arrays → transform(slice)
- Elemente in A aber nicht in B → transform(diff)
- Schnittmenge zweier Listen → transform(intersect)
- Zwei Listen zusammenführen ohne Duplikate → transform(union)
- Einzelnen Wert an flaches Array anhängen → transform(list_append)
- Länge einer Liste → transform(count)
- Objekte nach Key gruppieren oder numerische Werte sammeln → transform(group_by)
- Verschachteltes Array flach machen → transform(flatten)
- Array sortieren → transform(sort)
- Statistiken auf gruppierten numerischen Listen → transform(statistics)
- Einzelnes Feld aus JSON-Objekt extrahieren → transform(json_path)
- Wert aus XML extrahieren → transform(xml_extract)
- Wert aus Text per Regex → transform(regex_extract)
- Arithmetik zwischen Werten → transform(arithmetic)
- Numerischer Vergleich → transform(compare)
- Kurze Fakten im State speichern → state_write
- Lange Dokumente speichern → data_write
- Anderen Agent starten → trigger_agent (immer nach allen state_write-Steps)
- User benachrichtigen → notify_user

Was braucht ein LLM:
- Unstrukturierter Text der verstanden werden muss → llm_extract
- Bewertung, Urteil, Entscheidung mit Begründung → llm_decide
- Mengenoperationen zwischen zwei Listen (Differenz, neue Einträge, verlorene Einträge) → llm_decide
- Zusammenfassung für Menschen → llm_summarize
- Websuche wenn URL nicht bekannt → web_search
- Komplexe Analyse aus heterogenen Quellen, mehrstufiges Schlussfolgern, Gesamturteil aus widersprüchlichen Informationen → llm_analyze"""


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

llm_analyze:
{"id": "analyze", "type": "llm_analyze", "prompt": "Analysiere X auf Basis von {{source_a}} und {{source_b}}. Wäge widersprüchliche Informationen ab und bilde ein begründetes Gesamturteil. Antworte NUR mit rohem JSON: {\"field\": \"...\"}", "output_key": "analysis_result"}
Verwenden für: Fundamentalanalysen, komplexe Risikobewertungen, strategische Einschätzungen aus heterogenen Quellen. Capability: deep_reasoning.

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

xlsx_fetch:
{"id": "fetch_data", "type": "xlsx_fetch", "url": "https://example.com/data.xlsx", "sheet": 0, "columns": ["<spalte_1>", "<spalte_2>"], "filters": [{"column": "<spalte>", "operator": "<operator>"}, {"column": "<spalte>", "operator": "<operator>", "value": "<wert>"}], "output_key": "<key>"}
Verfügbare Operatoren für filters: "equals", "not_equals", "contains", "not_contains", "not_empty", "empty", "starts_with", "ends_with".
Mehrere filters werden mit AND verknüpft. Für einen einzelnen einfachen Gleichheitsfilter kann auch die Kurzform verwendet werden: "filter": {"column": "<spalte>", "value": "<wert>"}.
Wenn der Subtask ein "parameters"-Feld hat, müssen dessen Werte direkt übernommen werden — columns aus parameters.columns, filters aus parameters.filters — ohne Interpretation, Ergänzung oder Weglassen.
Hinweis: xlsx_fetch gibt direkt ein JSON-Array zurück. Kein http_fetch + transform für Excel-Dateien — immer xlsx_fetch verwenden.

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
Wichtig: array_push nur für einzelne skalare Werte an gruppierte numerische Arrays (z.B. Preishistorie). Nicht für Mengenoperationen zwischen Listen — dafür llm_decide verwenden.

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

transform map_field:
{"id": "extract_isins", "type": "transform", "operation": "map_field", "source_key": "companies", "field": "isin", "output_key": "isin_list"}
{"id": "extract_names", "type": "transform", "operation": "map_field", "source_key": "companies", "field": "name", "output_format": "comma_list", "output_key": "name_list"}
Extrahiert einen Key aus jedem Objekt eines JSON-Arrays. output_format: "json_array" (Standard) oder "comma_list".

transform filter:
{"id": "filter_active", "type": "transform", "operation": "filter", "source_key": "items", "filters": [{"field": "status", "operator": "equals", "value": "active"}, {"field": "isin", "operator": "not_empty"}], "output_key": "active_items"}
Filtert ein JSON-Array nach Bedingungen (AND-verknüpft). Operatoren: equals, not_equals, contains, not_contains, not_empty, empty, starts_with, ends_with, gt, lt.

transform first:
{"id": "get_first", "type": "transform", "operation": "first", "source_key": "items", "output_key": "first_item", "default": ""}

transform slice:
{"id": "take_ten", "type": "transform", "operation": "slice", "source_key": "items", "start": 0, "end": 10, "output_key": "batch"}

transform diff:
{"id": "find_new", "type": "transform", "operation": "diff", "source_key": "current_list", "subtract_key": "known_list", "output_key": "new_items"}
{"id": "find_new", "type": "transform", "operation": "diff", "source_key": "current_list", "subtract_key": "known_list", "output_format": "comma_list", "output_key": "new_items"}
Gibt Elemente zurück die in source_key aber nicht in subtract_key sind. output_format: "json_array" oder "comma_list".

transform intersect:
{"id": "common", "type": "transform", "operation": "intersect", "source_key": "list_a", "other_key": "list_b", "output_key": "common_items"}

transform union:
{"id": "merged", "type": "transform", "operation": "union", "source_key": "list_a", "other_key": "list_b", "output_key": "merged_list"}

transform list_append:
{"id": "append_isin", "type": "transform", "operation": "list_append", "value_key": "selected_isin", "target_key": "already_reviewed", "output_key": "updated_reviewed"}
Hängt einen einzelnen Wert an ein flaches Array an. Keine Duplikate. target_key ist der bestehende Array-Context-Key.

transform count:
{"id": "count_items", "type": "transform", "operation": "count", "source_key": "items", "output_key": "item_count"}

transform group_by:
{"id": "group_prices", "type": "transform", "operation": "group_by", "source_key": "listings", "group_field": "model", "value_field": "price_eur", "target_key": "historical_prices", "max_items": 500, "output_key": "historical_prices"}
Gruppiert Objekte eines Arrays nach group_field. Mit value_field: sammelt nur diesen Wert pro Gruppe (für statistics). Ohne value_field: sammelt ganze Objekte. target_key: bestehender Dict zum Zusammenführen.

transform flatten:
{"id": "flatten", "type": "transform", "operation": "flatten", "source_key": "nested_list", "output_key": "flat_list"}

transform sort:
{"id": "sort_items", "type": "transform", "operation": "sort", "source_key": "items", "field": "price", "reverse": false, "output_key": "sorted_items"}

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
- Dot-Notation für verschachtelte Felder: wenn ein LLM-Step {"merged_list": [...], "new_models": [...]} zurückgibt mit output_key "merge_result", kann ein nachfolgender Step source_key "merge_result.merged_list" oder condition_key "merge_result.new_models" verwenden — kein extra Transform-Step nötig
- trigger_agent Steps müssen immer nach allen state_write, state_write_external, data_write und data_write_external Steps stehen — nie davor, da der getriggerte Agent sonst auf veralteten State zugreift
- Wenn die Instruction einen anderen Agenten erwähnt der angestoßen, getriggert, benachrichtigt oder gestartet werden soll, muss ein trigger_agent Step generiert werden — auch wenn das Wort "trigger" nicht explizit vorkommt. Erkennungsmerkmale: "starte X", "löse X aus", "informiere X", "übergib an X", "X soll dann laufen"
- xlsx_fetch nie durch http_fetch + transform ersetzen — xlsx_fetch gibt direkt ein JSON-Array zurück"""


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


def next_agent_run_after(schedule: str, timezone: str) -> datetime:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(config.BOT_DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    next_run_local = croniter(schedule, now).get_next(datetime)
    return next_run_local.astimezone(ZoneInfo("UTC"))
