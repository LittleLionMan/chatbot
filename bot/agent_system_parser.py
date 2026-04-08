from __future__ import annotations
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from croniter import croniter
import asyncpg
from bot import brain, memory
from bot.agent_parser import _classify_work_capability, _generate_pipeline
from bot.models import CAPABILITY_BALANCED, CAPABILITY_DEEP_REASONING
from bot.utils import clean_llm_json

logger = logging.getLogger(__name__)

_SYSTEM_PARSER_PROMPT = """Du entwirfst ein koordiniertes Multi-Agent-System aus einem Freitext-Prompt.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "agents": Liste von Agent-Konfigurationen. Jeder Agent hat:
  - "name": Thematisch passender menschlicher Name
  - "instruction": Vollständige eigenständige Anweisung, maximal 400 Zeichen
  - "state_keys": Keys die zwischen Läufen im State bleiben. Immer "last_run_summary". Nur kompakte Daten — Listen, Flags, kurze Zusammenfassungen. Keine langen Texte.
  - "data_reads": Lesevorgänge vor jedem Lauf. Zwei Typen:
    - {"type": "state", "agent_name": "..."} — liest den State eines anderen Agents
    - {"type": "namespace", "namespace": "...", "agent_name": "...", "key": "..."} — liest DB-Namespace eines anderen Agents, key optional, Template-Variablen möglich
    Leer wenn keine fremden Daten nötig.
  - "type": Schlagwort: monitoring, research, finance, news, coding, market
  - "schedule": Cron-Expression (5 Felder)
  - "target": "same" oder "dm"
- "description": Menschenlesbare Zusammenfassung des geplanten Systems. Erkläre jeden Agent mit Name, Zeitplan und Aufgabe. Benenne explizit welcher Agent wessen State liest und welche Trigger-Beziehungen bestehen. Formuliere so dass Missverständnisse sofort auffallen. Schließe mit "Soll ich das so anlegen?"

Regeln für gute Architektur:
- Der erste Agent (Sammler/Screener) hat keine data_reads — er ist die primäre Datenquelle
- Folge-Agents lesen den State des Sammlers via type:state
- Lange Texte (Analysen, Berichte) gehören in einen Namespace, nicht in den State
- Trigger-Beziehungen entstehen wenn ein Agent einen anderen bei bestimmten Bedingungen anstoßen soll
- Schedules staffeln: Sammler früh, Analyst danach, Monitor abends
- Namen thematisch passend: Finance → Gordon/Warren, News → Wolf/Anna, Monitoring → Argus/HAL

Beispiel-Input: "Beobachte GPU-Preise täglich, analysiere interessante Funde und halte mich über Marktveränderungen auf dem Laufenden"

Beispiel-Output:
{"agents": [{"name": "Linus", "instruction": "Suche täglich nach RTX-GPU-Angeboten unter 300€ auf deutschen Secondhand-Plattformen. Pflege eine Liste aller bekannten Angebote mit Preis, Zustand und Link in deinem State.", "state_keys": ["last_run_summary", "known_listings", "price_baseline"], "data_reads": [], "type": "research", "schedule": "0 8 * * *", "target": "same"}, {"name": "Gordon", "instruction": "Lies Linus' Angebotsliste. Analysiere ob neue Angebote einen echten Deal darstellen — Preisvergleich, Zustand, Verkäufer-Reputation. Melde nur echte Schnäppchen.", "state_keys": ["last_run_summary", "analyzed_listings"], "data_reads": [{"type": "state", "agent_name": "Linus"}], "type": "market", "schedule": "0 9 * * *", "target": "same"}], "description": "Ich würde das so aufsetzen:\\n\\n**Linus** (täglich 8 Uhr) — sucht GPU-Angebote und pflegt eine Liste in seinem State.\\n\\n**Gordon** (täglich 9 Uhr) — liest Linus' State und bewertet neue Angebote.\\n\\nAbhängigkeiten: Gordon liest Linus.\\n\\nSoll ich das so anlegen?"}"""


async def parse_agent_system(
    text: str,
    user_id: int,
    source_chat_id: int,
    pool: asyncpg.Pool,
) -> dict | None:
    try:
        raw = await brain.chat(
            system=_SYSTEM_PARSER_PROMPT,
            messages=[{"role": "user", "content": text}],
            capability=CAPABILITY_DEEP_REASONING,
            caller="agent_system_parser",
            pool=pool,
        )
        logger.debug("System parser raw output: %r", raw[:200])
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            return None

        agents_raw: list[dict] = parsed.get("agents", [])
        description: str = parsed.get("description", "")

        if not agents_raw or not description:
            return None

        tz_str = await memory.get_user_timezone(pool, user_id)
        try:
            tz = ZoneInfo(tz_str)
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)
        agents_prepared: list[dict] = []

        for agent_raw in agents_raw:
            schedule = agent_raw.get("schedule")
            if not schedule or not croniter.is_valid(schedule):
                logger.warning("System parser: invalid schedule for agent %s: %s", agent_raw.get("name"), schedule)
                return None

            instruction = agent_raw.get("instruction", "").strip()
            if not instruction:
                return None

            work_capability = await _classify_work_capability(instruction)
            logger.info("Agent '%s' work_capability classified as: %s", agent_raw.get("name"), work_capability)

            state_keys: list[str] = agent_raw.get("state_keys", ["last_run_summary"])
            pipeline_result = await _generate_pipeline(instruction, work_capability, state_keys)
            if pipeline_result:
                has_tmpl = bool(pipeline_result.get("pipeline_template"))
                total = len(pipeline_result.get("pipeline", [])) + len(pipeline_result.get("pipeline_after_template", []))
                logger.info("Agent '%s' pipeline: %d fixed steps, template=%s", agent_raw.get("name"), total, has_tmpl)

            next_run_local = croniter(schedule, now).get_next(datetime)
            next_run_utc = next_run_local.astimezone(ZoneInfo("UTC"))

            agent_config: dict = {
                "instruction": instruction,
                "state_keys": state_keys,
                "data_reads": agent_raw.get("data_reads", []),
                "type": agent_raw.get("type", "default"),
                "work_capability": work_capability,
            }
            if pipeline_result:
                if pipeline_result.get("pipeline"):
                    agent_config["pipeline"] = pipeline_result["pipeline"]
                if pipeline_result.get("pipeline_template"):
                    agent_config["pipeline_template"] = pipeline_result["pipeline_template"]
                if pipeline_result.get("pipeline_after_template"):
                    agent_config["pipeline_after_template"] = pipeline_result["pipeline_after_template"]

            agents_prepared.append({
                "name": agent_raw.get("name", "Agent"),
                "config": agent_config,
                "schedule": schedule,
                "target_chat_id": user_id if agent_raw.get("target") == "dm" else source_chat_id,
                "next_run_at": next_run_utc,
            })

        return {
            "agents": agents_prepared,
            "description": description,
        }

    except Exception as e:
        logger.warning("Agent system parsing failed: %s", e)
        return None
