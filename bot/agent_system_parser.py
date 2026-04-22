from __future__ import annotations
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
import asyncpg

from bot import brain, memory
from bot.agent_parser import _decompose_task, _generate_pipeline
from bot.models import CAPABILITY_DEEP_REASONING
from bot.utils import clean_llm_json

logger = logging.getLogger(__name__)

_SYSTEM_PARSER_PROMPT = """Du entwirfst ein koordiniertes Multi-Agent-System aus einem Freitext-Prompt.

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "agents": Liste von Agent-Konfigurationen. Jeder Agent hat:
  - "name": Thematisch passender menschlicher Name
  - "instruction": Vollständige eigenständige Anweisung in natürlicher Sprache.
  - "schedule": Cron-Expression (5 Felder)
  - "target": "same" oder "dm"
- "description": Menschenlesbare Zusammenfassung des geplanten Systems. Erkläre jeden Agent mit Name, Zeitplan und Aufgabe. Benenne explizit welcher Agent wessen Daten liest und welche Trigger-Beziehungen bestehen. Schließe mit "Soll ich das so anlegen?"

Regeln für gute Architektur:
- Sammler-Agents haben keine Abhängigkeiten zu anderen Agents
- Analyse-Agents lesen Daten von Sammler-Agents
- Lange Texte (Analysen, Berichte) gehören in einen Namespace, nicht in den State
- Schedules staffeln: Sammler früh, Analyst danach
- Namen thematisch passend: Finance → Gordon/Warren, News → Wolf/Anna, Monitoring → Argus/HAL

Externe Monitor-Services:
Wenn ein Agent kontinuierlich auf neue Ereignisse reagieren soll, kann ein Monitor-Service helfen.
Verfügbare Monitor-Typen: "rss" für News/Artikel-Tracking.
Wenn relevant, füge in der description einen Hinweis ein:
"Hinweis: Für [Agent-Name] wird ein RSS-Monitor-Service empfohlen."

Beispiel-Output:
{"agents": [{"name": "Linus", "instruction": "Suche täglich nach GPU-Angeboten...", "schedule": "0 8 * * *", "target": "same"}, {"name": "Gordon", "instruction": "Analysiere Linus' gefundene Angebote...", "schedule": "0 9 * * *", "target": "same"}], "description": "Linus sammelt täglich um 8 Uhr GPU-Angebote. Gordon analysiert diese um 9 Uhr...\\n\\nSoll ich das so anlegen?"}"""


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
        logger.debug("system parser raw: %r", raw[:200])
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
                logger.warning("system parser: invalid schedule for %s: %s", agent_raw.get("name"), schedule)
                return None

            instruction = agent_raw.get("instruction", "").strip()
            if not instruction:
                return None

            decomposition = await _decompose_task(instruction)
            if decomposition is None:
                logger.warning("system parser: decomposition failed for %s", agent_raw.get("name"))
                return None

            pipeline_result = await _generate_pipeline(instruction, decomposition)
            agent_type: str = decomposition.get("type", "default")

            logger.info(
                "system parser: agent '%s' type=%s pipeline=%s",
                agent_raw.get("name"),
                agent_type,
                "yes" if pipeline_result else "no",
            )

            next_run_local = croniter(schedule, now).get_next(datetime)
            next_run_utc = next_run_local.astimezone(ZoneInfo("UTC"))

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
        logger.warning("agent system parsing failed: %s", e)
        return None
