from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg

from bot import brain
from bot.models import CAPABILITY_REASONING
from bot.utils import clean_llm_json, parse_agent_config

logger = logging.getLogger(__name__)

_DIRTY_KEY = "pipeline_patterns_dirty"
_PATTERNS_KEY = "pipeline_patterns"

RATINGS: list[str] = ["perfekt", "sehr_gut", "gut", "ausreichend", "ungenuegend"]

_RATING_WEIGHTS: dict[str, float] = {
    "perfekt": 1.0,
    "sehr_gut": 0.8,
    "gut": 0.5,
    "ausreichend": 0.1,
    "ungenuegend": -1.0,
}

_STABILITY_MIN_DAYS = 14
_STABILITY_MIN_RUNS = 3

_PATTERN_EXTRACTOR_SYSTEM = """Du analysierst eine Sammlung von Agent-Pipelines und destillierst daraus wiederverwendbare Muster.

Du bekommst:
1. Fertige, produktiv laufende Agent-Pipelines mit Qualitätsbewertungen
2. Manuell korrigierte Pipeline-Diffs (LLM-Output → menschlich korrigierter Stand)

Qualitätsstufen der Bewertungen:
- perfekt (1.0): Lerne alles von diesen Agents — exakte Ground Truth
- sehr_gut (0.8): Starkes Positivbeispiel, minimale LLM-Varianz ist normal
- gut (0.5): Schwaches Positivbeispiel — Muster übernehmen aber mit Vorbehalt
- ausreichend (0.1): Nicht als Vorbild verwenden, aber Fehler daraus extrahieren
- ungenuegend (-1.0): Explizite Negativbeispiele — was vermieden werden soll

Antworte NUR mit einem JSON-Objekt, kein anderer Text, keine Markdown-Backticks.

Felder:
- "step_patterns": Liste bewährter Muster. Jedes Muster hat:
  - "trigger": Wann taucht dieses Muster auf? Instruction-Situation in 1-2 Sätzen.
  - "pattern": Konkrete Step-Sequenz, z.B. "state_read → finance → llm_decide → notify_user"
  - "rationale": Warum ist das so? Was geht schief wenn man es anders macht?
  - "frequency": "always", "often", "sometimes"
  - "confidence": float 0.0-1.0 basierend auf Bewertungsgewichten der Quell-Agents

- "common_mistakes": Fehler die LLMs typischerweise machen. Jeder Eintrag:
  - "mistake": Was wird falsch generiert?
  - "correction": Was ist richtig?
  - "source": "llm_generated" wenn aus Diff, "low_rated" wenn aus ausreichend/ungenuegend Agent

- "step_ordering_rules": Reihenfolge-Regeln als Strings.

- "context_requirements": Dict step_type → Liste von Vorgänger-Step-Types.

- "anti_patterns": Was explizit vermieden werden soll (aus ausreichend/ungenuegend Agents und Notizen).
  Jeder Eintrag: "pattern" und "reason".

Nur konkrete, handlungsrelevante Erkenntnisse. Allgemeinplätze sind wertlos."""


async def _is_dirty(pool: asyncpg.Pool) -> bool:
    try:
        row = await pool.fetchrow(
            "SELECT value FROM skill_store WHERE key = $1", _DIRTY_KEY
        )
        if not row:
            return True
        val = row["value"]
        if isinstance(val, str):
            return val.strip('"') == "true"
        return bool(val)
    except Exception:
        return True


async def _set_dirty(pool: asyncpg.Pool, dirty: bool) -> None:
    await pool.execute(
        """
        INSERT INTO skill_store (key, value, updated_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """,
        _DIRTY_KEY,
        json.dumps(dirty),
    )


async def mark_dirty(pool: asyncpg.Pool) -> None:
    await _set_dirty(pool, True)


async def _load_agents_with_ratings(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, name, config, current_rating, current_rating_note,
               last_rated_at, last_run_at
        FROM agents
        WHERE is_active = TRUE
        ORDER BY created_at ASC
        """
    )
    result: list[dict] = []
    for r in rows:
        config = parse_agent_config(r["config"])
        steps = config.get("steps") or config.get("pipeline", []) + config.get(
            "pipeline_after_template", []
        )
        if not steps:
            continue
        result.append(
            {
                "id": r["id"],
                "name": r["name"],
                "instruction": config.get("instruction", ""),
                "type": config.get("type", "unknown"),
                "steps": steps,
                "rating": r["current_rating"],
                "rating_note": r["current_rating_note"],
                "last_run_at": r["last_run_at"],
            }
        )
    return result


async def _load_stable_edits(pool: asyncpg.Pool) -> list[dict]:
    try:
        rows = await pool.fetch(
            """
            SELECT pe.agent_type, pe.instruction, pe.original_steps, pe.edited_steps,
                   a.current_rating, a.current_rating_note
            FROM pipeline_edits pe
            LEFT JOIN agents a ON a.id = pe.agent_id
            WHERE pe.stable_since IS NOT NULL
            ORDER BY pe.updated_at DESC
            LIMIT 50
            """
        )
    except Exception:
        return []
    result: list[dict] = []
    for r in rows:
        orig = r["original_steps"]
        edited = r["edited_steps"]
        if isinstance(orig, str):
            orig = json.loads(orig)
        if isinstance(edited, str):
            edited = json.loads(edited)
        result.append(
            {
                "agent_type": r["agent_type"],
                "instruction": r["instruction"],
                "original_steps": orig,
                "edited_steps": edited,
                "rating": r["current_rating"],
                "rating_note": r["current_rating_note"],
            }
        )
    return result


def summarize_steps(steps: list[dict]) -> str:
    parts: list[str] = []
    for s in steps:
        entry = f"{s.get('id', '?')}:{s.get('type', '?')}"
        if s.get("operation"):
            entry += f"({s['operation']})"
        if s.get("output_key"):
            entry += f"→{s['output_key']}"
        parts.append(entry)
    return " → ".join(parts)


def _build_extraction_content(agents: list[dict], edits: list[dict]) -> str:
    lines: list[str] = ["# Produktiv laufende Agents\n"]
    for a in agents:
        rating = a.get("rating") or "nicht bewertet"
        weight = _RATING_WEIGHTS.get(rating, 0.3)
        lines.append(f"## {a['name']} | Bewertung: {rating} (Gewicht: {weight})")
        if a.get("rating_note"):
            lines.append(f"Notiz: {a['rating_note']}")
        lines.append(f"Instruction: {a['instruction'][:300]}")
        lines.append(f"Pipeline: {summarize_steps(a['steps'])}")
        lines.append(f"Steps ({len(a['steps'])}):")
        for s in a["steps"]:
            lines.append(f"  - {json.dumps(s, ensure_ascii=False)[:200]}")
        lines.append("")

    if edits:
        lines.append("\n# Manuell korrigierte Pipelines (nur stabile Edits)\n")
        for e in edits:
            rating = e.get("rating") or "nicht bewertet"
            lines.append(f"## Typ: {e['agent_type']} | Agent-Bewertung: {rating}")
            if e.get("rating_note"):
                lines.append(f"Notiz: {e['rating_note']}")
            lines.append(f"Instruction: {e['instruction'][:200]}")
            lines.append(f"LLM-Original: {summarize_steps(e['original_steps'])}")
            lines.append(f"Menschlich korrigiert: {summarize_steps(e['edited_steps'])}")
            lines.append("")

    return "\n".join(lines)


async def extract_and_store_patterns(pool: asyncpg.Pool, force: bool = False) -> bool:
    if not force and not await _is_dirty(pool):
        logger.debug("agent_skills: not dirty, skipping extraction")
        return False

    agents = await _load_agents_with_ratings(pool)
    if not agents:
        logger.info("agent_skills: no active agents with steps, skipping")
        await _set_dirty(pool, False)
        return False

    edits = await _load_stable_edits(pool)
    content = _build_extraction_content(agents, edits)

    logger.info(
        "agent_skills: extracting patterns from %d agents, %d stable edits",
        len(agents),
        len(edits),
    )

    try:
        raw = await brain.chat(
            system=_PATTERN_EXTRACTOR_SYSTEM,
            messages=[{"role": "user", "content": content}],
            capability=CAPABILITY_REASONING,
            caller="agent_skills:extract",
        )
        parsed = json.loads(clean_llm_json(raw))
        if not isinstance(parsed, dict):
            logger.warning("agent_skills: extractor returned non-dict")
            return False
    except Exception as e:
        logger.warning("agent_skills: extraction failed: %s", e)
        return False

    await pool.execute(
        """
        INSERT INTO skill_store (key, value, updated_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """,
        _PATTERNS_KEY,
        json.dumps(parsed),
    )
    await _set_dirty(pool, False)

    logger.info(
        "agent_skills: stored patterns (%d step_patterns, %d mistakes, %d anti_patterns)",
        len(parsed.get("step_patterns", [])),
        len(parsed.get("common_mistakes", [])),
        len(parsed.get("anti_patterns", [])),
    )
    return True


async def load_skill_context(pool: asyncpg.Pool) -> str:
    try:
        row = await pool.fetchrow(
            "SELECT value FROM skill_store WHERE key = $1", _PATTERNS_KEY
        )
        if not row:
            return ""
        parsed = row["value"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if not isinstance(parsed, dict):
            return ""
    except Exception as e:
        logger.warning("agent_skills: failed to load skill context: %s", e)
        return ""

    lines: list[str] = ["## Bekannte Pipeline-Muster aus produktiven Agents\n"]

    for p in parsed.get("step_patterns", []):
        conf = p.get("confidence", "")
        conf_str = f" (Konfidenz: {conf:.1f})" if isinstance(conf, float) else ""
        lines.append(
            f"- [{p.get('frequency', '')}]{conf_str} Wenn: {p.get('trigger', '')}"
        )
        lines.append(f"  Muster: {p.get('pattern', '')}")
        lines.append(f"  Grund: {p.get('rationale', '')}")
    if parsed.get("step_patterns"):
        lines.append("")

    mistakes = parsed.get("common_mistakes", [])
    if mistakes:
        lines.append("### Häufige Fehler — vermeide diese")
        for m in mistakes:
            lines.append(f"- Falsch: {m.get('mistake', '')}")
            lines.append(f"  Richtig: {m.get('correction', '')}")
        lines.append("")

    anti = parsed.get("anti_patterns", [])
    if anti:
        lines.append("### Anti-Patterns — explizit vermeiden")
        for a in anti:
            lines.append(f"- {a.get('pattern', '')}: {a.get('reason', '')}")
        lines.append("")

    ordering = parsed.get("step_ordering_rules", [])
    if ordering:
        lines.append("### Reihenfolge-Regeln")
        for r in ordering:
            lines.append(f"- {r}")
        lines.append("")

    reqs = parsed.get("context_requirements", {})
    if reqs:
        lines.append("### Voraussetzungen zwischen Steps")
        for step_type, preds in reqs.items():
            lines.append(f"- {step_type} braucht davor: {', '.join(preds)}")
        lines.append("")

    return "\n".join(lines)


async def record_pipeline_edit(
    pool: asyncpg.Pool,
    agent_id: int,
    agent_type: str,
    instruction: str,
    original_steps: list[dict],
    edited_steps: list[dict],
    session_id: str,
) -> None:
    try:
        existing = await pool.fetchrow(
            "SELECT id FROM pipeline_edits WHERE agent_id = $1 AND session_id = $2",
            agent_id,
            session_id,
        )
        if existing:
            await pool.execute(
                """
                UPDATE pipeline_edits
                SET edited_steps = $1, updated_at = NOW(), stable_since = NULL
                WHERE id = $2
                """,
                json.dumps(edited_steps),
                existing["id"],
            )
        else:
            await pool.execute(
                """
                INSERT INTO pipeline_edits
                    (agent_id, agent_type, instruction, original_steps,
                     edited_steps, session_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                agent_id,
                agent_type,
                instruction[:500],
                json.dumps(original_steps),
                json.dumps(edited_steps),
                session_id,
            )
        logger.info(
            "agent_skills: recorded edit for agent %d session %s", agent_id, session_id
        )
    except Exception as e:
        logger.warning("agent_skills: failed to record edit: %s", e)


async def update_stability(pool: asyncpg.Pool) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=_STABILITY_MIN_DAYS)
    try:
        rows = await pool.fetch(
            """
            SELECT pe.id, pe.agent_id
            FROM pipeline_edits pe
            JOIN agents a ON a.id = pe.agent_id
            WHERE pe.stable_since IS NULL
              AND pe.updated_at < $1
              AND a.is_active = TRUE
            """,
            cutoff,
        )
        for row in rows:
            run_count = await pool.fetchval(
                """
                SELECT COUNT(*) FROM agent_runs
                WHERE agent_id = $1 AND run_at > $2 AND status = 'ok'
                """,
                row["agent_id"],
                cutoff,
            )
            if (run_count or 0) >= _STABILITY_MIN_RUNS:
                await pool.execute(
                    "UPDATE pipeline_edits SET stable_since = NOW() WHERE id = $1",
                    row["id"],
                )
                logger.info(
                    "agent_skills: marked edit %d as stable (agent %d, %d runs)",
                    row["id"],
                    row["agent_id"],
                    run_count,
                )
    except Exception as e:
        logger.warning("agent_skills: stability update failed: %s", e)


async def set_rating(
    pool: asyncpg.Pool,
    agent_id: int,
    rating: str,
    note: str | None,
) -> None:
    current = await pool.fetchrow(
        "SELECT current_rating FROM agents WHERE id = $1", agent_id
    )
    if current and current["current_rating"] != rating:
        await pool.execute(
            """
            INSERT INTO agent_ratings (agent_id, rating, note)
            VALUES ($1, $2, $3)
            """,
            agent_id,
            rating,
            note,
        )

    await pool.execute(
        """
        UPDATE agents
        SET current_rating = $1, current_rating_note = $2, last_rated_at = NOW()
        WHERE id = $3
        """,
        rating,
        note,
        agent_id,
    )
    await mark_dirty(pool)
    logger.info("agent_skills: rated agent %d as %s", agent_id, rating)


async def get_rating_history(pool: asyncpg.Pool, agent_id: int) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT rating, note, rated_at
        FROM agent_ratings
        WHERE agent_id = $1
        ORDER BY rated_at DESC
        LIMIT 20
        """,
        agent_id,
    )
    return [
        {
            "rating": r["rating"],
            "note": r["note"],
            "rated_at": r["rated_at"].isoformat(),
        }
        for r in rows
    ]


def new_session_id() -> str:
    return str(uuid.uuid4())
