from __future__ import annotations
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pool
    _pool = await asyncpg.create_pool(
        host=os.environ.get("POSTGRES_HOST", "db"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    yield
    await _pool.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Pool not initialized")
    return _pool


def _parse_config(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            result = json.loads(raw)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    return {}


@app.get("/")
async def serve_dashboard() -> HTMLResponse:
    host = os.environ.get("DASHBOARD_HOST", "localhost")
    port = os.environ.get("DASHBOARD_PORT", "8001")
    with open(os.path.join(os.path.dirname(__file__), "index.html")) as f:
        html = f.read()
    return HTMLResponse(html)


class AgentPatch(BaseModel):
    name: str | None = None
    schedule: str | None = None
    instruction: str | None = None
    pipeline: list | None = None
    pipeline_after_template: list | None = None


class MemoryBody(BaseModel):
    content: str
    subject_type: str


class MemoryPatch(BaseModel):
    old_content: str
    new_content: str
    subject_type: str


class StatePatch(BaseModel):
    value: str


class AgentDataBody(BaseModel):
    namespace: str
    key: str
    value: str


class AgentDataPatch(BaseModel):
    value: str


@app.get("/api/capabilities")
async def get_capabilities() -> list[str]:
    rows = await pool().fetch(
        "SELECT DISTINCT unnest(capabilities) as cap FROM model_registry ORDER BY cap"
    )
    caps = [r["cap"] for r in rows]
    return caps


@app.get("/api/step-types")
async def get_step_types() -> list[str]:
    return [
        "router_match",
        "router_llm",
        "llm_extract",
        "llm_decide",
        "llm_summarize",
        "web_search",
        "finance",
        "state_read",
        "state_write",
        "state_read_external",
        "state_write_external",
        "data_read",
        "data_write",
        "data_read_external",
        "data_write_external",
        "transform",
        "trigger_agent",
        "notify_user",
    ]


@app.get("/api/agents")
async def get_agents() -> list[dict]:
    rows = await pool().fetch(
        """
        SELECT id, user_id, name, config, schedule, is_active,
               last_run_at, next_run_at, created_at, target_chat_id
        FROM agents
        ORDER BY is_active DESC, created_at DESC
        """
    )
    result = []
    for r in rows:
        config = _parse_config(r["config"])
        result.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "name": r["name"],
            "is_active": r["is_active"],
            "schedule": r["schedule"],
            "instruction": config.get("instruction", ""),
            "type": config.get("type", ""),
            "pipeline": config.get("pipeline", []),
            "pipeline_after_template": config.get("pipeline_after_template", []),
            "data_reads": config.get("data_reads", []),
            "last_run_at": r["last_run_at"].isoformat() if r["last_run_at"] else None,
            "next_run_at": r["next_run_at"].isoformat() if r["next_run_at"] else None,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "target_chat_id": r["target_chat_id"],
        })
    return result


@app.patch("/api/agents/{agent_id}")
async def patch_agent(agent_id: int, body: AgentPatch) -> dict:
    row = await pool().fetchrow(
        "SELECT name, config, schedule FROM agents WHERE id = $1", agent_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    config = _parse_config(row["config"])
    new_name = body.name if body.name is not None else row["name"]
    new_schedule = body.schedule if body.schedule is not None else row["schedule"]
    if body.instruction is not None:
        config["instruction"] = body.instruction
    if body.pipeline is not None:
        config["pipeline"] = body.pipeline
    if body.pipeline_after_template is not None:
        config["pipeline_after_template"] = body.pipeline_after_template
    await pool().execute(
        "UPDATE agents SET name = $1, schedule = $2, config = $3 WHERE id = $4",
        new_name, new_schedule, json.dumps(config), agent_id,
    )
    return {"ok": True}


@app.delete("/api/agents/{agent_id}")
async def deactivate_agent(agent_id: int) -> dict:
    await pool().execute(
        "UPDATE agents SET is_active = FALSE WHERE id = $1", agent_id
    )
    return {"ok": True}


@app.post("/api/agents/{agent_id}/trigger")
async def trigger_agent(agent_id: int) -> dict:
    row = await pool().fetchrow(
        "SELECT name FROM agents WHERE id = $1 AND is_active = TRUE", agent_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found or inactive")
    await pool().execute(
        """
        INSERT INTO agent_trigger_queue (source_agent_id, target_agent_name, payload, scheduled_for)
        VALUES ($1, $2, '{}', NOW())
        """,
        agent_id, row["name"],
    )
    return {"ok": True}


@app.get("/api/agents/{agent_id}/state")
async def get_agent_state(agent_id: int) -> list[dict]:
    rows = await pool().fetch(
        "SELECT key, value, updated_at FROM agent_state WHERE agent_id = $1 ORDER BY key",
        agent_id,
    )
    return [{"key": r["key"], "value": r["value"], "updated_at": r["updated_at"].isoformat()} for r in rows]


@app.patch("/api/agents/{agent_id}/state/{key}")
async def patch_agent_state(agent_id: int, key: str, body: StatePatch) -> dict:
    await pool().execute(
        """
        INSERT INTO agent_state (agent_id, key, value, updated_at)
        VALUES ($1, $2, $3, NOW())
        ON CONFLICT (agent_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """,
        agent_id, key, body.value,
    )
    return {"ok": True}


@app.delete("/api/agents/{agent_id}/state/{key}")
async def delete_agent_state(agent_id: int, key: str) -> dict:
    await pool().execute(
        "DELETE FROM agent_state WHERE agent_id = $1 AND key = $2",
        agent_id, key,
    )
    return {"ok": True}


@app.get("/api/agents/{agent_id}/memories")
async def get_agent_memories(agent_id: int) -> list[dict]:
    rows = await pool().fetch(
        """
        SELECT id, content, created_at FROM memories
        WHERE subject_type = 'agent' AND subject_id = $1
        ORDER BY created_at DESC LIMIT 50
        """,
        agent_id,
    )
    return [{"id": r["id"], "content": r["content"], "created_at": r["created_at"].isoformat()} for r in rows]


@app.post("/api/agents/{agent_id}/memories")
async def add_agent_memory(agent_id: int, body: MemoryBody) -> dict:
    await pool().execute(
        "INSERT INTO memories (subject_type, subject_id, content) VALUES ('agent', $1, $2)",
        agent_id, body.content,
    )
    return {"ok": True}


@app.patch("/api/agents/{agent_id}/memories")
async def patch_agent_memory(agent_id: int, body: MemoryPatch) -> dict:
    await pool().execute(
        "UPDATE memories SET content = $1 WHERE subject_type = 'agent' AND subject_id = $2 AND content = $3",
        body.new_content, agent_id, body.old_content,
    )
    return {"ok": True}


@app.delete("/api/agents/{agent_id}/memories")
async def delete_agent_memory(agent_id: int, body: MemoryBody) -> dict:
    await pool().execute(
        "DELETE FROM memories WHERE subject_type = 'agent' AND subject_id = $1 AND content = $2",
        agent_id, body.content,
    )
    return {"ok": True}


@app.get("/api/agents/{agent_id}/data")
async def get_agent_data(agent_id: int) -> list[dict]:
    rows = await pool().fetch(
        "SELECT namespace, key, value, updated_at FROM agent_data WHERE agent_id = $1 ORDER BY namespace, key",
        agent_id,
    )
    return [{"namespace": r["namespace"], "key": r["key"], "value": r["value"], "updated_at": r["updated_at"].isoformat()} for r in rows]


@app.post("/api/agents/{agent_id}/data")
async def add_agent_data(agent_id: int, body: AgentDataBody) -> dict:
    await pool().execute(
        """
        INSERT INTO agent_data (agent_id, namespace, key, value, updated_at)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (agent_id, namespace, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """,
        agent_id, body.namespace, body.key, body.value,
    )
    return {"ok": True}


@app.patch("/api/agents/{agent_id}/data/{namespace}/{key}")
async def patch_agent_data(agent_id: int, namespace: str, key: str, body: AgentDataPatch) -> dict:
    await pool().execute(
        """
        UPDATE agent_data SET value = $1, updated_at = NOW()
        WHERE agent_id = $2 AND namespace = $3 AND key = $4
        """,
        body.value, agent_id, namespace, key,
    )
    return {"ok": True}


@app.delete("/api/agents/{agent_id}/data/{namespace}/{key}")
async def delete_agent_data(agent_id: int, namespace: str, key: str) -> dict:
    await pool().execute(
        "DELETE FROM agent_data WHERE agent_id = $1 AND namespace = $2 AND key = $3",
        agent_id, namespace, key,
    )
    return {"ok": True}


@app.get("/api/users")
async def get_users() -> list[dict]:
    rows = await pool().fetch(
        "SELECT telegram_id, username, first_name, last_name, timezone, last_seen_at FROM users ORDER BY last_seen_at DESC"
    )
    return [
        {
            "id": r["telegram_id"],
            "username": r["username"],
            "first_name": r["first_name"],
            "last_name": r["last_name"],
            "timezone": r["timezone"],
            "last_seen_at": r["last_seen_at"].isoformat() if r["last_seen_at"] else None,
        }
        for r in rows
    ]


@app.get("/api/users/{user_id}/memories")
async def get_user_memories(user_id: int) -> list[dict]:
    rows = await pool().fetch(
        """
        SELECT id, subject_type, content, created_at FROM memories
        WHERE subject_type IN ('user', 'reflection') AND subject_id = $1
        ORDER BY created_at DESC LIMIT 100
        """,
        user_id,
    )
    return [{"id": r["id"], "type": r["subject_type"], "content": r["content"], "created_at": r["created_at"].isoformat()} for r in rows]


@app.post("/api/users/{user_id}/memories")
async def add_user_memory(user_id: int, body: MemoryBody) -> dict:
    await pool().execute(
        "INSERT INTO memories (subject_type, subject_id, content) VALUES ($1, $2, $3)",
        body.subject_type, user_id, body.content,
    )
    return {"ok": True}


@app.patch("/api/users/{user_id}/memories")
async def patch_user_memory(user_id: int, body: MemoryPatch) -> dict:
    await pool().execute(
        "UPDATE memories SET content = $1 WHERE subject_type = $2 AND subject_id = $3 AND content = $4",
        body.new_content, body.subject_type, user_id, body.old_content,
    )
    return {"ok": True}


@app.delete("/api/users/{user_id}/memories")
async def delete_user_memory(user_id: int, body: MemoryBody) -> dict:
    await pool().execute(
        "DELETE FROM memories WHERE subject_type = $1 AND subject_id = $2 AND content = $3",
        body.subject_type, user_id, body.content,
    )
    return {"ok": True}


@app.get("/api/groups")
async def get_groups() -> list[dict]:
    rows = await pool().fetch(
        "SELECT telegram_id, title, first_seen_at FROM groups ORDER BY first_seen_at DESC"
    )
    return [
        {
            "id": r["telegram_id"],
            "title": r["title"],
            "first_seen_at": r["first_seen_at"].isoformat() if r["first_seen_at"] else None,
        }
        for r in rows
    ]


@app.get("/api/groups/{group_id}/memories")
async def get_group_memories(group_id: int) -> list[dict]:
    rows = await pool().fetch(
        """
        SELECT id, subject_type, content, created_at FROM memories
        WHERE subject_type IN ('group', 'bot', 'reflection') AND subject_id = $1
        ORDER BY created_at DESC LIMIT 100
        """,
        group_id,
    )
    return [{"id": r["id"], "type": r["subject_type"], "content": r["content"], "created_at": r["created_at"].isoformat()} for r in rows]


@app.post("/api/groups/{group_id}/memories")
async def add_group_memory(group_id: int, body: MemoryBody) -> dict:
    await pool().execute(
        "INSERT INTO memories (subject_type, subject_id, content) VALUES ($1, $2, $3)",
        body.subject_type, group_id, body.content,
    )
    return {"ok": True}


@app.patch("/api/groups/{group_id}/memories")
async def patch_group_memory(group_id: int, body: MemoryPatch) -> dict:
    await pool().execute(
        "UPDATE memories SET content = $1 WHERE subject_type = $2 AND subject_id = $3 AND content = $4",
        body.new_content, body.subject_type, group_id, body.old_content,
    )
    return {"ok": True}


@app.delete("/api/groups/{group_id}/memories")
async def delete_group_memory(group_id: int, body: MemoryBody) -> dict:
    await pool().execute(
        "DELETE FROM memories WHERE subject_type = $1 AND subject_id = $2 AND content = $3",
        body.subject_type, group_id, body.content,
    )
    return {"ok": True}


@app.get("/api/usage")
async def get_usage() -> dict:
    total = await pool().fetchrow(
        "SELECT SUM(input_tokens) as input, SUM(output_tokens) as output FROM llm_usage"
    )
    by_caller = await pool().fetch(
        """
        SELECT caller, SUM(input_tokens) as input, SUM(output_tokens) as output, COUNT(*) as calls
        FROM llm_usage GROUP BY caller
        ORDER BY (SUM(input_tokens) + SUM(output_tokens)) DESC
        """
    )
    by_model = await pool().fetch(
        """
        SELECT
            u.model,
            SUM(u.input_tokens) as input,
            SUM(u.output_tokens) as output,
            COUNT(*) as calls,
            r.input_cost_per_mtok,
            r.output_cost_per_mtok
        FROM llm_usage u
        LEFT JOIN model_registry r ON r.api_model_name = u.model
        WHERE u.model IS NOT NULL
        GROUP BY u.model, r.input_cost_per_mtok, r.output_cost_per_mtok
        ORDER BY (SUM(u.input_tokens) + SUM(u.output_tokens)) DESC
        """
    )
    model_stats = []
    for r in by_model:
        input_cost = float(r["input_cost_per_mtok"] or 0)
        output_cost = float(r["output_cost_per_mtok"] or 0)
        total_cost = (r["input"] * input_cost / 1_000_000) + (r["output"] * output_cost / 1_000_000)
        model_stats.append({
            "model": r["model"],
            "input": r["input"],
            "output": r["output"],
            "calls": r["calls"],
            "input_cost_per_mtok": input_cost,
            "output_cost_per_mtok": output_cost,
            "estimated_cost_usd": round(total_cost, 4),
        })
    return {
        "total_input": total["input"] or 0,
        "total_output": total["output"] or 0,
        "by_caller": [{"caller": r["caller"], "input": r["input"], "output": r["output"], "calls": r["calls"]} for r in by_caller],
        "by_model": model_stats,
    }


@app.get("/api/usage/history")
async def get_usage_history(page: int = 0, limit: int = 10) -> dict:
    offset = page * limit
    rows = await pool().fetch(
        """
        SELECT caller, model, input_tokens, output_tokens, created_at
        FROM llm_usage
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit, offset,
    )
    total = await pool().fetchval("SELECT COUNT(*) FROM llm_usage")
    return {
        "items": [
            {
                "caller": r["caller"],
                "model": r["model"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ],
        "total": total,
        "page": page,
        "limit": limit,
    }


@app.get("/api/triggers")
async def get_triggers() -> list[dict]:
    rows = await pool().fetch(
        """
        SELECT id, source_agent_id, target_agent_name, payload, scheduled_for, created_at, processed_at
        FROM agent_trigger_queue ORDER BY created_at DESC LIMIT 100
        """
    )
    return [
        {
            "id": r["id"],
            "source_agent_id": r["source_agent_id"],
            "target_agent_name": r["target_agent_name"],
            "payload": r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"]),
            "scheduled_for": r["scheduled_for"].isoformat() if r["scheduled_for"] else None,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "processed_at": r["processed_at"].isoformat() if r["processed_at"] else None,
        }
        for r in rows
    ]


@app.get("/api/registry")
async def get_registry() -> dict:
    models = await pool().fetch(
        """
        SELECT
            r.provider, r.model_id, r.display_name, r.api_model_name,
            r.size_class, r.capabilities, r.input_cost_per_mtok,
            r.output_cost_per_mtok, r.context_window, r.max_output_tokens,
            r.is_local, r.notes, r.last_updated_at,
            COALESCE(a.is_available, false) as is_available,
            a.last_checked_at, a.error_message
        FROM model_registry r
        LEFT JOIN model_availability a USING (provider, model_id)
        ORDER BY r.is_local ASC, r.input_cost_per_mtok ASC NULLS LAST
        """
    )
    routing = await pool().fetch(
        """
        SELECT DISTINCT ON (capability)
            unnest(r.capabilities) as capability,
            r.api_model_name,
            r.display_name,
            r.provider,
            r.input_cost_per_mtok,
            r.is_local
        FROM model_registry r
        JOIN model_availability a USING (provider, model_id)
        WHERE a.is_available = TRUE
        ORDER BY capability, r.is_local DESC, r.input_cost_per_mtok ASC NULLS LAST
        """
    )
    return {
        "models": [
            {
                "provider": r["provider"],
                "model_id": r["model_id"],
                "display_name": r["display_name"],
                "api_model_name": r["api_model_name"],
                "size_class": r["size_class"],
                "capabilities": list(r["capabilities"]),
                "input_cost_per_mtok": float(r["input_cost_per_mtok"] or 0),
                "output_cost_per_mtok": float(r["output_cost_per_mtok"] or 0),
                "context_window": r["context_window"],
                "max_output_tokens": r["max_output_tokens"],
                "is_local": r["is_local"],
                "notes": r["notes"],
                "is_available": r["is_available"],
                "last_checked_at": r["last_checked_at"].isoformat() if r["last_checked_at"] else None,
                "error_message": r["error_message"],
            }
            for r in models
        ],
        "routing": [
            {
                "capability": r["capability"],
                "model": r["api_model_name"],
                "display_name": r["display_name"],
                "provider": r["provider"],
                "input_cost_per_mtok": float(r["input_cost_per_mtok"] or 0),
                "is_local": r["is_local"],
            }
            for r in routing
        ],
    }


@app.get("/api/scrapers")
async def get_scrapers() -> list[dict]:
    rows = await pool().fetch(
        """
        SELECT id, platform, category, query, filters, target_agent,
               poll_interval_seconds, is_active, last_scraped_at, created_at
        FROM scraper_configs
        ORDER BY is_active DESC, created_at DESC
        """
    )
    result = []
    for r in rows:
        filters = r["filters"]
        if isinstance(filters, str):
            filters = json.loads(filters or "{}")
        result.append({
            "id": r["id"],
            "platform": r["platform"],
            "category": r["category"],
            "query": r["query"],
            "filters": filters,
            "target_agent": r["target_agent"],
            "poll_interval_seconds": r["poll_interval_seconds"],
            "is_active": r["is_active"],
            "last_scraped_at": r["last_scraped_at"].isoformat() if r["last_scraped_at"] else None,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        })
    return result


@app.delete("/api/scrapers/{config_id}")
async def deactivate_scraper(config_id: int) -> dict:
    await pool().execute(
        "UPDATE scraper_configs SET is_active = FALSE WHERE id = $1", config_id
    )
    return {"ok": True}


@app.get("/api/listings")
async def get_listings(
    category: str | None = None,
    platform: str | None = None,
    limit: int = 50,
) -> list[dict]:
    rows = await pool().fetch(
        """
        SELECT id, platform, category, external_id, url, title, price, currency,
               location, condition, seller_name, seller_rating, attributes,
               first_seen_at, last_seen_at
        FROM listings
        WHERE ($1::text IS NULL OR category = $1)
          AND ($2::text IS NULL OR platform = $2)
        ORDER BY first_seen_at DESC
        LIMIT $3
        """,
        category, platform, limit,
    )
    result = []
    for r in rows:
        attrs = r["attributes"]
        if isinstance(attrs, str):
            attrs = json.loads(attrs or "{}")
        result.append({
            "id": r["id"],
            "platform": r["platform"],
            "category": r["category"],
            "url": r["url"],
            "title": r["title"],
            "price": float(r["price"]) if r["price"] is not None else None,
            "currency": r["currency"],
            "location": r["location"],
            "condition": r["condition"],
            "seller_name": r["seller_name"],
            "seller_rating": float(r["seller_rating"]) if r["seller_rating"] is not None else None,
            "attributes": attrs,
            "first_seen_at": r["first_seen_at"].isoformat() if r["first_seen_at"] else None,
            "last_seen_at": r["last_seen_at"].isoformat() if r["last_seen_at"] else None,
        })
    return result


@app.delete("/api/listings/{listing_id}")
async def delete_listing(listing_id: int) -> dict:
    await pool().execute("DELETE FROM listings WHERE id = $1", listing_id)
    return {"ok": True}
