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
    with open(os.path.join(os.path.dirname(__file__), "index.html")) as f:
        html = f.read().replace("__API_URL__", f"http://localhost:{os.environ.get('DASHBOARD_PORT', '8001')}")
    return HTMLResponse(html)


class AgentConfigPatch(BaseModel):
    instruction: str | None = None
    schedule: str | None = None
    name: str | None = None


class MemoryBody(BaseModel):
    content: str
    subject_type: str


class MemoryPatch(BaseModel):
    old_content: str
    new_content: str
    subject_type: str


class StatePatch(BaseModel):
    value: str


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
            "state_keys": config.get("state_keys", []),
            "data_reads": config.get("data_reads", []),
            "last_run_at": r["last_run_at"].isoformat() if r["last_run_at"] else None,
            "next_run_at": r["next_run_at"].isoformat() if r["next_run_at"] else None,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "target_chat_id": r["target_chat_id"],
        })
    return result


@app.patch("/api/agents/{agent_id}")
async def patch_agent(agent_id: int, body: AgentConfigPatch) -> dict:
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
    daily = await pool().fetch(
        """
        SELECT DATE(created_at) as day, SUM(input_tokens) as input, SUM(output_tokens) as output
        FROM llm_usage WHERE created_at > NOW() - INTERVAL '14 days'
        GROUP BY DATE(created_at) ORDER BY day ASC
        """
    )
    return {
        "total_input": total["input"] or 0,
        "total_output": total["output"] or 0,
        "by_caller": [{"caller": r["caller"], "input": r["input"], "output": r["output"], "calls": r["calls"]} for r in by_caller],
        "daily": [{"day": str(r["day"]), "input": r["input"], "output": r["output"]} for r in daily],
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
