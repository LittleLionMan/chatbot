CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    telegram_id BIGINT PRIMARY KEY,
    username TEXT,
    first_name TEXT,
    last_name TEXT,
    timezone TEXT DEFAULT 'UTC',
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS groups (
    telegram_id BIGINT PRIMARY KEY,
    title TEXT,
    first_seen_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    subject_type TEXT NOT NULL,
    subject_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS memories_subject_idx ON memories (subject_type, subject_id);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    user_id BIGINT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS messages_chat_idx ON messages (chat_id, created_at DESC);

CREATE TABLE IF NOT EXISTS group_cooldowns (
    group_id BIGINT PRIMARY KEY,
    last_spontaneous_at TIMESTAMPTZ DEFAULT '1970-01-01'
);

CREATE TABLE IF NOT EXISTS session_extractions (
    group_id BIGINT PRIMARY KEY,
    last_extracted_at TIMESTAMPTZ DEFAULT '1970-01-01',
    last_message_at TIMESTAMPTZ DEFAULT '1970-01-01'
);

CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    source_chat_id BIGINT NOT NULL,
    target_chat_id BIGINT NOT NULL,
    description TEXT NOT NULL,
    schedule TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS tasks_active_idx ON tasks (is_active, next_run_at);
CREATE INDEX IF NOT EXISTS tasks_user_idx ON tasks (user_id);

CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    target_chat_id BIGINT NOT NULL,
    name TEXT NOT NULL,
    config JSONB NOT NULL,
    schedule TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS agents_active_idx ON agents (is_active, next_run_at);
CREATE INDEX IF NOT EXISTS agents_user_idx ON agents (user_id);

CREATE TABLE IF NOT EXISTS agent_state (
    agent_id INT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (agent_id, key)
);

CREATE TABLE IF NOT EXISTS agent_data (
    id SERIAL PRIMARY KEY,
    agent_id INT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (agent_id, namespace, key)
);

CREATE INDEX IF NOT EXISTS agent_data_namespace_idx ON agent_data (agent_id, namespace);
CREATE INDEX IF NOT EXISTS agent_data_global_idx ON agent_data (namespace, key);

CREATE TABLE IF NOT EXISTS agent_trigger_queue (
    id SERIAL PRIMARY KEY,
    source_agent_id INT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    target_agent_name TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    scheduled_for TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS agent_trigger_queue_pending_idx ON agent_trigger_queue (processed_at) WHERE processed_at IS NULL;

CREATE TABLE IF NOT EXISTS llm_usage (
    id SERIAL PRIMARY KEY,
    caller TEXT NOT NULL,
    input_tokens INT NOT NULL,
    output_tokens INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS llm_usage_created_idx ON llm_usage (created_at DESC);
CREATE INDEX IF NOT EXISTS llm_usage_caller_idx ON llm_usage (caller, created_at DESC);
