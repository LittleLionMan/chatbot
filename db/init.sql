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
