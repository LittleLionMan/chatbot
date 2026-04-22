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
    schedule TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ,
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
    source_agent_id INT REFERENCES agents(id) ON DELETE CASCADE,
    target_agent_name TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    scheduled_for TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS agent_trigger_queue_pending_idx ON agent_trigger_queue (scheduled_for) WHERE processed_at IS NULL;

CREATE TABLE IF NOT EXISTS llm_usage (
    id SERIAL PRIMARY KEY,
    caller TEXT NOT NULL,
    model TEXT,
    input_tokens INT NOT NULL,
    output_tokens INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS llm_usage_created_idx ON llm_usage (created_at DESC);
CREATE INDEX IF NOT EXISTS llm_usage_caller_idx ON llm_usage (caller, created_at DESC);

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    api_model_name TEXT NOT NULL,
    size_class TEXT,
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    input_cost_per_mtok NUMERIC(10, 4),
    output_cost_per_mtok NUMERIC(10, 4),
    context_window INT,
    max_output_tokens INT,
    is_local BOOLEAN NOT NULL DEFAULT FALSE,
    notes TEXT,
    last_updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (provider, model_id)
);

CREATE INDEX IF NOT EXISTS model_registry_provider_idx ON model_registry (provider);
CREATE INDEX IF NOT EXISTS model_registry_capabilities_idx ON model_registry USING GIN (capabilities);

CREATE TABLE IF NOT EXISTS model_availability (
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    is_available BOOLEAN NOT NULL DEFAULT FALSE,
    last_checked_at TIMESTAMPTZ DEFAULT NOW(),
    error_message TEXT,
    PRIMARY KEY (provider, model_id)
);

CREATE TABLE IF NOT EXISTS monitor_configs (
    id SERIAL PRIMARY KEY,
    monitor_type TEXT NOT NULL,
    name TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    source_state_key TEXT NOT NULL,
    source_format TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    feed_templates TEXT[] NOT NULL DEFAULT '{}',
    poll_interval_seconds INT NOT NULL DEFAULT 900,
    extra_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitor_seen (
    config_id INT NOT NULL REFERENCES monitor_configs(id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    seen_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (config_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS monitor_seen_config_idx ON monitor_seen (config_id);

CREATE TABLE IF NOT EXISTS scraper_configs (
    id SERIAL PRIMARY KEY,
    platform TEXT NOT NULL,
    category TEXT NOT NULL,
    query TEXT NOT NULL,
    filters JSONB NOT NULL DEFAULT '{}',
    target_agent TEXT NOT NULL,
    poll_interval_seconds INT NOT NULL DEFAULT 3600,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_scraped_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS scraper_configs_active_idx ON scraper_configs (is_active, last_scraped_at);

CREATE TABLE IF NOT EXISTS listings (
    id SERIAL PRIMARY KEY,
    platform TEXT NOT NULL,
    category TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    price NUMERIC,
    currency TEXT,
    location TEXT,
    condition TEXT,
    seller_name TEXT,
    seller_rating NUMERIC,
    attributes JSONB NOT NULL DEFAULT '{}',
    raw_text TEXT,
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (platform, external_id)
);

CREATE INDEX IF NOT EXISTS listings_platform_category_idx ON listings (platform, category);
CREATE INDEX IF NOT EXISTS listings_first_seen_idx ON listings (first_seen_at DESC);
CREATE INDEX IF NOT EXISTS listings_category_idx ON listings (category, first_seen_at DESC);
