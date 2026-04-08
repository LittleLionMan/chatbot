from __future__ import annotations
import logging
import os
import httpx
import anthropic
import asyncpg

logger = logging.getLogger(__name__)

Capability = str

CAPABILITY_FAST = "fast"
CAPABILITY_BALANCED = "balanced"
CAPABILITY_SEARCH = "search"
CAPABILITY_REASONING = "reasoning"
CAPABILITY_DEEP_REASONING = "deep_reasoning"
CAPABILITY_CODING = "coding"
CAPABILITY_MULTIMODAL = "multimodal"
CAPABILITY_LONG_CONTEXT = "long_context"

ALL_CAPABILITIES = [
    CAPABILITY_FAST,
    CAPABILITY_BALANCED,
    CAPABILITY_SEARCH,
    CAPABILITY_REASONING,
    CAPABILITY_DEEP_REASONING,
    CAPABILITY_CODING,
    CAPABILITY_MULTIMODAL,
    CAPABILITY_LONG_CONTEXT,
]

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
}

_available_models: list[dict] = []
_capability_model_map: dict[str, str] = {}
_pool_ref: asyncpg.Pool | None = None


def set_pool(pool: asyncpg.Pool) -> None:
    global _pool_ref
    _pool_ref = pool


async def _check_anthropic() -> list[str]:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return []
    try:
        client = anthropic.AsyncAnthropic(api_key=key)
        await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return ["anthropic"]
    except anthropic.AuthenticationError:
        logger.warning("Anthropic API key invalid")
        return []
    except Exception as e:
        logger.warning("Anthropic health check failed: %s", e)
        return []


async def _check_openai_compatible(provider: str, base_url: str, api_key: str, test_model: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": test_model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return resp.status_code in (200, 400)
    except Exception as e:
        logger.warning("%s health check failed: %s", provider, e)
        return False


async def _check_ollama() -> list[str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.info("Ollama not available: %s", e)
        return []


def _build_capability_map() -> None:
    global _capability_model_map
    _capability_model_map = {}

    for capability in ALL_CAPABILITIES:
        candidates = [m for m in _available_models if capability in m["capabilities"]]

        local = [m for m in candidates if m["is_local"]]
        if local:
            _capability_model_map[capability] = local[0]["api_model_name"]
            continue

        if candidates:
            candidates.sort(key=lambda m: m["input_cost_per_mtok"])
            _capability_model_map[capability] = candidates[0]["api_model_name"]

    logger.info("━━━ Model routing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for capability in ALL_CAPABILITIES:
        model = _capability_model_map.get(capability, "— nicht verfügbar —")
        logger.info("  %-14s → %s", capability, model)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


async def run_availability_check(pool: asyncpg.Pool) -> None:
    global _available_models
    set_pool(pool)

    logger.info("Running model availability check...")

    all_models: list[asyncpg.Record] = await pool.fetch(
        "SELECT provider, model_id, api_model_name, capabilities, input_cost_per_mtok, output_cost_per_mtok, context_window, is_local FROM model_registry ORDER BY input_cost_per_mtok ASC NULLS LAST"
    )

    available_providers: set[str] = set()

    anthropic_providers = await _check_anthropic()
    available_providers.update(anthropic_providers)

    _PROVIDER_TEST_MODELS: dict[str, str] = {
        "openai": "gpt-4o-mini",
        "google": "gemini-2.0-flash",
        "mistral": "mistral-small-latest",
        "deepseek": "deepseek-chat",
        "xai": "grok-3-mini-beta",
    }

    for provider, base_url in _PROVIDER_BASE_URLS.items():
        key = os.getenv(_PROVIDER_ENV_KEYS.get(provider, ""))
        if not key:
            logger.info("Provider %s: no API key configured, skipping", provider)
            continue
        ok = await _check_openai_compatible(provider, base_url, key, _PROVIDER_TEST_MODELS[provider])
        if ok:
            available_providers.add(provider)
            logger.info("Provider %s: available", provider)
        else:
            logger.info("Provider %s: health check failed", provider)

    available_ollama_models = await _check_ollama()
    if available_ollama_models:
        available_providers.add("ollama")
        logger.info("Ollama: available with %d models", len(available_ollama_models))

    new_available: list[dict] = []
    for row in all_models:
        provider = row["provider"]
        model_id = row["model_id"]
        api_name = row["api_model_name"]

        is_available = False
        if provider == "ollama":
            is_available = any(
                api_name in m or m.startswith(api_name.split(":")[0])
                for m in available_ollama_models
            )
        else:
            is_available = provider in available_providers

        await pool.execute(
            """
            INSERT INTO model_availability (provider, model_id, is_available, last_checked_at, error_message)
            VALUES ($1, $2, $3, NOW(), NULL)
            ON CONFLICT (provider, model_id) DO UPDATE
            SET is_available = EXCLUDED.is_available, last_checked_at = NOW(), error_message = NULL
            """,
            provider, model_id, is_available,
        )

        if is_available:
            new_available.append({
                "provider": provider,
                "model_id": model_id,
                "api_model_name": api_name,
                "capabilities": list(row["capabilities"]),
                "input_cost_per_mtok": float(row["input_cost_per_mtok"] or 0),
                "output_cost_per_mtok": float(row["output_cost_per_mtok"] or 0),
                "context_window": row["context_window"],
                "is_local": row["is_local"],
            })

    _available_models = new_available
    logger.info(
        "Availability check complete: %d models available across providers: %s",
        len(_available_models),
        sorted(available_providers),
    )

    _build_capability_map()


def select_model(capability: Capability, fallback_capability: Capability | None = None) -> str:
    if capability in _capability_model_map:
        return _capability_model_map[capability]
    if fallback_capability and fallback_capability in _capability_model_map:
        return _capability_model_map[fallback_capability]
    return os.getenv("LLM_MODEL", "claude-sonnet-4-6")


def get_provider_for_model(api_model_name: str) -> str:
    for m in _available_models:
        if m["api_model_name"] == api_model_name:
            return m["provider"]
    return "anthropic"


def get_available_summary() -> list[dict]:
    return [
        {
            "provider": m["provider"],
            "model_id": m["model_id"],
            "api_model_name": m["api_model_name"],
            "capabilities": m["capabilities"],
            "input_cost_per_mtok": m["input_cost_per_mtok"],
            "is_local": m["is_local"],
        }
        for m in _available_models
    ]
