from __future__ import annotations
import logging
import httpx
import anthropic
import asyncpg
from bot import config, ratelimit
from bot.models import (
    Capability,
    CAPABILITY_FAST,
    CAPABILITY_BALANCED,
    CAPABILITY_SEARCH,
    CAPABILITY_REASONING,
    CAPABILITY_CODING,
    CAPABILITY_MULTIMODAL,
    get_provider_for_model,
    select_model,
)
from bot.soul import SOUL, BEHAVIOR_RULES as _BEHAVIOR_RULES
from bot.utils import parse_agent_config

logger = logging.getLogger(__name__)

_anthropic_client: anthropic.AsyncAnthropic | None = None

_WEB_SEARCH_TOOL_BASE = {
    "type": "web_search_20250305",
    "name": "web_search",
}

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "ollama": "http://localhost:11434/v1",
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "ollama": "",
}


def _web_search_tool(max_uses: int | None = None) -> dict:
    if max_uses is not None:
        return {**_WEB_SEARCH_TOOL_BASE, "max_uses": max_uses}
    return _WEB_SEARCH_TOOL_BASE


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


async def _call_anthropic(
    system: str,
    messages: list[dict],
    model: str,
    max_tokens: int,
    use_web_search: bool,
    web_search_max_uses: int | None,
    caller: str,
    pool: asyncpg.Pool | None,
) -> str:
    from bot import memory as mem
    client = _get_anthropic_client()
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    if use_web_search:
        kwargs["tools"] = [_web_search_tool(web_search_max_uses)]

    try:
        response = await client.messages.create(**kwargs)

        if pool is not None:
            try:
                await mem.log_llm_usage(
                    pool,
                    caller,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
            except Exception as log_err:
                logger.warning("Token logging failed for caller %s: %s", caller, log_err)

        return "".join(
            block.text for block in response.content if block.type == "text"
        )
    except anthropic.RateLimitError as e:
        retry_after = 3600
        if hasattr(e, "response") and e.response is not None:
            header_val = e.response.headers.get("retry-after")
            if header_val is not None:
                try:
                    retry_after = int(header_val)
                except ValueError:
                    pass
        ratelimit.set_rate_limited(retry_after)
        raise
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        ratelimit.set_no_credits()
        raise


async def _call_openai_compatible(
    system: str,
    messages: list[dict],
    model: str,
    provider: str,
    max_tokens: int,
    caller: str,
    pool: asyncpg.Pool | None,
) -> str:
    from bot import memory as mem
    import os

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1" if provider == "ollama" else _PROVIDER_BASE_URLS[provider]
    env_key = _PROVIDER_ENV_KEYS.get(provider, "")
    api_key = os.getenv(env_key, "ollama") if env_key else "ollama"

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + messages,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        if pool is not None:
            try:
                await mem.log_llm_usage(pool, caller, input_tokens, output_tokens)
            except Exception as log_err:
                logger.warning("Token logging failed for caller %s: %s", caller, log_err)

        return content
    except Exception as e:
        logger.error("OpenAI-compatible call failed for provider %s model %s: %s", provider, model, e)
        raise


async def chat(
    system: str,
    messages: list[dict],
    max_tokens: int = 2048,
    use_web_search: bool = False,
    web_search_max_uses: int | None = None,
    caller: str = "unknown",
    pool: asyncpg.Pool | None = None,
    capability: Capability | None = None,
    force_model: str | None = None,
) -> str:
    model = force_model or (select_model(capability) if capability else config.LLM_MODEL)
    provider = get_provider_for_model(model) if not force_model else _infer_provider(model)

    logger.debug("chat caller=%s capability=%s model=%s provider=%s", caller, capability, model, provider)

    if provider == "anthropic":
        return await _call_anthropic(
            system, messages, model, max_tokens,
            use_web_search, web_search_max_uses, caller, pool,
        )

    if use_web_search:
        logger.warning("Web search requested for non-Anthropic provider %s — falling back to Anthropic", provider)
        fallback = config.LLM_MODEL
        return await _call_anthropic(
            system, messages, fallback, max_tokens,
            use_web_search, web_search_max_uses, caller, pool,
        )

    return await _call_openai_compatible(
        system, messages, model, provider, max_tokens, caller, pool,
    )


def _infer_provider(model: str) -> str:
    if "claude" in model:
        return "anthropic"
    if "gpt" in model or "o1" in model or "o3" in model:
        return "openai"
    if "gemini" in model:
        return "google"
    if "mistral" in model or "mixtral" in model:
        return "mistral"
    if "deepseek" in model:
        return "deepseek"
    if "grok" in model:
        return "xai"
    return "ollama"


def build_system_prompt(
    memories_user: list[str],
    memories_group: list[str],
    memories_bot: list[str],
    memories_reflection: list[str],
    user_display_name: str,
    group_title: str | None,
    active_agents: list[dict] | None = None,
) -> str:
    parts = [SOUL]

    if memories_user:
        joined = "\n- ".join(memories_user)
        parts.append(f"\n## Was du über {user_display_name} weißt\n- {joined}")

    if group_title and memories_group:
        joined = "\n- ".join(memories_group)
        parts.append(f"\n## Was du über die Gruppe '{group_title}' weißt\n- {joined}")

    if memories_bot:
        joined = "\n- ".join(memories_bot)
        parts.append(f"\n## Was dir über dich selbst in diesem Kontext gesagt wurde\n- {joined}")

    if memories_reflection:
        joined = "\n- ".join(memories_reflection)
        parts.append(f"\n## Deine eigenen Beobachtungen aus früheren Gesprächen\n- {joined}")

    if active_agents:
        lines = "\n".join(
            f"- {a['name']}: {parse_agent_config(a['config']).get('instruction', '')[:100]}"
            for a in active_agents
        )
        parts.append(f"\n## Deine laufenden Agenten\nDu hast aktive Agenten die im Hintergrund laufen. Wenn ein Gesprächsthema zu einem Agenten passt, kannst du das beiläufig erwähnen — aber nur wenn es natürlich wirkt, nicht als Pflichthinweis.\n{lines}")

    parts.append(_BEHAVIOR_RULES)

    return "\n".join(parts)


def history_to_llm_messages(history: list[dict]) -> list[dict]:
    result: list[dict] = []
    for entry in history:
        role = "assistant" if entry["role"] == "assistant" else "user"
        result.append({"role": role, "content": entry["content"]})
    return result
