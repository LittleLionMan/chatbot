from __future__ import annotations
import logging
import os
import httpx
import anthropic
import asyncpg
from bot import config, ratelimit
from bot.models import (
    Capability,
    CAPABILITY_BALANCED,
    get_provider_for_model,
    get_max_output_tokens,
    select_model,
)
from bot.soul import SOUL, BEHAVIOR_RULES as _BEHAVIOR_RULES
from bot.utils import parse_agent_config

logger = logging.getLogger(__name__)


class ProviderRateLimitError(Exception):
    def __init__(self, provider: str, retry_after: int = 3600) -> None:
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limit hit for provider {provider}")


class ProviderAuthError(Exception):
    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Auth error for provider {provider}")


_anthropic_client: anthropic.AsyncAnthropic | None = None

_WEB_SEARCH_TOOL_BASE: dict = {
    "type": "web_search_20250305",
    "name": "web_search",
}

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
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


def _infer_provider(model: str) -> str:
    if "claude" in model:
        return "anthropic"
    if "gpt" in model or model.startswith("o1") or model.startswith("o3"):
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
                    pool, caller,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    model=model,
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
        raise ProviderRateLimitError("anthropic", retry_after) from e
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise ProviderAuthError("anthropic") from e


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

    if provider == "ollama":
        base_url = config.OLLAMA_BASE_URL + "/v1"
        api_key = "ollama"
    else:
        base_url = _PROVIDER_BASE_URLS[provider]
        env_key = _PROVIDER_ENV_KEYS.get(provider, "")
        api_key = os.getenv(env_key, "")

    payload: dict = {
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

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", 3600))
            raise ProviderRateLimitError(provider, retry_after)

        if resp.status_code in (401, 403):
            raise ProviderAuthError(provider)

        resp.raise_for_status()
        data = resp.json()
        content: str = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})

        if pool is not None:
            try:
                await mem.log_llm_usage(
                    pool, caller,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    model=model,
                )
            except Exception as log_err:
                logger.warning("Token logging failed for caller %s: %s", caller, log_err)

        return content

    except (ProviderRateLimitError, ProviderAuthError):
        raise
    except Exception as e:
        logger.error("OpenAI-compatible call failed for provider %s model %s: %s", provider, model, e)
        raise


async def chat(
    system: str,
    messages: list[dict],
    max_tokens: int | None = None,
    use_web_search: bool = False,
    web_search_max_uses: int | None = None,
    caller: str = "unknown",
    pool: asyncpg.Pool | None = None,
    capability: Capability | None = None,
    force_model: str | None = None,
    search_queries: list[str] | None = None,
) -> str:
    model = force_model or (select_model(capability) if capability else select_model(CAPABILITY_BALANCED))
    provider = get_provider_for_model(model) if not force_model else _infer_provider(model)
    resolved_max_tokens = max_tokens or get_max_output_tokens(capability or CAPABILITY_BALANCED)

    logger.debug("chat caller=%s capability=%s model=%s provider=%s max_tokens=%d", caller, capability, model, provider, resolved_max_tokens)

    if use_web_search:
        from bot import search as _search
        searxng_available = await _search.is_available()

        if searxng_available:
            augmented_messages = await _inject_search_results(messages, search_queries, _search)
            try:
                if provider == "anthropic":
                    return await _call_anthropic(
                        system, augmented_messages, model, resolved_max_tokens,
                        False, None, caller, pool,
                    )
                return await _call_openai_compatible(
                    system, augmented_messages, model, provider, resolved_max_tokens, caller, pool,
                )
            except (ProviderRateLimitError, ProviderAuthError):
                raise
        else:
            logger.warning("SearXNG not available — falling back to Anthropic native search for caller %s", caller)
            from bot.models import _available_models
            anthropic_candidates = [
                m for m in _available_models
                if m["provider"] == "anthropic" and (capability is None or capability in m["capabilities"])
            ]
            if anthropic_candidates:
                anthropic_candidates.sort(key=lambda m: m["input_cost_per_mtok"])
                model = anthropic_candidates[0]["api_model_name"]
            else:
                model = "claude-sonnet-4-6"
            provider = "anthropic"

    try:
        if provider == "anthropic":
            return await _call_anthropic(
                system, messages, model, resolved_max_tokens,
                use_web_search and not (await _searxng_available_cached()), web_search_max_uses, caller, pool,
            )
        return await _call_openai_compatible(
            system, messages, model, provider, resolved_max_tokens, caller, pool,
        )

    except ProviderRateLimitError as e:
        ratelimit.set_rate_limited(e.provider, e.retry_after)
        raise

    except ProviderAuthError as e:
        ratelimit.set_no_credits(e.provider)
        raise


_searxng_available: bool | None = None


async def _searxng_available_cached() -> bool:
    global _searxng_available
    if _searxng_available is None:
        from bot import search as _search
        _searxng_available = await _search.is_available()
    return _searxng_available


_QUERY_EXTRACTION_SYSTEM = """Extrahiere aus dieser Nutzernachricht 1-3 optimierte Suchanfragen.
Antworte NUR mit einem JSON-Array von Strings, kein anderer Text, keine Markdown-Backticks.
Optimiere die Queries für maximale Relevanz: präzise Begriffe, relevante Qualifier, aktuelle Jahreszahl wenn zeitkritisch.
Beispiele:
"Wie ist das Wetter in Berlin?" → ["Wetter Berlin aktuell"]
"Was kostet eine RTX 4090 gerade?" → ["RTX 4090 Preis 2026"]
"Neueste Nachrichten zu OpenAI" → ["OpenAI News April 2026"]
"Vergleich Python vs JavaScript für Backend" → ["Python vs JavaScript Backend Performance Vergleich 2026"]"""


async def _extract_search_queries(text: str) -> list[str]:
    from bot import brain as _brain
    from bot.models import CAPABILITY_FAST
    import json
    try:
        raw = await _brain.chat(
            system=_QUERY_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=100,
            capability=CAPABILITY_FAST,
            caller="search_query_extractor",
        )
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, str) and q.strip()][:3]
        return [text[:100]]
    except Exception:
        return [text[:100]]
    messages: list[dict],
    search_queries: list[str] | None,
    search_module,
) -> list[dict]:
    if not messages:
        return messages

    last_message = messages[-1]
    user_text: str = ""
    if isinstance(last_message.get("content"), str):
        user_text = last_message["content"]
    elif isinstance(last_message.get("content"), list):
        for block in last_message["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                user_text = block.get("text", "")
                break

    queries = search_queries or await _extract_search_queries(user_text)
    all_results: list[str] = []
    for query in queries:
        if not query.strip():
            continue
        result = await search_module.search(query)
        if result:
            all_results.append(f"Suchanfrage: {query}\n\n{result}")

    if not all_results:
        return messages

    search_context = "\n\n---\n\n".join(all_results)
    injection = (
        f"\n\n[Suchergebnisse aus dem Internet — der User wurde bereits informiert dass du suchst. "
        f"Antworte direkt mit dem Ergebnis, keine Einleitung wie 'ich habe gesucht' oder 'lass mich nachsehen'.]\n\n{search_context}"
    )

    augmented = list(messages[:-1])
    last = dict(last_message)
    if isinstance(last.get("content"), str):
        last["content"] = last["content"] + injection
    elif isinstance(last.get("content"), list):
        last["content"] = list(last["content"]) + [{"type": "text", "text": injection}]
    augmented.append(last)
    return augmented


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
        parts.append(
            f"\n## Deine laufenden Agenten\nDu hast aktive Agenten die im Hintergrund laufen. "
            f"Wenn ein Gesprächsthema zu einem Agenten passt, kannst du das beiläufig erwähnen — "
            f"aber nur wenn es natürlich wirkt, nicht als Pflichthinweis.\n{lines}"
        )

    parts.append(_BEHAVIOR_RULES)
    return "\n".join(parts)


def history_to_llm_messages(history: list[dict]) -> list[dict]:
    result: list[dict] = []
    for entry in history:
        role = "assistant" if entry["role"] == "assistant" else "user"
        result.append({"role": role, "content": entry["content"]})
    return result
