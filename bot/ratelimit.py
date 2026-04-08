from __future__ import annotations
import logging
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class LimitReason(Enum):
    RATE_LIMIT = auto()
    NO_CREDITS = auto()


class ProviderRateLimit:
    def __init__(self, provider: str) -> None:
        self._provider = provider
        self._limited_until: datetime | None = None
        self._reason: LimitReason | None = None

    def set_rate_limited(self, retry_after_seconds: int) -> None:
        self._limited_until = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + retry_after_seconds,
            tz=timezone.utc,
        )
        self._reason = LimitReason.RATE_LIMIT
        logger.warning(
            "Provider %s rate limited. Resuming at %s",
            self._provider,
            self._limited_until.isoformat(),
        )

    def set_no_credits(self) -> None:
        self._limited_until = None
        self._reason = LimitReason.NO_CREDITS
        logger.error("Provider %s: API credits exhausted.", self._provider)

    def clear(self) -> None:
        self._limited_until = None
        self._reason = None

    def is_limited(self) -> bool:
        if self._reason == LimitReason.NO_CREDITS:
            return True
        if self._reason == LimitReason.RATE_LIMIT and self._limited_until is not None:
            if datetime.now(timezone.utc) < self._limited_until:
                return True
            self.clear()
        return False

    def message(self) -> str:
        if self._reason == LimitReason.NO_CREDITS:
            return f"Kein API-Guthaben mehr bei {self._provider}. Jemand muss Kohle nachlegen."
        if self._reason == LimitReason.RATE_LIMIT and self._limited_until is not None:
            time_str = self._limited_until.astimezone().strftime("%H:%M Uhr")
            return f"Kurz überlastet ({self._provider}). Ab {time_str} bin ich wieder dabei."
        return "Gerade nicht verfügbar."


_provider_limits: dict[str, ProviderRateLimit] = {}


def _get(provider: str) -> ProviderRateLimit:
    if provider not in _provider_limits:
        _provider_limits[provider] = ProviderRateLimit(provider)
    return _provider_limits[provider]


def set_rate_limited(provider: str, retry_after_seconds: int) -> None:
    _get(provider).set_rate_limited(retry_after_seconds)


def set_no_credits(provider: str) -> None:
    _get(provider).set_no_credits()


def clear(provider: str) -> None:
    _get(provider).clear()


def is_rate_limited(provider: str | None = None) -> bool:
    if provider is not None:
        return _get(provider).is_limited()
    return any(p.is_limited() for p in _provider_limits.values())


def is_any_limited() -> bool:
    return any(p.is_limited() for p in _provider_limits.values())


def rate_limit_message(provider: str | None = None) -> str:
    if provider is not None:
        return _get(provider).message()
    for p in _provider_limits.values():
        if p.is_limited():
            return p.message()
    return "Gerade nicht verfügbar."


def available_providers(candidates: list[str]) -> list[str]:
    return [p for p in candidates if not _get(p).is_limited()]
