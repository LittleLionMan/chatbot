from __future__ import annotations
import logging
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class LimitReason(Enum):
    RATE_LIMIT = auto()
    NO_CREDITS = auto()


_rate_limited_until: datetime | None = None
_limit_reason: LimitReason | None = None


def set_rate_limited(retry_after_seconds: int) -> None:
    global _rate_limited_until, _limit_reason
    _rate_limited_until = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + retry_after_seconds,
        tz=timezone.utc,
    )
    _limit_reason = LimitReason.RATE_LIMIT
    logger.warning("Rate limit hit. Resuming at %s", _rate_limited_until.isoformat())


def set_no_credits() -> None:
    global _rate_limited_until, _limit_reason
    _rate_limited_until = None
    _limit_reason = LimitReason.NO_CREDITS
    logger.error("API credits exhausted.")


def clear() -> None:
    global _rate_limited_until, _limit_reason
    _rate_limited_until = None
    _limit_reason = None


def is_rate_limited() -> bool:
    if _limit_reason == LimitReason.NO_CREDITS:
        return True
    if _limit_reason == LimitReason.RATE_LIMIT and _rate_limited_until is not None:
        if datetime.now(timezone.utc) < _rate_limited_until:
            return True
        clear()
    return False


def rate_limit_message() -> str:
    if _limit_reason == LimitReason.NO_CREDITS:
        return "Kein API-Guthaben mehr. Jemand muss Kohle nachlegen."
    if _limit_reason == LimitReason.RATE_LIMIT and _rate_limited_until is not None:
        time_str = _rate_limited_until.astimezone().strftime("%H:%M Uhr")
        return f"Kurz überlastet. Ab {time_str} bin ich wieder dabei."
    return "Gerade nicht verfügbar."
