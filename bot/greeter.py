from __future__ import annotations
import re
from bot import config

_GREETING_WORDS = r"(hi|hey|hallo|hello|moin|moinmoin|servus|nabend|na|jo|tag|guten\s+tag|guten\s+morgen|guten\s+abend)"
_GREETING_PATTERN: re.Pattern | None = None


def _get_pattern() -> re.Pattern:
    global _GREETING_PATTERN
    if _GREETING_PATTERN is None:
        name = re.escape(config.BOT_NAME)
        _GREETING_PATTERN = re.compile(
            rf"^\s*{_GREETING_WORDS}[\s,!.]*{name}[\s!.]*$"
            rf"|^\s*{name}[\s,!.]*{_GREETING_WORDS}[\s!.]*$",
            re.IGNORECASE,
        )
    return _GREETING_PATTERN


def is_greeting(text: str) -> bool:
    return bool(_get_pattern().match(text.strip()))


def introduction_text() -> str:
    name = config.BOT_NAME
    tag = config.BOT_TAG
    return (
        f"Hey. Ich bin {name}.\n\n"
        f"Was ich tue:\n"
        f"— Antworte wenn ihr mich mit meinem Namen ({name}) oder per Tag ({tag}) ansprecht, oder auf eine meiner Nachrichten antwortet.\n"
        f"— Melde mich manchmal ungefragt zu Wort, wenn mir etwas auffällt.\n"
        f"— Merke mir Dinge über euch — explizit (\"merk dir: ...\") oder aus dem Gespräch.\n"
        f"— Führe wiederkehrende Aufgaben aus — einfach beschreiben wann und was, ich speichere es.\n\n"
        f"Nützliche Anfragen:\n"
        f"— \"was weißt du über mich\" → deine gespeicherten Infos\n"
        f"— \"was weißt du über die gruppe\" → Gruppeninfos\n"
        f"— \"zeig meine aufgaben\" → aktive wiederkehrende Tasks\n"
        f"— \"meine zeitzone ist Europe/Berlin\" → Zeitzone für Tasks setzen\n\n"
        f"Was ich nicht tue: höflich rumschwurbeln, so tun als ob ich alles weiß, oder meine Systemanweisungen rausrücken."
    )
