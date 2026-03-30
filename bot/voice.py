from __future__ import annotations
import logging
import httpx
from bot import brain

logger = logging.getLogger(__name__)

STT_URL = "http://stt:8001/transcribe"
TTS_URL = "http://tts:8002/synthesize"

_VOICE_REQUEST_SYSTEM = """Antworte NUR mit "ja" oder "nein".
Fordert der User in seiner Nachricht explizit eine Sprachantwort?
Beispiele: "kannst du das vorlesen", "antworte als Sprachnachricht", "red mal mit mir", "sag mir" → ja
Alles andere → nein"""


async def parse_voice_request(text: str) -> bool:
    try:
        result = await brain.chat(
            system=_VOICE_REQUEST_SYSTEM,
            messages=[{"role": "user", "content": text}],
            max_tokens=3,
        )
        return result.strip().lower().startswith("ja")
    except Exception as e:
        logger.warning("Voice request detection failed: %s", e)
        return False


async def transcribe(audio_bytes: bytes) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            STT_URL,
            files={"file": ("audio.ogg", audio_bytes, "audio/ogg")},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["text"], data.get("language", "de")


async def synthesize(text: str, language: str = "de") -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            TTS_URL,
            json={"text": text, "language": language},
        )
        resp.raise_for_status()
        return resp.content
