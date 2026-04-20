from __future__ import annotations
import logging
import httpx

logger = logging.getLogger(__name__)

STT_URL = "http://stt:8001/transcribe"
TTS_URL = "http://tts:8002/synthesize"


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
