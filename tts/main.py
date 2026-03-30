import io
import logging
import wave
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from piper.voice import PiperVoice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOICES: dict[str, PiperVoice] = {}

DE_MODEL = "/app/voices/de_DE-thorsten-medium.onnx"
EN_MODEL = "/app/voices/en_US-lessac-medium.onnx"

logger.info("Loading TTS voices...")
VOICES["de"] = PiperVoice.load(DE_MODEL)
VOICES["en"] = PiperVoice.load(EN_MODEL)
logger.info("TTS voices loaded.")

app = FastAPI()


class TTSRequest(BaseModel):
    text: str
    language: str = "de"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "voices": list(VOICES.keys())}


@app.post("/synthesize")
def synthesize(req: TTSRequest) -> Response:
    lang = req.language if req.language in VOICES else "de"
    voice = VOICES[lang]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize(req.text, wav_file)

    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
