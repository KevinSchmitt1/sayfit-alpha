"""
POST /transcribe
================
Accepts an uploaded audio file (WebM, Opus, WAV, MP3, etc.) and returns the
transcribed text using the local OpenAI Whisper model.

Recording stays in the browser (MediaRecorder API). This endpoint handles
transcription only — sounddevice / PortAudio are never touched.
"""

import tempfile
from pathlib import Path

import fastapi
from fastapi import File, UploadFile

import config
from api.schemas import TranscribeResponse

router = fastapi.APIRouter()

# ── Lazy model singleton ──────────────────────────────────────────────────────
# Loaded on the first /transcribe request, cached for the container's lifetime.
# Not loaded at startup so that unit tests that never hit /transcribe are unaffected.

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)
    return _whisper_model


# ── Validation allowlists ─────────────────────────────────────────────────────

_ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/webm",
    "audio/ogg",
    "audio/opus",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
    "application/octet-stream",  # browser blobs are often sent with this type
}

_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".opus", ".m4a", ".mp4", ".flac"}


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file with Whisper.

    Accepts any format supported by ffmpeg (WebM, Opus, WAV, MP3, etc.).
    Returns the transcribed text stripped of leading/trailing whitespace.
    """
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise fastapi.HTTPException(
            status_code=422,
            detail=f"Unsupported content type '{file.content_type}'. Expected audio.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise fastapi.HTTPException(status_code=422, detail="Uploaded file is empty.")

    # Determine a safe file extension for ffmpeg format detection.
    # Only use the client-supplied extension if it is in the allowlist.
    suffix = ".webm"  # safe default — MediaRecorder output on Chrome/Firefox
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext in _ALLOWED_EXTENSIONS:
            suffix = ext

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = _get_whisper_model()
        result = model.transcribe(tmp_path, fp16=False)
        text = result["text"].strip()
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return TranscribeResponse(text=text)
