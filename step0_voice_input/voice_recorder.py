"""
Step 0 – Voice Recorder & Transcriber
=======================================
Records audio from microphone (or loads a .wav file), applies audio
normalisation and dB adjustment, then transcribes with OpenAI Whisper.

Can run standalone:
    python -m step0_voice_input.run                    # record from mic
    python -m step0_voice_input.run --wav path/to.wav  # transcribe existing file

Input  : microphone audio  OR  .wav file
Output : JSON  {"text": "...", "date_time": "...", "UID": "..."}
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _log(*args, **kwargs):
    """Print only when developer mode is active."""
    if config.DEV_MODE:
        print(*args, **kwargs)

# ── Audio settings ───────────────────────────────────────────────────────────
SAMPLE_RATE = config.WHISPER_SAMPLE_RATE
RECORD_SECONDS = config.WHISPER_RECORD_SECONDS
TARGET_DB = config.WHISPER_TARGET_DB


# ── Audio processing helpers ────────────────────────────────────────────────

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalise audio to [-1.0, 1.0] range.

    This prevents clipping and ensures consistent input levels for Whisper
    regardless of microphone gain.
    """
    _log("   🔊 Normalising audio (peak normalisation) …")
    peak = np.max(np.abs(audio))
    if peak == 0:
        _log("   ⚠️  Audio is silent – nothing to normalise.")
        return audio
    normalised = audio / peak
    _log(f"   ✅ Peak was {peak:.4f} → normalised to 1.0")
    return normalised


def adjust_db(audio: np.ndarray, target_db: float = TARGET_DB) -> np.ndarray:
    """
    Adjust the audio level to a target dB (RMS-based).

    This ensures the recording sits at a comfortable loudness for
    transcription, boosting quiet recordings and attenuating loud ones.

    Parameters
    ----------
    audio : np.ndarray
        Audio samples in float range [-1.0, 1.0].
    target_db : float
        Target RMS level in dB (e.g. -20 dB is a good default).
    """
    _log(f"   🎚️  Adjusting audio level to {target_db} dB RMS …")
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        _log("   ⚠️  Audio RMS is 0 – skipping dB adjustment.")
        return audio

    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    adjusted = audio * gain_linear

    # soft-clip to prevent any values exceeding [-1, 1]
    adjusted = np.clip(adjusted, -1.0, 1.0)

    _log(f"   ✅ Current RMS: {current_db:.1f} dB → applied {gain_db:+.1f} dB gain")
    return adjusted


def record_audio(duration: int = RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Record audio from the default microphone.

    Parameters
    ----------
    duration : int
        Recording duration in seconds.
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray – mono audio samples as float32.
    """
    import sounddevice as sd

    print(f"\n🎙️  [Step 0] Recording for {duration} seconds (Ctrl+C to stop early) …")
    _log(f"   Sample rate: {sample_rate} Hz | Channels: 1 (mono)")
    print("   🔴 Recording …")

    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except KeyboardInterrupt:
        import sounddevice as sd
        sd.stop()
        # trim to what was actually recorded
        print("\n   ⏹️  Recording stopped early by user.")

    audio = audio.flatten()
    actual_duration = len(audio) / sample_rate
    _log(f"   ✅ Recorded {actual_duration:.1f}s of audio ({len(audio):,} samples)")

    # Explicitly terminate PortAudio before returning so its internal threads
    # and semaphores are released.  Without this, FAISS/OpenMP (loaded later in
    # the pipeline) conflicts with PortAudio's OS-level resources on macOS,
    # causing a segfault.
    try:
        sd._terminate()
    except Exception:
        pass

    return audio


def load_wav(wav_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load a .wav file and return (audio_float32, sample_rate).

    Handles both int16 and float32 wav files.
    """
    wav_path = Path(wav_path)
    _log(f"\n📂 [Step 0] Loading WAV file: {wav_path}")

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    sr, audio = wavfile.read(str(wav_path))
    _log(f"   Sample rate: {sr} Hz | Samples: {len(audio):,} | Duration: {len(audio)/sr:.1f}s")

    # convert to float32 if needed
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
        _log("   Converted int16 → float32")
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
        _log("   Converted int32 → float32")
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        _log("   Converted stereo → mono")

    return audio, sr


def transcribe(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Transcribe audio using OpenAI Whisper (local model).

    Parameters
    ----------
    audio : np.ndarray
        Audio samples as float32.
    sample_rate : int
        Sample rate of the audio.

    Returns
    -------
    str – transcribed text.
    """
    import whisper

    model_name = config.WHISPER_MODEL
    print(f"\n🤖 [Step 0] Transcribing with Whisper (model: {model_name}) …")

    model = whisper.load_model(model_name)

    # Whisper expects 16 kHz – resample if needed
    if sample_rate != 16000:
        _log(f"   Resampling {sample_rate} Hz → 16000 Hz …")
        from scipy.signal import resample
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = resample(audio, num_samples).astype(np.float32)

    result = model.transcribe(audio, fp16=False)
    text = result["text"].strip()
    print(f"   ✅ Transcription: \"{text}\"")
    return text


def process_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    uid: str = "PLACEHOLDER_UserID",
) -> dict:
    """
    Full audio processing pipeline: normalise → dB adjust → transcribe → JSON.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio samples.
    sample_rate : int
        Sample rate.
    uid : str
        User identifier.

    Returns
    -------
    dict with keys "text", "date_time", "UID".
    """
    # normalise
    audio = normalize_audio(audio)

    # dB adjustment
    audio = adjust_db(audio, target_db=TARGET_DB)

    # transcribe
    text = transcribe(audio, sample_rate=sample_rate)

    # build output JSON
    output = {
        "text": text,
        "date_time": datetime.now().isoformat(),
        "UID": uid,
    }

    return output


# ── Convenience wrappers ────────────────────────────────────────────────────

def record_and_transcribe(
    duration: int = RECORD_SECONDS,
    uid: str = "PLACEHOLDER_UserID",
) -> dict:
    """Record from microphone and return transcription JSON."""
    audio = record_audio(duration=duration)
    return process_audio(audio, sample_rate=SAMPLE_RATE, uid=uid)


def transcribe_wav(wav_path: str | Path, uid: str = "PLACEHOLDER_UserID") -> dict:
    """Load a .wav file and return transcription JSON."""
    audio, sr = load_wav(wav_path)
    return process_audio(audio, sample_rate=sr, uid=uid)
