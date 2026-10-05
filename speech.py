"""
speech.py
----------
Azure Cognitive Services Speech integration.

Converts a text summary of extracted label fields to spoken audio
(MP3/WAV bytes) using Azure Text-to-Speech.

Set in .env:
    AZURE_SPEECH_KEY     Your Speech resource subscription key
    AZURE_SPEECH_REGION  e.g. eastus, westeurope
    AZURE_SPEECH_VOICE   optional; default en-US-JennyNeural
"""

import io
import os

_ENABLED = None  # lazy: None = not yet checked


def _check_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = bool(
            os.getenv("AZURE_SPEECH_KEY") and os.getenv("AZURE_SPEECH_REGION")
        )
    return _ENABLED


def is_available() -> bool:
    """Return True when the Speech SDK env vars are configured."""
    return _check_enabled()


def fields_to_speech_text(fields: dict) -> str:
    """Build a natural-language summary of the extracted label fields."""
    LABELS = {
        "manufacturer": "Manufacturer",
        "model": "Model",
        "serial_number": "Serial number",
        "voltage": "Voltage",
        "frequency": "Frequency",
        "power": "Power",
        "manufacture_date": "Manufacture date",
    }
    parts = []
    for key, label in LABELS.items():
        value = fields.get(key)
        if value:
            parts.append(f"{label}: {value}")

    # Include any 'other' sub-fields
    other = fields.get("other")
    if isinstance(other, dict):
        for k, v in other.items():
            if v:
                parts.append(f"{k.replace('_', ' ').title()}: {v}")

    if not parts:
        return "No label fields were detected."

    return "Label summary. " + ". ".join(parts) + "."


def synthesise(text: str, voice: str | None = None) -> bytes:
    """
    Convert *text* to audio bytes using Azure TTS.

    Returns
    -------
    bytes
        Raw audio data (format determined by the SDK default — typically WAV).

    Raises
    ------
    RuntimeError
        If the Speech SDK is not available or synthesis fails.
    ImportError
        If the azure-cognitiveservices-speech package is not installed.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "azure-cognitiveservices-speech is not installed. "
            "Run: pip install azure-cognitiveservices-speech"
        ) from exc

    key = os.environ.get("AZURE_SPEECH_KEY", "")
    region = os.environ.get("AZURE_SPEECH_REGION", "")
    selected_voice = voice or os.getenv("AZURE_SPEECH_VOICE", "en-US-JennyNeural")

    if not key or not region:
        raise RuntimeError(
            "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env "
            "to use text-to-speech."
        )

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = selected_voice

    # Synthesise to an in-memory stream
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
    pull_stream = speechsdk.audio.PullAudioOutputStream()
    audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(
            f"Azure Speech synthesis cancelled: {details.reason}. "
            f"Error details: {details.error_details}"
        )
    else:
        raise RuntimeError(f"Azure Speech synthesis failed with reason: {result.reason}")
