import io
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image
from cscience.features.api.config.config_mode import ConfigMode
from cscience.features.asr_whisper import AsrWhisperConnector
from cscience.features.asr_whisper.asr_config import AsrConfig
from cscience.features.asr_whisper.asr_whisper_datatypes.whisper_transcription import WhisperTranscriptionData
from cscience.features.clip import ClipConnector
from cscience.features.clip.clip_config import ClipConfig

_connector: AsrWhisperConnector | None = None


def initialize_once(config_path: str, unified_config: bool) -> None:
    AsrConfig.set_default_config_directory(config_path)
    global _connector
    if _connector is not None:
        return
    mode = ConfigMode.UNIFIED_CONFIG if unified_config else ConfigMode.CONFIG_PER_FEATURE
    _connector = AsrWhisperConnector(AsrConfig(mode=mode))


def _get_connector() -> AsrWhisperConnector:
    if _connector is None:
        raise RuntimeError(
            "The ASR Whisper connector has not been initialized. "
            "Call initialize_once() first."
        )
    return _connector


def get_feature_info() -> str:
    data = _get_connector().get_feature_info()
    return json.dumps(asdict(data), default=vars)


def get_service_info() -> str:
    data = _get_connector().get_service_info()
    return json.dumps(asdict(data), default=vars)


def transcribe(audio: bytes) -> WhisperTranscriptionData:
    return _get_connector().transcribe_audio_bytes(audio)
