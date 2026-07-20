import io
import json
from dataclasses import asdict

from PIL import Image
from cscience.features.api.config.config_mode import ConfigMode
from cscience.features.clip import ClipConnector
from cscience.features.clip.clip_config import ClipConfig

_connector: ClipConnector | None = None


def initialize_once(config_path: str, unified_config: bool) -> None:
    global _connector
    if _connector is not None:
        return
    mode =  ConfigMode.CONFIG_PER_FEATURE if unified_config else ConfigMode.CONFIG_PER_FEATURE
    _connector = ClipConnector(ClipConfig(config_path=config_path, mode=mode))

def _get_connector() -> ClipConnector:
    if _connector is None:
        raise RuntimeError(
            "The CLIP connector has not been initialized. "
            "Call initialize_once() first."
        )
    return _connector

def get_feature_info() -> str:
    data = _get_connector().get_feature_info()
    return json.dumps(asdict(data),default=vars)

def get_service_info() -> str:
    data = _get_connector().get_service_info()
    return json.dumps(asdict(data), default=vars)

def embed_text(text: str) -> list[float]:
    return _get_connector().text(text)

def embed_image(encoded_image_bytes: bytes) -> list[float]:
    image = Image.open(io.BytesIO(encoded_image_bytes)).convert("RGB")
    return _get_connector().image(image)
