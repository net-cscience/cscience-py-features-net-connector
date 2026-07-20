import io
import json
from dataclasses import asdict

from PIL import Image
from cscience.features.api.config.config_mode import ConfigMode
from cscience.features.nsfw_image import NsfwImageConnector
from cscience.features.nsfw_image.nsfw_config import NsfwConfig

_connector: NsfwImageConnector | None = None


def initialize_once(config_path: str, unified_config: bool) -> None:
    global _connector
    if _connector is not None:
        return
    mode =  ConfigMode.CONFIG_PER_FEATURE if unified_config else ConfigMode.CONFIG_PER_FEATURE
    _connector = NsfwImageConnector(NsfwConfig(config_path=config_path, mode=mode))

def _get_connector() -> NsfwImageConnector:
    if _connector is None:
        raise RuntimeError(
            "The NSFW connector has not been initialized. "
            "Call initialize_once() first."
        )
    return _connector

def get_feature_info() -> str:
    data = _get_connector().get_feature_info()
    return json.dumps(asdict(data),default=vars)

def get_service_info() -> str:
    data = _get_connector().get_service_info()
    return json.dumps(asdict(data), default=vars)

def embed_image(encoded_image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(encoded_image_bytes)).convert("RGB")
    return json.dumps(asdict(_get_connector().classify(image)), default=vars)
