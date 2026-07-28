import io
import json
from dataclasses import asdict

from PIL import Image
from cscience.features.api.config.config_mode import ConfigMode
from cscience.features.nsfw_image import NsfwImageConnector
from cscience.features.nsfw_image.nsfw_config import NsfwConfig
from cscience.features.nsfw_image.nsfw_image_datatypes.nsfw_prediction_data import NsfwPredictionData

_connector: NsfwImageConnector | None = None


def initialize_once(config_path: str, unified_config: bool) -> None:
    NsfwConfig.set_default_config_directory(config_path)
    global _connector
    if _connector is not None:
        return
    mode =  ConfigMode.UNIFIED_CONFIG if unified_config else ConfigMode.CONFIG_PER_FEATURE
    _connector = NsfwImageConnector(NsfwConfig(mode=mode))

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

def classify_image(encoded_image_bytes: bytes) -> NsfwPredictionData:
    image = Image.open(io.BytesIO(encoded_image_bytes)).convert("RGB")
    return _get_connector().classify(image)