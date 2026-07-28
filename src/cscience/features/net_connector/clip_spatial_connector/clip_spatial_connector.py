import io
import json
import os
from dataclasses import asdict
from pathlib import Path

from PIL import Image
from cscience.features.api.config.config_mode import ConfigMode
from cscience.features.api.datatypes.base.structural.spatial_vector_batch_data import SpatialVectorBatchData
from cscience.features.clip import ClipConnector
from cscience.features.clip.clip_config import ClipConfig
from cscience.features.clip_spatial import ClipSpatialConnector
from cscience.features.clip_spatial.clip_spatial_config import ClipSpatialConfig
from cscience.features.clip_spatial.clip_spatial_datatypes.spatial_score_vector_batch import SpatialScoreVectorBatch

_connector: ClipSpatialConnector | None = None


def initialize_once(config_path: str, unified_config: bool) -> None:
    ClipSpatialConfig.set_default_config_directory(config_path)

    mode = ConfigMode.UNIFIED_CONFIG if unified_config else ConfigMode.CONFIG_PER_FEATURE
    cfg = ClipSpatialConfig(config_path=config_path, mode=mode)

    global _connector
    if _connector is not None:
        return

    _connector = ClipSpatialConnector(cfg)

def _get_connector() -> ClipSpatialConnector:
    if _connector is None:
        raise RuntimeError(
            "The CLIP SPATIAL connector has not been initialized. "
            "Call initialize_once() first."
        )
    return _connector

def get_feature_info() -> str:
    data = _get_connector().get_feature_info()
    return json.dumps(asdict(data),default=vars)

def get_service_info() -> str:
    data = _get_connector().get_service_info()
    return json.dumps(asdict(data), default=vars)


def image_regions(encoded_image_bytes: bytes) ->  SpatialVectorBatchData[list[float]]:
    image = Image.open(io.BytesIO(encoded_image_bytes)).convert("RGB")
    return _get_connector().image_regions(image)

def score_regions(text: list[str], vectors:  SpatialVectorBatchData[list[float]]) -> SpatialScoreVectorBatch:
        return _get_connector().score_regions(text, vectors)