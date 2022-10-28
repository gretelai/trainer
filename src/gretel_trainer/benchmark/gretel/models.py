from pathlib import Path
from typing import Dict, Union

from gretel_client.projects.models import read_model_config

GretelModelConfig = Union[str, Path, Dict]


def _config(blueprint: str, class_name: str) -> Dict:
    config = read_model_config(blueprint)
    config["name"] = f"benchmark-{class_name}"
    return config


class GretelModel:
    config: GretelModelConfig

    @property
    def name(self) -> str:
        return type(self).__name__


class GretelLSTM(GretelModel):
    config = _config("synthetics/default", "GretelLSTM")


class GretelCTGAN(GretelModel):
    config = _config("synthetics/high-dimensionality", "GretelCTGAN")


class GretelAuto(GretelModel):
    config = "AUTO"


class GretelGPTX(GretelModel):
    config = _config("synthetics/natural-language", "GretelGPTX")


class GretelAmplify(GretelModel):
    config = _config("synthetics/amplify", "GretelAmplify")
