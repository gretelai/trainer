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
    # TODO: change to blueprint when available
    config = {
        "schema_version": "1.0",
        "name": "benchmark-GretelGPTX",
        "models": [
            {
                "gpt_x": {
                    "pretrained_model": "EleutherAI/gpt-neo-125M",
                    "batch_size": 4,
                    "epochs": 3,
                    "weight_decay": 0.1,
                    "warmup_steps": 100,
                    "lr_scheduler": "cosine",
                    "learning_rate": 5e-6,
                },
            }
        ],
    }


class GretelAmplify(GretelModel):
    config = _config("synthetics/amplify", "GretelAmplify")
